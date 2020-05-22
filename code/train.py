import os
import tensorflow as tf
import tensorflow_addons as tfa
import sys
from shutil import copy

from transformers import TFBertModel, BertConfig
import yaml
import argparse
import numpy as np
from data import *
from models import *
from losses import *


MODELS = { "bert-small": BertBasic, "lstm": LSTMBasic, "distil-bert": DistilBert}
DATASET = {"UniformSampler": UniformSampler, "UniformSamplerUnique": UniformSamplerUnique}
LOSS = {"sigmoid": sigmoid_loss, "cross_entropy": cross_entropy}

args = argparse.ArgumentParser()
args.add_argument('config_file')
args = args.parse_args()

config = yaml.safe_load(open(args.config_file,'r'))

model = MODELS[config['base_arch']](config)
loss_fn = LOSS[config['loss']]()
dataset_train = DATASET[config['dataset']['sampler']](config['dataset'])
dataset_val = DATASET[config['dataset']['sampler']](config['dataset'],train=False)

input_spec, padded_spec = dataset_train.get_spec()
data_train = tf.data.Dataset.from_generator(dataset_train.generator, output_types=dataset_train.get_output_types(),output_shapes=dataset_train.get_output_shapes())
data_train = data_train.padded_batch(config['batch_size'],padded_shapes=padded_spec, padding_values=(0.0,-1)).prefetch(config['prefetch_size'])
data_val = tf.data.Dataset.from_generator(dataset_val.generator, output_types=dataset_val.get_output_types(),output_shapes=dataset_val.get_output_shapes())
data_val = data_val.batch(1)
num_steps = int(dataset_train.get_len()/(config['batch_size']*config['dataset']['samples_per_instance']))
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=config['lr'], decay_steps=config['epochs']*num_steps, end_learning_rate=config['lr']/1000.0)
optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)


train_loss = tf.keras.metrics.Mean(name='train_loss')
metrics_train = loss_fn.get_metrics()['train']
metrics_val = loss_fn.get_metrics()['val']


if not os.path.isdir(config['ckpt_dir']):
    os.makedirs(config['ckpt_dir'])
copy(args.config_file, config['ckpt_dir'])


ckpt = tf.train.Checkpoint(optimizer=optimizer,model = model)
manager = tf.train.CheckpointManager(ckpt, config['ckpt_dir'], 
                    max_to_keep=config['max_to_keep'], 
                    keep_checkpoint_every_n_hours=1)
summary = tf.summary.create_file_writer(config['ckpt_dir'])

@tf.function(input_signature=input_spec)
def train_step(inputs_embeds, labels):
    with tf.GradientTape() as tape:
        output, attention_mask = model(inputs_embeds, training=True)
        loss = loss_fn(output, labels)

    gradients = zip(tape.gradient(loss, model.trainable_variables),model.trainable_variables)
    #grads = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(gradients)
    train_loss(loss)
    for metric in metrics_train:
        metric(labels, output, sample_weight=attention_mask)
    #return grads

@tf.function(input_signature=input_spec)
def val_step(inputs_embeds, labels):
    output, _ = model(inputs_embeds, training=False)
    output = tf.nn.softmax(output)
    for metric in metrics_val:
        metric(labels, output)
  


ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("restored from {}".format(manager.latest_checkpoint))
else:
    print("Initialising from scratch")

for epoch in range(config['epochs']):
    print(epoch)
    for idx, sample in enumerate(data_train):
        inputs_embeds = sample[0]
        labels = sample[1]

        train_step(inputs_embeds, labels)
        if idx%config['ckpt_steps'] == 0:
            for sample in data_val:
                val_step(sample[0], sample[1])
            path = manager.save(int(epoch*num_steps+idx))
            print("Saved ckpt for {}/{}: {}".format(epoch,idx,path))
            with summary.as_default():
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch + 1,
                            train_loss.result()),end=',')
                for metric in metrics_train:
                    print(metric.name.upper(), metric.result().numpy(),end=',')
                    tf.summary.scalar(metric.name, metric.result(), step=int(epoch*num_steps+idx))
                    metric.reset_states()
                for metric in metrics_val:
                    print(metric.name.upper(), metric.result().numpy(),end=',')
                    tf.summary.scalar(metric.name, metric.result(), step=int(epoch*num_steps+idx))
                    metric.reset_states()
                
                tf.summary.scalar('loss', train_loss.result(), step=int(epoch*num_steps+idx))
                train_loss.reset_states()
                for variable in model.trainable_variables:
                    if not 'embeddings' in variable.name:
                        tf.summary.histogram(variable.name,variable.value().numpy(), step = int(epoch*num_steps+idx))
                """
                for g,var in zip(grads, model.trainable_variables):
                    if (not 'embeddings' in var.name) and (not 'pooler' in var.name):
                        tf.summary.histogram("grad/{}".format(var.name), g.numpy(),step=int(epoch*num_steps+idx))                 
                """
                summary.flush()
                print()
                sys.stdout.flush()
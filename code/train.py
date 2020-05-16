import os
import tensorflow as tf
import tensorflow_addons as tfa

from transformers import TFBertModel, BertConfig
import yaml
import argparse
import numpy as np
from data import *
from models import *
from losses import *


MODELS = { "bert-small": BertBasic}
DATASET = {"UniformSampler": UniformSampler, "UniformSamplerUnique": UniformSamplerUnique}
LOSS = {"sigmoid": sigmoid_loss, "cross_entropy": cross_entropy}

args = argparse.ArgumentParser()
args.add_argument('config_file')
args = args.parse_args()

config = yaml.safe_load(open(args.config_file,'r'))

model = MODELS[config['base_arch']](config)
loss_fn = LOSS[config['loss']]()
dataset = DATASET[config['dataset']['sampler']](config['dataset'])
num_steps = int(DATASET[config['dataset']['sampler']].get_len()/(config['batch_size']*config['dataset']['samples_per_instance']))
input_spec, padded_spec = DATASET[config['dataset']['sampler']].get_spec()
dataset = dataset.padded_batch(config['batch_size'],padded_shapes=padded_spec, padding_values=(0.0,-1)).prefetch(config['prefetch_size'])


lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config['lr'], decay_steps=config['epochs']*num_steps, end_learning_rate=config['lr']/1000.0,
    )

optimizer = tf.optimizers.Adam(
        learning_rate=lr_schedule)


train_loss = tf.keras.metrics.Mean(name='train_loss')

metrics = loss_fn.get_metrics()

if not os.path.isdir(config['ckpt_dir']):
    os.makedirs(config['ckpt_dir'])


ckpt = tf.train.Checkpoint(optimizer=optimizer,model = model, dataset=dataset)
manager = tf.train.CheckpointManager(ckpt, config['ckpt_dir'], 
                    max_to_keep=config['max_to_keep'], 
                    keep_checkpoint_every_n_hours=1)
summary = tf.summary.create_file_writer(config['ckpt_dir'])

@tf.function(input_signature=input_spec)
def train_step(inputs_embeds, labels):
    with tf.GradientTape(persistent=True) as tape:
        output, attention_mask = model(inputs_embeds, training=True)
        loss, probs = loss_fn(output, labels)

    gradients = zip(tape.gradient(loss, model.trainable_variables),model.trainable_variables)
    #grads = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(gradients)
    train_loss(loss)
    for metric in metrics:
        metric(labels, probs, sample_weight=attention_mask)
    #return grads



for epoch in range(config['epochs']):
    print(epoch)
    for idx, sample in enumerate(dataset):
        inputs_embeds = sample[0]
        labels = sample[1]

        #tf.summary.trace_on(graph=True)
        train_step(inputs_embeds, labels)
        #with summary.as_default():
        #    tf.summary.trace_export("{}".format(epoch), step = int(epoch*num_steps+idx))
        if idx%config['ckpt_steps'] == 0:
            path = manager.save(int(epoch*num_steps+idx))
            print("Saved ckpt for {}/{}: {}".format(epoch,idx,path))
            with summary.as_default():
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch + 1,
                            train_loss.result()),end=',')
                for metric in metrics:
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

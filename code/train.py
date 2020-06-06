import os
import tensorflow as tf
import tensorflow_addons as tfa
import sys
from shutil import copy
import time
from transformers import TFBertModel, BertConfig
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from data import *
from models import *
from losses import *


tf.autograph.set_verbosity(10)

MODELS = { "bert-small": BertBasic, "lstm": LSTMBasic, "distil-bert": DistilBert,"distil-bert-norm": DistilBertNorm, "mlp": MLP}
DATASET = {"UniformSampler": UniformSampler, "UniformSamplerUnique": UniformSamplerUnique}
LOSS = {"sigmoid": sigmoid_loss, "cross_entropy": cross_entropy,"kl": kl, "kl_weighted": kl_weighted, "cross_entropy_weighted": cross_entropy_weighted}

args = argparse.ArgumentParser()
args.add_argument('config_file')
args = args.parse_args()

config = yaml.safe_load(open(args.config_file,'r'))



model = MODELS[config['base_arch']](config)
loss_fn = LOSS[config['loss']]()
dataset_train = DATASET[config['dataset']['sampler']](config['dataset'])
dataset_val = DATASET[config['dataset']['sampler']](config['dataset'],train=False)

input_spec, padded_spec, pad_val = dataset_train.get_spec()
data_train = tf.data.Dataset.from_generator(dataset_train.generator, output_types=dataset_train.get_output_types(),output_shapes=dataset_train.get_output_shapes())
data_train = data_train.shuffle(config['batch_size']*config['prefetch_size']).padded_batch(config['batch_size'],padded_shapes=padded_spec, padding_values=pad_val).prefetch(tf.data.experimental.AUTOTUNE)
data_val = tf.data.Dataset.from_generator(dataset_val.generator, output_types=dataset_val.get_output_types(),output_shapes=dataset_val.get_output_shapes())
data_val = data_val.padded_batch(config['batch_size'],padded_shapes=padded_spec, padding_values=pad_val).prefetch(tf.data.experimental.AUTOTUNE)

num_steps = int(len(dataset_train)/(config['num_accum_steps']*config['batch_size']*config['dataset']['samples_per_instance']))
warmup_fraction = config['warmup_fraction']
weight_decay = config['weight_decay']
optimizer = tfa.optimizers.RectifiedAdam(lr=config['lr'],beta_2=0.98,epsilon=1e-6,
                            weight_decay=weight_decay,total_steps=config['epochs']*num_steps,
                            warmup_proportion=warmup_fraction,min_lr=config['end_lr'])

train_loss = tf.keras.metrics.Mean(name='train_loss')
model_output = tf.keras.metrics.MeanTensor(name='model_output')
metrics_train = loss_fn.get_metrics()['train']
metrics_val = loss_fn.get_metrics()['val']


if not os.path.isdir(config['ckpt_dir']):
    os.makedirs(config['ckpt_dir'])
copy(args.config_file, config['ckpt_dir'])

ckpt  = tf.train.Checkpoint(optimizer=optimizer,model = model) 
manager = tf.train.CheckpointManager(ckpt, config['ckpt_dir'], 
                    max_to_keep=config['max_to_keep'], 
                    keep_checkpoint_every_n_hours=3)
summary = tf.summary.create_file_writer(config['ckpt_dir'])

@tf.function(input_signature=input_spec)
def train_step(inputs_embeds, labels):
    with tf.GradientTape() as tape:
        output, attention_mask = model(inputs_embeds, training=True)
        loss, output = loss_fn(output, labels)

    train_loss(loss)
    
    if config['dataset']['weighted']:
        labels = tf.cast(labels[:,:,:-1], tf.int32)
    labels = tf.squeeze(labels)
    for metric in metrics_train:
        metric(labels, output, sample_weight=attention_mask)
    return tape.gradient(loss, model.trainable_variables)

@tf.function(input_signature=input_spec)
def val_step(inputs_embeds, labels):
    output, attention_mask = model(inputs_embeds, training=False)
    output = tf.nn.softmax(output)
    for metric in metrics_val:
        metric(labels, output, attention_mask)
  
ckpt.restore(manager.latest_checkpoint) 
if manager.latest_checkpoint:
    print("restored from {}".format(manager.latest_checkpoint))
else:
    print("Initialising from scratch")


@tf.function(experimental_relax_shapes=True)
def accumulate_gradients(num_accumulated, gradients, accum_gradients):
    if num_accumulated == 0:
        return gradients
    else:
        arr = []
        for idx, grads in enumerate(gradients):
            if grads == None:
                arr.append(None)
            else:
                arr.append((num_accumulated*accum_gradients[idx] + grads)/(num_accumulated+1))
        return arr

@tf.function
def apply_grads(grads):
    gradients = zip(grads, model.trainable_variables)
    optimizer.apply_gradients(gradients)
    
accumulated_gradients = None        
steps = 0
num_accum_steps = config['num_accum_steps']
for epoch in tqdm(range(config['epochs'])):
    print(epoch)
    for sample in data_train:
        inputs_embeds = sample[0]
        labels = sample[1]
        grads = train_step(inputs_embeds, labels)
        accumulated_gradients = accumulate_gradients(steps%num_accum_steps,grads,accumulated_gradients)
        
        if steps%num_accum_steps == num_accum_steps-1:
            print(int(steps/num_accum_steps))
            apply_grads(accumulated_gradients)
       
        if steps%(num_accum_steps*config['ckpt_steps']) == num_accum_steps*config['ckpt_steps']-1 :
            path = manager.save(steps)
            print("Saved ckpt for {}/{}: {}".format(epoch,steps/(num_accum_steps),path))
            with summary.as_default():
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch + 1, train_loss.result()),end=',')
             
                #with tqdm(total=len(dataset_val)) as progress_bar:
                #    for val_sample in data_val:
                #        val_step(val_sample[0], val_sample[1])
                #        progress_bar.update(config['batch_size'])
 
                for metric in metrics_train:
                    print(metric.name.upper(), metric.result().numpy(),end=',')
                    tf.summary.scalar(metric.name, metric.result(), step=int(steps/(num_accum_steps)))
                    metric.reset_states()
                for metric in metrics_val:
                    print(metric.name.upper(), metric.result().numpy(),end=',')
                    tf.summary.scalar(metric.name, metric.result(), step=int(steps/(num_accum_steps)))
                    metric.reset_states()
                
                tf.summary.scalar('loss', train_loss.result(), step=int(steps/(num_accum_steps)))
                train_loss.reset_states()
                #tf.summary.histogram('model_output', model_output.result(), step=int(epoch*num_steps + idx))
                #model_output.reset_states()
                for variable in model.trainable_variables:
                    if not 'embeddings' in variable.name:
                        tf.summary.histogram(variable.name,variable.value().numpy(), step = int(steps/(num_accum_steps)))
                for g,var in zip(accumulated_gradients, model.trainable_variables):
                    if (not 'embeddings' in var.name) and (not 'pooler' in var.name):
                        tf.summary.histogram("grad/{}".format(var.name), g.numpy(),step=int(steps/(num_accum_steps)))                 
                summary.flush()
                print()
        sys.stdout.flush()
        steps+=1

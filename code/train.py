import tensorflow as tf
import tensorflow_addons as tfa

from transformers import TFBertModel, BertConfig
import yaml
import argparse
import numpy as np
from data import *
from models import *


MODELS = {
        "bert-small": BertBasic
        }
DATASET = {"UniformSampler": UniformSampler}

args = argparse.ArgumentParser()
args.add_argument('config_file')
args = args.parse_args()

config = yaml.load(open(args.config_file,'r'))

model = MODELS[config['base_arch']](config)

dataset = DATASET[config['dataset']['sampler']](config['dataset'])
num_steps = int(DATASET[config['dataset']['sampler']].get_len()/(config['batch_size']*config['dataset']['samples_per_instance']))
input_spec = DATASET[config['dataset']['sampler']].get_spec()
dataset = dataset.padded_batch(config['batch_size'],padded_shapes=([None, 2048],[None,513])).prefetch(config['prefetch_size'])


lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config['lr'], decay_steps=config['epochs']*num_steps, end_learning_rate=config['lr']/10,
    )

optimizer = tfa.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.0001)


train_loss = tf.keras.metrics.Mean(name='train_loss')
METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]

ckpt = tf.train.Checkpoint(optimizer=optimizer,model = model, dataset=dataset)
manager = tf.train.CheckpointManager(ckpt, config['ckpt_dir'], max_to_keep=10)
summary = tf.summary.create_file_writer(config['ckpt_dir'])

@tf.function(input_signature=input_spec)
def train_step(inputs_embeds, labels):
    with tf.GradientTape() as tape:
        attention_mask = tf.keras.layers.Masking()(inputs_embeds)._keras_mask
        output = model(inputs_embeds, attention_mask, training=True)
        attention_mask = tf.expand_dims(tf.cast(attention_mask,tf.float32),-1)
        labels = tf.cast(labels, tf.float32)
        mask = ((1.0-labels) + labels *(200.0))*attention_mask
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=output)*mask)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    probs = tf.sigmoid(output)

    train_loss(loss)
    for metric in METRICS:
        metric(labels, probs)


       
for epoch in range(config['epochs']):
    print(epoch)
    for idx, sample in enumerate(dataset):
        inputs_embeds = sample[0]
        labels = sample[1]
        train_step(inputs_embeds, labels)
      
        if idx%config['ckpt_steps'] == 0:
            path = manager.save()
            print("Saved ckpt for {}/{}: {}".format(epoch,idx,path))
            with summary.as_default():
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch + 1,
                            train_loss.result()),end=',')
                for metric in METRICS:
                    print(metric.name.upper(), metric.result().numpy(),end=',')
                    tf.summary.scalar(metric.name, metric.result(), step=int(epoch*num_steps+idx))
                    metric.reset_states()
                tf.summary.scalar('loss', train_loss.result(), step=int(epoch*num_steps+idx))
                train_loss.reset_states()
                for variable in model.trainable_variables:
                    #print(variable.name)
                    if not 'embeddings' in variable.name:
                        tf.summary.histogram(variable.name,variable.value().numpy(), step = epoch*num_steps+idx)
                        summary.flush()

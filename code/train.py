import tensorflow as tf
from transformers import TFBertModel, BertConfig
import yaml
import argparse

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
dataset = dataset.padded_batch(config['batch_size'],padded_shapes=([None, None],[None,None, None])).prefetch(config['prefetch_size'])

loss_fn = tf.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(lr=config['lr'])

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_acc')


@tf.function
def train_step(inputs_embeds, labels):
    with tf.GradientTape() as tape:
        output = model(inputs_embeds, training=True)
        output = tf.reshape(output,[-1,2])
        labels = tf.reshape(labels,[-1,2])
        loss = loss_fn(labels,output)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    probs = tf.nn.softmax(output)

    train_loss(loss)
    train_accuracy(labels, probs)

for epoch in range(config['epochs']):
    train_loss.reset_states()
    train_accuracy.reset_states()
    for sample in dataset:
        inputs_embeds = sample[0]
        labels = sample[1]
        train_step(inputs_embeds, labels)
        
        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100))
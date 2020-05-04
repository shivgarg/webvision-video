import tensorflow as tf
from transformers import TFBertModel, BertConfig
from data import *
import yaml
import argparse

MODELS = {"bert-small": [TFBertModel, BertConfig]}
DATASET = {"UniformSampler": UniformSampler}

args = argparse.ArgumentParser()
args.add_argument('config_file')
args = args.parse_args()

config = yaml.load(open(args.config_file,'r'))

model_config = MODELS[config['base_arch']][1].from_dict(config['base_config'])
model = MODELS[config['base_arch']][0](model_config)
print(model)
dataset = DATASET[config['dataset']['sampler']](config['dataset'])
dataset = dataset.padded_batch(config['batch_size'],padded_shapes=([None, None],[None,None])).prefetch(config['prefetch_size'])
print(dataset)

for sample in dataset:
    inputs_embeds = sample[0]
    print("Input Embed")
    print(inputs_embeds.shape)
    labels = sample[1]
    inputs_embeds = tf.keras.layers.Dense(config['base_config']['hidden_size'],input_shape =(2048,),use_bias=True)(inputs_embeds)
    attention_mask = tf.keras.layers.Masking()(inputs_embeds)._keras_mask
    output = model(None,attention_mask=attention_mask, inputs_embeds=inputs_embeds)
    assert  tf.reduce_all(output[0]!=inputs_embeds)
    print(output[0].shape)
    print(output[1].shape)

import tensorflow as tf
from transformers import TFBertModel, BertConfig, TFDistilBertModel, DistilBertConfig
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from head import *
import numpy as np


HEADS = {
            'default' : BertHead
        }



class BertBasic(Model):
    def __init__(self, config={}):
        super(BertBasic,self).__init__()
        self.masking = tf.keras.layers.Masking()
        self.fc1 = Dense(config['base_config']['hidden_size'], activation='relu')
        self.model_config = BertConfig.from_dict(config['base_config'])
        self.base = TFBertModel(self.model_config)
        self.head = HEADS[config['head']['name']]()

    def call(self, embeds, training=False):
        attention_mask = self.masking.compute_mask(embeds)
        x = self.fc1(embeds)
        x,_ = self.base(None,  attention_mask = attention_mask, inputs_embeds=x,training=training)
        x = self.head(x)
        return x, attention_mask

class LSTMBasic(Model):
    def __init__(self, config={}):
        super(LSTMBasic,self).__init__()
        self.masking = tf.keras.layers.Masking()
        self.fc1 = Dense(config['base_config']['num_units'], activation='relu')
        self.base = tf.keras.layers.LSTM(config['base_config']['num_units'],return_sequences=True)
        self.head = HEADS[config['head']['name']]()

    def call(self, embeds, training=False):
        attention_mask = self.masking.compute_mask(embeds)
        x = self.fc1(embeds)
        x = self.base(x,  mask = attention_mask,training=training)
        x = self.head(x)
        return x, attention_mask

class DistilBert(Model):
    def __init__(self, config={}):
        super(DistilBert,self).__init__()
        self.masking = tf.keras.layers.Masking()
        self.fc1 = Dense(config['base_config']['dim'], activation='relu')
        self.model_config = DistilBertConfig.from_dict(config['base_config'])
        self.base = TFDistilBertModel(self.model_config)
        self.head = HEADS[config['head']['name']]()

    def call(self, embeds, training=False):
        attention_mask = self.masking.compute_mask(embeds)
        x = self.fc1(embeds)
        x = self.base(None,  attention_mask = attention_mask, inputs_embeds=x,training=training)
        x = self.head(x[0])
        return x, attention_mask

class DistilBertNorm(Model):
    def __init__(self, config={}):
        super(DistilBertNorm,self).__init__()
        self.masking = tf.keras.layers.Masking()
        self.fc1 = Dense(config['base_config']['dim'])
        self.norm = tf.keras.layers.LayerNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.model_config = DistilBertConfig.from_dict(config['base_config'])
        self.base = TFDistilBertModel(self.model_config)
        
        self.fc2 = Dense(1024)
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.gelu2 = tf.keras.layers.ReLU()
        
        
        self.head = HEADS[config['head']['name']]()

    def call(self, embeds, training=False):
        attention_mask = self.masking.compute_mask(embeds)
        x = self.fc1(embeds)
        x = self.norm(x, training=training)
        x = self.relu(x)
        x = self.base(None,  attention_mask = attention_mask, inputs_embeds=x,training=training)
        x = self.fc2(x[0])
        x = self.norm2(x,training=training)
        x = self.gelu2(x)
        x = self.head(x)
        return x, attention_mask

class MLP(Model):
    def __init__(self, config={}):
        super(MLP,self).__init__()
        self.masking = tf.keras.layers.Masking()
        self.fc1 = Dense(config['base_config']['dim'])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.gelu1 = tf.keras.layers.ReLU()
        
        self.fc2 = Dense(3072)
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.gelu2 = tf.keras.layers.ReLU()
        
        self.head = HEADS[config['head']['name']]()

    def call(self, embeds, training=False):
        attention_mask = self.masking.compute_mask(embeds)
        x = self.fc1(embeds)
        x = self.norm1(x, training=training)
        x = self.gelu1(x)
        x = self.fc2(x)
        x = self.norm2(x, training=training)
        x = self.gelu2(x)
        x = self.head(x)
        return x, attention_mask

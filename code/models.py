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
        self.fc1 = Dense(config['base_config']['dim'], activation='relu')
        self.norm = tf.keras.layers.LayerNormalization()
        self.model_config = DistilBertConfig.from_dict(config['base_config'])
        self.base = TFDistilBertModel(self.model_config)
        self.head = HEADS[config['head']['name']]()

    def call(self, embeds, training=False):
        attention_mask = self.masking.compute_mask(embeds)
        x = self.fc1(embeds)
        x = self.norm(x, training=training)
        x = self.base(None,  attention_mask = attention_mask, inputs_embeds=x,training=training)
        x = self.head(x[0])
        return x, attention_mask


class MLP(Model):
    def __init__(self, config={}):
        super(MLP,self).__init__()
        self.masking = tf.keras.layers.Masking()
        self.fc1 = Dense(config['base_config']['dim'], activation='relu')
        self.batch_norm = tf.keras.layers.LayerNormalization()
        self.head = HEADS[config['head']['name']]()

    def call(self, embeds, training=False):
        attention_mask = self.masking.compute_mask(embeds)
        x = self.fc1(embeds)
        x = self.batch_norm(x, training=training)
        x = self.head(x)
        return x, attention_mask
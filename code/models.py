import tensorflow as tf
from transformers import TFBertModel, BertConfig
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
        self.base = tf.keras.layers.LSTM(config['base_config']['num_units'],return_sequences=True)
        self.head = HEADS[config['head']['name']]()

    def call(self, embeds, training=False):
        attention_mask = self.masking.compute_mask(embeds)
        x = self.base(embeds,  mask = attention_mask,training=training)
        x = self.head(x)
        return x, attention_mask

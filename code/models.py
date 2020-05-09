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
        self.fc1 = Dense(768, activation='relu')
        self.model_config = BertConfig.from_dict(config['base_config'])
        self.base = TFBertModel(self.model_config)
        self.head = HEADS[config['head']['name']]()

    def call(self, embeds):
        attention_mask = tf.keras.layers.Masking()(embeds)._keras_mask
        x = self.fc1(embeds)
        x,_ = self.base(None,  attention_mask = attention_mask, inputs_embeds=x)
        x = self.head(x)
        return x

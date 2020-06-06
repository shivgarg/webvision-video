import tensorflow as tf
from transformers import TFBertModel, BertConfig
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import numpy as np


class BertHead(Model):
    def __init__(self, cfg):
        super(BertHead,self).__init__()
        self.fc1 = Dense(513)

    def call(self, x):
        x = self.fc1(x)
        return x


class EmbedHead(Model):
    def __init__(self, cfg):
        super(EmbedHead,self).__init__()
        self.fc1 = Dense(768)
        self.embeddings = tf.constant(np.load(cfg['weights_file']))

    def call(self, x):
        x = self.fc1(x)
        x = tf.matmul(x, self.embeddings)
        return x

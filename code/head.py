import tensorflow as tf
from transformers import TFBertModel, BertConfig
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class BertHead(Model):
    def __init__(self):
        super(BertHead,self).__init__()
        self.fc1 = Dense(513*2, activation='relu')

    def call(self, x):
        x = self.fc1(x)
        x = tf.reshape(x,[x.shape[0],x.shape[1],513,2])
        return x
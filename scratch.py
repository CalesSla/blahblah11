import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gensim.downloader as api
from gensim import corpora
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Dense, Embedding, Input, Lambda


dataset = api.load('text8')

data =[]
for word in dataset:
  data.append(word)

i = 2
inputs = []
targets = []
counter = 0

for cur_data in data:
  i = 2
  while i < len(cur_data) - 2:
    temp_data = cur_data[i-2:i+3]
    target = temp_data[3-1::3]
    inputs.append(temp_data[0:2] + temp_data[3:])
    targets.append(temp_data[2])
    i += 1


tokenizer = Tokenizer(20000, oov_token = '<OOV>')
tokenizer.fit_on_texts(data)
X = tokenizer.texts_to_sequences(inputs)
Y = tokenizer.texts_to_sequences(targets)
X = tf.constant(X)
Y = np.array([item for sublist in Y for item in sublist])

# Y = tf.constant(Y)


i = Input(shape = (4,))
x = Embedding(20000, 50)(i)
x = Lambda(lambda t: tf.reduce_mean(t, axis = 1))(x)
x = Dense(20000, use_bias = False,  activation = 'softmax')(x)

model = tf.keras.Model(inputs = i, outputs = x)
model.summary()


model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X, Y, epochs = 3)
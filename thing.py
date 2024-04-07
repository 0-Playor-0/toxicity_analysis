import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy



df = pd.read_csv('train.csv')
X = df['comment_text']
y = df[df.columns[2:]].values
MAX_FEATURES = 250000 # number of words in the vocab

vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=2500,
                               output_mode='int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

#MCSHBAP - map, chache, shuffle, batch, prefetch  from_tensor_slices, list_file
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) # helps bottlenecks
train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

model = Sequential()
# Create the embedding layer 
model.add(Embedding(MAX_FEATURES+1, 32))
# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(32, activation='tanh')))
# Feature extractor Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# Final layer 
model.add(Dense(6, activation='sigmoid'))



def results(x):
    pre = Precision()
    re = Recall()
    acc = CategoricalAccuracy()
    for batch in test.as_numpy_iterator(): 
        # Unpack the batch 
        X_true, y_true = batch
        # Make a prediction 
        yhat = model.predict(X_true)
        
        # Flatten the predictions
        y_true = y_true.flatten()
        yhat = yhat.flatten()
        
        pre.update_state(y_true, yhat)
        re.update_state(y_true, yhat)
        acc.update_state(y_true, yhat)
    print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

    loc = 'toxicity'+str(x)+'.h5'

    model.save(loc)


model.compile(loss='BinaryCrossentropy', optimizer='Adam')
model.summary()
history = model.fit(train, epochs=1, verbose = 1, validation_data=val)
results(1)
history = model.fit(train, epochs=1, verbose = 1, validation_data=val)
results(2)
history = model.fit(train, epochs=1, verbose = 1, validation_data=val)
results(3)

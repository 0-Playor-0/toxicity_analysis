from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import uvicorn
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    StringInput: str

import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
df = pd.read_csv('train.csv')
X = df['comment_text']
y = df[df.columns[2:]].values
MAX_FEATURES = 100000 # number of words in the vocab
X = X.astype(str)

# Now use TextVectorization
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1000,
                               output_mode='int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

toxicity_model = tf.keras.models.load_model('toxicity.h5')

def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = toxicity_model.predict(vectorized_comment)

    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)

    return text

@app.post('/toxicity_pred')
def toxicity_pred(input_parameters: model_input):

    result = score_comment(input_parameters.StringInput)

    return result

#!ngrok config add-authtoken 2YGsL60FAmS6YHrWlS9vusY8uNC_54mhbWhaFjHVhwCFsx6oi


#ngrok_tunnel = ngrok.connect(8000)
#print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
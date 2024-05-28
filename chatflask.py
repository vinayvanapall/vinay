# Ml_project_chatbot
from flask import*
import os
import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
#colorama.init()
from colorama import Fore, Style, Back

import random
import pickle
import sys
import time

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def res():
    if request.method=="POST":
        inp=request.form['Msg']
        with open(r'path/globaldeploy.json') as file:
            data = json.load(file)
        model = keras.models.load_model('path/model')
        # return 'hii'
        
        # load tokenizer object
        with open('path/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # load label encoder object
        with open('path/label_encoder.pickle', 'rb') as enc:
            lbl_encoder = pickle.load(enc)

        # parameters
        max_len = 20
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        f=0
        for i in data['intents']:
            if i['tag'] == tag:
                f=1
                #res=np.random.choice(i['responses'])
                #break
                if inp.lower() in i['patterns']:
                    res=np.random.choice(i['responses'])
                    break
                else:
                    res="Please check the spellings in your question.If the spellings are correct then the question is not related to this bot"
                    break
        return (res)
    else:
        return render_template("path/index.html")

if __name__ == '__main__':
    app.run(debug=True)

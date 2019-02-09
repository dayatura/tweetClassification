from tensorflow.keras.models import model_from_json
from flask import Flask, render_template,request
import numpy
import re
import sys 
import os
from load import * 

app = Flask(__name__)

global char_to_int, graph, embedding_model, ann_model, rf_model, svm_model

char_to_int, graph, embedding_model, ann_model, rf_model, svm_model = init()
	

def preProcess(tweet):
    #tokenization
    tweet = tweet.split()

    # cleaning
    for i in range(len(tweet)):
        tweet[i] = re.sub("\W", " ", tweet[i])
        tweet[i] = re.sub("_", " ", tweet[i])
        if re.search(r"\bhttp\w+", tweet[i]) != None:
            tweet[i] = ""
        if re.search(r"\d", tweet[i]) != None:
            tweet[i] = ""
    tweet = ' '.join(tweet)
    tweet = tweet.split()
    tweet = ' '.join(tweet) 
    tweet = [char_to_int.get(char,0) for char in tweet.lower().split()]
    tweet = numpy.asarray(tweet)
    tweet = tweet.reshape(1,tweet.shape[0])
    tweet = sequence.pad_sequences(tweet, maxlen=31)

    # one hot vector
    with graph.as_default():
        tweet = embedding_model.predict(tweet)

    return tweet


def predict(tweet):

    # dict for prediction
    hasil = {0:'Keluhan', 1:'Respon', 2:'Bukan Keluhan/Respon'}
    
    X = tweet
    result = {}


    # ============= ANN =====================
   
    with graph.as_default():
        score = ann_model.predict(X)
        result['ann'] = hasil[numpy.argmax(score)]


    # ============= Random Forest ==================
    score = rf_model.predict(X)  
    result['rf'] = hasil[score[0]]



    # ============= SVM ==================
    score = svm_model.predict(X)
    result['svm'] = hasil[score[0]]

    return result


# ================================================
# ================ WEB APP =======================
# ================================================

@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        
        tweet = request.form.get('tweet')
        ori_tweet = tweet
        # tweet = "Tiap hari lewat bolak balik pa @ridwankamil uang habis buat bayar jalan aja ini mah 😞😞😞😞 ga ada cara lain ? https://t.co/fydVITBqVv"
    
        tweet = preProcess(tweet)
        result = predict(tweet)
        result['tweet'] = ori_tweet

        return render_template("result.html", result=result) 
    
    else:
    
        return render_template("index.html")



if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
	#app.run(debug=True)

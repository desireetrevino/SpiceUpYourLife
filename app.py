from flask import Flask, jsonify, render_template, request
import string
# Import dependencies
import pandas as pd
from pathlib import Path
import gensim
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__) 

def word2vec_model(result):
	wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
	return result	

@app.route("/") 
def home(): 
	return render_template('index.html') 

@app.route('/process', methods=['POST']) 
def process(): 
	data = request.form.get('data')
	user_input = data.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
	result = [word.lower().strip() for word in user_input.split()]
	word2vec_model(data, result)
	return result

if __name__ == '__main__': 
	app.run(debug=True) 

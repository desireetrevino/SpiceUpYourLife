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

def word2vec_model(data, result):
	wvmodel = KeyedVectors.load("Resources/descriptions.model", mmap='r')
	user_vectors = gensim.utils.simple_preprocess(data, deacc=False, min_len=2, max_len=15)
	#wvmodel.wv.most_similar(result)
	# calculate vector representations of descriptions - same as jupyter notebook
	# sum the vectors
	# take their mean
	# load the model
	# predict using the vector that we created
	# model.predict(the mean of the vectors we did with the used description)

	return result

@app.route("/") 
def home(): 
	return render_template('index.html') 

@app.route('/process', methods=['POST']) 
def process(): 
	data = request.form.get('data')
	recommendation = ""
	user_input = data.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
	result = [word.lower().strip() for word in user_input.split()]
	word2vec_model(data, result)
	return result

if __name__ == '__main__': 
	app.run(debug=True) 


#import itertools
# import collections
# counter = collections.Counter(df["country"].values)
# {k:v for k,v in counter.items() if v>5000}
# sentence = collections.Counter("the grape wine was made with grape and it tasted of wine".split())
# doc_accumulator = None
# for word,count in sentence:
#     if word in wv.model
#         if doc_accumulator == None:
#             doc_accumulator = wv.model[word] * count
#         else:
#             doc_accumulator = doc_accumulator + wv.model[word] * count






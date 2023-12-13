from flask import Flask, jsonify, render_template, request
import string
# Import dependencies
import pandas as pd
import numpy as np
from pathlib import Path
import gensim
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__) 

def data_processing(data, result):
	# load word2vec model:
	wvmodel = KeyedVectors.load("Resources/descriptions.model", mmap='r')
	# calculate vector representations of descriptions - same as jupyter notebook:
	text = drop_stop_words(data)
	text = lemmatize_words(data)
	text = gensim.utils.simple_preprocess(data)
	vectors = get_desc_vec(text, wvmodel)
	# sum the vectors

	# take their mean

	# load the knn model
	
	# predict using the vector that we created
	
	# model.predict(the mean of the vectors we did with the used description)

	return result

def drop_stop_words(text):
	nltk.download('stopwords')
	stop_words = set(stopwords.words('english'))
	lambda x: ' '.join([word for word in x.split() if word not in (stop_words)])

def lemmatize_words(text):
	nltk.download('wordnet')
	lemmatizer = WordNetLemmatizer()
	words = text.split()
	words = [lemmatizer.lemmatize(word,pos='v') for word in words]
	return ' '.join(words)


def get_desc_vec(document, wvmodel):
    return np.array(sum(wvmodel.wv[word] for word in document)/len(document))

@app.route("/") 
def home(): 
	return render_template('index.html') 

@app.route('/process', methods=['POST']) 
def process(): 
	data = request.form.get('data')
	user_input = data.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
	result = [word.lower().strip() for word in user_input.split()]
	data_processing(data, result)
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






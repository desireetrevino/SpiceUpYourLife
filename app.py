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
nltk.download('punkt')

app = Flask(__name__) 

def data_processing(data, recommendation):
	# load word2vec model:
	model = KeyedVectors.load("Resources/descriptions.model", mmap='r')
	user_input = data.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
	description = [word.lower().strip() for word in user_input.split()]
	clean_description = [w for w in description if w in model.wv]
	sum(model.wv[word] for word in clean_description)	
	# calculate vector representations of descriptions - same as jupyter notebook:
	#filtered_sentence = []
	#drop_stop_words(data, filtered_sentence)
	#print(filtered_sentence)
	#lemmatize_words(filtered_sentence)
	#text = gensim.utils.simple_preprocess(text)
	vectors = get_desc_vec(clean_description, model)
	print(vectors)
	#data = vectors
	# sum the vectors

	# take their mean

	# load the knn model
	
	# predict using the vector that we created

	# model.predict(the mean of the vectors we did with the used description)

	return data

def drop_stop_words(text, filtered_sentence):
	nltk.download('stopwords')
	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(text)
	# converts the words in word_tokens to lower case and then checks whether 
	#they are present in stop_words or not
	filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
	#with no lower case conversion
	for w in word_tokens:
		if w not in stop_words:
			filtered_sentence.append(w)
	return filtered_sentence

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
	results = "You might like: Chardonnay, Sauvignon Blanc, Riesling, Rosé, White Blend"
	return results

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






# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 23:27:49 2020

@author: Raman
"""
import numpy as np
import re
from flask_mysqldb import MySQL
from geotext import GeoText
from flask import Flask, request, jsonify, render_template
import pickle
import io
import random
from geopy.geocoders import Nominatim
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import TfidfVectorizer
from csv import DictReader
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
with open("C:/Users/Raman/Documents/kw.csv", 'rt') as f:
    b1 = [row["Sym"] for row in DictReader(f)]
    keywords = [x.upper() for x in b1]
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Raman/Documents/GoogleNews-vectors-negative300-SLIM.bin', binary=True) #Create model from word2vec file

#print(model.most_similar('hello')) #Find words most similar to given word
#print(model.similarity('man', 'woman')) #Find cosine similarity between the 2 words i.e. from 0-1 how close they are, with 1 being most

import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize
text = word_tokenize("Stomach Pain")
pos = nltk.pos_tag(text)

# print(pos)
wanted_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
pos_tags = [item for item in pos if item[1] in wanted_tags]
nv_words = []
for item in pos_tags:
   nv_words.append(item[0])

#declaring variables

priority_dict = {}
import csv
#opens key_words.csv as key_words
with open('C:/Users/Raman/Documents/kw.csv', 'rt') as key_words:
    #for each row, assigns symptoms as the first column and priority as the second column
    for row in key_words.readlines():
        if ',' in row:
            array = row.split(',')
            symptoms = array[1].lower().strip()
            priority = array[0].strip()
            #creates a dictionary as priority_dict with the key being the key words and the value is the priority
            
            priority_dict.update({symptoms:priority})         
        else:
            continue

app = Flask(__name__) #Initialize the flask App
model1 = pickle.load(open('model1.pkl', 'rb'))
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'MyDB'

mysql = MySQL(app)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    
    '''
    vectorizer=pickle.load(open('vectorizer','rb'))
    int_features = request.form['dd']
    data = [int_features]
    pred = model1.predict(vectorizer.transform(data))
    output=pred
    
    
    max_avg = 0
    max_label = ''
    sum_cos = 0 
    count = 0
    labelswords = []
    word_call_list = []
    exit = 'false'
    for call in data:  #for each call in the dataset
      call_tokens = word_tokenize(call)  #
      call_pos = nltk.pos_tag(call_tokens) #finds the pos and stores it in a list
      call_words = [item[0].lower() for item in call_pos if item[1] in wanted_tags]
      print(call_words)
      sum_cos = 0
      count = 0
      exit = 'false'
      max_label = ''
      for word_call in call_words:
        word_call_list = []
        for labels in keywords:
          labels = word_tokenize(labels)
          labelswords = []
          count = 0
          sum_cos = 0
          for word_mpds in labels:
            if word_mpds.isalpha() and word_mpds != '>' and word_mpds != 'of' and word_mpds != 'and' and word_mpds != 'a'and word_mpds != 'monixide':
              labelswords.append(word_mpds.lower())
          word_call_list.append(word_call)
          common = set(labelswords).intersection(set(word_call_list))
          if len(common) > 0:
            max_label = labels 
            exit = 'true'
          else:
            for word_mpds in labels:
              try:
                callcorrelation = model.similarity(word_call, word_mpds.lower())
                if callcorrelation > 0.2 or callcorrelation < -0.2: 
                  sum_cos += callcorrelation
                  count += 1
              except KeyError:
                continue  
          if count == 0:
             count = 1
          avg = sum_cos/count
          if avg >= max_avg:
            max_avg = avg
            max_label = labels
          if exit == 'true':
            break
#      print('Matching label for ', call, 'is ', max_label)
#      print (max_avg)
    s=' '.join(max_label)
    r=s.lower()
    x=priority_dict[r+'"']
    y=''.join(data)
    tokenized_doc  = word_tokenize(y)
    tagged_sentences = nltk.pos_tag(tokenized_doc )
    NE= nltk.ne_chunk(tagged_sentences )
    #NE.draw()
    named_entities = []
    for tagged_tree in NE:
        if hasattr(tagged_tree, 'label'):
            entity_name = ' '.join(c[0] for c in tagged_tree.leaves()) #
            entity_type = tagged_tree.label() # get NE category
            named_entities.append((entity_name, entity_type))
    for tag in named_entities:
        if tag[1]=='GPE' or 'JJ' or 'PERSON':
            f=tag[0]
            geolocator = Nominatim(user_agent="specify_your_app_name_here")
            location = geolocator.geocode(f)
#    print(location.address)
#    print((location.latitude, location.longitude))
    
    #plt.savefig('C:/Users/Raman/Desktop/Deepmodel/static/Images/image.png')
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO MyUsers(descr,emergency,prio,location) VALUES (%s, %s, %s, %s)", (y,output, x,f))
            mysql.connection.commit()
            cur.close()
            
            return render_template('index1.html', prediction_text='{},{},{},{}'.format(output,x,location.latitude,location.longitude))
        else:
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO MyUsers(descr,emergency,prio,location) VALUES (%s, %s, %s, %s)", (y,output, x,"India"))
            mysql.connection.commit()
            cur.close()
            return render_template('index1.html', prediction_text="{},{},{},{}".format(output,x,22.3511148,78.6677428))
            
            
if __name__ == "__main__":
    app.run(port=5000,debug=True)


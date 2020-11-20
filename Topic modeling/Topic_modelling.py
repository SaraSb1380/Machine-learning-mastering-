#!/usr/bin/env python
# coding: utf-8

#importing libraries and modules
import sys
import nltk
import sklearn
import pandas as pd
import numpy as np

#text processing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import spacy
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

#LDA
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()


# function for loading data
def load(data_path):
    df = pd.read_csv(data_path)
    
    return df


#text preprocessing function
nlp = spacy.load('en')
def spacy_tokenizer(sentence):

    return [word.lemma_.replace('datum','data') for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word) == 1)]
  
    
#Topic modeling and topic visualization object  
class TopicModelling():
    def __init__(self,vectorizer,corpus,number_of_topics):
        self.corpus = corpus
        self.vectorizer = vectorizer
        self.number_of_topics = number_of_topics
        
    #fitting LDA    
    def fit_transform(self):
        #vectorizing the JD
        self.X = self.vectorizer.fit_transform(self.corpus)
        #Topic modeling of JDs using LDA methods, n_components= defines the number of topics
        self.lda = LDA(n_components=self.number_of_topics, random_state=0)
        return self.lda.fit_transform(self.X)          
       
    #print 25 top words of any topics   
    def print_top_words(self,n_top_words):
        self.feature_names = self.vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(self.lda.components_):
            message = "\nTopic #%d: " % (topic_idx+1)
            message += ",".join([self.feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()
        
    # LDA visualization    
    def visualization(self):
        zit = pyLDAvis.sklearn.prepare(self.lda,self.X,self.vectorizer)
        return( pyLDAvis.display(zit))
   
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:57:18 2017

@author: wyh0117
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import numpy as np
import scipy as sc
from datetime import datetime
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('D:/School Work/Data Science Project/face-rating-data-1 cleaned.csv')

# Group data by the faceID
data_face = data.groupby('faceID')

data_face_non = data_face['non_physical_description'].apply(lambda x: ' '.join(x)).reset_index()
data_face_phy = data_face['physical_description'].apply(lambda x: ' '.join(x)).reset_index()

# Seperate physical and non-physical description
non_physical_description = data_face_non['non_physical_description']
physical_description = data_face_phy['physical_description']

# Open Self-defined stopwords file
with open('D:/School Work/Data Science Project/Stopwords.txt') as stopwords:
    stop_words = [word.strip('\n') for word in stopwords.readlines()]
    
# Specify parameters for the vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = stop_words)

# Transform the responses
tfidf_nphy = tfidf_vectorizer.fit_transform(non_physical_description)
non_phy = tfidf_vectorizer.transform(data['non_physical_description'])


#%%
# ------------------------------
# Grid Search Testing Version 2
# ------------------------------

def entropy(data, base):    
    entropy = [sc.stats.entropy(dist, base=base) for dist in data]
    average_entropy = sum(entropy)/len(entropy)
    return average_entropy

def cosine_sim(data):
    similarities = []
    data = np.matrix(data)
    for i in range(0, len(data)):
        matrix = np.delete(data, i, axis = 0)
        vector = data[i]
        cosine_similarities = cosine_similarity(matrix, vector)
        avg_cosine_sim = sum(cosine_similarities)/len(cosine_similarities)
        similarities.append(avg_cosine_sim)
    average = sum(similarities)/len(similarities)
    return average[0]

def lda_model(data_by_image, data, num_topics, d_t_p, t_w_p):
    lda = LatentDirichletAllocation(n_components = num_topics, learning_method='online', doc_topic_prior = d_t_p, topic_word_prior = t_w_p)
    lda.fit(data_by_image)
    topics_dist = lda.transform(data)
    avg_entropy = entropy(topics_dist, num_topics)
    avg_cos_sim = cosine_sim(topics_dist)
    return str(num_topics) + ',' +  str(d_t_p) + ',' + str(t_w_p) + ',' + str(avg_entropy) + ',' + str(avg_cos_sim) + '\n'

def grid_search(data_by_image, data):
    file = open('D:/School Work/Data Science Project/grid-search-results.txt', 'w')
    for num_topics in np.arange(5, 27, 2):
        for d_t_p in np.arange(0,1.05,0.05):
            for t_w_p in np.arange(0,1.05,0.05):
                result = lda_model(data_by_image, data, num_topics, d_t_p, t_w_p)
                file.write(result)
    file.close()

start=datetime.now()
grid_search(tfidf_nphy, non_phy)
print (datetime.now()-start)

#%%
# -----------------------------------------
# Visualization of the grid search results
# -----------------------------------------
results = pd.read_csv('D:/School Work/Data Science Project/grid-search-results.txt', sep = ',', names = ['Topic_Num', 'document_prior', 'topic_prior', 'entropy', 'cosine_sim'])

sns.regplot(x='entropy', y = 'cosine_sim', data = results, scatter=True, fit_reg=False)
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

#%% 
# ------------------------------------------------------------------
# Parse out all the key values from demographics and responses columns
# ------------------------------------------------------------------

data = pd.read_json('C:/Users/u656772/Downloads/Data Science Project/Guess-Who-both-versions/parsed_json/face-rating-data-2.json',encoding = 'utf-8')

data['country'] = list(map(lambda x: x['country'], data['demographics']))
data['age'] = list(map(lambda x: x['age'], data['demographics']))
data['ethnicity'] = list(map(lambda x: x['ethnicity'], data['demographics']))
data['ethnicity-details'] = list(map(lambda x: x['ethnicity-details'], data['demographics']))
data['gender'] = list(map(lambda x: x['gender'], data['demographics']))
data['responses_age'] = list(map(lambda x: x['age'], data['responses']))
data['attractive'] = list(map(lambda x: x['attractive'], data['responses']))
data['non_physical_description'] = list(map(lambda x: x['non-physical-description'], data['responses']))
data['physical_description'] = list(map(lambda x: x['physical-description'], data['responses']))
data['photo-gender'] = list(map(lambda x: x['photo-gender'], data['responses']))
#data['responses_ethnicity'] = list(map(lambda x: x['ethnicity'], data['responses']))
#data['responses_ethnicity_details'] = list(map(lambda x: x['ethnicity_details'], data['responses']))
data['eye'] = list(map(lambda x: x['eye'], data['responses']))
data['hair'] = list(map(lambda x: x['hair'], data['responses']))
data['occupation'] = list(map(lambda x: x['occupation'], data['responses']))
data['typical'] = list(map(lambda x: x['typical'], data['responses']))

# drop the columns with dictionaries
data.drop(['demographics', 'responses'], axis = 1, inplace = True)


#%% 
# -------------------------------------
# Text Pre-processing
# -------------------------------------

def text_preprocessing(text):
    text = text.lower()
    if '[comma]' in text:
        text = text.replace('[comma]', '')
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

data['physical_description'] = data.physical_description.apply(text_preprocessing)
data['non_physical_description'] = data.non_physical_description.apply(text_preprocessing)
data.to_csv('C:/Users/u656772/Downloads/Data Science Project/face-rating-data-1 cleaned.csv', index = False, encoding = 'utf-8')

#%%
# -------------------------------------
# Group by faceID and Tfidf Transform
# -------------------------------------

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
# LDA Grid Search 
# ------------------------------

def entropy(data, base):
    # base is the number of topics to normalize entropy
    entropy = [sc.stats.entropy(dist, base=base) for dist in data]
    average_entropy = sum(entropy)/len(entropy)
    return average_entropy


def cosine_sim(data):
    similarities = []
    data = np.matrix(data)
    for i in range(0, len(data)):
        # obtain the matrix without current vector
        matrix = np.delete(data, i, axis = 0)
        # obtain the current vector
        vector = data[i]
        # compute cosine similarities for the current vector against the matrix above
        cosine_similarities = cosine_similarity(matrix, vector)
        # take the average of the result
        avg_cosine_sim = sum(cosine_similarities)/len(cosine_similarities)
        similarities.append(avg_cosine_sim)
    average = sum(similarities)/len(similarities)
    return average[0]

def lda_model(data_by_image, data, num_topics, d_t_p, t_w_p):
    # Define LDA model
    lda = LatentDirichletAllocation(n_components = num_topics, learning_method='online', doc_topic_prior = d_t_p, topic_word_prior = t_w_p)
    lda.fit(data_by_image)
    # Obtain topic distribution for all responses
    topics_dist = lda.transform(data)
    avg_entropy = entropy(topics_dist, num_topics)
    avg_cos_sim = cosine_sim(topics_dist)
    return str(num_topics) + ',' +  str(d_t_p) + ',' + str(t_w_p) + ',' + str(avg_entropy) + ',' + str(avg_cos_sim) + '\n'

def grid_search(data_by_image, data):
    file = open('D:/School Work/Data Science Project/grid-search-results.txt', 'w')
    for num_topics in np.arange(5, 25, 2):
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

result = results[(results.entropy > 0.7) & (results.cosine_sim < 0.7)]

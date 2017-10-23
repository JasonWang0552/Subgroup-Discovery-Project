# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#with open('C:/Users/u656772/Downloads/Data Science Project/Guess-Who-both-versions/parsed_json/face-rating-data-2.json') as text_file:
#    test = json.load(text_file)

#data = pd.read_json('C:/Users/u656772/Downloads/Data Science Project/Guess-Who-both-versions/parsed_json/face-rating-data-2.json',encoding = 'utf-8')

#data = pd.read_csv('C:/Users/u656772/Downloads/Data Science Project/face-rating-data-1 cleaned.csv')
#%% 
# ------------------------------------------------------------------
# Parse out all the key values from demographics and responses columns
# ------------------------------------------------------------------

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
# -------------------------------
# LDA Topic Modeling
# -------------------------------


# Read the cleaned csv file
data = pd.read_csv('D:/School Work/Data Science Project/face-rating-data-1 cleaned.csv')

# Group data by the faceID
data_face = data.groupby('faceID')

data_face_non = data_face['non_physical_description'].apply(lambda x: ' '.join(x)).reset_index()
data_face_phy = data_face['physical_description'].apply(lambda x: ' '.join(x)).reset_index()

# Seperate physical and non-physical description
non_physical_description = data_face_non['non_physical_description']
physical_description = data_face_phy['physical_description']

# Define function to print topics
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

# Open Self-defined stopwords file
with open('D:/School Work/Data Science Project/Stopwords.txt') as stopwords:
    stop_words = [word.strip('\n') for word in stopwords.readlines()]

# Specify parameters for the LDA model
tfidf_vectorizer = TfidfVectorizer(stop_words = stop_words)
lda_non = LatentDirichletAllocation(n_topics = 5, learning_method='online')

# Fit non-physical data
tfidf_nphy = tfidf_vectorizer.fit_transform(non_physical_description)
lda_non.fit(tfidf_nphy)
tf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(lda_non, tf_feature_names, 10)

# Using the trained LDA Model to predict topics distribution for all responses
non_phy = tfidf_vectorizer.transform(data['non_physical_description'])
topics_dist = np.matrix(lda_non.transform(non_phy))


#%%
# ----------------------
# Fit physical data
# ----------------------

# train the model again for the physical descriptions
tfidf_vectorizer = TfidfVectorizer(stop_words = stop_words)
lda_phy = LatentDirichletAllocation(n_topics = 20, learning_method='online')

# Fit physical data
tfidf_phy = tfidf_vectorizer.fit_transform(physical_description)
lda_phy.fit(tfidf_phy)
tf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(lda_phy, tf_feature_names, 10)

# Using the trained LDA Model to predict topics distribution for all responses
physical = tfidf_vectorizer.transform(data['physical_description'])
topics_dist_phy = np.matrix(lda_phy.transform(physical))
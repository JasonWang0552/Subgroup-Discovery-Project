# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:00:25 2017

@author: u656772
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/u656772/Downloads/Data Science Project/face-rating-data-1 cleaned.csv', encoding = 'utf-8')

# plot the distribution of gender
sns.countplot(x='gender', data = data)

#%%
# plot the distribution of country
sns.countplot(x='country', data = data)
plt.tight_layout()
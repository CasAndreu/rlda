#==============================================================================
# exmaple.py
# Purpose: fitting multiple LDA models to one-minute floor speeches and then
#   clustering the resulting topics.
# Author: Andreu Casas
#==============================================================================


import rlda
import random
from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# A path to a directory where we will save some example files
path = "/Users/andreucasas/Desktop/rlda_example/"


# Getting all one-minute floor speeches from House representatives of the
#   113th Congress (n  = 9,704). List of dictionaries with the following keys:
#   'bioguide_id'
#   'speech'
#   'date'
#   'party'
#   'id'
#   'capitolwords_url'
sample_data = rlda.speeches_data

# Getting only the text ('speech') of the speeches.
speeches = [d['speech'] for d in sample_data]
random.seed(1)
rand_vector = random.sample(xrange(len(speeches)), 1000)
sample = [speeches[s] for s in rand_vector]

# Pre-processing the speeches:
#   - Parsing speeches into words
#   - Removing punctuation
#   - Removing stopwords
#   - Removing words shorter than 3 characters
#   - Stemming remaining words (Porter Stemmer)
clean_speeches = rlda.pre_processing(sample)

# Creating an object of class RLDA so that we can implement all functions
#    in this 'rlda' module
robust_model = rlda.RLDA()

# Getting a TDM matrix from the speeches
robust_model.get_tdm(clean_speeches)

# A list of the number of topics (k) of the models we want to estimate
k_list = [45, 50, 55]

# Number of times we want the algorithm estimating LDA models to iterate
n_iter = 300

# Fitting multiple LDA models (n = len(k_list))
robust_model.fit_models(k_list = k_list, n_iter = n_iter)

# Getting the feature-topic-probability vectors for each topic, and also
#   the top keywords for each topic. 
robust_model.get_all_ftp(features_top_n = 50)

# You can explore now the top keywords of topic in the console by using 
#   this funciton and specifying the topic label: "k-t" where k = the number 
#   of topics of that topic, and t = the topic number. For example, "45-1" is 
#   the first topic of the topic-model with 45 topics.
robust_model.show_top_kws('45-1')

# Creating a cosine similarity matrix. Dimensions = TxT where 
#       T = (#topics from all topic models)
robust_model.get_cosine_matrix()

# Clustering the topics into N clusters, e.g. 50 clusters,
#   using Spectral Clustering.
clusters = robust_model.cluster_topics(clusters_n = 10)





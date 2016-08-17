#==============================================================================
# exmaple.py
# Purpose: fitting multiple LDA models to one-minute floor speeches and then
#   clustering the resulting topics.
# Author: Andreu Casas
#==============================================================================


import rlda

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
sample = speeches[:100]

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
robust_model.get_tdm(sample)

# A list of the number of topics (k) of the models we want to estimate
k_list = [45, 50, 55]

# Number of times we want the algorithm estimating LDA models to iterate
n_iter = 300

# Fitting multiple LDA models (n = len(k_list))
robust_model.fit_models(k_list = k_list, n_iter = n_iter)

# Creating a cosine similarity matrix. Dimensions = TxT where 
#       T = (#topics from all topic models)
robust_model.get_cosine_matrix()

# Also just creating a list with all the cosine similarities
robust_model.get_cosine_list()

robust_model.get_all_ftp(features_top_n = 15)
clusters = robust_model.cluster_topics(clusters_n = 40)
fcps = robust_model.get_fcp(clusters, features_top_n = 15)


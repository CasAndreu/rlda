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

# Clustering
clusters = robust_model.cluster_topics(clusters_n = 50)

# Using multidimension sclaing to translate cosine similarity matrix into
#   2-dimension coordinates
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=1)
results = mds.fit(robust_model.cos_X)
coords = results.embedding_

# Creating a data frame with the data for the plot
# Data Frame with scatter plot data
df = pd.DataFrame(dict(x=coords[:, 0], y=coords[:, 1], cluster = clusters))
groups = df.groupby("cluster")

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12)
ax.legend()

plt.show()

plt.figure(figsize=(10, 8))
plt.scatter(df.x, df.y, c=clusters, cmap='prism')  # plot points with cluster dependent colors
plt.show()

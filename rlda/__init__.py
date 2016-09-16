# -*- coding: utf-8 -*-

# Copyright (C) 2016 Andreu Casas
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details: 
# https://www.gnu.org/licenses/gpl-3.0.en.html

"""
This python module provides a set of functions to fit multiple LDA models to a 
text corpus and then search for the robust topics present in multiple models.

Please cit as:
Wilkerson, John and Andreu Casas. 2016. "Large-scale Computerized 
Text Analysis in Political Science: Opportunities and Challenges." 
Annual Review of Political Science, AA:x-x.
"""


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
import re
import json
from tqdm import tqdm
import lda
import textmining
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
import operator
import csv


# GLOBAL OBJECTS

punct = string.punctuation
Pstemmer = nltk.stem.PorterStemmer()
stopw = stopwords.words("english") + ["speaker"]
path_to_dir = re.sub("__init__.py", "", __file__)
with open(path_to_dir + "data.json") as data_file:
    speeches_data = json.load(data_file)  


#FUNCTIONS

def pre_processing(list_docs, remove_punct = True,
                  remove_stopwords = True, stopwords_list = stopw,
                  remove_words_shorter_than = 3, steming = True):
    """
    Runs some text pre-processing tasks. By default:
        - Word-tokenizes
        - Removes punctuation
        - Removes stopwords
        - Removes words shorter than 3 characters
        - Stems the remaining words (Porter Stemmer)
        
    text_docs = [doc1, doc2, ..., docN]
    """
    cleanText = []
    for t in tqdm(list_docs):
        words = word_tokenize(t.decode('ascii','ignore').lower())
        if remove_punct:
            words = [w for w in words if w not in punct]
        if remove_stopwords:
            words = [w for w in words if w not in stopwords_list]
        words = [w for w in words if len(w) > 3]
        if steming:
            words = [Pstemmer.stem(w) for w in words]
        statement = ' '.join(words)
        cleanText.append(statement)
    return cleanText

    
def sorted_topic_words(model, wordsrow):
    """
    Takes an <LDA> class object and the 
    topic-model and the row with document features (words) and 
    returns, for each topic in the model, a sorted list of pairs of words 
    and probabilites where words with higher probabilities are at the top
    of the list.
    """
    n_topics = model.n_topics
    wtp = [] # Word-Topic-Probability Info
    for topic in range(0,n_topics):
        dic = {}
        for i in range(0, len(wordsrow)):
            dic[wordsrow[i]] = model.topic_word_[topic,i]
        sortedWords = sorted(dic, key=dic.get, reverse=True)
        sortedValues = [dic[w] for w in sortedWords]
        newInfo = []
        for i in range(0,len(sortedWords)):
            newInfo.append((sortedWords[i],sortedValues[i]))
        wtp.append(newInfo)
    return wtp

def get_ftp(model, features, features_top_n = 50):
        """
        Creates a list of lists with the most predictive features for each
        topic: Feature-Topic Probablities (ftp)
        """
        topics_n = model.n_topics
        ftp = []
        for i in range(0, topics_n):
            dic = {}
            for j in range(0, len(features)):
                dic[features[j]] = model.topic_word_[i, j]
            sorted_features = sorted(dic, key = dic.get, reverse = True)
            sorted_values = [dic[f] for f in sorted_features]
            new_info = []
            for z in range(0, features_top_n):
                new_info.append((sorted_features[z], sorted_values[z]))
            ftp.append(new_info)
        return ftp
        
        
class RLDA(object):

    def __init__(self):
        self.X = None
        self.features = None
        self.tdm = textmining.TermDocumentMatrix()
        self.models_list = []
        self.k_list = None
        self.topics_n = None
        self.topic_labels = []
        self.models_matrix = None
        self.cos_X = None
        self.cos_list = []
        self.ftps = []
        
    def get_tdm(self, list_docs):
        """
        Creates a Term Document Matrix (X) from a list of documents and 
        stores a list with the features of X.
        
        list_docs = [doc1, doc2, ..., docN]
        """
        for d in list_docs:
            self.tdm.add_doc(d)
        temp = list(self.tdm.rows())
        self.features = tuple(temp[0])
        self.X = np.array(temp[1:])
        
    def fit_models(self, k_list, n_iter = 500):
        """
        Fits multiple LDA models to X. Implements <lda> module.
        
        k_list = [10, 20, 25, ..., 90]
        """
        self.k_list = k_list
        self.topics_n = sum(k_list)
        models_k = reduce(lambda x,y: x+y, [[k] * k for k in self.k_list])
        for i in k_list:
            for j in range(0, i):
                self.topic_labels.append(str(i) + "-" + str(j+1))
        self.models_matrix = np.matrix([0] * len(self.features))
        for k in k_list:
            model = lda.LDA(n_topics = k, n_iter = n_iter, random_state = 1)
            model.fit(self.X)
            self.models_list.append(model)  
            self.models_matrix = np.vstack((self.models_matrix, model.nzw_))
        self.models_matrix = self.models_matrix[1:]
        
    def get_all_ftp(self, features_top_n = 50):
        output = []
        ftp = []
        for model in self.models_list:
            new = get_ftp(model, self.features, features_top_n = features_top_n)
            ftp = ftp + new
        for i in range(0, self.topics_n):
            dic = {}
            dic["topic"] = self.topic_labels[i]
            dic["top_features"]= ftp[i]
            output.append(dic)
        self.ftps = output
        
    def show_top_kws(self, topic_label):
        ftps_index = self.topic_labels.index(topic_label)
        top_kws = self.ftps[ftps_index]
        print('')
        print('Top keywords for topic ' + topic_label)
        print('======================================')
        kw_counter = 1
        for kw in top_kws['top_features']:
            print(str(kw_counter) + ') ' + str(kw[0]) + ': ' + str("%.4f" % kw[1]))
            kw_counter += 1
        print('')
        
        
    def get_cosine_matrix(self):
        """
        Calculates pairwise cosine similarities between all topics (t) from 
        all models (m) in models_list and creates a matrix of cosine
        similarities of size (topics_n X topics_n).
        """
        self.cos_X = np.matrix([0] * self.topics_n)
        for i in range(0, self.topics_n):
            t = self.models_matrix[i,]
            t_cos = cosine_similarity(t, self.models_matrix)
            self.cos_X = np.vstack((self.cos_X, t_cos))
        self.cos_X = self.cos_X[1:]
        
    def get_cosine_list(self):
        """
        Creates a list of dictionaries with information about all pairwise 
        cosine similarities in cos_X.
        """
        for i in range(0, self.topics_n):
            for j in range(0, self.topics_n):
                dic = {}
                dic["t1"] = self.topic_labels[i]
                dic["t2"] = self.topic_labels[j]
                dic["cos_sim"] = self.cos_X[i,j]
                self.cos_list.append(dic)
                
    def save_cosine_list_to_csv(self, path_to_the_file):
        """
        Saves the cosine similarities [cos_list (list of dictionaries)] into
        a csv file.
        """
        f = open(path_to_the_file, 'wb')
        w = csv.DictWriter(f, self.cos_list[0].keys())
        w.writeheader()
        w.writerows(self.cos_list)
        f.close()

        
    def cluster_topics(self, clusters_n, random_state = 1):
        """
        Clusters topics into clusters_n number of clusters using 
        Spectral Clustering.
        
        randon_state = setting the random seed. 1 by default.
        """
        clusters = list(SpectralClustering(clusters_n,
                                random_state = 1).fit_predict(self.cos_X))
        clusters = [x + 1 for x in clusters]
        return clusters
        
    def get_fcp(self, clusters, features_top_n = 50):
        """
        Creates a list of lists with the most predictive features for each
        cluster: Feature-Cluster Probablities (fcp)
        """
        unique_clusters = list(set(clusters))
        fcps = []
        for c in unique_clusters:
            dic = {}
            dic["cluster"] = c
            all_c_features = []
            for i in range(0, len(clusters)):
                if clusters[i] == c:
                    t_ftps = self.ftps[i]["top_features"]
                    all_c_features.extend(t_ftps)
                    all_c_features.sort(key = operator.itemgetter(1), 
                                        reverse = True)
            unique_c_features = []
            for f in all_c_features:
                if f[0] not in unique_c_features:
                    unique_c_features.append(str(f[0]))
            dic["top_features"] = unique_c_features[0:features_top_n]
            fcps.append(dic)
        return fcps


        

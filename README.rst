`rlda`: Robust Latent Dirichlet Allocation models 
-------------------------

This python module provides a set of functions to fit multiple LDA models to a 
text corpus and then search for the robust topics present in multiple models.

LDA models are often used to classify text into topics. However, the substance of
topcis often varies depending on model specification, making them
quite unstable (see Chuang_ 2015.). This `python` module implements a method 
proposed by Wilkerson and Casas (2016) to add a level of robustness when using
unsupervised topic models.

Please cite as:
Wilkerson, John and Andreu Casas. 2016. ``Large-scale Computerized Text
Analysis in Political Science: Opportunities and Challenges." *Annual Review
of Political Science*, AA:x-x. 

.. _Chuang: http://www.aclweb.org/anthology/N15-1018  

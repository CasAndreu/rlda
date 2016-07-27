import rlda

sample_data = rlda.speeches_data

speeches = [d['speech'] for d in sample_data]
sample = speeches[:100]

clean_speeches = rlda.pre_processing(speeches)

robust_model = rlda.RLDA()
robust_model.get_tdm(sample)
k_list = [45, 50]
n_iter = 50
robust_model.fit_models(k_list = k_list, n_iter = n_iter)
robust_model.get_cosine_matrix()
robust_model.get_cosine_list()

robust_model.get_all_ftp(features_top_n = 15)
clusters = robust_model.cluster_topics(clusters_n = 40)
fcps = robust_model.get_fcp(clusters, features_top_n = 15)


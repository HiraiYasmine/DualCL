img_embeddings_file = '../data/wn9/embeddings_vgg_19_avg_normalized.pkl'
txt_embeddings_file = '../data/wn9/Embeddings_Glove_normalized.pkl'
entity_embeddings_file =  "../data/wn9/k2b_unif_l1_100_normalized.pkl"
relation_embeddings_file =  "../data/wn9/k2b_unif_l1_100_normalized.pkl"

all_triples_file =   "../data/wn9/all.txt"
test_triples_file =  "../data/wn9/test.txt"



model_id = "wn9-model"

# where to load the weights for the model
checkpoint_best_valid_dir = "../weights/"+model_id+"/"
model_weights_best_valid_file = checkpoint_best_valid_dir + model_id + "_best_hits"
best_valid_model_meta_file = checkpoint_best_valid_dir + model_id + "_best_hits.meta"

import tensorflow as tf
import os

#wn18
relation_embeddings_size = 100
entity_embeddings_size = 100
img_embeddings_size = 4096
txt_embeddings_size = 300

mapping_size = 256

dropout_ratio = 0.2
training_epochs = 500
batch_size = 256
display_step = 1
activation_function = tf.nn.tanh
initial_learning_rate = 0.001

margin = 10
temperature_e = 2
temperature_t = 1
cl_ett_num = 32
infonce_weight = 200
infonce_weight_2 = 100

model_id = "wn18_70" 

all_triples_file =   "../data/wn9/all.txt" #"
train_triples_file = "../data/wn9/train.txt" #
test_triples_file =  "../data/wn9/test.txt"
valid_triples_file =  "../data/wn9/valid.txt"

entities_similarity_file = "../data/wn9/wn9_sort_by_concate.pkl"
img_embeddings_file = '../data/wn9/embeddings_vgg_19_avg_normalized.pkl'
txt_embeddings_file = '../data/wn9/Embeddings_Glove_normalized.pkl'
entity_embeddings_file =  "../data/wn9/k2b_unif_l1_100_normalized.pkl"
relation_embeddings_file =  "../data/wn9/k2b_unif_l1_100_normalized.pkl"


checkpoint_best_valid_dir = "weights/best_"+model_id+"/"
checkpoint_current_dir ="weights/current_"+model_id+"/"
results_dir = "results/results_"+model_id+"/"

if not os.path.exists(checkpoint_best_valid_dir):
    os.makedirs(checkpoint_best_valid_dir)

if not os.path.exists(checkpoint_current_dir):
    os.makedirs(checkpoint_current_dir)


if not os.path.exists(results_dir):
    os.makedirs(results_dir)


model_current_weights_file = checkpoint_current_dir + model_id + "_current"
current_model_meta_file = checkpoint_current_dir + model_id + "_current.meta"

model_weights_best_valid_file = checkpoint_best_valid_dir + model_id + "_best_hits"
best_valid_model_meta_file = checkpoint_best_valid_dir + model_id + "_best_hits.meta"


result_file = results_dir+model_id+"_results.txt"
log_file = results_dir+model_id+"_log.txt"


import os
import numpy as np
import tensorflow as tf
import parameters as param
import util as u
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="0" 
logs_path = "log"
# .... Loading the data ....
print("load all triples")
relation_embeddings = u.load_binary_file(param.relation_embeddings_file)
entity_embeddings = u.load_binary_file(param.entity_embeddings_file)
img_embeddings = u.load_binary_file(param.img_embeddings_file)
txt_embeddings = u.load_binary_file(param.txt_embeddings_file)
entity_similarity = u.load_binary_file(param.entities_similarity_file)

all_train_test_valid_triples, entity_list = u.load_training_triples(param.all_triples_file)
triples_set = [t[0] + "_" + t[1] + "_" + t[2] for t in all_train_test_valid_triples]
triples_set = set(triples_set)
entity_list_filtered = []
for e in entity_list:
    if e in entity_embeddings and e in img_embeddings and e in txt_embeddings:
        entity_list_filtered.append(e)
entity_list = entity_list_filtered
print("#entities", len(entity_list), "#total triples", len(all_train_test_valid_triples))

training_data = u.load_triple_data(param.train_triples_file, entity_embeddings, relation_embeddings, entity_list)
print("#training data", len(training_data))


valid_data = u.load_triple_data(param.valid_triples_file, entity_embeddings, relation_embeddings, entity_list)
print("valid_data",len(valid_data))

def max_norm_regulizer(threshold,axes=1,name="max_norm",collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights,clip_norm=threshold,axes=axes)
        clip_weights = tf.assign(weights,clipped,name=name)
        tf.add_to_collection(collection,clip_weights)
        return None
    return max_norm

max_norm_reg = max_norm_regulizer(threshold=1.0)

def my_dense(x, nr_hidden, scope, activation_fn=param.activation_function,reuse=None):
    with tf.variable_scope(scope):
        h = tf.contrib.layers.fully_connected(x, nr_hidden,
                                              activation_fn=activation_fn,
                                              reuse=reuse,
                                              scope=scope#, weights_regularizer= max_norm_reg
                                              )

        return h



# ........... Creating the model
with tf.name_scope('input'):
    r_input = tf.placeholder(dtype=tf.float32, shape=[None, param.relation_embeddings_size],name="r_input")

    h_pos_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_embeddings_size], name="h_pos_input")
    h_neg_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_embeddings_size], name="h_neg_input")

    h_pos_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.img_embeddings_size], name="h_pos_img_input")
    h_neg_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.img_embeddings_size], name="h_neg_img_input")

    h_pos_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.txt_embeddings_size], name="h_pos_txt_input")
    h_neg_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.txt_embeddings_size], name="h_neg_txt_input")

    t_pos_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_embeddings_size], name="t_pos_input")
    t_neg_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_embeddings_size], name="t_neg_input")

    t_pos_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.img_embeddings_size], name="t_pos_img_input")
    t_neg_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.img_embeddings_size], name="t_neg_img_input")

    t_pos_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.txt_embeddings_size], name="t_pos_txt_input")
    t_neg_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.txt_embeddings_size], name="t_neg_txt_input")

    h_cl_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_embeddings_size], name="h_cl_input")
    h_cl_input_img = tf.placeholder(dtype=tf.float32, shape=[None, param.img_embeddings_size], name="h_cl_input_img")
    h_cl_input_txt = tf.placeholder(dtype=tf.float32, shape=[None, param.txt_embeddings_size], name="h_cl_input_txt")
    t_cl_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_embeddings_size], name="t_cl_input")
    t_cl_input_img = tf.placeholder(dtype=tf.float32, shape=[None, param.img_embeddings_size], name="t_cl_input_img")
    t_cl_input_txt = tf.placeholder(dtype=tf.float32, shape=[None, param.txt_embeddings_size], name="t_cl_input_txt")

    triple_h_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_embeddings_size], name="triple_h_input")
    triple_h_input_img = tf.placeholder(dtype=tf.float32, shape=[None, param.img_embeddings_size], name="triple_h_input_img")
    triple_h_input_txt = tf.placeholder(dtype=tf.float32, shape=[None, param.txt_embeddings_size], name="triple_h_input_txt")
    triple_r_input = tf.placeholder(dtype=tf.float32, shape=[None, param.relation_embeddings_size], name="triple_r_input")

    labels = tf.placeholder(dtype=tf.int64, shape=[None], name="labels_input")

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

with tf.name_scope('head_relation'):
    # relation
    r_mapped = my_dense(r_input, param.mapping_size, activation_fn=param.activation_function, scope="rel_proj", reuse=None)
    r_mapped = tf.nn.dropout(r_mapped, keep_prob)

    # head
    h_pos_mapped = my_dense(h_pos_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj", reuse=None)
    h_pos_mapped = tf.nn.dropout(h_pos_mapped, keep_prob)
    h_pos_img_mapped = my_dense(h_pos_img_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_img", reuse=None)
    h_pos_img_mapped = tf.nn.dropout(h_pos_img_mapped, keep_prob)
    h_pos_txt_mapped = my_dense(h_pos_txt_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_txt", reuse=None)
    h_pos_txt_mapped = tf.nn.dropout(h_pos_txt_mapped, keep_prob)
    h_pos_stacked = tf.stack([h_pos_mapped, h_pos_img_mapped, h_pos_txt_mapped], axis=0)
    h_pos_avg = tf.reduce_mean(h_pos_stacked, axis=0)

    h_neg_mapped = my_dense(h_neg_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj", reuse=True)
    h_neg_mapped = tf.nn.dropout(h_neg_mapped, keep_prob)
    h_neg_img_mapped = my_dense(h_neg_img_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_img", reuse=True)
    h_neg_img_mapped = tf.nn.dropout(h_neg_img_mapped, keep_prob)
    h_neg_txt_mapped = my_dense(h_neg_txt_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_txt", reuse=True)
    h_neg_txt_mapped = tf.nn.dropout(h_neg_txt_mapped, keep_prob)
    h_neg_stacked = tf.stack([h_neg_mapped, h_neg_img_mapped, h_neg_txt_mapped], axis=0)
    h_neg_avg = tf.reduce_mean(h_neg_stacked, axis=0)

    # Tail 
    t_pos_mapped = my_dense(t_pos_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj", reuse=True)
    t_pos_mapped = tf.nn.dropout(t_pos_mapped, keep_prob)
    t_pos_img_mapped = my_dense(t_pos_img_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_img", reuse=True)
    t_pos_img_mapped = tf.nn.dropout(t_pos_img_mapped, keep_prob)
    t_pos_txt_mapped = my_dense(t_pos_txt_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_txt", reuse=True)
    t_pos_txt_mapped = tf.nn.dropout(t_pos_txt_mapped, keep_prob)
    t_pos_stacked = tf.stack([t_pos_mapped, t_pos_img_mapped, t_pos_txt_mapped], axis=0)
    t_pos_avg = tf.reduce_mean(t_pos_stacked, axis=0)

    t_neg_mapped = my_dense(t_neg_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj", reuse=True)
    t_neg_mapped = tf.nn.dropout(t_neg_mapped, keep_prob)
    t_neg_img_mapped = my_dense(t_neg_img_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_img", reuse=True)
    t_neg_img_mapped = tf.nn.dropout(t_neg_img_mapped, keep_prob)
    t_neg_txt_mapped = my_dense(t_neg_txt_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_txt", reuse=True)
    t_neg_txt_mapped = tf.nn.dropout(t_neg_txt_mapped, keep_prob)
    t_neg_stacked = tf.stack([t_neg_mapped, t_neg_img_mapped, t_neg_txt_mapped], axis=0)
    t_neg_avg = tf.reduce_mean(t_neg_stacked, axis=0)

    #cl part
    h_cl_mapped = my_dense(h_cl_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj", reuse=True)
    h_cl_mapped = tf.nn.dropout(h_cl_mapped, keep_prob)
    h_cl_mapped_img = my_dense(h_cl_input_img, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_img", reuse=True)
    h_cl_mapped_img = tf.nn.dropout(h_cl_mapped_img, keep_prob)
    h_cl_mapped_txt = my_dense(h_cl_input_txt, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_txt", reuse=True)
    h_cl_mapped_txt = tf.nn.dropout(h_cl_mapped_txt, keep_prob)
    h_cl_stacked = tf.stack([h_cl_mapped, h_cl_mapped_img, h_cl_mapped_txt], axis=0)
    h_cl_avg = tf.reduce_mean(h_cl_stacked, axis=0)

    t_cl_mapped = my_dense(t_cl_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj", reuse=True)
    t_cl_mapped = tf.nn.dropout(t_cl_mapped, keep_prob)
    t_cl_mapped_img = my_dense(t_cl_input_img, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_img", reuse=True)
    t_cl_mapped_img = tf.nn.dropout(t_cl_mapped_img, keep_prob)
    t_cl_mapped_txt = my_dense(t_cl_input_txt, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_txt", reuse=True)
    t_cl_mapped_txt = tf.nn.dropout(t_cl_mapped_txt, keep_prob)
    t_cl_stacked = tf.stack([t_cl_mapped, t_cl_mapped_img, t_cl_mapped_txt], axis=0)
    t_cl_avg = tf.reduce_mean(t_cl_stacked, axis=0)

    h_cl_final = tf.transpose(h_cl_avg)
    t_cl_final = tf.transpose(t_cl_avg)

    triple_h_mapped = my_dense(triple_h_input, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj", reuse=True)
    triple_h_mapped = tf.nn.dropout(triple_h_mapped, keep_prob)
    triple_h_mapped_img = my_dense(triple_h_input_img, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_img", reuse=True)
    triple_h_mapped_img = tf.nn.dropout(triple_h_mapped_img, keep_prob)
    triple_h_mapped_txt = my_dense(triple_h_input_txt, param.mapping_size, activation_fn=param.activation_function, scope="ett_proj_txt", reuse=True)
    triple_h_mapped_txt = tf.nn.dropout(triple_h_mapped_txt, keep_prob)
    triple_h_stacked = tf.stack([triple_h_mapped, triple_h_mapped_img, triple_h_mapped_txt], axis=0)
    triple_h_avg = tf.reduce_mean(triple_h_stacked, axis=0)

    triple_r_mapped = my_dense(triple_r_input, param.mapping_size, activation_fn=param.activation_function, scope="rel_proj", reuse=True)
    triple_r_mapped = tf.nn.dropout(triple_r_mapped, keep_prob)

    h_plus_r_cl = tf.transpose(triple_h_avg + triple_r_mapped)



with tf.name_scope('cosine'):

    # Head model
    h_model_pos = tf.reduce_sum(abs(h_pos_avg + r_mapped - t_pos_avg), 1, keep_dims=True, name="h_model_pos")
    h_model_neg = tf.reduce_sum(abs(h_pos_avg + r_mapped - t_neg_avg), 1, keep_dims=True, name="h_model_neg")
    # Tail model
    t_model_pos = tf.reduce_sum(abs(t_pos_avg - r_mapped - h_pos_avg), 1, keep_dims=True, name="t_model_pos")
    t_model_neg = tf.reduce_sum(abs(t_pos_avg - r_mapped - h_neg_avg), 1, keep_dims=True, name="t_model_neg")
    
    kbc_loss_h = tf.maximum(0., param.margin - h_model_neg + h_model_pos)
    kbc_loss_t = tf.maximum(0., param.margin - t_model_neg + t_model_pos)


    kbc_loss = kbc_loss_h + kbc_loss_t

    h_infonce_neg = tf.einsum("nc,ck->nk", h_pos_avg, h_cl_final)
    
    h_infonce_pos_ei = tf.reduce_sum(h_pos_mapped * h_pos_img_mapped, axis=1, keepdims=True)
    h_logits_ei = tf.concat([h_infonce_pos_ei, h_infonce_neg], axis=1)
    h_logits_ei /= param.temperature_e
    h_infonce_loss_ei = tf.keras.losses.sparse_categorical_crossentropy(labels, h_logits_ei)

    h_infonce_pos_et = tf.reduce_sum(h_pos_mapped * h_pos_txt_mapped, axis=1, keepdims=True)
    h_logits_et = tf.concat([h_infonce_pos_et, h_infonce_neg], axis=1)
    h_logits_et /= param.temperature_e
    h_infonce_loss_et = tf.keras.losses.sparse_categorical_crossentropy(labels, h_logits_et)

    h_infonce_pos_it = tf.reduce_sum(h_pos_img_mapped * h_pos_txt_mapped, axis=1, keepdims=True)
    h_logits_it = tf.concat([h_infonce_pos_it, h_infonce_neg], axis=1)
    h_logits_it /= param.temperature_e
    h_infonce_loss_it = tf.keras.losses.sparse_categorical_crossentropy(labels, h_logits_ei)

    h_infonce_loss = h_infonce_loss_ei[0] + h_infonce_loss_et[0] + h_infonce_loss_it[0]

    t_infonce_neg = tf.einsum("nc,ck->nk", t_pos_avg, t_cl_final)
    
    t_infonce_pos_ei = tf.reduce_sum(t_pos_mapped * t_pos_img_mapped, axis=1, keepdims=True)
    t_logits_ei = tf.concat([t_infonce_pos_ei, t_infonce_neg], axis=1)
    t_logits_ei /= param.temperature_e
    t_infonce_loss_ei = tf.keras.losses.sparse_categorical_crossentropy(labels, t_logits_ei)
    t_infonce_pos_et = tf.reduce_sum(t_pos_mapped * t_pos_txt_mapped, axis=1, keepdims=True)
    t_logits_et = tf.concat([t_infonce_pos_et, t_infonce_neg], axis=1)
    t_logits_et /= param.temperature_e
    t_infonce_loss_et = tf.keras.losses.sparse_categorical_crossentropy(labels, t_logits_et)
    t_infonce_pos_it = tf.reduce_sum(t_pos_img_mapped * t_pos_txt_mapped, axis=1, keepdims=True)
    t_logits_it = tf.concat([t_infonce_pos_it, t_infonce_neg], axis=1)
    t_logits_it /= param.temperature_e
    t_infonce_loss_it = tf.keras.losses.sparse_categorical_crossentropy(labels, t_logits_ei)
    t_infonce_loss = t_infonce_loss_ei[0] + t_infonce_loss_et[0] + t_infonce_loss_it[0]

    hr_cl_pos = tf.reduce_sum((h_pos_avg + r_mapped) * t_pos_avg, axis=1, keepdims=True)
    hr_cl_neg = tf.einsum("nc,ck->nk", (h_pos_avg + r_mapped), h_plus_r_cl)
    hr_logits = tf.concat([hr_cl_pos, hr_cl_neg], axis=1)
    hr_logits /= param.temperature_t
    hr_cl_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, hr_logits)

    tf.summary.histogram("loss", kbc_loss)
    tf.summary.histogram("h_infonce_loss", h_infonce_loss)
    tf.summary.histogram("t_infonce_loss", t_infonce_loss)
    total_loss = kbc_loss + (h_infonce_loss + t_infonce_loss ) / param.infonce_weight + hr_cl_loss[0] / param.infonce_weight_2
    #total_loss = kbc_loss + hr_cl_loss[0] / param.infonce_weight_2

optimizer = tf.train.AdamOptimizer().minimize(total_loss)

summary_op = tf.summary.merge_all()

#..... start the training
saver = tf.train.Saver()
log_file = open(param.log_file,"w")

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True


h_data_valid, h_img_data_valid, h_txt_data_valid, r_data_valid, \
t_data_valid, t_img_data_valid, t_txt_data_valid, \
t_neg_data_valid, t_neg_img_data_valid, t_neg_txt_data_valid, \
h_neg_data_valid, h_neg_img_data_valid, h_neg_txt_data_valid, \
h_cl_data_valid, h_cl_img_data_valid, h_cl_txt_data_valid, \
t_cl_data_valid, t_cl_img_data_valid, t_cl_txt_data_valid, \
triple_h_data_valid, triple_h_img_data_valid, triple_h_txt_data_valid, triple_r_data_valid = u.get_batch_with_neg_heads_tails_relation_and_cl_entity(
                        valid_data, triples_set, entity_list, 0, len(valid_data), param.cl_ett_num, 
                        entity_embeddings, img_embeddings, txt_embeddings, relation_embeddings, entity_similarity)

with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())

    if os.path.isfile(param.best_valid_model_meta_file):
        print("restore the weights",param.checkpoint_best_valid_dir)
        saver = tf.train.import_meta_graph(param.best_valid_model_meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(param.checkpoint_best_valid_dir))
    else:
        print("no weights to load :(")


    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    initial_valid_loss = 10000


    for epoch in range(param.training_epochs):

        np.random.shuffle(training_data)
        training_loss = 0.
        total_batch = len(training_data) // param.batch_size

        for i in tqdm(range(total_batch), desc='Processing:'):

            batch_loss = 0
            start = i * param.batch_size
            end = (i + 1) * param.batch_size

            h_data, h_img_data, h_txt_data, r_data, \
            t_data, t_img_data, t_txt_data, \
            t_neg_data, t_neg_img_data, t_neg_txt_data, \
            h_neg_data, h_neg_img_data, h_neg_txt_data, \
            h_cl_data, h_cl_img_data, h_cl_txt_data, \
            t_cl_data, t_cl_img_data, t_cl_txt_data, \
            triple_h_data, triple_h_img_data, triple_h_txt_data, trilple_r_data = u.get_batch_with_neg_heads_tails_relation_and_cl_entity(
                                              training_data, triples_set, entity_list, start, end, param.cl_ett_num, 
                                              entity_embeddings, img_embeddings, txt_embeddings, relation_embeddings, entity_similarity)

            labels_input = np.zeros(param.batch_size)
            _, loss, summary = sess.run(
                [optimizer, total_loss, summary_op],
                feed_dict={r_input: r_data,
                           h_pos_input: h_data,
                           t_pos_input: t_data,

                           h_pos_img_input: h_img_data,
                           t_pos_img_input: t_img_data,

                           h_pos_txt_input: h_txt_data,
                           t_pos_txt_input: t_txt_data,

                           h_neg_input: h_neg_data,
                           t_neg_input: t_neg_data,

                           h_neg_img_input: h_neg_img_data,
                           t_neg_img_input: t_neg_img_data,

                           h_neg_txt_input: h_neg_txt_data,
                           t_neg_txt_input: t_neg_txt_data,

                           h_cl_input: h_cl_data,
                           h_cl_input_img: h_cl_img_data,
                           h_cl_input_txt: h_cl_txt_data,
                           t_cl_input: t_cl_data,
                           t_cl_input_img: t_cl_img_data,
                           t_cl_input_txt: t_cl_txt_data,

                           triple_h_input: triple_h_data,
                           triple_h_input_img: triple_h_img_data,
                           triple_h_input_txt: triple_h_txt_data,
                           triple_r_input: trilple_r_data,

                           labels: labels_input,

                           keep_prob: 1 - param.dropout_ratio
                           })

            batch_loss = np.sum(loss)/param.batch_size

            training_loss += batch_loss

            writer.add_summary(summary, epoch * total_batch + i)

        training_loss = training_loss / total_batch

        # validating by sampling every epoch

        labels_input = np.zeros(r_data_valid.shape[0])
        val_loss = sess.run([kbc_loss],
                            feed_dict={ r_input: r_data_valid,
                                        h_pos_input: h_data_valid,
                                        t_pos_input: t_data_valid,

                                        h_pos_img_input: h_img_data_valid,
                                        t_pos_img_input: t_img_data_valid,

                                        h_pos_txt_input: h_txt_data_valid,
                                        t_pos_txt_input: t_txt_data_valid,

                                        t_neg_input: t_neg_data_valid,
                                        h_neg_input: h_neg_data_valid,

                                        h_neg_img_input: h_neg_img_data_valid,
                                        t_neg_img_input: t_neg_img_data_valid,

                                        h_neg_txt_input: h_neg_txt_data_valid,
                                        t_neg_txt_input: t_neg_txt_data_valid,

                                        h_cl_input: h_cl_data_valid,
                                        h_cl_input_img: h_cl_img_data_valid,
                                        h_cl_input_txt: h_cl_txt_data_valid,
                                        t_cl_input: t_cl_data_valid,
                                        t_cl_input_img: t_cl_img_data_valid,
                                        t_cl_input_txt: t_cl_txt_data_valid,

                                        triple_h_input: triple_h_data_valid,
                                        triple_h_input_img: triple_h_img_data_valid,
                                        triple_h_input_txt: triple_h_txt_data_valid,
                                        triple_r_input: triple_r_data_valid,

                                        labels: labels_input,
                                        keep_prob: 1
                                       })

        val_score = np.sum(val_loss) / len(valid_data)


        print("Epoch:", (epoch + 1), "loss=", str(round(training_loss, 4)), "val_loss", str(round(val_score, 4)))

        if val_score < initial_valid_loss :
            saver.save(sess, param.model_weights_best_valid_file)
            log_file.write("save model best validation loss: " + str(initial_valid_loss) + "==>" + str(val_score) + "\n")
            print("save model valid loss: ", str(initial_valid_loss), "==>", str(val_score))
            initial_valid_loss = val_score


        saver.save(sess, param.model_current_weights_file)

        log_file.write("Epoch:\t" + str(epoch + 1) + "\tloss:\t" + str(round(training_loss, 5)) + "\tval_loss:\t" + str(
            round(val_score, 5)) + "\n")
        log_file.flush()





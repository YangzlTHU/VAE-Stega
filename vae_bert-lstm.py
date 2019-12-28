import os
import tensorflow as tf
import numpy as np
import zhusuan as zs
from zhusuan import reuse
from tensorflow.python import debug as tf_debug

import huffman
from bert import create_input
from bert import tokenization
from bert import modeling
from utils import data_process, model, parameters_bert


def rnn_placeholders(state):
    """Convert RNN state tensors to placeholders with the zero state as default."""
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        c, h = state
        c = tf.placeholder_with_default(c, c.shape, c.op.name)
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return tf.contrib.rnn.LSTMStateTuple(c, h)
    elif isinstance(state, tf.Tensor):
        h = state
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return h
    else:
        structure = [rnn_placeholders(x) for x in state]
        return tuple(structure)


def online_inference(sess, data_dict, sample, seq, dataset, in_state=None, out_state=None, seed="<BOS>", length=None):
    """Generate sequence one word at a time, based on the previous word."""
    statistics1 = huffman.get_all_start_words("./data/train_"+params.dataset+".txt")
    start_word = huffman.pro_start_word(statistics1)
    sentence = [seed, start_word]
    state = None
    for _ in range(params.gen_length):
        if "<EOS>" in sentence:
            break
        input_sent_vect = [data_dict.word2idx[word] for word in sentence]
        feed = {seq: np.array(input_sent_vect).reshape([1, len(input_sent_vect)]), length: [len(input_sent_vect)]}
        # for the first decoder step, the state is None
        if state is not None:
             feed.update({in_state: state})
        index, state = sess.run([sample, out_state], feed)
        sentence += [data_dict.idx2word[int(index)]]
    sentence = " ".join([word for word in sentence if word not in ["<BOS>", "<EOS>"]])
    print(sentence)
    return sentence


def build_bert_input(encoder_input, max_seq_length):
    texts = []
    for line in encoder_input:
        texts.append(" ".join([word for word in line[1:-1]]))

    tokenizer = tokenization.FullTokenizer(vocab_file="pre/uncased_L-12_H-768_A-12/vocab.txt")
    input_idsList = []
    input_masksList = []
    segment_idsList = []
    for t in texts:
        single_input_id, single_input_mask, single_segment_id = create_input.convert_single_example(max_seq_length,
                                                                                                    tokenizer, t)
        input_idsList.append(single_input_id)
        input_masksList.append(single_input_mask)
        segment_idsList.append(single_segment_id)

    input_idsList = np.asarray(input_idsList, dtype=np.int32)
    input_masksList = np.asarray(input_masksList, dtype=np.int32)
    segment_idsList = np.asarray(segment_idsList, dtype=np.int32)

    return input_idsList, input_masksList, segment_idsList


def bert_encoder(input_ids, input_mask, segment_ids, bert_config):
    with zs.BayesianNet() as encoder:
        model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False
        )

        output_layer = model.get_pooled_output()
        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable("output_weights", [params.latent_size * 2, hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable("output_bias", [params.latent_size * 2],
                                      initializer=tf.zeros_initializer())

        if params.mode == "train":
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        lz_mean, lz_logstd = tf.split(logits, 2, axis=1)

        # define latent variable`s Stochastic Tensor
        z = zs.Normal("z", mean=lz_mean, logstd=lz_logstd, group_ndims=1)
        tf.summary.histogram("latent_space", z)
        return z


@reuse("decoder")
def vae_lstm(observed, batch_size, d_seq_length, embedding, d_inputs_ps, vocab_size, gen_mode=False):
    """decoder for vae"""
    with zs.BayesianNet(observed=observed):
        z_mean = tf.zeros([batch_size, params.latent_size])
        z = zs.Normal("z", mean=z_mean, std=0.1, group_ndims=0)

        with tf.device("/cpu:0"):
            dec_inps = tf.nn.embedding_lookup(embedding, d_inputs_ps)
        if params.dec_keep_rate < 1 and not gen_mode:
            dec_inps = tf.nn.dropout(dec_inps, params.dec_keep_rate)

        base_cell = tf.contrib.rnn.LSTMCell
        cell = model.make_rnn_cell([params.decoder_hidden for _ in range(params.decoder_rnn_layers)],
                                   base_cell=base_cell)

        for i in range(params.highway_lc):
            with tf.variable_scope("hw_layer_dec{0}".format(i)):
                if i == 0:
                    prev_y = tf.layers.dense(z, params.decoder_hidden * 2)
                elif i == params.highway_lc - 1:
                    z_dec = tf.layers.dense(prev_y, params.decoder_hidden * 2)
                else:
                    prev_y = model.highway_network(prev_y, params.highway_ls)
        inp_h, inp_c = tf.split(z_dec, 2, axis=1)
        initial = cell.zero_state(batch_size, dtype=tf.float32)
        initial_state = rnn_placeholders((tf.contrib.rnn.LSTMStateTuple(inp_c, inp_h),)) + initial[0:-1]

        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=dec_inps,
                                                 sequence_length=d_seq_length,
                                                 initial_state=initial_state,
                                                 swap_memory=True,
                                                 dtype=tf.float32)

        if gen_mode:
            outputs = outputs[:, -1, :]
        outputs_r = tf.reshape(outputs, [-1, params.decoder_hidden])
        x_logits = tf.layers.dense(outputs_r, units=vocab_size, activation=None)
        sample = tf.multinomial(x_logits / params.temperature, 1)[0][0]
        return x_logits, (initial_state, final_state), sample


def main(params):
    train_data_raw = data_process.data_read(params)
    data, labels_arr, embed_arr, data_dict = data_process.prepare_data(train_data_raw, params)
    input_idsList, input_masksList, segment_idsList = build_bert_input(train_data_raw, params.max_seq_length)

    with tf.Graph().as_default():
        d_inputs_ps = tf.placeholder(shape=[None, None], dtype=tf.int32)
        labels = tf.placeholder(shape=[None, None], dtype=tf.int32)

        with tf.device("/cpu:0"):
            if not params.pre_trained_embed:
                embedding = tf.get_variable("embedding", [data_dict.vocab_size, params.embed_size], dtype=tf.float32)
            else:
                embedding = tf.Variable(embed_arr, trainable=params.fine_tune_embed, name="embedding", dtype=tf.float32)

        seq_length = tf.placeholder_with_default([0.0], shape=[None])
        d_seq_length = tf.placeholder(shape=[None], dtype=tf.float32)

        input_ids = tf.placeholder(shape=[params.batch_size, params.max_seq_length], dtype=tf.int32, name="input_ids")
        input_mask = tf.placeholder(shape=[params.batch_size, params.max_seq_length], dtype=tf.int32, name="input_mask")
        segment_ids = tf.placeholder(shape=[params.batch_size, params.max_seq_length], dtype=tf.int32,
                                     name="segment_ids")

        bert_config = modeling.BertConfig.from_json_file("pre/uncased_L-12_H-768_A-12/bert_config.json")

        qz = bert_encoder(input_ids, input_mask, segment_ids, bert_config)
        x_logits, _, _ = vae_lstm({"z": qz}, params.batch_size, d_seq_length, embedding, d_inputs_ps,
                                  vocab_size=data_dict.vocab_size)

        labels_flat = tf.reshape(labels, [-1])
        cross_entr = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x_logits, labels=labels_flat)
        mask_labels = tf.sign(tf.to_float(labels_flat))
        masked_losses = mask_labels * cross_entr
        masked_losses = tf.reshape(masked_losses, tf.shape(labels))
        mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / d_seq_length
        rec_loss = tf.reduce_mean(mean_loss_by_example)
        perplexity = tf.exp(rec_loss)
        tf.summary.scalar("perplexity", perplexity)
        kld = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + tf.log(tf.square(qz.distribution.std) + 0.0001)
                    - tf.square(qz.distribution.mean)
                    - tf.square(qz.distribution.std), 1))
        tf.summary.scalar("kl_divergence", kld)
        anneal = tf.placeholder(tf.int32)
        annealing = (tf.tanh((tf.to_float(anneal) - 3500)/1000) + 1)/2
        lower_bound = rec_loss + tf.multiply(tf.to_float(annealing), tf.to_float(kld)) / 10

        gradients = tf.gradients(lower_bound, tf.trainable_variables())
        opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        clipped_grad, _ = tf.clip_by_global_norm(gradients, 5)
        optimize = opt.apply_gradients(zip(clipped_grad, tf.trainable_variables()))

        logits, states, smpl = vae_lstm({}, 1, d_seq_length, embedding, d_inputs_ps, vocab_size=data_dict.vocab_size,
                                        gen_mode=True)
        init_state = states[0]
        fin_output = states[1]

        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        init_checkpoint = "pre/uncased_L-12_H-768_A-12/bert_model.ckpt"
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            if params.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            if params.mode == "generate stega text":
                model_file = tf.train.latest_checkpoint(params.LOG_DIR)
                print("restoring " + model_file)
                saver.restore(sess, model_file)
                print("Done")
                saml_huffman = tf.nn.softmax(logits)
                for i in range(1, 10):
                    huffman.gen_sentence_by_huffman(sess,
                                                    data_dict,
                                                    params,
                                                    sample=saml_huffman,
                                                    seq=d_inputs_ps,
                                                    bit_num=i,
                                                    in_state=init_state,
                                                    out_state=fin_output,
                                                    length=d_seq_length,)

            elif params.mode == "generate text":
                model_file = tf.train.latest_checkpoint(params.LOG_DIR)
                print("restoring " + model_file)
                saver.restore(sess, model_file)
                print("Done")
                if not os.path.exists(params.GEN_DIR):
                    os.makedirs(params.GEN_DIR)
                for i in range(params.gen_num):
                    gen_sentence = online_inference(sess,
                                                    data_dict,
                                                    sample=smpl,
                                                    seq=d_inputs_ps,
                                                    dataset=params.dataset,
                                                    in_state=init_state,
                                                    out_state=fin_output,
                                                    length=d_seq_length)

                    with open(os.path.join(params.GEN_DIR, "vae_test.txt"), "a+") as f:
                        f.write(gen_sentence + "\n")

            else:
                # model_file = tf.train.latest_checkpoint(params.LOG_DIR)
                # saver.restore(sess, model_file)
                summary_writer = tf.summary.FileWriter(params.LOG_DIR, sess.graph)
                summary_writer.add_graph(sess.graph)
                num_iters = len(data) // params.batch_size
                cur_it = 0
                kld_arr, coeff, ppl = [], [], []
                for e in range(params.num_epochs):
                    for it in range(num_iters):
                        batch = data[it * params.batch_size: (it + 1) * params.batch_size]
                        l_batch = labels_arr[it * params.batch_size:(it + 1) * params.batch_size]
                        pad = len(max(batch, key=len))
                        length_ = np.array([len(sent) for sent in batch]).reshape(params.batch_size)
                        batch = np.array([sent + [0] * (pad - len(sent)) for sent in batch])
                        l_batch = np.array([(sent + [0] * (pad - len(sent))) for sent in l_batch])

                        batch_input_idsList = input_idsList[it * params.batch_size: (it + 1) * params.batch_size]
                        batch_input_masksList = input_masksList[it * params.batch_size: (it + 1) * params.batch_size]
                        batch_segment_idsList = segment_idsList[it * params.batch_size: (it + 1) * params.batch_size]

                        feed = {input_ids: batch_input_idsList,
                                input_mask: batch_input_masksList,
                                segment_ids: batch_segment_idsList,
                                d_inputs_ps: batch,
                                labels: l_batch,
                                seq_length: length_,
                                d_seq_length: length_,
                                anneal: cur_it}
                        lb, _, kld_, ann_, r_loss, perplexity_ = sess.run(
                            [lower_bound, optimize, kld, annealing, rec_loss, perplexity], feed_dict=feed)
                        cur_it += 1
                        kld_arr.append(kld_)
                        coeff.append(ann_)
                        ppl.append(perplexity_)
                        summary = sess.run(merged, feed_dict=feed)
                        summary_writer.add_summary(summary, cur_it)

                        if cur_it % 100 == 0 and cur_it != 0:
                            print("VLB after {} ({}) iterations (epoch): {} KLD: {} Annealing Coeff: {} CE: {}".format(
                                cur_it, e, lb, kld_, ann_, r_loss))
                            print("Perplexity: {}".format(perplexity_))
                            online_inference(sess,
                                             data_dict,
                                             sample=smpl,
                                             seq=d_inputs_ps,
                                             dataset=params.dataset,
                                             in_state=init_state,
                                             out_state=fin_output,
                                             length=d_seq_length)

                        if cur_it % 40000 == 0 and cur_it != 0:
                            saver.save(sess, os.path.join(params.LOG_DIR, "bertlstm_model.ckpt"), cur_it)

                np.save(os.path.join(params.LOG_DIR, "kld_arr.npy"), np.array(kld_arr))
                np.save(os.path.join(params.LOG_DIR, "coeff.npy"), np.array(coeff))
                np.save(os.path.join(params.LOG_DIR, "ppl.npy"), np.array(ppl))


if __name__ == "__main__":
    params = parameters_bert.Parameters()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(params)

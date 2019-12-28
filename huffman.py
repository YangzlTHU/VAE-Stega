import collections
import numpy as np
import os
import random

import Huffman_Encoding


def get_all_start_words(dataset):
    start_words = []
    data = open(dataset, "r", encoding="UTF-8").readlines()
    for i in range(len(data)):
        line_ste = data[i].strip().split()
        if len(line_ste) > 0:
            start_word = line_ste[0]
            start_words.append(start_word)
    statistics = collections.Counter(start_words)
    statistics1 = sorted(statistics.items(), key=lambda item: item[1], reverse=True)
    return statistics1


def pro_start_word(statistics1):
    sel_word_sta = []
    sel_value_sta = []
    for i in range(100):
        k = statistics1[i]
        key = k[0]
        value = k[1]
        sel_word_sta.append(key)
        sel_value_sta.append(value)
    sel_value_sta = np.array(sel_value_sta)
    sel_value_sta = sel_value_sta/float(sum(sel_value_sta))
    start = np.random.choice(sel_word_sta, 1, p=sel_value_sta)
    start_word = start[0]
    while not start_word.islower():
        start = np.random.choice(sel_word_sta, 1, p=sel_value_sta)
        start_word = start[0]
    return start_word


def gen_sentence_by_huffman(sess, data_dict, params, sample, seq, bit_num=0,
                            in_state=None, out_state=None, length=None):
    statistics1 = get_all_start_words("./data/train_" + params.dataset + ".txt")
    bit_stream = open("./bit_stream/bit_stream.txt", "r").readline()
    bit_index = random.randint(0, 1000)

    if not os.path.exists(params.GEN_DIR):
        os.makedirs(params.GEN_DIR)
    outfile = open(os.path.join(params.GEN_DIR, params.dataset + "_" + str(bit_num) + "bit.txt"), "w")
    bitfile = open(os.path.join(params.GEN_DIR, params.dataset + "_" + str(bit_num) + "bit.bit"), "w")

    k = 0
    while k < params.gen_num:
        state = None
        start_word = pro_start_word(statistics1)
        sentence = ["<BOS>", start_word]

        input_sent_vect = [data_dict.word2idx[word] for word in sentence]
        feed = {seq: np.array(input_sent_vect).reshape([1, len(input_sent_vect)]), length: [len(input_sent_vect)]}
        if state is not None:
            feed.update({in_state: state})
        prob, state = sess.run([sample, out_state], feed)

        gen = np.random.choice(data_dict.vocab_size, 1, p=prob.reshape(-1))
        sentence.append(data_dict.idx2word[int(gen)])
        bit = ""

        for _ in range(params.gen_length - 2):
            if "<EOS>" in sentence:
                break
            input_sent_vect = [data_dict.word2idx[word] for word in sentence]
            feed = {seq: np.array(input_sent_vect).reshape([1, len(input_sent_vect)]), length: [len(input_sent_vect)]}
            if state is not None:
                feed.update({in_state: state})
            prob, state = sess.run([sample, out_state], feed)

            p = {}
            for i in range(len(prob[0])):
                p[i] = prob[0][i]

            prob_sort = sorted(p.items(), key=lambda x: x[1], reverse=True)

            m = 2**int(bit_num)
            word_prob = []
            for i in range(m + 1):
                if len(word_prob) == m:
                    break
                if prob_sort[i][0] == data_dict.word2idx["unknown"]:
                    continue
                else:
                    word_prob.append(prob_sort[i])

            nodes = Huffman_Encoding.createNodes([item[1] for item in word_prob])
            root = Huffman_Encoding.createHuffmanTree(nodes)
            codes = Huffman_Encoding.huffmanEncoding(nodes, root)

            for i in range(m):
                if bit_stream[bit_index:bit_index + i + 1] in codes:
                    code_index = codes.index(bit_stream[bit_index:bit_index + i + 1])
                    gen = word_prob[code_index][0]
                    sentence += [data_dict.idx2word[int(gen)]]
                    if data_dict.idx2word[int(gen)] == "<EOS>":
                        break
                    bit += bit_stream[bit_index:bit_index + i + 1]
                    bit_index = bit_index + i + 1
                    break
        if len(sentence) < params.gen_min_length + 2:
            continue
        sentence = " ".join([word for word in sentence if word not in ["<BOS>", "<EOS>"]])
        k = k + 1
        outfile.write(sentence + "\n")
        bitfile.write(bit)

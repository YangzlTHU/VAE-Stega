import os
import multiprocessing
import pickle
import collections
import numpy as np


def data_read(params):
    corpus_file = "data/train_"+params.dataset+".txt"
    sent_file = "trained_embeddings/sent_"+params.dataset+"_train.pickle"
    if os.path.exists(sent_file):
        print("Loading sentences file")
        with open(sent_file, "rb") as rf:
            sentences = pickle.load(file=rf)
        return sentences

    if not os.path.exists("trained_embeddings"):
        os.makedirs("trained_embeddings")

    sentences = []
    with open(corpus_file, encoding="utf-8") as rf:
        for line in rf:
            sentences.append(["<BOS>"] + line.strip().split(" ") + ["<EOS>"])   # strip移除字符串开头结尾
    with open(sent_file, "wb") as wf:
        pickle.dump(sentences, file=wf)
    return sentences


def train_w2vec(embed_fn, embed_size, w2vec_it=5, sentences=None, model_path="./trained_embeddings/1"):
    from gensim.models import KeyedVectors, Word2Vec
    embed_fn += ".embed"
    print(os.path.join(model_path, embed_fn))
    print("Corpus contains {0:,} tokens".format(sum(len(sent) for sent in sentences)))
    if os.path.exists(os.path.join(model_path, embed_fn)):
        print("Loading existing embeddings file")
        return KeyedVectors.load_word2vec_format(os.path.join(model_path, embed_fn))
    # sample parameter-downsampling for frequent words
    w2vec = Word2Vec(sg=0,
                     workers=multiprocessing.cpu_count(),
                     size=embed_size, min_count=0, window=5, iter=w2vec_it)
    w2vec.build_vocab(sentences=sentences)
    print("Training w2vec")
    w2vec.train(sentences=sentences, total_examples=w2vec.corpus_count, epochs=w2vec.iter)
    # Save it to model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    w2vec.wv.save_word2vec_format(os.path.join(model_path, embed_fn))
    return KeyedVectors.load_word2vec_format(os.path.join(model_path, embed_fn))


class Dictionary(object):
    def __init__(self, sentences, vocab_drop):
        # sentences - array of sentences
        self._vocab_drop = vocab_drop
        if vocab_drop < 0:
            raise ValueError
        self._sentences = sentences
        self._word2idx = {}
        self._idx2word = {}
        self._words = []
        self.get_words()
        self._words.append("<unk>")
        self.build_vocabulary()
        self._mod_sentences()

    @property
    def vocab_size(self):
        return len(self._idx2word)

    @property
    def sentences(self):
        return self._sentences

    @property
    def word2idx(self):
        return self._word2idx

    @property
    def idx2word(self):
        return self._idx2word

    def seq2dx(self, sentence):
        return [self.word2idx[wd] for wd in sentence]

    def get_words(self):
        for sent in self.sentences:
            for word in sent:
                word = word if word in ["<EOS>", "<BOS>", "<PAD>", "<unk>"] else word.lower()
                self._words.append(word)

    def _mod_sentences(self):
        # for every sentence, if word not in vocab set to <unk>
        for i in range(len(self._sentences)):
            sent = self._sentences[i]
            for j in range(len(sent)):
                try:
                    self.word2idx[sent[j]]
                except:
                    sent[j] = "<unk>"
            self._sentences[i] = sent

    def build_vocabulary(self):
        counter = collections.Counter(self._words)
        # words, that occur less than 5 times dont include
        sorted_dict = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        sorted_dict = [(wd, count) for wd, count in sorted_dict
                       if count >= self._vocab_drop or wd in ["<unk>", "<BOS>", "<EOS>"]]
        # after sorting the dictionary, get ordered words
        words, _ = list(zip(*sorted_dict))
        self._word2idx = dict(zip(words, range(1, len(words) + 1)))
        self._idx2word = dict(zip(range(1, len(words) + 1), words))
        # add <PAD> as zero
        self._idx2word[0] = "<PAD>"
        self._word2idx["<PAD>"] = 0

    def __len__(self):
        return len(self.idx2word)


def prepare_data(data_raw, params):
    # get embeddings, prepare data
    print("building dictionary")
    data_dict = Dictionary(data_raw, params.vocab_drop)
    embed_arr = None

    if params.pre_trained_embed:
        w2_vec = train_w2vec(params.dataset, params.embed_size, w2vec_it=5,
                             sentences=data_dict.sentences, model_path="./trained_embeddings")
        embed_arr = np.zeros([data_dict.vocab_size, params.embed_size])
        for i in range(embed_arr.shape[0]):
            if i == 0:
                continue
            embed_arr[i] = w2_vec.word_vec(data_dict.idx2word[i])

    data = [[data_dict.word2idx[word] for word in sent[:-1]] for sent in data_dict.sentences
            if len(sent) < params.sent_max_size - 2]
    labels = [[data_dict.word2idx[word] for word in sent[1:]] for sent in data_dict.sentences
              if len(sent) < params.sent_max_size - 2]
    print("----Corpus_Information--- \n "
          "Raw data size: {} sentences \n Vocabulary size {}"
          "\n Limited data size {} sentences \n".format(len(data_raw), data_dict.vocab_size, len(data)))

    return data, labels, embed_arr, data_dict

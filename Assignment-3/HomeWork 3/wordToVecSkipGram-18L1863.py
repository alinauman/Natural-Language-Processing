import numpy as np
import re
import string
from collections import defaultdict
from nltk.corpus import stopwords 
from neuralnetwork.Sigmoid import Sigmoid


# --- CONSTANTS --------------------    --------------------------------------------+


class word2vec():
    def __init__(self):
        self.n = settings['n']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
        
        pass
    # Preprocessing of the corpus
    def preprocessing(corpus): 
        stop_words = set(stopwords.words('english'))     
        training_data = [] 
        sentences = corpus.split(".") 
        for i in range(len(sentences)): 
            sentences[i] = sentences[i].strip() 
            sentence = sentences[i].split() 
            x = [word.strip(string.punctuation) for word in sentence 
                                     if word not in stop_words] 
            x = [word.lower() for word in x] 
            training_data.append(x) 
        return training_data 

    # GENERATE TRAINING DATA
    def generate_training_data(self, settings, corpus):

        # GENERATE WORD COUNTS
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1
                     
        #word_counts = negative_sampling(word_counts, 5, settings)
        self.v_count = len(word_counts.keys()) 

        # GENERATE LOOKUP DICTIONARIES
        self.words_list = sorted(list(word_counts.keys()), reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        # CYCLE THROUGH EACH SENTENCE IN CORPUS
        for sentence in corpus:
            sent_len = len(sentence)

            # CYCLE THROUGH EACH WORD IN SENTENCE
            for i, word in enumerate(sentence):

                # w_target  = sentence[i]
                w_target = self.word2onehot(sentence[i])

                # CYCLE THROUGH CONTEXT WINDOW
                w_context = []
                for j in range(i - self.window, i + self.window + 1):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])
        return np.array(training_data)

    # SOFTMAX ACTIVATION FUNCTION
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    # Negative Sampling - SkipGram
    def negative_sampling(self, x, K, settings):
        output_pos = 0
        inn = 0
        output_neg = 0
        for row in corpus:
            for word in row:
                # K Negative Samples
                neg_word = np.random.choice(x.keys(), size = K, p = x.values())
                
                # Compute Gradients
                h = np.dot(self.w1.T, x)
                c_p = np.dot(self.w2.T, h)
                gradient_out = (Sigmoid(c_p * h) - 1) * self.w1
                gradient_input = (Sigmoid(c_p * h) - 1) * c_p
                neg_list = []
                for i in neg_word:
                    neg_list.append(Sigmoid(i * h) * h)
                    gradient_input += Sigmoid(i * h) * i
                
                output_pos = output_pos - settings * gradient_out
                inn = inn - settings * gradient_input
                for j in neg_list:
                    output_neg = output_neg - self.eta * j
        return output_pos, output_neg
    
    # CONVERT WORD TO ONE HOT ENCODING
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec

    # FORWARD PASS
    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    # BACKPROPAGATION
    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # UPDATE WEIGHTS
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)
        pass

    # TRAIN W2V model
    def train(self, training_data):
        # INITIALIZE WEIGHT MATRICES
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.n))  # embedding matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count))  # context matrix

        # CYCLE THROUGH EACH EPOCH
        for i in range(0, self.epochs):

            self.loss = 0

            # CYCLE THROUGH EACH TRAINING SAMPLE
            for w_t, w_c in training_data:
                # FORWARD PASS
                y_pred, h, u = self.forward_pass(w_t)

                # CALCULATE ERROR
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                # BACKPROPAGATION
                self.backprop(EI, h, w_t)

                # CALCULATE LOSS
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                # self.loss += -2*np.log(len(w_c)) -np.sum([u[word.index(1)] for word in w_c]) + (len(w_c) * np.log(np.sum(np.exp(u))))

            print
            'EPOCH:', i, 'LOSS:', self.loss
        pass

    # input a word, returns a vector (if available)
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    # input a vector, returns nearest word(s)
    def vec_sim(self, vec, top_n):

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(vec, v_w2)
            theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda word, sim: sim, reverse=True)

        for word, sim in words_sorted[:top_n]:
            print
            word, sim

        pass

    # input word, returns top [n] most similar words
    def word_sim(self, word, top_n):

        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta
         #   print(word, theta)

        words_sorted = sorted(word_sim.items(),  key=lambda x: x[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print (word, sim)

        pass
    
 
# --- EXAMPLE RUN --------------------------------------------------------------+

settings = {}
settings['n'] = 5  # dimension of word embeddings
settings['window_size'] = 2  # context window +/- center word
settings['min_count'] = 0  # minimum word count
settings['epochs'] = 5000  # number of training epochs
settings['neg_samp'] = 10  # number of negative words to use during training
settings['learning_rate'] = 0.01  # learning rate
np.random.seed(0)  # set the seed for reproducibility

# Adding a sentence/corpus
#corpus = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
corpus = "" 
corpus += "The earth revolves around the sun. The moon revolves around the earth"
training_data = word2vec.preprocessing(corpus)

# INITIALIZE W2V MODEL
w2v = word2vec()

# generate training data
training_data = w2v.generate_training_data(settings, training_data)

# train word2vec model
w2v.train(training_data)

# Top 5 words for a given word in a corpus
w2v.word_sim('sun',5)


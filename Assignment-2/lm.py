import os.path
import sys
import random
import math
from operator import itemgetter
from collections import defaultdict
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef


# Preprocess the corpus
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor

    #replace rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if freqDict[word] < 2:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor
    
    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):
        print("""Your task is to implement four kinds of n-gram language models:
      a) an (unsmoothed) unigram model (UnigramModel)
      b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
      c) an unsmoothed bigram model (BigramModel)
      d) a bigram model smoothed using kneser-ney smoothing (SmoothedBigramModelKN)
      """)
    #enddef

    # Generate a sentence by drawing words according to the 
    # model's probability distribution
    # Note: think about how to set the length of the sentence 
    #in a principled way
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."
    #emddef

    # Given a sentence (sen), return the probability of 
    # that sentence under the model
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0
    #enddef

    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        return 0.0
    #enddef

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + " ".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)
            
	#endfor
    #enddef
#endclass

# Unigram language model
class UnigramModel(LanguageModel):
    def __init__(self, corpus):
        # print("Subtask: implement the unsmoothed unigram language model")
        self.unigramdist = UnigramDist(corpus)
    
    def generateSentence(self):
        sentence = [start]
        while(1):
            word = self.unigramdist.draw()
            if (word == start):
                continue
            if (word != end):
                sentence.append(word)
            else:
                break
        sentence.append(end)
        return (" ".join(sentence))
    
    def getSentenceProbability(self, sen):
        p = 0.0
        sentence = sen.split()
        length = len(sentence)
        for word in range(1,length):
            p += self.unigramdist.prob(sentence[word])
        return p
    
    def getPerplexity(self, corpus):
        pp = 0.0
        N = 0.0
        for sentence in corpus:
            for word in sentence:
                if word == start:
                    continue
                pp += self.unigramdist.prob(word)
                N += 1
        pp = -pp/N
        pp = math.exp(pp)
        return pp
    #endddef
#endclass

#Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus):
        self.sunigramdist = SmoothedUnigramDist(corpus)
        
    def generateSentence(self):
        sentence = [start]
        while(1):
            word = self.sunigramdist.draw()
            if (word == start):
                continue
            if (word != end):
                sentence.append(word)
            else:
                break
        sentence.append(end)
        return (" ".join(sentence))
    
    
    def getSentenceProbability(self, sen):
        p = 0.0
        sentence = sen.split()
        length = len(sentence)
        for word in range(1,length):
            p += self.sunigramdist.prob(sentence[word])
        return p
    
    def getPerplexity(self, corpus):
        pp = 0.0
        N = 0.0
        for sentence in corpus:
            for word in sentence:
                if word == start:
                    continue
                pp += self.sunigramdist.prob(word)
                N += 1
        pp = -pp/N
        pp = math.exp(pp)
        return pp 
    #endddef
#endclass

# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def __init__(self, corpus):
        # print("Subtask: implement the unsmoothed bigram language model")
        self.bigramdist = BigramDist(corpus)
    
    def generateSentence(self):
        sentence = [start]
        word = self.bigramdist.draw(start)
        while word != end:
            sentence.append(word)
            word = self.bigramdist.draw(word)
        sentence.append(end)
        return sentence
    
    # Given a sentence, return the probability of that sentence
    def getSentenceProbability(self, sen):
        prev_word = start
        p = 1.0
        for word in sen[1:]:
            p *= self.bigramdist.prob(word, prev_word)
            prev_word = word
        return p
    
    # Calculate the perplexity - (normalized inverse log probability)
    def getPerplexity(self, corpus):
        prev_word = start
        words = []
        for sentence in corpus:
            for word in sentence:
                words.append(word)
        next(words)
        N = 0
        log_sum = 0.0
        for i in words:
            try:
                log_sum += math.log(self.bigramdist.prob(i, prev_word))
                N += 1
            except:
                pass
            prev_word = i
        pp = math.exp(log_sum/-N)
        return pp
    #endddef
#endclass



# Smoothed bigram language model (use absolute discounting and kneser-ney for smoothing)
class SmoothedBigramModelKN(LanguageModel):
    def __init__(self, corpus):
        # print("Subtask: implement the smoothed bigram language model with kneser-ney smoothing")
        self.smoothdist = SmoothedBigramDist(corpus)
    
    def generateSentence(self):
        sentence = [start]
        word = self.smoothdist.draw(start)
        while word != end:
            sentence.append(word)
            word = self.smoothdist.draw(word)
        sentence.append(end)
        return sentence
    
    def getSentenceProbability(self,sen):
        p = 0.0
        sentence = sen.split()
        length = len(sentence)
        for word in range(1,length):
            p += self.bigramdist.prob(sentence[word],sentence[word-1])
        return p
    
    def getPerplexity(self, corpus):
        prev_word = start
        words = []
        for sentence in corpus:
            for word in sentence:
                words.append(word)
        next(words)
        N = 0
        log_sum = 0.0
        for i in words:
            try:
                log_sum += math.log(self.bigramdist.prob(i, prev_word))
                N += 1
            except:
                pass
            prev_word = i
        pp = math.exp(log_sum/-N)
        return pp
    #endddef
#endclass



# Sample class for a unsmoothed unigram probability distribution
# Note: 
#       Feel free to use/re-use/modify this class as necessary for your 
#       own code (e.g. converting to log probabilities after training). 
#       This class is intended to help you get started
#       with your implementation of the language models above.
    
class UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
                
    # Returns the probability of word in the distribution
    def prob(self, word):
        return self.counts[word]/self.total

    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word

class SmoothedUnigramDist:
    def __init__(self,corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
        
    # Add observed counts from corpus to the distribution  
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
        self.total -= self.counts[start]
        self.counts.pop(start)
        self.S = S = len(self.counts.keys())
        
    # Returns the probability of word in the distribution
    def prob(self,word):
        return (self.counts[word]+1.)/(self.total+self.S)
    
    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word

class BigramDist:
    def __init__(self,corpus):
        self.unicounts = defaultdict(float)
        self.bicounts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
        
    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                self.unicounts[corpus[i][j]] += 1.0
                if j != 0:
                    self.bicounts[corpus[i][j],corpus[i][j-1]] += 1.0
                self.total += 1.0
                
    # Returns the probability of word given that another word has ocurred
    def prob(self, word, prev_word):
        return math.log(self.bicounts[word,prev_word]/self.unicounts[prev_word])
    
    # Generate a random bigram according to the distribution
    def draw(self, prev_word):
        rand = self.unicounts[prev_word]*random.random()
        for i, j in self.bicounts:
            if j == prev_word:
                rand -= self.bicounts[i, j]
                if rand <= 0.0:
                    return i
        
class SmoothedBigramDist:
    def __init__(self,corpus):
        self.unicounts = defaultdict(float)
        self.bicounts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
        self.D = 0.0
        self.Cc = defaultdict(float)
        self.unigramDist = UnigramDist(corpus)
        self.setD(self.bicounts)
        self.setCc(self.bicounts)
        
    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                self.unicounts[corpus[i][j]] += 1.0
                if j != 0:
                    self.bicounts[corpus[i][j],corpus[i][j-1]] += 1.0
                self.total += 1.0
				
    # Discount factor
    def setD(self, bicounts):
        a = sum( x == 1 for x in self.bicounts.values() )
        b  = sum( x == 2 for x in self.bicounts.values() )
        self.D = float(a) / ( a + (2 * b) )
        
    # Continuation Counts
    def setCc(self, bicounts):
        for word, prev_word in bicounts:
            for k in prev_word:
                if prev_word[k] <= 4:
                    self.Cc[word+1][int(prev_word[k]-1)] += 1
            
    def prob(self, word, prev_word):
        p = max((self.bicounts[word, prev_word] - self.D), 0)/self.unicounts[prev_word]
        p += self.D * self.Cc[prev_word] * (math.exp(self.unigramDist.prob(word)))/self.unicounts[prev_word]
        p = math.log(p)
        return p
    
    
    def knn(self, word, prev_word):
        return max((self.bicounts[word,prev_word] - self.D), 0) + self.D * self.Cc[prev_word] * math.exp(self.unigramDist.prob(word))
    
    def draw(self, prev_word):
        rand = self.unicounts[prev_word]*random.random()
        for i in self.unicounts:
            if i == start:
                continue
            rand -= self.knn(i, prev_word)
            if rand <= 0.0:
                return rand
#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    #read your corpora
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)
    
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')
    
    vocab = set()
    # Please write the code to create the vocab over here before the function preprocessTest
    print("""Task 0: create a vocabulary(collection of word types) for the train corpus""")


    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)
    
    # Unigram Model
    u =  UnigramModel(trainCorpus)
    print("Unigram Model")
    u.generateSentencesToFile(20, "Unigram_output.txt")
    perplexity_test1 = u.getPerplexity(posTestCorpus)
    perplexity_test2 = u.getPerplexity(negTestCorpus)
    print("Perplexity of positive test corpus: "+ str(perplexity_test1))
    print("Perplexity of negative test corpus: "+ str(perplexity_test2))
    # Smoothed Unigram Model
    su = SmoothedUnigramModel(trainCorpus)
    print("SmoothedUnigramModel Model")
    su.generateSentencesToFile(20, "Smooth_Unigram_output.txt")
    perplexity_test1 = su.getPerplexity(posTestCorpus)
    perplexity_test2 = su.getPerplexity(negTestCorpus)
    print("Perplexity of positive test corpus: "+ str(perplexity_test1))
    print("Perplexity of negative test corpus: "+ str(perplexity_test2))
    # Bigram Model
    b = BigramModel(trainCorpus)
    print("BigramModel Model")
    b.generateSentencesToFile(20, "Bigram_output.txt")
    perplexity_test1 = b.getPerplexity(posTestCorpus)
    perplexity_test2 = b.getPerplexity(negTestCorpus)
    print("Perplexity of positive test corpus: "+ str(perplexity_test1))
    print("Perplexity of negative test corpus: "+ str(perplexity_test2))
    # Smoothed Bigram Model - Kneser Ney
    sb = SmoothedBigramModelKN(trainCorpus)
    print("SmoothedBigramModel Model")
    sb.generateSentencesToFile(20, "Smooth_Bigram_output.txt")
    perplexity_test1 = sb.getPerplexity(posTestCorpus)
    perplexity_test2 = sb.getPerplexity(negTestCorpus)
    print("Perplexity of positive test corpus: "+ str(perplexity_test1))
    print("Perplexity of negative test corpus: "+ str(perplexity_test2))

    # Run sample unigram dist code
    unigramDist = UnigramDist(trainCorpus)
    print("Sample UnigramDist output:")
    print("Probability of \"picture\": ", unigramDist.prob("picture"))
    print("Probability of \""+UNK+"\": ", unigramDist.prob(UNK))
    print("\"Random\" draw: ", unigramDist.draw())


############################## Question No.1 #####################################
# When generating sentences with the unigram model, what controls the length of the generated sentences?
# How does this differ from the sentences produced by the bigram models?

# Ans: The length of the generated sentence is controlled by the end of the sentence "</s>". As soon
    # it is spotted the sentence is stopped. Whereas in the bigram models this is determined by the probability
    # of the word given a certain word is already there. 
    
############################# Question No. 2 ####################################
#Consider the probability of the generated sentences according to your models. 
#Do your models assign drastically different probabilities to the different sets of sentences? 
#Why do you think that is?

# Ans: Yes, the models assign drastically different probabilities to different sets of sentences. 
    # The reason being that the n-gram model predicts the occurrence of a word based on the occurrence
    # of (n-1) word previous words. Thus, it is dependent on the occurrence of the previous word. In case
    # of the bigram model, predicts the probability based on the occurrence of its only previous word. Whereas
    # in the case of Unigram model, it is dependent on the probability of the word in the training data
    # and is not dependent on any previous word occurrence. 
    
############################# Question No. 3 ####################################
# Generate additional sentences using your bigram and smoothed bigram models. 
# In your opinion, which model produces better / more realistic sentences?
    
# Ans:In my opinion, Smoothed Bigram performs much better, as it takes into account the discounted value and the 
    # continuation count as well. 
    
############################# Question No. 4 ####################################
# For each of the four models, which test corpus has a higher perplexity? 
# Why? Make sure to include the perplexity values in the answer.

# Ans: Please find below 
#Unigram Model:
#Positive Test - Perplexity:604.9933549170498
#Negative Test - Perplexity:579.2887665578976

#Smooth_Unigram Model:
#Positive Test - Perplexity:1127.266784986728
#Negative Test - Perplexity:1141.9708068510158

#Bigram Model:
#Positive Test - Perplexity:74.19571898384349
#Negative Test - Perplexity:78.1559203822373

#Smooth_Bigram Model:
#Positive Test - Perplexity:374.7337154269931
#Negative Test - Perplexity:392.09823143798945
    
# As we can see the Smooth_Unigram Model, has the highest perplexity, which means the sentences
# have the lowest probability of occurrence. The reason being the Laplace Add-One smoothing, which
# is not a popular smoothing option for the n-grams. Though,it gives a count to those, which have '0'
# but it also takes a proportional amount from the rest. This causes an imbalance in the values. 
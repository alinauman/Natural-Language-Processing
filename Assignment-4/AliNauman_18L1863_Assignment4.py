# SemEval - CommonSense Task
# Determine out of the two given sentences, which sentence is against common
# sense and predict the accuracy

import os
import numpy as np
import pandas as pd
import csv
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List
import torch.nn as nn
import torch
from transformers import BertPreTrainedModel
from transformers import BertModel
from torch.nn.parameter import Parameter

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-large-uncased')

# Put the model in "evaluation" mode, meaning feed-forward operation.
model = model.eval()

# Loading the data
train_dir_path = 'subtaskA_data_all.csv'
data = pd.read_csv(train_dir_path, sep=',')
# Loadint the labels
labels_dir_path = 'subtaskA_answers_all.csv'
true_labels = pd.read_csv(labels_dir_path, sep=',', header=None)
true_labels = true_labels[true_labels.columns[1]]

label = []
for i in range(len(data)):
    # Extracting sentences
    sent1 = data.sent0[i]
    sent2 = data.sent1[i]
    # Masking the sentences with 'CLS' & 'SEP'
    sent1 = marked_text = "[CLS] " + sent1 + " [SEP]"
    sent2 = marked_text = "[CLS] " + sent2 + " [SEP]"
    # Tokenizing the sentences
    tokenized_text1 = tokenizer.tokenize(sent1)
    tokenized_text2 = tokenizer.tokenize(sent2)
    # Indexing the tokenized text
    indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
    indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
    # Segment Ids with '0' & '1'
    segments_ids1 = [0] * len(tokenized_text1)
    segments_ids2 = [1] * len(tokenized_text2)

    # Convert inputs to PyTorch tensors
    tokens_tensor1 = torch.tensor([indexed_tokens1])
    segments_tensors1 = torch.tensor([segments_ids1])
    # Convert inputs to PyTorch tensors
    tokens_tensor2 = torch.tensor([indexed_tokens2])
    segments_tensors2 = torch.tensor([segments_ids2])
    


    # Predict hidden states features for each layer - sent1
    with torch.no_grad():
        encoded_layers1, _ = model(tokens_tensor1, segments_tensors1)
    
    # Predict hidden states features for each layer - sent2
    with torch.no_grad():
        encoded_layers2, _ = model(tokens_tensor2, segments_tensors2)
     
    # Creating sentence embeddings
    sentence_embedding1 = torch.mean(encoded_layers1[11], 1)
    sentence_embedding2 = torch.mean(encoded_layers2[11], 1)
    
    # Concatening both sentence embeddings to create a fused sentence
    concat_sent = torch.cat((sentence_embedding1+sentence_embedding1, 
                             sentence_embedding1*sentence_embedding1),dim=1)
    reduce_fuse_linear = nn.Linear(2048, 1024)
    concat_sent = reduce_fuse_linear(concat_sent)
    # Using cosine similarity to compare the vectors
    cosine = nn.CosineSimilarity()
    cos_sim1 = cosine(concat_sent, sentence_embedding1)
    cos_sim2 = cosine(concat_sent, sentence_embedding2)
    
    # Label with class '0' or '1'
    if cos_sim1 < cos_sim2:
        label.append(0)
    else:
        label.append(1)


# Calculating the accuracy using sklearn
from sklearn.metrics import accuracy_score
print(accuracy_score(true_labels, label))

# bert-large-uncased is giving an accuracy of 93.4%

#################################################################################
# In this task we have Multiple Choice
# We have tp select the most corresponding reason why this statement 
# is against common sense (A, B, C)

train_dir_path = 'subtaskB_data_all.csv'
data = pd.read_csv(train_dir_path, sep=',')
labels_dir_path = 'subtaskB_answers_all.csv'
true_labels = pd.read_csv(labels_dir_path, sep=',', header=None)
true_labels = true_labels[true_labels.columns[1]]

label = []
for i in range(len(data)):
    # Extracting sentences
    falsesent = data.FalseSent[i]
    sent1 = data.OptionA[i]
    sent2 = data.OptionB[i]
    sent3 = data.OptionC[i]
    
    # Masking the sentences with 'CLS' & 'SEP'
    falsesent = marked_text = "[CLS] " + falsesent + " [SEP]"
    sent1 = marked_text = "[CLS] " + sent1 + " [SEP]"
    sent2 = marked_text = "[CLS] " + sent2 + " [SEP]"
    sent3 = marked_text = "[CLS] " + sent3 + " [SEP]"
    
    # Tokenizing the sentences
    tokenized_text1 = tokenizer.tokenize(falsesent)
    tokenized_text2 = tokenizer.tokenize(sent1)
    tokenized_text3 = tokenizer.tokenize(sent2)
    tokenized_text4 = tokenizer.tokenize(sent3)
    
    # Indexing the tokenized text
    indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
    indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
    indexed_tokens3 = tokenizer.convert_tokens_to_ids(tokenized_text3)
    indexed_tokens4 = tokenizer.convert_tokens_to_ids(tokenized_text4)
    # Segment Ids with '0' & '1'
    segments_ids1 = [0] * len(tokenized_text1)
    segments_ids2 = [1] * len(tokenized_text2)
    segments_ids3 = [1] * len(tokenized_text3)
    segments_ids4 = [1] * len(tokenized_text4)
    
    # Convert inputs to PyTorch tensors
    tokens_tensor1 = torch.tensor([indexed_tokens1])
    segments_tensors1 = torch.tensor([segments_ids1])
    tokens_tensor2 = torch.tensor([indexed_tokens2])
    segments_tensors2 = torch.tensor([segments_ids2])
    tokens_tensor3 = torch.tensor([indexed_tokens3])
    segments_tensors3 = torch.tensor([segments_ids3])
    tokens_tensor4 = torch.tensor([indexed_tokens4])
    segments_tensors4 = torch.tensor([segments_ids4])
    
    # Predict hidden states features for each layer - sent1
    with torch.no_grad():
        encoded_layers1, _ = model(tokens_tensor1, segments_tensors1)
    
    with torch.no_grad():
        encoded_layers2, _ = model(tokens_tensor2, segments_tensors2)
        
    with torch.no_grad():
        encoded_layers3, _ = model(tokens_tensor3, segments_tensors3)
        
    with torch.no_grad():
        encoded_layers4, _ = model(tokens_tensor4, segments_tensors4)
        
    # Creating sentence embeddings
    sentence_embedding1 = torch.mean(encoded_layers1[11], 1)
    sentence_embedding2 = torch.mean(encoded_layers2[11], 1)
    sentence_embedding3 = torch.mean(encoded_layers3[11], 1)
    sentence_embedding4 = torch.mean(encoded_layers4[11], 1)
    
    cosine = nn.CosineSimilarity()
    cos_simA = cosine(sentence_embedding1, sentence_embedding2)
    cos_simB = cosine(sentence_embedding1, sentence_embedding3)
    cos_simC = cosine(sentence_embedding1, sentence_embedding4)
    
    if(cos_simA > cos_simB and cos_simA > cos_simC):
        label.append('A')
    if(cos_simB > cos_simA and cos_simB > cos_simC):
        label.append('B')
    if(cos_simC > cos_simA and cos_simC > cos_simB):
        label.append('C')
        
# Calculating the accuracy using sklearn
from sklearn.metrics import accuracy_score
print(accuracy_score(true_labels, label))

# bert-large-uncased is giving an accuracy of 82.6%
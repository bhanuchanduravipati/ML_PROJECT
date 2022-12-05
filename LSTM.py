!pip install pytorch_pretrained_bert
!pip install scikit-multilearn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import gc
import re
import operator

import sys
from sklearn import metrics
from sklearn import model_selection
from nltk.stem import PorterStemmer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, tqdm_notebook

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import string

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertAdam, BertConfig
from nltk.corpus import stopwords
import csv

from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgbm
import numpy as np




# print(os.listdir("../input/nvidiaapex/repository/NVIDIA-apex-39e153a"))
from google.colab import drive
drive.mount('/content/drive')

device=torch.device('cuda')
BERT_MODEL = 'bert-base-uncased'
import numpy as np 
import pandas as pd 

train_data = pd.read_csv("drive/My Drive/diaster_dataset.tsv",delimiter=',')


englishTokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=None)

class InputExample(object):
    def __init__(self, guid, text_a, label=None):
        self.guid = guid
        self.text_a = text_a
        self.label = label

# Base Feature structure in BERT
class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode):
    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example "+str(ex_index)+" of "+ str(len(examples)))
                  
        tokens_a = tokenizer.tokenize(example.text_a)
        if len(tokens_a) > max_seq_length - 2:
                  tokens_a = tokens_a[:(max_seq_length - 2)]
        
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(InputFeatures(input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id))
    return features

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

def transformFeaturestoTensorSets(train_features):
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

from sklearn.model_selection import train_test_split

print(train_data[0:1])
print(train_data.head(n=10))

xTrain, xTest = train_test_split(train_data, test_size = 0.2, random_state = 0)

max_seq_length = 300
label_list=['affected_people','caution_and_advice','deaths_reports','disease_signs_or_symptoms','disease_transmission','displaced_people_and_evacuations','donation_needs_or_offers_or_volunteering_services','infrastructure_and_utilities_damage','injured_or_dead_people','missing_trapped_or_found_people','not_related_or_irrelevant','other_useful_information','prevention','sympathy_and_emotional_support','treatment']
testModel = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_list))
X_train = xTrain[xTrain.label != 'NaN']
train_examples = [InputExample('train', row.tweet_text, row.label) for row in X_train.itertuples()]
train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, englishTokenizer,'classification')

# batch gradient decent
train_batch_size = 16
# Fine Tune
num_train_epochs = 1
print('train', len (train_examples))

device = torch.device("cuda")
n_gpu = torch.cuda.device_count()

# gradient optimize period
gradient_accumulation_steps = 1

print(train_batch_size)
print(gradient_accumulation_steps)
train_batch_size = train_batch_size // gradient_accumulation_steps
print(train_batch_size)
num_train_optimization_steps = int(len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
learning_rate = 2e-5
warmup_proportion = 0.1
testModel.to(device)

param_optimizer = list(testModel.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

optimizer = BertAdam(optimizer_grouped_parameters,lr=learning_rate,warmup=warmup_proportion,t_total=num_train_optimization_steps)

train_data = transformFeaturestoTensorSets(train_features)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
testModel.train()

from tqdm import tqdm, trange

global_step = 0
nb_tr_steps = 0
tr_loss = 0

for _ in trange(int(num_train_epochs), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    total_step = len(train_data)// train_batch_size
    twenty_percent_step = total_step // 5
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        loss = testModel(input_ids, segment_ids, input_mask, label_ids)
        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
        if step % twenty_percent_step == 0:
            print("Fininshed: {:.2f}% ({}/{})".format(step/total_step*100, step, total_step))
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
## Constant variables
output_dir = 'output'
work_dir = "../working/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_to_save = testModel.module if hasattr(testModel, 'module') else testModel 
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
torch.save(model_to_save.state_dict(), output_model_file)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

with open(output_config_file, 'w') as f:
    f.write(model_to_save.config.to_json_string())


config = BertConfig(output_config_file)
model = BertForSequenceClassification(config, num_labels=len(label_list))
model.load_state_dict(torch.load(output_model_file))
model.to(device)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def predict(model, tokenizer, examples, label_list, eval_batch_size=128):
    model.to(device)
    eval_examples = examples
    eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer, 'classification')
    eval_data = transformFeaturestoTensorSets(eval_features)
    
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    
    res = []
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
        logits = logits.detach().cpu().numpy()
        res.extend(logits.argmax(-1))
        nb_eval_steps += 1
        if nb_eval_steps % 1000 == 0:
            print("nb_eval_steps: ", nb_eval_steps)
    return res
testSet = xTest

testSet.fillna('UNKNOWN', inplace=True)
test_examples = [InputExample('test', row.tweet_text, 'affected_people') for row in testSet.itertuples()]
predict_results = predict(model, englishTokenizer, test_examples, label_list)

labelArr = {idx:lab for idx, lab in enumerate(label_list)}
label_results = [labelArr[c] for c  in predict_results]

#　For Submission
testSet['Category'] = label_results
submission = testSet.loc[:, ['Category']].reset_index()
submission.columns = ['Id', 'Category']
submission.to_csv('submission.csv', index=False)

label_results[:10]
y_test = xTest["label"].tolist()
from sklearn.metrics import f1_score,accuracy_score,precision_score,precision_score,recall_score

print('f1_score macro: ', f1_score(y_test,label_results, average='macro'))
print('f1_score micro: ', f1_score(y_test,label_results, average='micro'))

from sklearn.metrics import f1_score, accuracy_score, precision_score, precision_score, recall_score
print('accuracy_score: ', accuracy_score(y_test,label_results))


print('precision_score macro: ', precision_score(y_test,label_results, average='macro'))
print('precision_score micro: ', precision_score(y_test,label_results, average='micro'))
print('recall_score macro: ', recall_score(y_test,label_results, average='macro'))
print('recall_score micro: ', recall_score(y_test,label_results, average='micro'))
print('f1_score macro: ', f1_score(y_test,label_results, average='macro'))
print('f1_score micro: ', f1_score(y_test,label_results, average='micro'))
#!/usr/bin/python

#from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import preprocess_df, print_squad_sample, from_df_to_model_dict
import json
import sys


# Load pre-trained model tokenizer (vocabulary)
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#import tensorflow as tf

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Input file:', str(sys.argv[1]))

try:
    with open(sys.argv[1]) as f:
        json_data = json.load(f)
except:
    print('Argument not valid')
    exit()

data = pd.json_normalize(json_data['data'])

df, context_dict = preprocess_df(data)
print(df.head())
print_squad_sample(df, context_dict)

'''
We load the model and we pass to it the input in the expected format
From the output we build the json output file

Input: 

Output:
'''
#train_dict, train_starts, train_ends = from_df_to_model_dict(df, context_dict)

output_data = {}
for i, question in df.iterrows():
    # here we put prediction for given answer
    output_data[question['question_id']] = "answer"

with open('pred.json', 'w') as outfile:
    json.dump(output_data, outfile)

#!/usr/bin/python
import transformers
from transformers import BertTokenizer
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
transformers.logging.set_verbosity_error() # suppress tokenizer sentences' length warnings

import numpy as np
import pandas as pd
from utils import create_output_dict, preprocess_df, print_squad_sample, from_df_to_model_dict
from model import build_model
import json
import sys

import tensorflow as tf

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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print(df.head())
print_squad_sample(df, context_dict)

print("\n\n")
print("#"*20)
print("Building model input")
print("#"*20)
print()
'''
We load the model and we pass to it the input in the expected format
From the output we build the json output file

Input: 

Output:
'''


test = from_df_to_model_dict(df, context_dict, tokenizer)
print("\n")
print("#"*20)
print("Loading model")
print("#"*20)
print("\n")
model = build_model()
model.load_weights('weights.h5')

# Predict test set
pred = model.predict([test["input_ids"],
                      test["attention_mask"],
                      test["token_type_ids"]],
                     verbose=1)

pred_start_ids = np.argmax(pred[0], axis=-1)
pred_end_ids = np.argmax(pred[1], axis=-1)

output_data = create_output_dict(
    test, context_dict, tokenizer, pred_start_ids, pred_end_ids)

with open('pred.json', 'w') as outfile:
    json.dump(output_data, outfile)

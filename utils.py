import random
import pandas as pd
import numpy as np


def preprocess_df(df):
    temp = []
    title_dict = {}
    contexts = []

    for i, row in df.iterrows():
        for context in row['paragraphs']:
            contexts.append(context['context'])
            for qa in context['qas']:
                question_id = qa['id']
                question = qa['question']
                temp.append([question_id, question, i, len(contexts)-1])

    context_dict = dict(enumerate(contexts))

    df = pd.DataFrame(
        temp, columns=['question_id', 'question_text', 'title_id', 'context_id'])

    return df, context_dict


def print_squad_sample(data, context_dict, line_length=120, separator_length=150):
    sample = data.sample(frac=1).head(1)
    context = context_dict[sample['context_id'].item()]
    print('='*separator_length)
    print('CONTEXT: ')
    print('='*separator_length)
    lines = [''.join(context[idx:idx+line_length])
             for idx in range(0, len(context), line_length)]
    for l in lines:
        print(l)
    # print(context)
    print('='*separator_length)
    questions = data[data['context_id']
                     == sample['context_id'].item()]
    print('QUESTIONS:\n')
    for idx, row in questions.iterrows():
        question = row.question_text
        print(question + '\n')


def from_df_to_model_dict(df, context_dict):
    return

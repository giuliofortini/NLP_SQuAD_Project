import random
import pandas as pd
import numpy as np
import transformers
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
transformers.logging.set_verbosity_error() # suppress tokenizer sentences' length warnings

SEQUENCE_LIMIT = 512
STRIDE = 256


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


def print_progress(value, total, bar_width=100):
    perc = int(value/total*bar_width)
    rest = bar_width - perc
    print("\r{:>3} % [ {}{} ]".format(perc, perc*'■', rest*'─'), end="")

# ensure to preserve all sub-sequences of length limi-stride.


def split_long_sequence(my_sequence, limit, stride):

    if len(my_sequence) <= limit:
        return [my_sequence]

    rest = my_sequence
    split = []
    while len(rest) > limit:
        left_hand = rest[:limit]
        rest = rest[stride:]
        split.append(left_hand)

    split.append(rest)
    return split


def from_df_to_model_dict(df, context_dict, tokenizer, verbose=False):

    # initialize structures
    question_id = []
    question_text = []
    context_id = []
    input_ids = []
    input_mask = []
    input_type_ids = []

    max_iter = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        # print progress
        print_progress(i+1, max_iter)

        # encode question and context
        if verbose:
            print("\tTokenizing question and context...", end="")
        encoded_question = tokenizer.encode(row['question_text'])
        encoded_context = tokenizer.encode(context_dict[row['context_id']])[1:]

        # concatenate input data
        if verbose:
            print("\tChecking sequence length...")
        if len(encoded_question + encoded_context) > SEQUENCE_LIMIT:
            # if the sequence is too long, split it in n subsequences of length <= SEQUENCE_LIMIT
            encoded_contexts = split_long_sequence(encoded_context,
                                                   limit=SEQUENCE_LIMIT -
                                                   len(encoded_question),
                                                   stride=STRIDE)
        else:
            encoded_contexts = [encoded_context]

        # for each too long sequence, the context has been split in n parts. We need to process them separately, creating new entries for the input
        for context_piece in encoded_contexts:

            encoded_input = encoded_question + context_piece

            # create mask of ones
            ones_mask = tf.ones_like(encoded_input)

            # add padding and convert to tensor
            if verbose:
                print("\tPadding...", end="")
            encoded_input = tf.keras.preprocessing.sequence.pad_sequences(
                [encoded_input], maxlen=512, padding='pre')
            encoded_input = tf.squeeze(tf.convert_to_tensor(encoded_input))

            # create input_type_ids
            if verbose:
                print("\tInput types creation...", end="")

            type_ids = tf.concat([tf.zeros_like(encoded_question, dtype=tf.int32),
                                  tf.ones_like(context_piece, dtype=tf.int32)],
                                 axis=-1)

            type_ids = tf.keras.preprocessing.sequence.pad_sequences(
                [type_ids], maxlen=512, padding='pre')

            type_ids = tf.squeeze(tf.convert_to_tensor(type_ids))

            # create mask of zeros
            if verbose:
                print("\tMask creation...", end="")
            zeros_mask = tf.zeros(
                SEQUENCE_LIMIT - len(ones_mask), dtype=tf.int32)
            mask = tf.concat([zeros_mask, ones_mask], axis=-1)

            # append elements to lists
            if verbose:
                print("\tAppending inputs...", end="")
            question_id.append(row['question_id'])
            question_text.append(row['question_text'])
            context_id.append(row['context_id'])
            input_ids.append(encoded_input)
            input_mask.append(mask)
            input_type_ids.append(type_ids)

    # save input data as dictionary
    data = {
        'question_id': question_id,
        'question_text': question_text,
        'context_id': context_id,
        'input_ids': tf.convert_to_tensor(input_ids),
        'attention_mask': tf.convert_to_tensor(input_mask),
        'token_type_ids': tf.convert_to_tensor(input_type_ids),
    }

    return data


def most_similar_answer(context, answer):
    original_answer = answer

    answer = answer.replace(" %", "%")

    answer = answer.replace(" . ", "")
    answer = answer.replace(" '", "")
    answer = answer.replace(' "', '')
    answer = answer.replace(" “", "")
    answer = answer.replace(" ”", "")
    answer = answer.replace(" , ", ",")
    # answer = answer.replace(" ,", "")
    answer = answer.replace(" ;", "")
    answer = answer.replace(" - ", "-")
    answer = answer.replace(" – ", "–")
    answer = answer.replace(" ( ", "")
    answer = answer.replace(" ) ", "")

    context = context.replace(".", "")
    context = context.replace(", ", " ")
    # context = context.replace(",", "")
    context = context.replace(";", "")
    context = context.replace('"', '')
    context = context.replace("“", "")
    context = context.replace("”", "")
    context = context.replace("'", " ")
    context = context.replace("\"", "")
    context = context.replace("(", "")
    context = context.replace(")", "")

    answer_list = answer.split(" ")
    context_list = context.split(" ")

    for i, element in enumerate(context_list):
        lowered = []
        for token in context_list[i:i + len(answer_list)]:
            lowered.append(token.lower())

        if lowered == answer_list:
            return True, ' '.join(context_list[i:i + len(answer_list)])

    return False, original_answer


def get_text_from_token_ids(context, tokenizer, start_id, end_id):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(context[start_id:end_id+1]))


def create_output_dict(test, test_context_dict, tokenizer, pred_start_ids, pred_end_ids):
    # output dictionary in requested format ('question_id': 'answer_text')
    outputs_dict = {}

    for i, input_ids in enumerate(test['input_ids']):
        # Extract variables
        question_id = test['question_id'][i]
        context = test_context_dict[test["context_id"][i]]
        pred_start = pred_start_ids[i]
        pred_end = pred_end_ids[i]

        predicted_answer = get_text_from_token_ids(
            input_ids, tokenizer, pred_start, pred_end)

        # Convert answer (if possible)
        convert_flag, converted_answer = most_similar_answer(
            context, predicted_answer)

        # If answer not found
        if pred_start == 0 and pred_end == 0:
            converted_answer = ""

        # Write answer dictionary
        outputs_dict[question_id] = converted_answer

    return outputs_dict

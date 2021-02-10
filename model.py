import tensorflow as tf

from keras import Model
from keras.utils import plot_model
from keras.layers import Input, Dense, Flatten, Dropout, Activation

import transformers
from transformers import BertTokenizer, TFBertModel
# suppress tokenizer sentences' length warnings
transformers.logging.set_verbosity_error()


@tf.autograph.experimental.do_not_convert
def build_model():

    # Define Input layers
    input_ids_layer = Input(shape=(512,), dtype='int64')
    attention_mask_layer = Input(shape=(512,), dtype='int64')
    token_ids_layer = Input(shape=(512,), dtype='int64')

    # Load Bert model
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    bert_model.trainable = True

    # Initialize bert model
    bert = bert_model([input_ids_layer,
                       attention_mask_layer,
                       token_ids_layer],
                      return_dict=False,
                      output_attentions=False,
                      output_hidden_states=False
                      ).last_hidden_state

    # Dropout
    dropout = Dropout(0.2)(bert)

    # Classifiers
    start_classifier = Dense(units=1, name="Start_classifier")(dropout)
    end_classifier = Dense(units=1, name="End_classifier")(dropout)

    # Flatteners
    start_flat = Flatten(name="Start_flattener")(start_classifier)
    end_flat = Flatten(name="End_flattener")(end_classifier)

    # Activations
    start_softmax = Activation(
        tf.keras.activations.softmax, name="Start_softmax")(start_flat)
    end_softmax = Activation(tf.keras.activations.softmax,
                             name="End_softmax")(end_flat)

    # Build model
    model = Model(inputs=[input_ids_layer, attention_mask_layer, token_ids_layer],
                  outputs=[start_softmax, end_softmax])

    model.summary()
    return model


model = build_model()
plot_model(model, show_shapes=True)

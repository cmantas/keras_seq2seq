# -*- coding: utf-8 -*-
from __future__ import print_function

# ignore some Keras warnings regarding deprecations and model saving
import warnings
warnings.filterwarnings('ignore')

from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, Dense, RepeatVector, \
                         TimeDistributed, Activation, GRU, Dropout,\
                         Bidirectional, Convolution1D, Dot,\
                         Concatenate
import pickle

from helpers import *
import importlib
from importlib import reload
from scipy.stats import hmean

batch_size = 512  # Batch size for training.
latent_dim = 512  # Latent dimensionality of the encoding space.
num_samples =200000 # Number of samples to train on.
# Path to the datatxt file on disk.
data_path = 'data/sentences.txt'

noise = .05
misspellings_count = 2
chunk_size = 10000

optimizer= 'adam'
loss_fn='categorical_crossentropy'

# Hand-pick maximum sequence lengths
max_encoder_seq_length = 25
max_decoder_seq_length = 30


all_phrases = load_preprocessed(data_path, max_encoder_seq_length)
input_phrases = all_phrases[:num_samples]
test_phrases = all_phrases[num_samples:]
print('All phrases in dataset: ', len(all_phrases))
print('Training phrases: ', len(input_phrases))
print('Test phrases: ', len(test_phrases))

print("\n * ".join(['Examples:'] + all_phrases[:10]))

# create doken indices out of all phrases
input_token_index = token_index(all_phrases)
num_encoder_tokens = len(input_token_index)
target_token_index = {'\t': num_encoder_tokens,
                      '\n': num_encoder_tokens+1,
                      **input_token_index}
num_decoder_tokens = len(target_token_index)

# Keep the count of all the possible input characters
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)




def vectorize_batch(texts, token_index, max_seq_len, offset=False):
    num_tokens = len(token_index)
    example_count = len(texts)

    # Generate 1-hot encoding
    data = np.zeros((example_count, max_seq_len, num_tokens), dtype='float')

    for i, text in enumerate(texts):
        start_t = 1 if offset else 0
        for t, char in enumerate(text[start_t:]):
            idx = token_index[char]
            data[i, t, idx] = 1.
    return data


def models(num_encoder_tokens, num_decoder_tokens, latent_dim):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens),
                           name='encoder_inputs')
    encoder = LSTM(latent_dim, return_state=True, name='encoder',
                   dropout=.05)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,
                        name='decoder_lstm', dropout=.05)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax',
                          name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_input_h')
    decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_input_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
        )
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
        )

    return model, encoder_model, decoder_model

def cnn_attention_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
  encoder_inputs = Input(shape=(None, num_encoder_tokens))
  # Encoder
  x_encoder = Convolution1D(256, kernel_size=3, activation='relu',
                            padding='causal')(encoder_inputs)
  x_encoder = Convolution1D(256, kernel_size=3, activation='relu',
                            padding='causal', dilation_rate=2)(x_encoder)
  x_encoder = Convolution1D(256, kernel_size=3, activation='relu',
                            padding='causal', dilation_rate=4)(x_encoder)

  decoder_inputs = Input(shape=(None, num_decoder_tokens))
  # Decoder
  x_decoder = Convolution1D(256, kernel_size=3, activation='relu',
                            padding='causal')(decoder_inputs)
  x_decoder = Convolution1D(256, kernel_size=3, activation='relu',
                            padding='causal', dilation_rate=2)(x_decoder)
  x_decoder = Convolution1D(256, kernel_size=3, activation='relu',
                            padding='causal', dilation_rate=4)(x_decoder)
  # Attention
  attention = Dot(axes=[2, 2])([x_decoder, x_encoder])
  attention = Activation('softmax')(attention)

  context = Dot(axes=[2, 1])([attention, x_encoder])
  decoder_combined_context = Concatenate(axis=-1)([context, x_decoder])

  decoder_combined_context = Dropout(.03)(decoder_combined_context)

  decoder_outputs = Convolution1D(64, kernel_size=3, activation='relu',
                                  padding='causal')(decoder_combined_context)
  decoder_outputs = Convolution1D(64, kernel_size=3, activation='relu',
                                  padding='causal')(decoder_outputs)
  # Output
  decoder_dense = TimeDistributed(
    Dense(num_decoder_tokens, activation='softmax')
  )
  decoder_outputs = decoder_dense(decoder_outputs)

  # Define the model that will turn
  # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
  model.summary()

  # Run training
  model.compile(optimizer='adam', loss='categorical_crossentropy')
  return model

model, encoder_model, decoder_model = models(num_encoder_tokens, num_decoder_tokens, latent_dim)
#model = test_attention_model(num_encoder_tokens, num_decoder_tokens, -1)
model.compile(optimizer=optimizer, loss=loss_fn,  metrics=['accuracy'])





def evaluate_correct(texts, corrector):
    errors = 0.0
    for t in texts:
        if t != corrector(t): errors += 1
    return errors / len(texts)

def evaluate_misspelled(texts, corrector):
    errors = 0.0
    for t in texts:
        errored = add_noise_to_string(t, 0.05)
        if t != corrector(errored): errors += 1
    return errors / len(texts)



model.compile(optimizer=optimizer, loss=loss_fn,  metrics=['accuracy'])

texts = input_phrases[:1_000]

target_texts = wrap_with_delims(texts)
encoder_input_data = vectorize_batch(
    texts, input_token_index,
    max_encoder_seq_length
)
decoder_input_data = vectorize_batch(
    wrap_with_delims(texts), target_token_index,
    max_decoder_seq_length
)
# same as decoder input data, but offset by one
decoder_output_data = vectorize_batch(
    target_texts, target_token_index,
    max_decoder_seq_length, True
)
X = [encoder_input_data, decoder_input_data]
Y = decoder_output_data

model.fit(X, Y, batch_size=batch_size)

#exit()

import gc

loss, val, val_acc = [], [], []
# for epoch in range(60):
#     print("(Real) Epoch: %s" % epoch)
#     # Initialize the generator
#     gen = training_generator()


#     l, v, va = [], [], []
#     i = 0
#     for X, Y in gen:
#         h = model.fit(X, Y,
#                       batch_size=batch_size,epochs=1, validation_split=0.1, verbose=1)
#         l.append(h.history['loss'][0])
#         v.append(h.history['val_loss'][0])
#         va.append(h.history['val_accuracy'][0])

#     del(gen)
#     gc.collect()

#     loss.append(hmean(l))
#     val.append(hmean(v))
#     val_acc.append(hmean(va))

#     print('loss:',hmean(l), 'val loss:', hmean(v), 'val acc:', hmean(va))
    #print('running eval')
    #test_err = evaluate_misspelled(test_phrases[:100], corrector)
    #print('Test Error:', test_err)
    #test.append(hmean(t))


corrector = translate_fn(encoder_model, decoder_model,
                         input_token_index, target_token_index,
                        max_encoder_seq_length)

evaluate_misspelled(test_phrases[:1000], corrector)

er_phrases = [add_noise_to_string(p, .05) for p in test_phrases[10:25]]
[(p , corrector(p)) for p in er_phrases]

[(corrector(p)) for p in ['cal me please', 'sotp it!', "helo i'm herre", 'is ths onn?'] ]

model.save('/drive/My Drive/ML/custom_functional_train.h5')
encoder_model.save('/drive/My Drive/ML/custom_functional_encoder.h5')
decoder_model.save('/drive/My Drive/ML/custom_functional_decoder.h5')

plt.plot(loss)
plt.plot(val)
plt.plot(test)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation', 'test'], loc='upper left')
plt.show()





test[-3:]

evaluate_correct(input_phrases[:1000], corrector)

evaluate_correct(input_phrases[-1000:], corrector)

evaluate_misspelled(input_phrases[:1000], corrector)

evaluate_correct(test_phrases[:1000], corrector)

evaluate_misspelled(test_phrases[:1000], corrector)



def evaluate_vect(input_texts, target_texts, model, training_vectorizer):
    target_texts = wrap_with_delims(target_texts)

    #wrapped_target_texts = wrap_with_delims(target_texts)
    X, Y = training_vectorizer(input_texts, target_texts)
    loss = model.evaluate(X, Y)
    print('\nTesting loss: ', loss)

misspelled = [add_noise_to_string(p, .05) for p in test_phrases[:1000]]
evaluate_vect(misspelled, test_phrases[:1000],
              model, training_vectorizer)

















# find max encoder seq legth
#max_encoder_seq_length = encoder_model.get_layer('encoder_inputs').input_shape[-1]
phrases = ['fire', 'stp', 'comein', 'get ot', 'i cant go','im sorry',
           'h is busi', 'hes drunk', 'ill be lat', 'hold mi beer', 'pus the buton',
          'coll me on my phone', 'helo boys and girls']

[corrector(phrase) for phrase in phrases]







def save():
    """quick-n-dirty helper for saving models"""
    print("Saving model")
    model.save('training.h5')
    encoder_model.save('encoder.h5')
    decoder_model.save('decoder.h5')

    model_metadata = { 'input_token_index': input_token_index,
                       'target_token_index': target_token_index,
                       'max_encoder_seq_length': max_encoder_seq_length }

    with open('model_metadata.pickle', 'wb') as f:
        pickle.dump(model_metadata, f)

save()

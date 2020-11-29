
# coding: utf-8

# In[2]:

# ignore some Keras warnings regarding deprecations and model saving
import warnings
warnings.filterwarnings('ignore')

from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, Dense, RepeatVector,                          TimeDistributed, Activation, GRU, Dropout,                          Bidirectional
from keras.callbacks import EarlyStopping
import pickle
from helpers import *


# In[3]:

# from google.colab import drive
# drive.mount('/drive')


# In[ ]:




# Sentences from the [tatoeba dataset](https://tatoeba.org/eng/downloads)

# In[4]:

epochs = 100  # Number of epochs to train for.
train_size =39500#0 # Number of samples to train on.
val_size = 2000
# Path to the txt file on disk.
prefix ='' #'/drive/My Drive/ML/'
data_path = prefix + 'sentences.txt'

noise = .05
misspellings_count = 3
batch_size = 1000  # Batch size for training.

optimizer= 'adam'
loss_fn='categorical_crossentropy'
# Hand-pick maximum sequence lengths
max_string_length = 25 # max (allowed) input string length
max_seq_length = 28 # accomodate for the delimiters + spelling correction


# In[ ]:




# In[5]:

all_phrases = load_preprocessed(data_path, max_string_length)
assert len(all_phrases) > train_size + val_size
train_phrases = all_phrases[:train_size]
test_phrases = all_phrases[train_size: train_size + val_size]

print('All phrases in dataset: ', len(all_phrases))
print('Training phrases: ', len(train_phrases))
print('Test phrases: ', len(test_phrases))

print("\n * ".join(['Examples:'] + all_phrases[:10]))


# In[6]:

# create doken indices out of all phrases
token_idx = token_index(all_phrases + ['\t', '\n'])
# ^^ \t and \n are our [START] and [END] delimiters. With this trick
# we are adding them to the token index

num_encoder_tokens = len(token_idx)

print('Number of unique tokens:', num_encoder_tokens)


# In[7]:

def simple_lstm(output_len, token_count, latent_dim = 128):
    """Generate the model"""
    initializer = 'he_normal'

    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    encoder = LSTM(latent_dim, input_shape=(None, token_count),
                   kernel_initializer=initializer)

    # For the decoder's input, we repeat the encoded input for each time step
    repeater = RepeatVector(output_len)

    decoder = LSTM(latent_dim, return_sequences=True, kernel_initializer=initializer)

    # For each of step of the output sequence, decide which character should be chosen
    time_dist = TimeDistributed(Dense(token_count, kernel_initializer=initializer))
    activation = Activation('softmax')

    model = Sequential([encoder, repeater, decoder, time_dist, activation])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[8]:

def deep_lstm(output_len, token_count):
    """Generate the model"""
    latent_dim = 256  # Latent dimensionality of the encoding space.
    initializer = 'he_normal'

    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    encoder_1 = LSTM(latent_dim, input_shape=(None, token_count),
                     return_sequences=True,
                     kernel_initializer=initializer)

    dropout1 = Dropout(.05)

    encoder_2 = LSTM(latent_dim, input_shape=(None, token_count),
                 kernel_initializer=initializer)

    # For the decoder's input, we repeat the encoded input for each time step
    repeater = RepeatVector(output_len)

    decoder_1 = LSTM(latent_dim, return_sequences=True, kernel_initializer=initializer)
    dropout2 = Dropout(.05)
    decoder_2 = LSTM(latent_dim, return_sequences=True, kernel_initializer=initializer)

    # For each of step of the output sequence, decide which character should be chosen
    time_dist = TimeDistributed(Dense(token_count, kernel_initializer=initializer))
    activation = Activation('softmax')

    model = Sequential([encoder_1, dropout1, encoder_2,
                        repeater,
                        decoder_1, dropout2, decoder_2,
                        time_dist, activation])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def bidrectional_lstm(output_len, token_count):
  """Generate the model"""
  latent_dim = 512  # Latent dimensionality of the encoding space.

  layers = [
      # something like an encoder
      Bidirectional(LSTM(latent_dim, return_sequences=True, dropout =.05),
                    input_shape=(output_len, token_count)),

      # ~decoder
      LSTM(latent_dim, return_sequences=True),

      # time-distr activation
      TimeDistributed(Dense(token_count,  activation='softmax')),
  ]

  model = Sequential(layers)

  model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])
  return model


# In[9]:

gen_model = simple_lstm


# In[ ]:




# In[10]:

val_misspellings, val_correct = create_misspellings(test_phrases[:2000],
                                            noise, misspellings_count,
                                            max_seq_length)

val_X = vectorize_batch(val_misspellings, token_idx,
                  max_seq_length, dtype=np.bool)
val_Y = vectorize_batch(wrap_with_delims(val_correct), token_idx,
                  max_seq_length, dtype=np.bool)


def batcher(phrases, batch_size):
  for i in range(0, len(phrases), batch_size):
    frrom = i
    to = i+batch_size
    yield phrases[frrom:to]

def chunkify(lst,n):
  return [lst[i::n] for i in range(n)]

def misspelled_gen(phrases, batch_size, noise, misspellings_count,
                  max_seq_length):
  gen = batcher(phrases, batch_size)
  for batch in gen:
    misspelled, correct = create_misspellings(batch,
                                      noise, misspellings_count,
                                      max_seq_length)
    mis_chunks = chunkify(misspelled, misspellings_count + 1)
    cor_chunks = chunkify(correct, misspellings_count + 1)

    yield from zip(mis_chunks, cor_chunks)


# In[13]:

def training_gen(phrases, batch_size, noise, misspellings_count,
                  max_seq_length, token_idx):

  while True:
    Random().shuffle(phrases)
    mis_gen = misspelled_gen(phrases, batch_size, noise, misspellings_count,
                             max_seq_length)
    for mis, cor in mis_gen:
      X = vectorize_batch(mis, token_idx,
                          max_seq_length, dtype=np.bool)
      Y = vectorize_batch(wrap_with_delims(cor), token_idx,
                          max_seq_length, dtype=np.bool)
      yield (X, Y)


# In[14]:

steps_per_epoch = (len(train_phrases) *(misspellings_count + 1) / batch_size)#30 <-for test


# In[15]:

gen = training_gen(train_phrases, batch_size, noise, misspellings_count,
                   max_seq_length, token_idx)

model = gen_model(max_seq_length, len(token_idx))

print(model.summary())

epochs = 1
hist = model.fit_generator(gen, validation_data=(val_X, val_Y),
                    steps_per_epoch=steps_per_epoch,
                    verbose=1, max_queue_size=1, epochs=epochs)


# In[16]:

print("MY eval:", dual_evaluate_batch(val_correct, val_misspellings, model, token_idx,
                    max_seq_length))


# In[17]:

model.save(prefix+'model_dim512_rd.05.h5')


# In[18]:

pickle.dump(hist, open('training_history.pickle', 'wb'))


# In[19]:

#plot_history(hist)


# In[20]:

corrector = sequential_translator_fn(model, token_idx, max_seq_length)


# In[23]:

# find max encoder seq legth
#max_encoder_seq_length = encoder_model.get_layer('encoder_inputs').input_shape[-1]
phrases = ['fire', 'stp', 'comein', 'get ot', 'i cant go','im sorry',
           'h is busi', 'hes drunk', 'ill be lat', 'hold mi beer', 'pus the buton',
          'coll me on my phone', 'helo boys and girls']

[corrector(phrase) for phrase in phrases]


# In[ ]:

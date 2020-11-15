from math import ceil


from helpers import *
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, Dense, RepeatVector,                                     TimeDistributed, Activation, GRU, Dropout,                            Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)

BATCH_SIZE = 100 #1000
OPTIMIZER = 'adam'
LOSS_FN = 'sparse_categorical_crossentropy'

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

class S2SModel:
    def __init__(self, max_string_length=25):
        # accomodate for the delimiters + spelling correction
        self.max_seq_length = max_string_length #+ 3
        self.token_idx = None
        self.inverse_token_index = None
        self.num_encoder_tokens = None
        self.tokenizer = None

    def init_from_texts(self, texts):
        # \t and \n are our [START] and [END] delimiters.
        # With this trick we are adding them to the token index
        self.token_idx = token_index(texts + ['\t', '\n'])
        self.inverse_token_index =  {
          v: k for k, v in self.token_idx.items()
        }
        self.num_encoder_tokens = len(self.token_idx)
        self.tokenizer = Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(texts + ['\t', '\n'])

    def vectorize_batch(self, texts):
        return vectorize_batch(
          texts, self.token_idx, self.max_seq_length, dtype='bool')

    def vectorize_output_batch(self, texts):
        #texts = wrap_with_delims(texts)
        #return self.vectorize_batch(delim_texts)
        seqs = self.tokenizer.texts_to_sequences(texts)
        pseqs = pad_sequences(seqs, self.max_seq_length, padding='post')
        return pseqs.reshape((*pseqs.shape, 1))

    def vectorize_phrase(self, txt):
      return vectorize_phrase(txt, self.token_idx,
                              self.max_seq_length)

    def training_gen(self, texts):
        while True:
            Random().shuffle(texts)
            X = self.vectorize_batch(texts)
            Y = self.vectorize_output_batch(texts)
            yield (X, Y)

    def create_model(self, latent_dim = 128):
      token_count = len(self.tokenizer.word_index)
      output_len = self.max_seq_length

      encoder = Bidirectional(
        LSTM(latent_dim, return_sequences=True),
        input_shape=(output_len, token_count)
      )
      # ~decoder
      decoder = Bidirectional(
        LSTM(latent_dim, return_sequences=True)
      )
      time_dist = Dense(token_count, activation='softmax')


      model = Sequential(
        [encoder, decoder, time_dist]
      )

      model.compile(loss=LOSS_FN,
                    optimizer=OPTIMIZER,
                    metrics=['accuracy'])
      return model


    def train(self, texts, epochs=1, init=True, test_size=None, verbose=1):
      if init:
        self.model = self.create_model()

      if len(texts) < 100:
        test_size = 0
      elif test_size is None:
        test_size = .1

      train_txts, test_txts = train_test_split(
        texts, test_size=test_size
      )

      val_X = self.vectorize_batch(test_txts)
      val_Y = self.vectorize_output_batch(test_txts)

      steps_per_epoch = ceil(len(texts) / BATCH_SIZE)

      gen = self.training_gen(texts)

      hist = self.model.fit(
        gen, validation_data=(val_X, val_Y),
        steps_per_epoch=steps_per_epoch,
        verbose=verbose, max_queue_size=1, epochs=epochs
      )

    def seq_to_text(self, seq):
      chars = [self.tokenizer.index_word.get(i, '#') for i in seq]
      return ''.join(chars)

    def predict(self, in_txts):
      wrap = isinstance(in_txts, str)

      txts = [in_txts] if wrap else in_txts

      x = self.vectorize_batch(txts)
      pred_seqs = self.model.predict_classes(x, verbose=0)
      out_txts = [self.seq_to_text(seq) for seq in pred_seqs]

      return out_txts[0] if wrap else out_txts

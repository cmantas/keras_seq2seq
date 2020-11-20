from math import ceil


from helpers import *
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, Dense, RepeatVector, \
  TimeDistributed, Activation, GRU, Dropout, Bidirectional, \
  Embedding, Lambda
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import sparse_categorical_accuracy

from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)

# dirty hack for renaming the accuracy metric
def acc(y, y_h):
    return sparse_categorical_accuracy(y, y_h)

def seq_acc(y_true, y_pred):
    a,b,_1 = y_true.shape
    y = y_true.numpy().reshape(a,b)
    y_h= np.argmax(y_pred.numpy(), 2)
    seq_acc = np.all((y == y_h), axis=1).sum() / len(y)
    return seq_acc

class S2SModel:
    BATCH_SIZE = 1000
    LATENT_DIM = 128
    OPTIMIZER = 'adam'
    LOSS_FN = 'sparse_categorical_crossentropy'

    def __init__(self, max_string_length=25, latent_dim=LATENT_DIM):
        # accomodate for the delimiters + spelling correction
        self.max_seq_length = max_string_length #+ 3
        self.token_idx = None
        self.inverse_token_index = None
        self.num_encoder_tokens = None
        self.tokenizer = None
        self.latent_dim = latent_dim
        self.hist = None

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
      seqs = self.tokenizer.texts_to_sequences(texts)
      return pad_sequences(seqs, self.max_seq_length, padding='post')

    def vectorize_output_batch(self, texts):
      #texts = wrap_with_delims(texts)
      seqs = self.vectorize_batch(texts)
      # reshape to a 3d array of (?, vocab_size, 1)
      return seqs.reshape((*seqs.shape, 1))


    def vectorize_phrase(self, txt):
      return vectorize_phrase(txt, self.token_idx,
                              self.max_seq_length)

    def training_gen(self, texts):
        while True:
            Random().shuffle(texts)
            for batch in batcher(texts, self.BATCH_SIZE):
              X = self.vectorize_batch(batch)
              Y = self.vectorize_output_batch(batch)
              yield (X, Y)

    @classmethod
    def one_hot_layer(cls, token_count):
      # Alternatively, with an embedding layer:
      # Embedding(input_dim=token_count, output_dim=token_count,
      #                    input_length=None, trainable=False,
      #                    embeddings_initializer='identity',
      #                    dtype='float16')
      # This could conceivably be trainable too

      # maybe: one_hot(K.cast(x,'uint8'), token_count))
      return Lambda(lambda x:K.one_hot(x, token_count))

    def create_model(self, latent_dim = 128):
      token_count = len(self.tokenizer.word_index)
      output_len = self.max_seq_length

      layers = [
          self.one_hot_layer(token_count),
          Bidirectional(
              LSTM(latent_dim, return_sequences=True),
              input_shape=(output_len, token_count),
          ),
          Bidirectional(
              LSTM(latent_dim, return_sequences=True)
          ),
          Bidirectional(
              LSTM(latent_dim, return_sequences=True)
          ),
          TimeDistributed(
              Dense(token_count, activation='softmax')
          )
      ]


      model = Sequential(layers)

      model.compile(loss=self.LOSS_FN,
                    optimizer=self.OPTIMIZER,
                    run_eagerly=True,
                    metrics=[acc, seq_acc])
      return model

    def steps_per_epoch(self, size):
        return ceil(size / self.BATCH_SIZE)

    def validation_data(self, val_texts):
        val_X = self.vectorize_batch(val_texts)
        val_Y = self.vectorize_output_batch(val_texts)
        return (val_X, val_Y)

    def train(self, texts, epochs=1, init=True, val_size=None, verbose=1):
      if init:
        self.model = self.create_model()

      if len(texts) < 100:
        val_size = 0
      elif val_size is None:
        val_size = .1

      train_txts, test_txts = train_test_split(
        texts, test_size=val_size
      )

      print(f"Training on {len(train_txts)} examples,"\
            f"validating on {len(test_txts)}")

      gen = self.training_gen(texts)

      self.hist = self.model.fit(
          gen, validation_data=self.validation_data(test_txts),
          steps_per_epoch=self.steps_per_epoch(len(texts)),
          verbose=verbose, max_queue_size=1, epochs=epochs
      )

    def seq_to_text(self, seq):
      chars = [self.tokenizer.index_word.get(i, '') for i in seq]
      return ''.join(chars)

    def predict(self, in_txts):
      wrap = isinstance(in_txts, str)

      txts = [in_txts] if wrap else in_txts

      x = self.vectorize_batch(txts)
      pred_seqs = self.model.predict_classes(x, verbose=0)
      out_txts = [self.seq_to_text(seq) for seq in pred_seqs]

      return out_txts[0] if wrap else out_txts

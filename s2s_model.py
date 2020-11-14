from helpers import *
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, Dense, RepeatVector,                                     TimeDistributed, Activation, GRU, Dropout,                            Bidirectional

BATCH_SIZE = 100 #1000
OPTIMIZER = 'adam'
LOSS_FN = 'categorical_crossentropy'

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
        self.max_seq_length = max_string_length + 3
        self.token_idx = None
        self.inverse_token_index = None
        self.num_encoder_tokens = None

    def init_from_texts(self, texts):
        # \t and \n are our [START] and [END] delimiters.
        # With this trick we are adding them to the token index
        self.token_idx = token_index(texts + ['\t', '\n'])
        self.inverse_token_index =  {
          v: k for k, v in self.token_idx.items()
        }
        self.num_encoder_tokens = len(self.token_idx)

    def vectorize_batch(self, texts):
        return vectorize_batch(texts, self.token_idx,
                               self.max_seq_length,
                               dtype=np.bool)

    def vectorize_output_batch(self, texts):
        delim_texts = wrap_with_delims(texts)
        return self.vectorize_batch(delim_texts)

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
      token_count = len(self.token_idx)
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


    def train(self, texts, epochs=1, init=True, test_size=None):
      if init:
        self.model = self.create_model()

      if test_size is None:
        test_size = .1

      train_txts, test_txts = train_test_split(
        texts, test_size=test_size
      )

      val_X = self.vectorize_batch(test_txts)
      val_Y = self.vectorize_output_batch(test_txts)

      steps_per_epoch = len(texts) / BATCH_SIZE

      gen = self.training_gen(texts)

      hist = self.model.fit(
        gen, validation_data=(val_X, val_Y),
        steps_per_epoch=steps_per_epoch,
        verbose=1, max_queue_size=1, epochs=epochs
      )

    def predict(self, txt):
      x = self.vectorize_phrase(txt)
      pred_idxes = self.model.predict_classes(x, verbose=0)[0]
      chars = [self.inverse_token_index[i] for i in pred_idxes]
      txt = ''.join(chars)
      end_idx = txt.find("\n")
      return txt[1:end_idx]






texts = ['abc', 'cbd'] * 10

model = S2SModel(3)
model.init_from_texts(texts)

m = model.create_model()

model.train(texts, 10)

pred = model.predict('abc')

max_len = 15
all_phrases = load_preprocessed('sentences.txt', max_len)
all_phrases = all_phrases[:3000]
BATCH_SIZE = 250
model = S2SModel(max_len)
model.init_from_texts(all_phrases)
model.train(all_phrases, 100)

print([model.predict(p) for p in all_phrases[:10]])

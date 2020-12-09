from math import ceil
from copy import copy
import pickle


from helpers import *
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    RepeatVector,
    TimeDistributed,
    Activation,
    GRU,
    Dropout,
    Bidirectional,
    Embedding,
    Lambda,
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import sparse_categorical_accuracy

from keras_adabound import AdaBound

from numpy.random import seed

seed(1)
from tensorflow.random import set_seed

set_seed(2)

# dirty hack for renaming the accuracy metric
def acc(y, y_h):
    return sparse_categorical_accuracy(y, y_h)


def seq_acc(y_true, y_pred):
    a, b = y_true.shape[:2]
    if len(y_pred.shape) == 3:
        y = y_true.numpy().reshape(a, b)
    else:
        y = y_true
    y_h = y_pred.numpy().argmax(2)
    seq_acc = np.all((y == y_h), axis=1).sum() / len(y)
    return seq_acc

def load_s2s_model(fname):
    s2s_model = pickle.load(open(fname + '.pickle', 'rb'))
    s2s_model.create_model()
    s2s_model.model.load_weights(fname + '.h5')
    return s2s_model

class S2SModel:
    BATCH_SIZE = 1000
    LATENT_DIM = 128
    OPTIMIZER = 'nadam' #AdaBound(lr=1e-2, final_lr=0.1) # also: nadam, adamax
    LOSS_FN = "sparse_categorical_crossentropy"

    def __init__(self, max_string_length=25, latent_dim=LATENT_DIM,
                 optimizer=OPTIMIZER):
        # accomodate for the delimiters + spelling correction
        self.max_seq_length = max_string_length  # + 3
        self.token_idx = None
        self.inverse_token_index = None
        self.tokenizer = None
        self.latent_dim = latent_dim
        self.optimizer=optimizer
        self.model = None
        self.token_count = None
        self.history = None

    def init_from_texts(self, texts):
        print(f"Creating a {self.__class__.__name__} Model with \n"\
              f"latent_dim={self.latent_dim}, optmizer={self.optimizer}")
        # \t and \n are our [START] and [END] delimiters.
        # With this trick we are adding them to the token index
        self.tokenizer = Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(texts + ["\t", "\n"])
        self.token_count = len(self.tokenizer.word_index)
        print(f"Tokenizing with {self.token_count} tokens")

    def vectorize_batch(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, self.max_seq_length, padding="post")

    def vectorize_input_batch(self, texts):
        return self.vectorize_batch(texts)

    def vectorize_output_batch(self, texts):
        # texts = wrap_with_delims(texts)
        seqs = self.vectorize_batch(texts)
        # reshape to a 3d array of (?, vocab_size, 1)
        return seqs.reshape((*seqs.shape, 1))

    def vectorize_phrase(self, txt):
        return vectorize_phrase(txt, self.token_idx, self.max_seq_length)

    def training_gen(self, texts):
        while True:
            Random().shuffle(texts)
            for batch in batcher(texts, self.BATCH_SIZE):
                yield self.vectorize_pairs(batch, batch)

    def one_hot_layer(self):
        # Create a simple embedding Layer that only does
        # one-hot encoding
        return Embedding(input_dim=self.token_count,
                         output_dim=self.token_count,
                         input_length=None, trainable=False,
                         embeddings_initializer='identity')
        # This embeddings could conceivably be trainable too

        # Alternatively:
        # Lambda(lambda x: K.one_hot(x, self.token_count))
        # Or casted to uint:
        # maybe: one_hot(K.cast(x,'uint8'), token_count))

    def output_layer(self):
        return TimeDistributed(
            Dense(self.token_count, activation='softmax')
        )

    def compile_model(self):
        self.model.compile(
            loss=self.LOSS_FN, optimizer=self.optimizer,
            run_eagerly=True,
            metrics=[acc, seq_acc],
        )

    def create_model(self):
        output_len = self.max_seq_length

        layers = [
            self.one_hot_layer(),
            Bidirectional(
                LSTM(self.latent_dim, return_sequences=True),
                input_shape=(output_len, self.token_count),
            ),
            Bidirectional(LSTM(self.latent_dim, return_sequences=True)),
            Bidirectional(LSTM(self.latent_dim, return_sequences=True)),
            self.output_layer()
        ]

        self.model = Sequential(layers)
        self.compile_model()

    def steps_per_epoch(self, size):
        return ceil(size / self.BATCH_SIZE)

    def vectorize_pairs(self, in_texts, out_texts):
        X = self.vectorize_batch(in_texts)
        Y = self.vectorize_output_batch(out_texts)
        return (X, Y)

    def validation_data(self, val_texts):
        # For the identity model, return the same texts as output
        return self.vectorize_pairs(val_texts, val_texts)

    def train(self, texts, epochs=1, init=False, val_size=None, verbose=1):
        if init or self.model is None:
            self.create_model()

        if len(texts) < 100:
            val_size = 0
        elif val_size is None:
            val_size = 0.1

        if val_size > 0:
            train_txts, test_txts = train_test_split(texts, test_size=val_size)
        else:
            train_txts, test_txts = texts, []

        print(
            f"Training on {len(train_txts)} examples," f"validating on {len(test_txts)}"
        )

        gen = self.training_gen(train_txts)

        try:
            self.model.fit(
                gen,
                validation_data=self.validation_data(test_txts),
                steps_per_epoch=self.steps_per_epoch(len(texts)),
                verbose=verbose,
                max_queue_size=1,
                epochs=epochs,
            )
        except KeyboardInterrupt:
            print("\n\nUnpacient are we?")
        finally:
            self.history = self.model.history.history

        print(self.last_training_metrics())

    def seq_to_text(self, seq):
        chars = [self.tokenizer.index_word.get(i, "") for i in seq]
        return "".join(chars)

    def predict(self, in_txts, confidence=False):
        wrap = isinstance(in_txts, str)

        txts = [in_txts] if wrap else in_txts

        x = self.vectorize_batch(txts)
        predictions = self.model.predict(x, verbose=0)
        pred_seqs = predictions.argmax(axis=2)
        out_txts = [self.seq_to_text(seq) for seq in pred_seqs]
        if not confidence:
            return out_txts[0] if wrap else out_txts

        pred_probs = predictions.max(axis=2).min(axis=1)
        zipped = list(zip(out_txts, list(pred_probs)))
        return zipped[0] if wrap else zipped

    def evaluate(self, in_txts, target_txts):
        predicted = self.predict(in_txts)
        right = sum([1 for yh, y in zip(predicted, target_txts) if yh == y])
        return float(right)/len(in_txts)

    def report(self, txts):
        acc = self.evaluate(txts, txts)
        print("Accuracy was: ", acc)

    def last_training_metrics(self):
        return {k: v[-1] for k,v in self.history.items()}

    def save(self, fpath):
        self.model.save_weights(fpath + '.h5')
        # creating a dummy copy, so that we don't mutate the current instance
        dummy = copy(self)
        dummy.model = None
        with open(fpath + '.pickle', 'wb') as f:
            pickle.dump(dummy, f)

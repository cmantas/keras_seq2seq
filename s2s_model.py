from math import ceil


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


class S2SModel:
    BATCH_SIZE = 1000
    LATENT_DIM = 128
    OPTIMIZER = "nadam"
    LOSS_FN = "sparse_categorical_crossentropy"

    def __init__(self, max_string_length=25, latent_dim=LATENT_DIM):
        # accomodate for the delimiters + spelling correction
        self.max_seq_length = max_string_length  # + 3
        self.token_idx = None
        self.inverse_token_index = None
        self.tokenizer = None
        self.latent_dim = latent_dim
        self.model = None
        self.token_count = None
        self.hist = None

    def init_from_texts(self, texts):
        print(f"Creating a {self.__class__.__name__} Model")
        # \t and \n are our [START] and [END] delimiters.
        # With this trick we are adding them to the token index
        self.tokenizer = Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(texts + ["\t", "\n"])
        self.token_count = len(self.tokenizer.word_index)

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
        # Alternatively, with an embedding layer:
        # Embedding(input_dim=token_count, output_dim=token_count,
        #                    input_length=None, trainable=False,
        #                    embeddings_initializer='identity',
        #                    dtype='float16')
        # This could conceivably be trainable too

        # maybe: one_hot(K.cast(x,'uint8'), token_count))
        return Lambda(lambda x: K.one_hot(x, self.token_count))

    def output_layer(self):
        return TimeDistributed(
            Dense(self.token_count, activation='softmax')
        )

    def compile_model(self):
        self.model.compile(
            loss=self.LOSS_FN, optimizer=self.OPTIMIZER,
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

        self.hist = self.model.fit(
            gen,
            validation_data=self.validation_data(test_txts),
            steps_per_epoch=self.steps_per_epoch(len(texts)),
            verbose=verbose,
            max_queue_size=1,
            epochs=epochs,
        )

    def seq_to_text(self, seq):
        chars = [self.tokenizer.index_word.get(i, "") for i in seq]
        return "".join(chars)

    def predict(self, in_txts):
        wrap = isinstance(in_txts, str)

        txts = [in_txts] if wrap else in_txts

        x = self.vectorize_batch(txts)
        pred_seqs = self.model.predict(x, verbose=0).argmax(axis=2)
        out_txts = [self.seq_to_text(seq) for seq in pred_seqs]

        return out_txts[0] if wrap else out_txts

    def evaluate(self, txts):
        predicted = self.predict(txts)
        right = sum([1 for yh, y in zip(predicted, txts) if yh == y])
        return float(right)/len(txts)

    def report(self, txts):
        acc = self.evaluate(txts)
        print("Accuracy was: ", acc)

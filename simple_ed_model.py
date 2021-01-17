from s2s_model import S2SModel
from spelling_model import SpellingModel

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    RepeatVector,
    Input,
    TimeDistributed,
    Dense,
    Attention,
    Activation,
    concatenate,
    dot,
    Conv1D,
    Reshape,
    Embedding,
    Layer,
    Dropout,
)
import tensorflow as tf


from tensorflow.keras import backend as K
import numpy as np

class SEDModel(S2SModel):
    BATCH_SIZE = 250
    def create_model(self):
        output_len = self.max_seq_length

        layers = [
            self.one_hot_layer(),
            Bidirectional(
                LSTM(self.latent_dim, return_sequences=False),
                input_shape=(output_len, self.token_count),
            ),
            RepeatVector(output_len),
            Bidirectional(LSTM(self.latent_dim, return_sequences=True)),
            self.output_layer()
        ]

        self.model = Sequential(layers)
        self.compile_model()


class SEDSpellingModel(SEDModel, SpellingModel):
    pass

class SEDAModel(S2SModel):
    def create_model(self):
        output_len = self.max_seq_length

        encoder_input = Input(shape=(self.max_seq_length), dtype='int32')
        one_hot_emb = self.one_hot_layer()
        lstm_input = one_hot_emb(encoder_input)

        encoder = Bidirectional(
            LSTM(self.latent_dim, return_sequences=True),
            input_shape=(output_len, self.token_count),
        )

        # Due to `return_sequences` the encoder outputs are of shape
        # (X, sequence_length, 2 x LSTM hidden dim).
        # we only need the last timestep for our decoder input
        encoder_output = encoder(lstm_input)
        encoder_last = encoder_output[:,-1,:]

        repeated = RepeatVector(output_len)(encoder_last)

        decoder = Bidirectional(LSTM(self.latent_dim, return_sequences=True))
        decoder_output = decoder(repeated)

        # custom attention
        #attention = dot([decoder_output, encoder_output], axes=[2, 2])
        #attention = Activation('softmax', name='attention')(attention)
        #context = dot([attention, encoder_output], axes=[2,1])
        #decoder_combined_context = concatenate([context, decoder_output])

        attention = Attention()
        decoder_combined_context = attention([decoder_output, encoder_output])
        #decoder_combined_context = concatenate([context, decoder_output])

        td_dense = TimeDistributed(
            Dense(self.latent_dim, activation='tanh')
        )
        output_1 = td_dense(decoder_combined_context)
        output = self.output_layer()(output_1)

        self.model =  Model(inputs=encoder_input, outputs=output)
        self.compile_model()

class SEDASpellingModel(SEDAModel, SpellingModel):
    pass

class SEDSpellingModel(SEDModel, SpellingModel):
    pass


class CSEDAModel(S2SModel):
    # https://blog.codecentric.de/en/2019/07/move-n-gram-extraction-into-your-keras-model/
    def ngram_block(self, n):
        alphabet_size = self.token_count

        def wrapped(inputs):
            layer = Conv1D(1, n, use_bias=False, trainable=False)
            x = Reshape((-1, 1))(inputs)
            x = layer(x)
            kernel = np.power(alphabet_size, range(0, n),
                              dtype=K.floatx())
            layer.set_weights([kernel.reshape(n, 1, 1)])
            return Reshape((-1,))(x)

        return wrapped

    def create_model(self):
        output_len = self.max_seq_length

        inputt = Input(shape=(self.max_seq_length), dtype='float32')
        n = 2
        ngrams = self.ngram_block(n)(inputt)
        embedded = Embedding(pow(self.token_count, n), 100)(ngrams)

        encoder = Bidirectional(
            LSTM(self.latent_dim, return_sequences=True),
            input_shape=(output_len, self.token_count),
        )

        # Due to `return_sequences` the encoder outputs are of shape
        # (X, sequence_length, 2 x LSTM hidden dim).
        # we only need the last timestep for our decoder input
        encoder_output = encoder(embedded)
        encoder_last = encoder_output[:,-1,:]

        repeated = RepeatVector(output_len)(encoder_last)

        decoder_output = Bidirectional(
            LSTM(self.latent_dim, return_sequences=True)
        )(repeated)

        # custom attention
        attention = dot([decoder_output, encoder_output], axes=[2, 2])
        attention = Activation('softmax', name='attention')(attention)
        context = dot([attention, encoder_output], axes=[2,1])
        decoder_combined_context = concatenate([context, decoder_output])

        td_dense = TimeDistributed(
            Dense(self.latent_dim, activation='tanh')
        )
        output_1 = td_dense(decoder_combined_context)
        output = self.output_layer()(output_1)

        self.model =  Model(inputs=inputt, outputs=output)
        self.compile_model()


class CSEDASpellingModel(CSEDAModel, SpellingModel):
    pass


class ECCNNModel(S2SModel):
    def create_model(self):
        output_len = self.max_seq_length

        inputt = Input(shape=(self.max_seq_length), dtype='int32')
        emb = self.one_hot_layer()
        emb.trainable = True
        embedded = emb(inputt)

        conv2 = Conv1D(
            self.latent_dim, kernel_size=2, activation='tanh', padding='same'
        )(embedded)

        lstm_input = concatenate([embedded, conv2])

        encoder_output = Bidirectional(
            LSTM(self.latent_dim, return_sequences=True),
            input_shape=(output_len, self.token_count),
        )(lstm_input)
        # Due to `return_sequences` the encoder outputs are of shape
        # (X, sequence_length, 2 x LSTM hidden dim).
        # we only need the last timestep for our decoder input
        encoder_last = encoder_output[:,-1,:]

        repeated = RepeatVector(output_len)(encoder_last)

        decoder_output = Bidirectional(
            LSTM(self.latent_dim, return_sequences=True)
        )(repeated)

        # custom attention
        attention = dot([decoder_output, encoder_output], axes=[2, 2])
        attention = Activation('softmax', name='attention')(attention)
        context = dot([attention, encoder_output], axes=[2,1])
        decoder_combined_context = concatenate([context, decoder_output])

        td_dense = TimeDistributed(
            Dense(self.latent_dim, activation='tanh')
        )
        output_1 = td_dense(decoder_combined_context)
        output = self.output_layer()(output_1)

        self.model =  Model(inputs=inputt, outputs=output)
        self.compile_model()


class ECCNNSpellingModel(ECCNNModel, SpellingModel ):
    pass

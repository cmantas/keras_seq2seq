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
    dot
)

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
        raise 'this model does not really work...'
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
        attention = dot([decoder_output, encoder_output], axes=[2, 2])
        attention = Activation('softmax', name='attention')(attention)
        context = dot([attention, encoder_output], axes=[2,1])
        decoder_combined_context = concatenate([context, decoder_output])

        #attention = Attention()
        #decoder_combined_context = attention([decoder_output, encoder_output])

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

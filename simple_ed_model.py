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
    Attention
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
            LSTM(self.latent_dim, return_sequences=False),
            input_shape=(output_len, self.token_count),
        )

        encoder_output = encoder(lstm_input)

        repeated = RepeatVector(output_len)(encoder_output)

        decoder = Bidirectional(LSTM(self.latent_dim, return_sequences=True))
        decoder_output = decoder(repeated)

        attention = Attention()
        decoder_combined_context = attention([decoder_output, encoder_output])

        td_dense = TimeDistributed(Dense(self.latent_dim, activation='tanh'))
        output_1 = td_dense(decoder_combined_context)
        output = self.output_layer()(output_1)

        self.model =  Model(inputs=encoder_input, outputs=output)
        self.compile_model()

class SEDASpellingModel(SEDAModel, SpellingModel):
    pass

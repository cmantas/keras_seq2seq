from encoder_decoder_model import *
from spelling_model import SpellingModel
from tensorflow.keras.layers import dot, concatenate, Attention

class EDAModel(EDModel):
    def create_model(self):
        # Encoder
        encoder_input = Input(shape=(self.max_seq_length), dtype='int32')
        decoder_input = Input(shape=(self.max_seq_length), dtype='int32')

        embedding = self.one_hot_layer()
        lstm_input = embedding(encoder_input)

        encoder = LSTM(self.latent_dim, return_sequences=True, unroll=True)
        encoder_output = encoder(lstm_input)
        encoder_last_output = encoder_output[:,-1,:]


        one_hot = self.one_hot_layer()
        decoder_data = one_hot(decoder_input)
        decoder = LSTM(self.latent_dim, return_sequences=True, unroll=True)
        decoder_output = decoder(
            decoder_data,
            initial_state=[encoder_last_output, encoder_last_output]
        )

        #attention = dot([decoder, encoder], axes=[2, 2])
        #attention = Activation('softmax', name='attention')(attention)

        #context = dot([attention, encoder], axes=[2,1])
        #print('context', context)

        #decoder_combined_context = concatenate([context, decoder])

        attention = Attention()
        decoder_combined_context = attention([decoder_output, encoder_output])

        td_dense = TimeDistributed(Dense(self.latent_dim, activation='tanh'))
        output_1 = td_dense(decoder_combined_context)
        output = self.output_layer()(output_1)

        self.model = Model(
            inputs=[encoder_input, decoder_input],
            outputs=[output]
        )
        self.compile_model()

class EDASpellModel(EDAModel, SpellingModel):
    pass

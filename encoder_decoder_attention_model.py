from encoder_decoder_model import *
from keras.layers import dot, concatenate, Attention

class EDAModel(EDModel):
    def create_model(self, latent_dim=128):
        # Encoder
        encoder_input = Input(shape=(self.max_seq_length), dtype='int32')
        decoder_input = Input(shape=(self.max_seq_length), dtype='int32')

        embedding = self.one_hot_layer()
        lstm_input = embedding(encoder_input)

        encoder = LSTM(64, return_sequences=True, unroll=True)(lstm_input)
        encoder_last = encoder[:,-1,:]


        one_hot = self.one_hot_layer()
        decoder_data = one_hot(decoder_input)
        decoder = LSTM(
            64, return_sequences=True, unroll=True
        )(decoder_data, initial_state=[encoder_last, encoder_last])

        #attention = dot([decoder, encoder], axes=[2, 2])
        #attention = Activation('softmax', name='attention')(attention)

        #context = dot([attention, encoder], axes=[2,1])
        #print('context', context)

        #decoder_combined_context = concatenate([context, decoder])

        decoder_combined_context = Attention()([decoder, encoder])

        output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)
        output = self.output_layer()(output)

        model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
        model.compile(optimizer=self.OPTIMIZER,
                      loss=self.LOSS_FN,
                      metrics=['sparse_categorical_accuracy'])

        self.model=model

class EDASpellModel(EDAModel, EDSpellModel):
    pass

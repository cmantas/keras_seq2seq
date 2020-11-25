from encoder_decoder_model import *
from keras.layers import dot, concatenate

class EDAModel(EEModel):
    def create_model(self, latent_dim=128):
        token_count = len(self.tokenizer.word_index)

        # Encoder
        encoder_input = Input(shape=(self.max_seq_length), dtype='int32')
        decoder_input = Input(shape=(self.max_seq_length), dtype='int32')

        embedding = self.one_hot_layer(token_count)
        lstm_input = embedding(encoder_input)

        encoder = LSTM(64, return_sequences=True, unroll=True)(lstm_input)
        encoder_last = encoder[:,-1,:]


        one_hot = self.one_hot_layer(token_count)
        decoder_data = one_hot(decoder_input)
        decoder = LSTM(
            64, return_sequences=True, unroll=True
        )(decoder_data, initial_state=[encoder_last, encoder_last])

        attention = dot([decoder, encoder], axes=[2, 2])
        attention = Activation('softmax', name='attention')(attention)

        context = dot([attention, encoder], axes=[2,1])
        print('context', context)

        decoder_combined_context = concatenate([context, decoder])

        output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)
        output = TimeDistributed(Dense(token_count, activation="softmax"))(output)

        model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

        self.model=model

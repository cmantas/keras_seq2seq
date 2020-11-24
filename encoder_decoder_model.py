from s2s_model import *

class EEModel(S2SModel):
    def create_model(self, latent_dim=128):
        encoder_input = Input(shape=(self.max_seq_length,), dtype='uint8')
        decoder_input = Input(shape=(self.max_seq_length,), dtype='uint8')

        token_count = len(self.tokenizer.word_index)
        embedding = self.one_hot_layer(token_count)
        # Embedding(input_dim=token_count, output_dim=token_count,
        #                        input_length=None, trainable=False,
        #                        embeddings_initializer='identity',mask_zero=True)
        lstm_input = embedding(encoder_input)
        encoder = LSTM(latent_dim, return_sequences=False)(lstm_input)

        # decoder = Embedding(input_dim=token_count, output_dim=token_count,
        #                     input_length=None, mask_zero=True)(decoder_input)
        decoder = self.one_hot_layer(token_count)(decoder_input)

        decoder = LSTM(
            latent_dim, return_sequences=True
        )(decoder, initial_state=[encoder, encoder])

        decoder = TimeDistributed(Dense(token_count, activation="softmax"))(decoder)

        model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder])
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        self.model=model

    def vectorize_sketo(self, texts):
        token_count = len(self.tokenizer.word_index)
        encoded_input  = self.vectorize_batch(texts)
        encoded_output = self.vectorize_batch(texts)


        encoder_input = encoded_input
        decoder_input = np.zeros_like(encoded_output)
        decoder_input[:, 1:] = encoded_output[:, :-1]
        #decoder_input[:, 0] = self.tokenizer.word_index['\t']

        decoder_output = np.eye(token_count)[encoded_output.astype('int')]
        X = [encoder_input, decoder_input]
        Y = [decoder_output]
        return(X, Y)

    def train(self, texts, epochs=1, init=False, val_size=None, verbose=1):
        if init or self.model is None:
            self.create_model()

        X, Y = self.vectorize_sketo(texts)

        self.hist = self.model.fit(
            X, Y, epochs=epochs, batch_size = 10
        )

    def predict(self, texts):

        (encoder_input, decoder_input), _= self.vectorize_sketo(texts)
        preds = self.model.predict([encoder_input, decoder_input]).argmax(axis=2)

        return [self.seq_to_text(seq) for seq in preds]


        print(preds.shape)
        exit()
        for i in range(1, self.max_seq_length):
            output = self.model.predict([encoder_input, decoder_input]).argmax(axis=2)
            decoder_input[:, i] = output[:, i]

        return [self.seq_to_text(seq) for seq in decoder_input]

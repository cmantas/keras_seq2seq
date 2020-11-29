from s2s_model import *

class EDModel(S2SModel):
    def create_model(self, latent_dim=128):
        token_count = len(self.tokenizer.word_index)
        # Inputs
        encoder_input = Input(shape=(self.max_seq_length), dtype='int32')
        decoder_input = Input(shape=(self.max_seq_length), dtype='int32')

        # Encoder Input Data: we are not using embeddings but simply one-hot
        # char vectors
        one_hot_enc = self.one_hot_layer(token_count)
        lstm_input = one_hot_enc(encoder_input)

        # Encoder
        encoder = LSTM(latent_dim, return_sequences=False)
        encoder_output = encoder(lstm_input)

        # Decoder Input Data
        one_hot_dec = self.one_hot_layer(token_count)
        decoder_data = one_hot_dec(decoder_input)

        decoder = LSTM(latent_dim, return_sequences=True)
        decoder_output = decoder(
            decoder_data, initial_state=[encoder_output, encoder_output]
        )

        # Dense
        t_dense = TimeDistributed(Dense(token_count, activation="softmax"))
        output = t_dense(decoder_output)

        model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
        model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS_FN,
                      metrics=['sparse_categorical_accuracy'])

        self.model=model

    def vectorize_sketo(self, texts):
        encoder_input = self.vectorize_batch(texts)

        encoded_output = self.vectorize_batch(texts)
        decoder_input = np.zeros_like(encoded_output)
        decoder_input[:, 1:] = encoded_output[:, :-1]
        #decoder_input[:, 0] = self.tokenizer.word_index['\t']

        decoder_output = encoded_output
        X = (encoder_input, decoder_input)
        Y = decoder_output
        return(X, Y)

    def train(self, texts, epochs=1, init=False, val_size=None, verbose=1):
        if init or self.model is None:
            self.create_model()

        X, Y = self.vectorize_sketo(texts)

        self.hist = self.model.fit(
            X, Y, epochs=epochs, batch_size = 100
        )

    def predict(self, texts):

        (encoder_input, decoder_input), _ = self.vectorize_sketo(texts)
        preds = self.model.\
            predict([encoder_input, decoder_input]). \
            argmax(axis=2)

        return [self.seq_to_text(seq) for seq in preds]

        # TODO: incorporate this ...
        for i in range(1, self.max_seq_length):
            output = self.model.predict([encoder_input, decoder_input]).argmax(axis=2)
            decoder_input[:, i] = output[:, i]

        return [self.seq_to_text(seq) for seq in decoder_input]

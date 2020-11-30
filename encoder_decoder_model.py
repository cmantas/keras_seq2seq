from s2s_model import *
from spelling_model import SpellingModel

class EDModel(S2SModel):
    BATCH_SIZE = 100 # smaller batches seem to have fastest conversion

    def create_model(self, latent_dim=128):
        # Inputs
        encoder_input = Input(shape=(self.max_seq_length), dtype='int32')
        decoder_input = Input(shape=(self.max_seq_length), dtype='int32')

        # Encoder Input Data: we are not using embeddings but simply one-hot
        # char vectors
        one_hot_enc = self.one_hot_layer()
        lstm_input = one_hot_enc(encoder_input)

        # Encoder
        encoder = LSTM(latent_dim, return_sequences=False)
        encoder_output = encoder(lstm_input)

        # Decoder Input Data
        one_hot_dec = self.one_hot_layer()
        decoder_data = one_hot_dec(decoder_input)

        decoder = LSTM(latent_dim, return_sequences=True)
        decoder_output = decoder(
            decoder_data, initial_state=[encoder_output, encoder_output]
        )

        # Dense
        t_dense = self.output_layer()
        output = t_dense(decoder_output)

        self.model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
        self.compile_model()

    def vectorize_pairs(self, in_texts, out_texts):
        encoder_input = self.vectorize_batch(in_texts)

        # TODO: maybe the encoder input is not like encoder input
        decoder_input = np.zeros_like(encoder_input)
        decoder_input[:, 1:] = encoder_input[:, :-1]
        #decoder_input[:, 0] = self.tokenizer.word_index['\t']

        X = (encoder_input, decoder_input)
        if out_texts is None:
            return X, None

        decoder_output = self.vectorize_output_batch(out_texts)
        Y = decoder_output
        return(X, Y)


    def predict(self, texts):
        (encoder_input, decoder_input), _ = self.vectorize_pairs(texts, None)
        preds = self.model.\
            predict([encoder_input, decoder_input]). \
            argmax(axis=2)

        return [self.seq_to_text(seq) for seq in preds]

        # TODO: incorporate this ...
        for i in range(1, self.max_seq_length):
            output = self.model.predict([encoder_input, decoder_input]).argmax(axis=2)
            decoder_input[:, i] = output[:, i]

        return [self.seq_to_text(seq) for seq in decoder_input]


class EDSpellModel(EDModel, SpellingModel):
    pass

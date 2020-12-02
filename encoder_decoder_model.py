from s2s_model import *
from spelling_model import SpellingModel

class EDModel(S2SModel):
    BATCH_SIZE = 100 # smaller batches seem to have fastest conversion
    CHAR_CODE_START = '\t'

    def init_from_texts(self, texts):
         super().init_from_texts(texts + [self.CHAR_CODE_START])

    def start_token(self):
        return self.tokenizer.word_index[self.CHAR_CODE_START]

    def create_model(self):
        # Inputs
        encoder_input = Input(shape=(self.max_seq_length), dtype='int32')
        decoder_input = Input(shape=(self.max_seq_length), dtype='int32')

        # Encoder Input Data: we are not using embeddings but simply one-hot
        # char vectors
        one_hot_enc = self.one_hot_layer()
        lstm_input = one_hot_enc(encoder_input)

        # Encoder
        encoder = LSTM(self.latent_dim, return_sequences=False)
        encoder_output = encoder(lstm_input)

        # Decoder Input Data
        one_hot_dec = self.one_hot_layer()
        decoder_data = one_hot_dec(decoder_input)

        decoder = LSTM(self.latent_dim, return_sequences=True)
        decoder_output = decoder(
            decoder_data, initial_state=[encoder_output, encoder_output]
        )

        # Dense
        t_dense = self.output_layer()
        output = t_dense(decoder_output)

        self.model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
        self.compile_model()

    def vectorize_output_batch(self, texts):
        padded = ["\t" + t for t in texts]
        return self.vectorize_batch(padded)

    def vectorize_pairs(self, in_texts, out_texts):
        encoder_input = self.vectorize_batch(in_texts)

        decoder_output = self.vectorize_output_batch(out_texts)

        # TODO: maybe the encoder input is not like encoder input
        decoder_input = np.zeros_like(encoder_input)

        # teacher forcing: use as decoder inputs the ground-truth outputs,
        # shifted by +1.
        decoder_input[:, 1:] = decoder_output[:, :-1]
        decoder_input[:, 0] = self.start_token()

        X = (encoder_input, decoder_input)
        if out_texts is None:
            return X, None

        #decoder_output = self.vectorize_output_batch(out_texts)
        Y = decoder_output
        return(X, Y)

    def seq_to_text(self, seq):
        return super().seq_to_text(seq)[1:]

    def predict(self, in_texts):
        encoder_input = self.vectorize_batch(in_texts)
        decoder_input = np.zeros((len(in_texts), self.max_seq_length))

        # start with the decoder input sequences all containing the start token
        decoder_input[:, 0] = self.start_token()

        # each char of the output needs to be predicted based on the previous chars
        for i in range(1, self.max_seq_length):
            # for each
            output =  self.model.predict([encoder_input, decoder_input]).\
                                 argmax(axis=2)
            decoder_input[:, i] = output[:, i]
            # Note: this is practically identical to
            # output = decoder_input = self.model.predict([encoder_input, decoder_input])
            # but that doesn't read as well....

        decoder_output = decoder_input # why not just use `output` ?

        return [self.seq_to_text(seq) for seq in decoder_output]


class EDSpellModel(EDModel, SpellingModel):
    pass

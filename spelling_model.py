from s2s_model import S2SModel
from helpers import *
from math import ceil


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def misspelled_gen(phrases, batch_size, noise, misspellings_count, max_seq_length):
    gen = batcher(phrases, batch_size)
    for batch in gen:
        misspelled, correct = create_misspellings(
            batch, noise, misspellings_count, max_seq_length
        )
        mis_chunks = chunkify(misspelled, misspellings_count + 1)
        cor_chunks = chunkify(correct, misspellings_count + 1)

        generated = list(zip(mis_chunks, cor_chunks))
        shuffle(generated)
        yield from generated


class SpellingModel(S2SModel):
    MISSPELLING_RATIO = 3

    def training_gen(self, texts):
        while True:
            Random().shuffle(texts)
            mis_gen = misspelled_gen(
                texts, self.BATCH_SIZE, 0.05, self.MISSPELLING_RATIO,
                self.max_seq_length
            )
            for mis, cor in mis_gen:
                X = self.vectorize_batch(mis)
                Y = self.vectorize_output_batch(cor)
                yield (X, Y)

    def steps_per_epoch(self, size):
        e_size = size * (self.MISSPELLING_RATIO + 1)
        return ceil(e_size / self.BATCH_SIZE)

    def validation_data(self, val_texts):
        wrong_texts, right_texts = create_misspellings(
            val_texts, .05, self.MISSPELLING_RATIO,
            self.max_seq_length
        )
        val_X = self.vectorize_batch(wrong_texts)
        val_Y = self.vectorize_output_batch(right_texts)
        return (val_X, val_Y)

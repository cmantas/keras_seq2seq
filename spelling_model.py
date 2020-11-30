from s2s_model import S2SModel
from helpers import *
from math import ceil


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


class SpellingModel(S2SModel):
    MISSPELLING_RATIO = 3

    def generate_synthetic_pairs(self, texts, noise=.05):
        misspelled, correct = create_misspellings(
            texts, .05, self.MISSPELLING_RATIO, self.max_seq_length
        )

        generated = list(zip(misspelled, correct))
        Random(1).shuffle(generated)
        return generated

    def training_gen(self, texts):
        print(f"Generating Noisy texts on a ratio of 1/{self.MISSPELLING_RATIO}")
        while True:
            generated = self.generate_synthetic_pairs(texts)
            for batch in batcher(generated, self.BATCH_SIZE):
                miss, corr = zip(*batch) # unzip
                yield self.vectorize_pairs(miss, corr)

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

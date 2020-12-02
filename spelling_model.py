from s2s_model import S2SModel
from helpers import *
from math import ceil


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


class SpellingModel(S2SModel):
    MISSPELLING_RATIO = 3
    NOISE = .05

    def generate_synthetic_pairs(self, texts, noise=NOISE):
        misspelled, correct = create_misspellings(
            texts, .05, self.MISSPELLING_RATIO, self.max_seq_length
        )

        generated = list(zip(misspelled, correct))
        Random(1).shuffle(generated)
        return generated

    def training_gen(self, texts):
        print(f"Generating Noisy texts on a ratio of 1/{self.MISSPELLING_RATIO} (in batches of {self.BATCH_SIZE})")
        while True:
            generated = self.generate_synthetic_pairs(texts)
            for batch in batcher(generated, self.BATCH_SIZE):
                miss, corr = zip(*batch) # unzip
                yield self.vectorize_pairs(miss, corr)

    def steps_per_epoch(self, size):
        e_size = size * (self.MISSPELLING_RATIO + 1)
        return ceil(e_size / self.BATCH_SIZE)

    def validation_data(self, val_texts):
        generated = self.generate_synthetic_pairs(val_texts)
        miss, corr = zip(*generated) # unzip
        return self.vectorize_pairs(miss, corr)

    def report(self, txts):
        identity_acc = self.evaluate(txts, txts)
        misspelled = [add_noise_to_string(t, self.NOISE) for t in txts]
        misspelled_acc = self.evaluate(misspelled, txts)
        print('Accuracy on correct:', identity_acc,
              ' on misspelled:', misspelled_acc)

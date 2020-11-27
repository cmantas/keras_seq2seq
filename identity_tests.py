from  s2s_model import *
from helpers import *
from encoder_decoder_model import *
from encoder_decoder_attention_model import *


def acc(src, trgt):
    pairs = list(zip(src,trgt))
    return sum([1 for p,t in pairs if p == t]) / float(len(pairs))

def report(model, phrases, targets, legend='Report'):
    preds = model.predict(phrases)
    pairs = list(zip(preds, phrases))
    print(legend)
    print(pairs)
    print(legend, "Accuracy:", acc(preds, phrases))


texts = ['abc', 'bcd', 'cde', 'efg'] * 10

model_class = EDAModel
#model_class = S2SModel

model = model_class(5)
model.init_from_texts(texts)

print(model.vectorize_output_batch(texts[:2]))

model.train(texts, 100, verbose=0)
model.model.summary()

phrases = texts[:10]

print(phrases)
print(model.predict(phrases))
#report(model, phrases, phrases, 'Identity')



max_len = 20
all_phrases = load_preprocessed('data/sentences.txt', max_len)
all_phrases = all_phrases[:30_000]
BATCH_SIZE = 250
model = model_class(max_len)
model.init_from_texts(all_phrases)
model.train(all_phrases, 50, val_size=1_000)

phrases = all_phrases[:10]

report(model, phrases, phrases, 'Identity, larger')

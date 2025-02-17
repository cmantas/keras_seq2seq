from  s2s_model import *
from helpers import *
from encoder_decoder_model import *
from encoder_decoder_attention_model import *
from simple_ed_model import *


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

#model_class = S2SModel
model_class = EDModel
#model_class = EDAModel
#model_class = S2SModel
#model_class = SEDAModel
#model_class = EDModel
model = model_class(4, 64)
model.batch_size = 10
model.init_from_texts(texts)

print(model.vectorize_output_batch(texts[:2]))

print("training on toy dataset")
model.train(texts, 200, verbose=0)
model.report(texts)

print(texts)
print(model.predict(texts))
#report(model, phrases, phrases, 'Identity')



max_len = 20
all_phrases = load_preprocessed('data/sentences.txt', max_len)
all_phrases = all_phrases[:10_000]
BATCH_SIZE = 50
model = model_class(max_len)
model.init_from_texts(all_phrases)

try:
    model.train(all_phrases, 30)
except KeyboardInterrupt:
    print("\n\nUnpacient are we?")



phrases = all_phrases[:10]

report(model, phrases, phrases, 'Identity, larger')
model.report(phrases[:1000])

from spelling_model import *
from s2s_transformer_model import *
from s2s_attention_model import *

mclass = SpellingAttention #SpellingTransformer

def acc(src, trgt):
    pairs = list(zip(src,trgt))
    return sum([1 for p,t in pairs if p == t]) / float(len(pairs))

def report(model, phrases, targets, legend='Report'):
    preds = model.predict(phrases)
    pairs = list(zip(preds, phrases))[:10]
    print(legend)
    print(pairs)
    print(legend, "Accuracy:", acc(preds, phrases))



max_len = 20
all_phrases = load_preprocessed('data/sentences.txt', max_len)
all_phrases = all_phrases[:30_000]
BATCH_SIZE = 250
model = mclass(max_len, 256)
model.init_from_texts(all_phrases)
model.train(all_phrases, 500, val_size=2_000)

phrases = all_phrases[:100]

miss_phrases = [add_noise_to_string(p, .05) for p in phrases]

report(model, miss_phrases, phrases, 'Spelling')

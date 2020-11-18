from spelling_model import *

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
model = SpellingModel(max_len)
model.init_from_texts(all_phrases)
model.train(all_phrases, 5, val_size=1_000)

phrases = all_phrases[:100]

miss_phrases = [add_noise_to_string(p, .05) for p in phrases]

report(model, miss_phrases, phrases, 'Spelling')

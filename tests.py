from  s2s_model import *

texts = ['abc', 'cbd'] * 10

model = S2SModel(3)
model.init_from_texts(texts)

m = model.create_model()

model.train(texts, 10)

#pred = model.predict('abc')

max_len = 15
all_phrases = load_preprocessed('sentences.txt', max_len)
all_phrases = all_phrases[:3000]
BATCH_SIZE = 250
model = S2SModel(max_len)
model.init_from_texts(all_phrases)
model.train(all_phrases, 100)

print([model.predict(p) for p in all_phrases[:10]])


token_idx = token_index(texts + ['\t', '\n'])


max_seq_length = 3
texts = ['aaa', 'bbb', 'ccc']
token_idx = token_index(texts)
vectorize_batch(texts, token_idx, max_seq_length, 'int')

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

tok = Tokenizer(char_level=True)
tok.fit_on_texts(texts)
tok.texts_to_sequences()

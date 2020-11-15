from  s2s_model import *

texts = ['abc', 'bcd', 'cde', 'efg'] * 10

model = S2SModel(3)
model.init_from_texts(texts)

print(model.vectorize_output_batch(texts[:2]))

model.train(texts, 100, verbose=2)


phrases = texts[:10]
print(list(zip(model.predict(phrases), phrases)))


max_len = 15
all_phrases = load_preprocessed('sentences.txt', max_len)
all_phrases = all_phrases[:5_000]
BATCH_SIZE = 250
model = S2SModel(max_len)
model.init_from_texts(all_phrases)
model.train(all_phrases, 5)

phrases = all_phrases[:10]
print(list(zip(model.predict(phrases), phrases)))




# token_idx = token_index(texts + ['\t', '\n'])


# max_seq_length = 3
# texts = ['aaa', 'bbb', 'ccc']
# token_idx = token_index(texts)
# vectorize_batch(texts, token_idx, max_seq_length, 'int')

# from keras.preprocessing.text import Tokenizer
# from keras.utils import to_categorical
# from keras.preprocessing.sequence import pad_sequences

# tok = Tokenizer(char_level=True)
#tok.fit_on_texts(texts)
#tok.texts_to_sequences()

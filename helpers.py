import numpy as np
from random import Random
from numpy.random import choice, randint, shuffle, seed, rand
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')


def read_data(fname, delimiter="\n"):
    """Helper reading a file with the input and 
    target texts.
    Returns a tuple with 2 lists of phrases (input, target)"""
    input_phrases = []
    target_phrases = []
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.read().split(delimiter)
    
        for line in lines:
            pair = line.split('\t')
            if len(pair) !=2: continue 

            input_text, target_text = pair
            input_phrases.append(input_text)
            target_phrases.append(target_text)

        return (input_phrases, target_phrases)

def wrap_with_delims(texts, start='\t', end='\n'):
    """Helper wrapping the input texts with the start
    and and end sequence delimiters.""" 
    return [start + t + end for t in texts]

def text_preprocess(texts):
    """Some minimal text preprocessing:
    * downcase
    * remove trailing periods
    * remove a weird unicode char"""
    return [t.strip().lower().\
            rstrip('.').strip().\
            replace(u'\u202f', '').\
            replace(u'\ufeff','') for t in texts]

#spelling

CHARS = list("abcdefghijklmnopqrstuvwxyz .")
def add_noise_to_string(a_string, amount_of_noise):
    """Add some artificial spelling mistakes to the string"""
    # Assume no errors on strings of 1 or 2 chars
    if len(a_string) <= 2: return a_string
    # The probability for each permutation (assume equal)
    perm_prob = amount_of_noise / 4.0 # divide by the possible permutations
    threshold = perm_prob * len(a_string) # threshold for each permutation
    if rand() < threshold:
        # Replace a character with a random character
        # Random character position
        i = randint(len(a_string))
        a_string = a_string[:i] + choice(CHARS[:-1]) + a_string[i + 1:]
    if rand() < threshold:
        # Delete a character
        i = randint(len(a_string))
        a_string = a_string[:i] + a_string[i + 1:]
    if rand() < threshold:
        # Add a random character
        i = randint(len(a_string))
        a_string = a_string[:i] + choice(CHARS[:-1]) + a_string[i:]
    if rand() < threshold:
        # Transpose 2 characters
        i = randint(len(a_string) - 1)
        a_string = (a_string[:i] + a_string[i + 1] + a_string[i] +
                    a_string[i + 2:])
    return a_string

def create_misspellings(phrases, noise, misspelled_times):
    """given a list of N phrases it appends to this list 
    another misspelled_times*N phrases that permutations of the
    input ones with some random spelling errors introduced"""
    misspelled = [add_noise_to_string(p, noise) for p in phrases * misspelled_times]
    all_phrases = phrases + misspelled
    target_phrases = phrases * (misspelled_times + 1)
    return all_phrases, target_phrases

def token_index(texts):
    """Create a dictionary with all characters in the `texts` corpus
    and a number signifying their index in an 1-hot encoding."""
    vocab = set()
    for txt in texts:
        for char in txt: vocab.add(char)
    vocab = sorted(list(vocab))
    return dict([(char, i) for i, char in enumerate(vocab)])

def vectorize_batch(texts, token_index, max_seq_len, offset=False):
    num_tokens = len(token_index)
    example_count = len(texts)

    # Generate 1-hot encoding
    data = np.zeros((example_count, max_seq_len, num_tokens), dtype='float32')

    for i, text in enumerate(texts):
        start_t = 1 if offset else 0
        for t, char in enumerate(text[start_t:]):
            idx = token_index[char]
            data[i, t, idx] = 1.
    return data

def vectorize_dataset(input_texts, target_texts,
                      input_token_index, target_token_index,
                      max_encoder_seq_length, max_decoder_seq_length):
    return ([vectorize_batch(input_texts, input_token_index,
                             max_encoder_seq_length),
             vectorize_batch(target_texts, target_token_index,
                             max_decoder_seq_length)],
            # same as decoder input data, but offset by one
            vectorize_batch(target_texts, target_token_index,
                            max_decoder_seq_length, True))

def vectorize_phrase(phrase, token_index, max_seq_len):
    num_tokens = len(token_index)
    # Generate 1-hot encoding
    # -> maximum phrase size x # of possible characters
    vectorized = np.zeros((1, max_seq_len, num_tokens),dtype='float32')

    # for each of the characters
    for char_no, char in enumerate(phrase):
        idx = token_index[char]
        vectorized[0, char_no, idx] = 1.
    return vectorized

def decode_sequence(input_seq, target_token_index, encoder_model, decoder_model):
    # construct the reverse 
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    reverse_target_char_index = {v: k for k, v in target_token_index.items()} 
    # the count of possible target tokens
    num_decoder_tokens = len(target_token_index)

    # the output sequence's length can be found by
    # the last layer's output shape
    max_decoder_seq_length = decoder_model.layers[-1].output_shape[-1]

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    sampled_char = None
    decoded_sentence = ''
    while sampled_char != '\n' and \
    len(decoded_sentence) <= max_decoder_seq_length:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def translate_fn(encoder_model, decoder_model,
                 input_token_index, target_token_index,
                 max_encoder_seq_length):
    def translate(phrase):
        vect = vectorize_phrase(phrase, input_token_index,
                                max_encoder_seq_length)
        decoded = decode_sequence(vect,target_token_index, 
                                  encoder_model, decoder_model)
        return decoded[:-1]
    return translate

def plot_history(history):
    """credit: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/"""
    plt.plot(history.history['loss'])
    val_loss = history.history.get('val_loss', [])
    plt.plot(val_loss)
    plt.plot(history.history.get('acc', []))
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation', 'accuracy'], loc='upper left')
    plt.show()

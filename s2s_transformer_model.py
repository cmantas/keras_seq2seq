from math import ceil

import tensorflow as tf
import keras

from helpers import *
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, Dense, RepeatVector, \
  TimeDistributed, Activation, GRU, Dropout, Bidirectional, \
  Embedding, Lambda, Layer
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import sparse_categorical_accuracy

from s2s_model import *
from spelling_model import SpellingModel

# https://www.kaggle.com/fareise/multi-head-self-attention-for-text-classification
class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        #self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.token_emb = Lambda(lambda x:K.one_hot(x, embed_dim))
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class S2STransformerModel(S2SModel):
    def create_model(self):
      print("creating transformer")
      token_count = len(self.tokenizer.word_index)
      output_len = self.max_seq_length

      inputs = layers.Input(shape=(output_len,),dtype='int32')
      embedding_layer = TokenAndPositionEmbedding(output_len, token_count, token_count)
      x = embedding_layer(inputs)
      # embed_dim, num_heads, ff_dim, rate=0.1):
      # TMP: token count as num_heads
      transformer_block = TransformerBlock(token_count, token_count, self.latent_dim, 0)
      x = transformer_block(x)
      transformer_block = TransformerBlock(token_count, token_count, self.latent_dim, 0)
      x = transformer_block(x)

      t_dense = TimeDistributed(Dense(token_count, activation="softmax"))
      output = t_dense(x)

      model = Model(inputs=inputs, outputs=output)

      model.compile(loss=self.LOSS_FN,
                    optimizer=self.OPTIMIZER,
                    run_eagerly=True,
                    metrics=[acc, seq_acc])
      self.model = model

    def predict(self, in_txts):
        wrap = isinstance(in_txts, str)

        txts = [in_txts] if wrap else in_txts

        x = self.vectorize_batch(txts)
        preds = self.model.predict(x).argmax(axis=2)
        out_txts = [self.seq_to_text(seq) for seq in preds]

        return out_txts[0] if wrap else out_txts


class SpellingTransformer(SpellingModel, S2STransformerModel):
    pass

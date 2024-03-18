import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

# Custom Layer: Positional Encoding
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, maxlen, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(maxlen, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        pos_encoding = tf.slice(self.pos_encoding, [0, 0, 0], [1, seq_length, -1])
        return inputs + pos_encoding

    


# Custom Layer: Transformer Encoder Layer
class TransformerEncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential([
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dense(d_model)
        ])

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Define Transformer Model
def transformer_model(maxlen, d_model, num_heads, dff, input_vocab_size, target_vocab_size, num_layers, dropout_rate):
    inputs = keras.Input(shape=(maxlen,))
    x = inputs

    # Add positional encoding
    position_embedding = PositionalEncoding(maxlen, d_model)
    x = position_embedding(x)

    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)(x)

    outputs = keras.layers.Dense(target_vocab_size)(x)
    return keras.Model(inputs, outputs)


# Example usage:
maxlen = 100  # Maximum sequence length
d_model = 128  # Dimensionality of model
num_heads = 8  # Number of attention heads
dff = 512  # Dimensionality of feed-forward layer
input_vocab_size = 10000  # Input vocabulary size
target_vocab_size = 10000  # Target vocabulary size
num_layers = 6  # Number of transformer encoder layers
dropout_rate = 0.1  # Dropout rate

# Create the model
model = transformer_model(maxlen, d_model, num_heads, dff, input_vocab_size, target_vocab_size, num_layers, dropout_rate)

# Compile and train the model (you need to provide data and labels)
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# model.fit(dataset, epochs=10)

#model.summary()
import tensorflow as tf
from tensorflow.keras import layers, models

def get_positional_encoding(seq_len, d_model):
    positions = tf.range(seq_len, dtype=tf.float32)
    positions = tf.expand_dims(positions, 1)
    i = tf.range(d_model, dtype=tf.float32)
    angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / d_model)
    angle_rads = positions * angle_rates
    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = models.Sequential([layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        attention_output = self.attention(inputs, inputs)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config

def build_cnn_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    shape = x.shape
    seq_len = shape[1] 
    embed_dim = shape[2] * shape[3]
    x = layers.Reshape((seq_len, embed_dim))(x)
    
    pos_encoding = get_positional_encoding(seq_len, embed_dim)
    x = x + pos_encoding
    
    x = TransformerEncoder(embed_dim=embed_dim, dense_dim=embed_dim*2, num_heads=4)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_Transformer")
    return model

if __name__ == "__main__":
    model = build_cnn_transformer_model((126, 40, 1))
    model.summary()

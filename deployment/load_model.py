import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.models import model_from_json

# Convert the variables to the correct data type
# Load the variables from the JSON file
json_file_path = "variables.json"
with open(json_file_path, 'r') as json_file:
    variables_dict = json.load(json_file)

# Epsilon value for layer normalisation
LAYER_NORM_EPS = variables_dict['LAYER_NORM_EPS']

# final embedding and transformer embedding size
UNITS_ENCODER = variables_dict['UNITS_ENCODER']
UNITS_DECODER = variables_dict['UNITS_DECODER']

# Transformer
NUM_BLOCKS_ENCODER = variables_dict['NUM_BLOCKS_ENCODER']
NUM_BLOCKS_DECODER = variables_dict['NUM_BLOCKS_DECODER']
NUM_HEADS = variables_dict['NUM_HEADS']
MLP_RATIO = variables_dict['MLP_RATIO']

# Dropout
EMBEDDING_DROPOUT = variables_dict['EMBEDDING_DROPOUT']
MLP_DROPOUT_RATIO = variables_dict['MLP_DROPOUT_RATIO']
MHA_DROPOUT_RATIO = variables_dict['MHA_DROPOUT_RATIO']
CLASSIFIER_DROPOUT_RATIO = variables_dict['CLASSIFIER_DROPOUT_RATIO']

# Number of Frames to resize recording to
N_TARGET_FRAMES = variables_dict['N_TARGET_FRAMES']
N_UNIQUE_CHARACTERS = variables_dict['N_UNIQUE_CHARACTERS']
N_UNIQUE_CHARACTERS0 = variables_dict['N_UNIQUE_CHARACTERS0']
PAD_TOKEN = variables_dict['PAD_TOKEN']
SOS_TOKEN = variables_dict['SOS_TOKEN']

# Length of Phrase + EOS Token
MAX_PHRASE_LENGTH = variables_dict['MAX_PHRASE_LENGTH']

# Mean/Standard Deviations of data used for normalizing
MEANS = np.array(variables_dict['MEANS'])
STDS = np.array(variables_dict['STDS'])

# Initiailizers
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
INIT_ZEROS = tf.keras.initializers.constant(0.0)
# Activations
GELU = tf.keras.activations.gelu


class Embedding(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.positional_embedding = tf.Variable(
            initial_value=tf.zeros([N_TARGET_FRAMES, UNITS_ENCODER], dtype=tf.float32),
            trainable=True, name='embedding_positional_encoder')
        self.dominant_hand_embedding = LandmarkEmbedding(UNITS_ENCODER, 'dominant_hand')

    def call(self, x, training=False):
        x = tf.where(tf.math.equal(x, 0.0), 0.0, (x - MEANS) / STDS)
        x = self.dominant_hand_embedding(x)
        x = x + self.positional_embedding
        return x

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Encoder(tf.keras.Model):
    def __init__(self, num_blocks, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_blocks = num_blocks
        self.support_masking = True
        self.blocks = [
            EncoderTransformerBlock(UNITS_ENCODER, NUM_HEADS, MLP_RATIO, MHA_DROPOUT_RATIO, MLP_DROPOUT_RATIO) for _ in
            range(num_blocks)]

        if UNITS_ENCODER != UNITS_DECODER:
            self.dense_out = tf.keras.layers.Dense(UNITS_DECODER, kernel_initializer=INIT_GLOROT_UNIFORM,
                                                   use_bias=False)
            self.apply_dense_out = True
        else:
            self.apply_dense_out = False

    def call(self, x, x_inp, training=False):
        attention_mask = tf.where(tf.math.reduce_sum(x_inp, axis=[2]) == 0.0, 0.0, 1.0)
        attention_mask = tf.expand_dims(attention_mask, axis=1)
        attention_mask = tf.repeat(attention_mask, repeats=N_TARGET_FRAMES, axis=1)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, training=training)

        if self.apply_dense_out:
            x = self.dense_out(x)

        return x, attention_mask

    def get_config(self):
        config = super().get_config()
        config.update({"num_blocks": self.num_blocks})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Decoder(tf.keras.Model):
    def __init__(self, num_blocks, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.num_blocks = num_blocks
        self.supports_masking = True
        self.positional_embedding = tf.Variable(
            initial_value=tf.zeros([N_TARGET_FRAMES, UNITS_DECODER], dtype=tf.float32),
            trainable=True, name='embedding_positional_encoder')
        self.char_emb = tf.keras.layers.Embedding(N_UNIQUE_CHARACTERS, UNITS_DECODER, embeddings_initializer=INIT_ZEROS)
        self.pos_emb_mha = MultiHeadAttention(UNITS_DECODER, NUM_HEADS, MHA_DROPOUT_RATIO)
        self.pos_emb_ln = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
        self.blocks = [
            DecoderTransformerBlock(UNITS_DECODER, NUM_HEADS, MLP_RATIO, MHA_DROPOUT_RATIO, MLP_DROPOUT_RATIO) for _ in
            range(num_blocks)]

    def get_causal_attention_mask(self, B):
        ones = tf.ones((N_TARGET_FRAMES, N_TARGET_FRAMES))
        mask = tf.linalg.band_part(ones, 0, -1)
        mask = tf.transpose(mask)
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.tile(mask, [B, 1, 1])
        mask = tf.cast(mask, tf.float32)
        return mask

    def call(self, encoder_outputs, attention_mask, phrase, training=False):
        B = tf.shape(encoder_outputs)[0]
        phrase = tf.cast(phrase, tf.int32)
        phrase = tf.pad(phrase, [[0, 0], [1, 0]], constant_values=SOS_TOKEN, name='prepend_sos_token')
        phrase = tf.pad(phrase, [[0, 0], [0, N_TARGET_FRAMES - MAX_PHRASE_LENGTH - 1]], constant_values=PAD_TOKEN,
                        name='append_pad_token')
        causal_mask = self.get_causal_attention_mask(B)
        x = self.positional_embedding + self.char_emb(phrase)
        x = self.pos_emb_ln(x + self.pos_emb_mha(x, x, x, attention_mask=causal_mask))

        for block in self.blocks:
            x = block(x, encoder_outputs, attention_mask=attention_mask, training=training)

        x = tf.slice(x, [0, 0, 0], [-1, MAX_PHRASE_LENGTH, -1])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"num_blocks": self.num_blocks})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Embeds a landmark using fully connected layers
class LandmarkEmbedding(tf.keras.Model):
    def __init__(self, units, name):
        super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
        self.units = units
        self.supports_masking = True

    def build(self, input_shape):
        # Embedding for missing landmark in frame, initizlied with zeros
        self.empty_embedding = self.add_weight(
            name=f'{self.name}_empty_embedding',
            shape=[self.units],
            initializer=INIT_ZEROS,
        )
        # Embedding
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_1', use_bias=False,
                                  kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU),  # Can change activation
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=False,
                                  kernel_initializer=INIT_HE_UNIFORM),
        ], name=f'{self.name}_dense')

    def call(self, x):
        return tf.where(
            # Checks whether landmark is missing in frame
            tf.reduce_sum(x, axis=2, keepdims=True) == 0,
            # If so, the empty embedding is used
            self.empty_embedding,
            # Otherwise the landmark data is embedded
            self.dense(x),
        )

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "name": self.name})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EncoderTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, units, num_heads, mlp_ratio, mha_dropout_ratio, mlp_dropout_ratio, **kwargs):
        super(EncoderTransformerBlock, self).__init__(**kwargs)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
        self.mha = MultiHeadAttention(units, num_heads, mha_dropout_ratio)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(units * mlp_ratio, activation=GELU, kernel_initializer=INIT_GLOROT_UNIFORM,
                                  use_bias=False),
            tf.keras.layers.Dropout(mlp_dropout_ratio),
            tf.keras.layers.Dense(units, kernel_initializer=INIT_HE_UNIFORM, use_bias=False),
        ])

    def call(self, inputs, attention_mask, training=False):
        x = self.layer_norm_1(inputs + self.mha(inputs, inputs, inputs, attention_mask=attention_mask))
        x = self.layer_norm_2(x + self.mlp(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "num_heads": self.num_heads, "mlp_ratio": self.mlp_ratio,
                       "mha_dropout_ratio": self.mha_dropout_ratio, "mlp_dropout_ratio": self.mlp_dropout_ratio})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# replaced softmax with softmax layer to support masked softmax
def scaled_dot_product(q, k, v, softmax, attention_mask):
    # calculates Q . K(transpose)
    qkt = tf.matmul(q, k, transpose_b=True)
    # calculates scaling factor
    dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=tf.float32))
    scaled_qkt = qkt / dk
    softmax = softmax(scaled_qkt, mask=attention_mask)
    z = tf.matmul(softmax, v)
    # shape: (m,Tx,depth), same shape as q,k,v
    return z


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_of_heads, dropout, d_out=None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads  # Can change
        self.wq = [tf.keras.layers.Dense(self.depth, use_bias=False) for i in
                   range(num_of_heads)]  # depth//2 isn't common, we can try different numbers
        self.wk = [tf.keras.layers.Dense(self.depth, use_bias=False) for i in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth, use_bias=False) for i in range(num_of_heads)]
        self.softmax = tf.keras.layers.Softmax()
        self.do = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True
        self.wo = tf.keras.layers.Dense(d_model if d_out is None else d_out, use_bias=False)

    def call(self, q, k, v, attention_mask=None, training=False):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](q)
            K = self.wk[i](k)
            V = self.wv[i](v)
            multi_attn.append(scaled_dot_product(Q, K, V, self.softmax, attention_mask))

        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        multi_head_attention = self.do(multi_head_attention, training=training)

        return multi_head_attention

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_of_heads": self.num_of_heads, "dropout": self.dropout})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DecoderTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, units, num_heads, mlp_ratio, mha_dropout_ratio, mlp_dropout_ratio, **kwargs):
        super(DecoderTransformerBlock, self).__init__(**kwargs)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
        self.mha = MultiHeadAttention(units, num_heads, mha_dropout_ratio)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(units * mlp_ratio, activation=GELU, kernel_initializer=INIT_GLOROT_UNIFORM,
                                  use_bias=False),
            tf.keras.layers.Dropout(mlp_dropout_ratio),
            tf.keras.layers.Dense(units, kernel_initializer=INIT_HE_UNIFORM, use_bias=False),
        ])

    def call(self, inputs, encoder_outputs, attention_mask, training=False):
        x = self.layer_norm_1(
            inputs + self.mha(inputs, encoder_outputs, encoder_outputs, attention_mask=attention_mask))
        x = self.layer_norm_2(x + self.mlp(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "num_heads": self.num_heads, "mlp_ratio": self.mlp_ratio,
                       "mha_dropout_ratio": self.mha_dropout_ratio, "mlp_dropout_ratio": self.mlp_dropout_ratio})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


custom_objects = {'Embedding': Embedding,
                  'Encoder': Encoder,
                  'Decoder': Decoder,
                  'LandmarkEmbedding': LandmarkEmbedding,
                  'EncoderTransformerBlock': EncoderTransformerBlock,
                  'MultiHeadAttention': MultiHeadAttention,
                  'DecoderTransformerBlock': DecoderTransformerBlock}

# load json and create model
json_file = open('model_architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# load model from JSON file
loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)

# load weights into the new model
loaded_model.load_weights("model.h5")

# loaded_model.summary(expand_nested=True, show_trainable=True, )

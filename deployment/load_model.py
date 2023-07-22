import tensorflow as tf

# Epsilon value for layer normalisation
LAYER_NORM_EPS = 1e-5

# final embedding and transformer embedding size
UNITS_ENCODER = 384
UNITS_DECODER = 384

# Transformer
NUM_BLOCKS_ENCODER = 3
NUM_BLOCKS_DECODER = 2
NUM_HEADS = 4
MLP_RATIO = 2

# Dropout
EMBEDDING_DROPOUT = 0.2
MLP_DROPOUT_RATIO = 0.2
MHA_DROPOUT_RATIO = 0.2
CLASSIFIER_DROPOUT_RATIO = 0.2

# Number of Frames to resize recording to
N_TARGET_FRAMES = 128
N_UNIQUE_CHARACTERS = 62
N_UNIQUE_CHARACTERS0 = 59
PAD_TOKEN = N_UNIQUE_CHARACTERS0
SOS_TOKEN = N_UNIQUE_CHARACTERS0 + 1

# Length of Phrase + EOS Token
MAX_PHRASE_LENGTH = 31 + 1

# Mean/Standard Deviations of data used for normalizing
MEANS = [0.69352776, 0.60659605, 0.53412515, 0.4970676, 0.48584947, 0.5761701,
         0.5300588, 0.49778917, 0.47764367, 0.6305243, 0.5822572, 0.55222154,
         0.53908557, 0.68544865, 0.63951194, 0.6104323, 0.5991277, 0.7378051,
         0.7018211, 0.6776119, 0.6673842, 0.76445776, 0.7457853, 0.7062872,
         0.67484325, 0.6514734, 0.6304427, 0.5906848, 0.5854317, 0.5849309,
         0.6276549, 0.5890438, 0.59771925, 0.6047316, 0.6383216, 0.60959125,
         0.6295764, 0.6437836, 0.6588292, 0.6397078, 0.65018004, 0.65816236,
         0.26357186, 0.35093567, 0.4236605, 0.45704976, 0.4634739, 0.37947592,
         0.4234214, 0.45306972, 0.4717593, 0.3199842, 0.36261505, 0.38926786,
         0.40241373, 0.26189587, 0.30273047, 0.3301876, 0.34255308, 0.20624675,
         0.23920882, 0.263005, 0.27461466, 0.75472385, 0.73504084, 0.6943852,
         0.6608657, 0.63613355, 0.6144105, 0.5700216, 0.56217206, 0.5597008,
         0.611077, 0.56800383, 0.575002, 0.5811821, 0.62163454, 0.59134597,
         0.61230445, 0.6277079, 0.64273566, 0.6216118, 0.6318555, 0.63973725,
         0.56342137, 0.5647059, 0.5649758, 0.5657689, 0.54460865, 0.52689284,
         0.51569146, 0.5043293, 0.51033896, 0.52668756, 0.53708506, 0.54991424,
         0.5468167, 0.55006754, 0.5267238, 0.5178957, 0.51888436, 0.5099791,
         0.53717476, 0.5305108, 0.5081805, 0.51886874, 0.58258605, 0.6024338,
         0.6155048, 0.6306914, 0.6245343, 0.6058631, 0.59408224, 0.58018464,
         0.5852319, 0.5804903, 0.60605526, 0.61589545, 0.61500907, 0.6246284,
         0.59435004, 0.6024958, 0.6250273, 0.61513, 0.508501, 0.5193109,
         0.52219623, 0.53701967, 0.5069547, 0.51169485, 0.51677644, 0.5253185,
         0.5245756, 0.521367, 0.5199756, 0.51932734, 0.5365361, 0.5221106,
         0.5230684, 0.53079647, 0.5238175, 0.52800494, 0.5223436, 0.5342269,
         0.5212379, 0.52289945, 0.506347, 0.5106173, 0.51533395, 0.5235456,
         0.5230225, 0.52027595, 0.51917976, 0.5189014, 0.5361387, 0.5216965,
         0.5220167, 0.52960336, 0.5225625, 0.5264617, 0.5215638, 0.53341466,
         0.51952803, 0.5216051]

STDS = [0.10834738, 0.10391748, 0.10296664, 0.10752504, 0.12336373, 0.10313869,
        0.10744168, 0.11199072, 0.1193621, 0.10597368, 0.11260378, 0.1170811,
        0.12447591, 0.11238337, 0.12130429, 0.12248141, 0.1267081, 0.1224081,
        0.13301295, 0.13806877, 0.1437398, 0.08867608, 0.08839962, 0.08913112,
        0.09358086, 0.09968524, 0.08439907, 0.09381164, 0.10565417, 0.11996002,
        0.08592986, 0.1002507, 0.11805841, 0.13548768, 0.08893858, 0.1042807,
        0.11806193, 0.13066797, 0.09283979, 0.1044982, 0.11446757, 0.12410894,
        0.08575833, 0.08688664, 0.08871841, 0.09452496, 0.11280894, 0.08605019,
        0.09069607, 0.09625262, 0.10480069, 0.08209087, 0.08907479, 0.09521613,
        0.10375828, 0.0827678, 0.09389319, 0.09721766, 0.10260603, 0.0892784,
        0.10309231, 0.11121955, 0.11911318, 0.08014706, 0.07939664, 0.07666104,
        0.07640523, 0.07845239, 0.06779566, 0.06928173, 0.07995176, 0.09609538,
        0.06776656, 0.07411631, 0.09502285, 0.11704809, 0.06976698, 0.07840788,
        0.09568293, 0.11219386, 0.07334771, 0.07997227, 0.09204492, 0.10471888,
        0.1324311, 0.13287905, 0.13296498, 0.13300247, 0.13251117, 0.13296743,
        0.13352127, 0.13476767, 0.13467269, 0.13386367, 0.13339657, 0.13304512,
        0.13318144, 0.13313657, 0.13394693, 0.13404495, 0.1343446, 0.13446471,
        0.13349241, 0.13355125, 0.13414721, 0.13430822, 0.13283393, 0.13377732,
        0.1346423, 0.13602652, 0.13584861, 0.13470158, 0.1339573, 0.13331288,
        0.13342074, 0.133372, 0.13473015, 0.13483934, 0.13534908, 0.13551436,
        0.13399816, 0.13405652, 0.1354323, 0.13537434, 0.06685787, 0.06737807,
        0.06767439, 0.06927998, 0.06658512, 0.06643137, 0.0663855, 0.06645988,
        0.06653237, 0.06679216, 0.06700299, 0.06721594, 0.06899743, 0.06748881,
        0.06692849, 0.06752784, 0.06670087, 0.06690367, 0.06722134, 0.06834918,
        0.06637124, 0.06663854, 0.06680202, 0.06691353, 0.06701645, 0.06724831,
        0.06726662, 0.06730385, 0.06735906, 0.06739713, 0.06924284, 0.06767783,
        0.06744281, 0.06815296, 0.06732813, 0.0676265, 0.06758311, 0.06880609,
        0.06710069, 0.0672657]

# Initiailizers
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
INIT_ZEROS = tf.keras.initializers.constant(0.0)
# Activations
GELU = tf.keras.activations.gelu

# Learning Rate
LEARNING_RATE = 1e-4


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
    # caculates scaling factor
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

# Load architecture
from tensorflow.keras.models import model_from_json

# load json and create model
json_file = open('model_architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# load model from JSON file
loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)

# load weights into the new model
loaded_model.load_weights("model.h5")

# loaded_model.summary(expand_nested=True, show_trainable=True, )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sn
import tensorflow as tf
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from leven import levenshtein
import glob
import sys
import os
import math
import gc
import sys
import sklearn
import time
import json
import kaggle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



class EDA_Preprocess:
    def __init__(self):
        self.download_path = '.'
        self.VERBOSE = 1 
        self.SEED = 20 
        self.N_TARGET_FRAMES = 128
        self.DEBUG = False
        self.USE_VAL = True
        self.BATCH_SIZE = 64
        self.N_EPOCHS = 2 if IS_INTERACTIVE else 100
        self.N_WARMUP_EPOCHS = 10
        self.LR_MAX = 1e-3
        self.WD_RATIO = 0.05
        self.MAX_PHRASE_LENGTH = 31 + 1
        self.TRAIN_MODEL = True
        self.LOAD_WEIGHTS = False
        self.WARMUP_METHOD = 'exp'
        
        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['axes.labelsize'] = 18
        mpl.rcParams['axes.titlesize'] = 24
        
        with open('asl-fingerspelling/character_to_prediction_index.json') as json_file:
            self.CHAR2ORD = json.load(json_file)
        self.ORD2CHAR = {j: i for i, j in self.CHAR2ORD.items()}
        self.N_UNIQUE_CHARACTERS0 = len(self.CHAR2ORD)
        self.PAD_TOKEN = self.N_UNIQUE_CHARACTERS0
        self.SOS_TOKEN = self.N_UNIQUE_CHARACTERS0 + 1
        self.EOS_TOKEN = self.N_UNIQUE_CHARACTERS0 + 2
        self.N_UNIQUE_CHARACTERS = self.N_UNIQUE_CHARACTERS0 + 3
        self.df = None
        self.MAX_FRAMES = None
        self.train = None
        self.train_sequence_id = None
        self.N_SAMPLES = None
        self.INFERENCE_FILE_PATHS = None

    def download_kaggle_dataset_ASL_Fingerspelling(self):
        # Download the "ASL Fingerspelling" competition dataset
        kaggle.api.competition_download_files('asl-fingerspelling', path=self.download_path)
        
        # Optionally, you can unzip the downloaded file
        with zipfile.ZipFile(f'{self.download_path}/asl-fingerspelling.zip', 'r') as zip_ref:
            zip_ref.extractall(self.download_path)

    def download_kaggle_dataset_ASL_dataset(self):
		dataset_name = 'markwijkhuizen/aslfr-preprocessing-dataset' asl-fingerspelling


		# Download the dataset
		kaggle.api.dataset_download_files(dataset_name, path=self.download_path, unzip=True)

        # After unzipping, the dataset will be available in the current directory

    def load_train_dataframe(self):
        if self.DEBUG:
            self.train = pd.read_csv('asl-fingerspelling/train.csv').head(5000)
        else:
            self.train = pd.read_csv('asl-fingerspelling/train.csv')

        # Set Train Indexed By sqeuence_id
        self.train_sequence_id = self.train.set_index('sequence_id')

        # Number Of Train Samples
        self.N_SAMPLES = len(self.train)
        print(f'N_SAMPLES: {self.N_SAMPLES}')

    def get_file_path(self, path):
        return f'asl-fingerspelling/{path}'

    def get_inference_files(self):
        self.INFERENCE_FILE_PATHS = pd.Series(
            glob.glob('aslfr-preprocessing-dataset/train_landmark_subsets/*')
        )
        print(f'Found {len(self.INFERENCE_FILE_PATHS)} Inference Pickle Files')

    def run(self):
        tqdm.pandas()

        print(f'Tensorflow Version {tf.__version__}')
        print(f'Python Version: {sys.version}')

        self.download_kaggle_dataset_ASL_Fingerspelling()
		self.download_kaggle_dataset_ASL_dataset()
        self.load_train_dataframe()
        self.get_inference_files()
        
# Training class
class Training:

    def main(self):
		# Train/Validation
		if USE_VAL:
			# TRAIN
			X_train = np.load('/kaggle/input/aslfr-preprocessing-dataset/X_train.npy')
			y_train = np.load('/kaggle/input/aslfr-preprocessing-dataset/y_train.npy')[:,:MAX_PHRASE_LENGTH]
			N_TRAIN_SAMPLES = len(X_train)
			# VAL
			X_val = np.load('/kaggle/input/aslfr-preprocessing-dataset/X_val.npy')
			y_val = np.load('/kaggle/input/aslfr-preprocessing-dataset/y_val.npy')[:,:MAX_PHRASE_LENGTH]
			N_VAL_SAMPLES = len(X_val)
			# Shapes
			print(f'X_train shape: {X_train.shape}, X_val shape: {X_val.shape}')
		# Train On All Data
		else:
			# TRAIN
			X_train = np.load('/kaggle/input/aslfr-preprocessing-dataset/X.npy')
			y_train = np.load('/kaggle/input/aslfr-preprocessing-dataset/y.npy')[:,:MAX_PHRASE_LENGTH]
			N_TRAIN_SAMPLES = len(X_train)
			print(f'X_train shape: {X_train.shape}')


		X_test,X_val = X_val[:1000], X_val[1000:]
		y_test,y_val = y_val[:1000], y_val[1000:]
		X_test.shape, y_test.shape, X_val.shape, y_val.shape

		# Example Batch For Debugging
		N_EXAMPLE_BATCH_SAMPLES = 1024
		N_EXAMPLE_BATCH_SAMPLES_SMALL = 32
		# Example Batch
		X_batch = {
			'frames': np.copy(X_train[:N_EXAMPLE_BATCH_SAMPLES]),
			'phrase': np.copy(y_train[:N_EXAMPLE_BATCH_SAMPLES]),
		#     'phrase_type': np.copy(y_phrase_type_train[:N_EXAMPLE_BATCH_SAMPLES]),
		}
		y_batch = np.copy(y_train[:N_EXAMPLE_BATCH_SAMPLES])
		# Small Example Batch
		X_batch_small = {
			'frames': np.copy(X_train[:N_EXAMPLE_BATCH_SAMPLES_SMALL]),
			'phrase': np.copy(y_train[:N_EXAMPLE_BATCH_SAMPLES_SMALL]),
		#     'phrase_type': np.copy(y_phrase_type_train[:N_EXAMPLE_BATCH_SAMPLES_SMALL]),
		}
		y_batch_small = np.copy(y_train[:N_EXAMPLE_BATCH_SAMPLES_SMALL])

		# Read First Parquet File
		# example_parquet_df = pd.read_parquet(train['file_path'][0])
		example_parquet_df = pd.read_parquet(INFERENCE_FILE_PATHS[0])

		# Each parquet file contains 1000 recordings
		print(f'# Unique Recording: {example_parquet_df.index.nunique()}')
		# Display DataFrame layout
		print(example_parquet_df.head())

		# Get indices in original dataframe
		def get_idxs(df, words_pos, words_neg=[], ret_names=True, idxs_pos=None):
			"""
			Returns column indices, or both column indices and names
			Input: dataframe, body_name, words/letters to exclude, get names or not, exact positions to get 
			"""
			idxs = []
			names = []
			for w in words_pos:
				for col_idx, col in enumerate(example_parquet_df.columns):
					# Exclude Non Landmark Columns
					if col in ['frame']:
						continue
						
					col_idx = int(col.split('_')[-1])
					# Check if column name contains all words
					if (w in col) and (idxs_pos is None or col_idx in idxs_pos) and all([w not in col for w in words_neg]):
						idxs.append(col_idx)
						names.append(col)
			# Convert to Numpy arrays
			idxs = np.array(idxs)
			names = np.array(names)
			# Returns either both column indices and names
			if ret_names:
				return idxs, names
			# Or only columns indices
			else:
				return idxs


		# Lips Landmark Face Ids
		# #AM we could try adding more face landmarks.
		LIPS_LANDMARK_IDXS = np.array([
				61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
				291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
				78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
				95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
			])

		# Landmark Indices for Left/Right hand without z axis in raw data
		LEFT_HAND_IDXS0, LEFT_HAND_NAMES0 = get_idxs(example_parquet_df, ['left_hand'], ['z'])
		RIGHT_HAND_IDXS0, RIGHT_HAND_NAMES0 = get_idxs(example_parquet_df, ['right_hand'], ['z'])
		LIPS_IDXS0, LIPS_NAMES0 = get_idxs(example_parquet_df, ['face'], ['z'], idxs_pos=LIPS_LANDMARK_IDXS)
		COLUMNS0 = np.concatenate((LEFT_HAND_NAMES0, RIGHT_HAND_NAMES0, LIPS_NAMES0))
		N_COLS0 = len(COLUMNS0)
		# Only X/Y axes are used
		N_DIMS0 = 2

		print(f'N_COLS0: {N_COLS0}')


		# Landmark Indices in subset of dataframe with only COLUMNS selected
		LEFT_HAND_IDXS = np.argwhere(np.isin(COLUMNS0, LEFT_HAND_NAMES0)).squeeze()
		RIGHT_HAND_IDXS = np.argwhere(np.isin(COLUMNS0, RIGHT_HAND_NAMES0)).squeeze()
		LIPS_IDXS = np.argwhere(np.isin(COLUMNS0, LIPS_NAMES0)).squeeze()
		HAND_IDXS = np.concatenate((LEFT_HAND_IDXS, RIGHT_HAND_IDXS), axis=0)
		N_COLS = N_COLS0
		# Only X/Y axes are used
		N_DIMS = 2

		print(f'N_COLS: {N_COLS}')

		# Indices in processed data by axes with only dominant hand
		HAND_X_IDXS = np.array(
				[idx for idx, name in enumerate(LEFT_HAND_NAMES0) if 'x' in name]
			).squeeze()
		HAND_Y_IDXS = np.array(
				[idx for idx, name in enumerate(LEFT_HAND_NAMES0) if 'y' in name]
			).squeeze()
		# Names in processed data by axes
		HAND_X_NAMES = LEFT_HAND_NAMES0[HAND_X_IDXS]
		HAND_Y_NAMES = LEFT_HAND_NAMES0[HAND_Y_IDXS]
		HAND_X_NAMES.shape, HAND_Y_NAMES.shape

		# Mean/Standard Deviations of data used for normalizing
		MEANS = np.load('aslfr-preprocessing-dataset/MEANS.npy').reshape(-1)
		STDS = np.load('aslfr-preprocessing-dataset/STDS.npy').reshape(-1)

		"""
			Tensorflow layer to process data in TFLite
			Data needs to be processed in the model itself, so we can not use Python
		""" 
		class PreprocessLayer(tf.keras.layers.Layer):
			def __init__(self):
				super(PreprocessLayer, self).__init__()
				self.normalisation_correction = tf.constant(
							# Add 0.50 to x coordinates of left hand (original right hand) and substract 0.50 of right hand (original left hand)
							 [0.50 if 'x' in name else 0.00 for name in LEFT_HAND_NAMES0],
						dtype=tf.float32,
					)
			
			@tf.function(
				input_signature=(tf.TensorSpec(shape=[None,N_COLS0], dtype=tf.float32),),
			)
			def call(self, data0, resize=True):
				# Fill NaN Values With 0
				data = tf.where(tf.math.is_nan(data0), 0.0, data0)
				
				# Hacky
				data = data[None]
				
				# Empty Hand Frame Filtering
				hands = tf.slice(data, [0,0,0], [-1, -1, 84])
				hands = tf.abs(hands)
				mask = tf.reduce_sum(hands, axis=2)
				mask = tf.not_equal(mask, 0)
				data = data[mask][None]
				
				# Pad Zeros
				N_FRAMES = len(data[0])
				if N_FRAMES < N_TARGET_FRAMES:
					data = tf.concat((
						data,
						tf.zeros([1,N_TARGET_FRAMES-N_FRAMES,N_COLS], dtype=tf.float32)
					), axis=1)
				# Downsample
				data = tf.image.resize(
					data,
					[1, N_TARGET_FRAMES],
					method=tf.image.ResizeMethod.BILINEAR,
				)
				
				# Squeeze Batch Dimension
				data = tf.squeeze(data, axis=[0])
				
				return data
			
		preprocess_layer = PreprocessLayer()
			


		# Function To Test Preprocessing Layer
		def test_preprocess_layer():
			demo_sequence_id = example_parquet_df.index.unique()[15]
			demo_raw_data = example_parquet_df.loc[demo_sequence_id, COLUMNS0]
			data = preprocess_layer(demo_raw_data)

			print(f'demo_raw_data shape: {demo_raw_data.shape}')
			print(f'data shape: {data.shape}')
			
			return data
			
		if IS_INTERACTIVE:
			data = test_preprocess_layer()

		def get_train_dataset(X, y, batch_size=BATCH_SIZE):
			sample_idxs = np.arange(len(X))
			while True:
				# Get random indices
				random_sample_idxs = np.random.choice(sample_idxs, batch_size)
				
				inputs = {
					'frames': X[random_sample_idxs],
					'phrase': y[random_sample_idxs],
				}
				outputs = y[random_sample_idxs]
				
				yield inputs, outputs


		# Train Dataset
		train_dataset = get_train_dataset(X_train, y_train)


		# Training Steps Per Epoch
		TRAIN_STEPS_PER_EPOCH = math.ceil(N_TRAIN_SAMPLES / BATCH_SIZE)
		print(f'TRAIN_STEPS_PER_EPOCH: {TRAIN_STEPS_PER_EPOCH}')

		# Validation Set
		def get_val_dataset(X, y, batch_size=BATCH_SIZE):
			offsets = np.arange(0, len(X), batch_size)
			while True:
				# Iterate over whole validation set
				for offset in offsets:
					inputs = {
						'frames': X[offset:offset+batch_size],
						'phrase': y[offset:offset+batch_size],
					}
					outputs = y[offset:offset+batch_size]

					yield inputs, outputs



		# Validation Dataset
		print(USE_VAL)
		if USE_VAL:
			val_dataset = get_val_dataset(X_val, y_val)
				

		if USE_VAL:
			N_VAL_STEPS_PER_EPOCH = math.ceil(N_VAL_SAMPLES / BATCH_SIZE)
			print(f'N_VAL_STEPS_PER_EPOCH: {N_VAL_STEPS_PER_EPOCH}')
			

		# Model Config

		# Epsilon value for layer normalisation
		LAYER_NORM_EPS = 1e-6

		# final embedding and transformer embedding size
		UNITS_ENCODER = 384
		UNITS_DECODER = 256

		# Transformer
		NUM_BLOCKS_ENCODER = 3
		NUM_BLOCKS_DECODER = 2
		NUM_HEADS = 4
		MLP_RATIO = 2

		# Dropout
		EMBEDDING_DROPOUT = 0.00
		MLP_DROPOUT_RATIO = 0.30
		MHA_DROPOUT_RATIO = 0.20
		CLASSIFIER_DROPOUT_RATIO = 0.10

		# Initiailizers
		INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
		INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
		INIT_ZEROS = tf.keras.initializers.constant(0.0)
		# Activations
		GELU = tf.keras.activations.gelu
			

		# Landmarks Embedding

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
					tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU), # Can change activation
					tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
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

		# Embedding

		# Creates embedding for each frame
		class Embedding(tf.keras.Model):
			def __init__(self):
				super(Embedding, self).__init__()
				self.supports_masking = True
			
			def build(self, input_shape):
				# Positional embedding for each frame index
				self.positional_embedding = tf.Variable(
					initial_value=tf.zeros([N_TARGET_FRAMES, UNITS_ENCODER], dtype=tf.float32),
					trainable=True,
					name='embedding_positional_encoder',
				)
				# Embedding layer for Landmarks
				self.dominant_hand_embedding = LandmarkEmbedding(UNITS_ENCODER, 'dominant_hand')

			def call(self, x, training=False):
				# Normalize
				x = tf.where(
						tf.math.equal(x, 0.0),
						0.0,
						(x - MEANS) / STDS,
					)
				# Dominant Hand
				x = self.dominant_hand_embedding(x)
				# Add Positional Encoding
				x = x + self.positional_embedding
				
				return x

		# Transfomer

		# replaced softmax with softmax layer to support masked softmax
		def scaled_dot_product(q,k,v, softmax, attention_mask):
			#calculates Q . K(transpose)
			qkt = tf.matmul(q,k,transpose_b=True)
			#caculates scaling factor
			dk = tf.math.sqrt(tf.cast(q.shape[-1],dtype=tf.float32))
			scaled_qkt = qkt/dk
			softmax = softmax(scaled_qkt, mask=attention_mask)
			z = tf.matmul(softmax,v)
			#shape: (m,Tx,depth), same shape as q,k,v
			return z

		class MultiHeadAttention(tf.keras.layers.Layer):
			def __init__(self,d_model, num_of_heads, dropout, d_out=None):
				super(MultiHeadAttention,self).__init__()
				self.d_model = d_model
				self.num_of_heads = num_of_heads
				self.depth = d_model//num_of_heads
				self.wq = [tf.keras.layers.Dense(self.depth//2, use_bias=False) for i in range(num_of_heads)] # depth//2 isn't common, we can try different numbers
				self.wk = [tf.keras.layers.Dense(self.depth//2, use_bias=False) for i in range(num_of_heads)]
				self.wv = [tf.keras.layers.Dense(self.depth//2, use_bias=False) for i in range(num_of_heads)]
				self.wo = tf.keras.layers.Dense(d_model if d_out is None else d_out, use_bias=False)
				self.softmax = tf.keras.layers.Softmax()
				self.do = tf.keras.layers.Dropout(dropout)
				self.supports_masking = True
				
			def call(self, q, k, v, attention_mask=None, training=False):
				
				multi_attn = []
				for i in range(self.num_of_heads):
					Q = self.wq[i](q)
					K = self.wk[i](k)
					V = self.wv[i](v)
					multi_attn.append(scaled_dot_product(Q,K,V, self.softmax, attention_mask))
					
				multi_head = tf.concat(multi_attn, axis=-1)
				multi_head_attention = self.wo(multi_head)
				multi_head_attention = self.do(multi_head_attention, training=training)
				
				return multi_head_attention

		# Encoder

		class Encoder(tf.keras.Model):
			def __init__(self, num_blocks):
				super(Encoder, self).__init__(name='encoder')
				self.num_blocks = num_blocks
				self.support_masking = True
			
			def build(self, input_shape):
				self.ln_1s = []
				self.mhas = []
				self.ln_2s = []
				self.mlps = []
				for i in range(self.num_blocks):
					# Normalization Layer
					self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
					# MultiHeads Layer
					self.mhas.append(MultiHeadAttention(UNITS_ENCODER, NUM_HEADS, MHA_DROPOUT_RATIO))
					# Normalization Layer 
					self.ln_2s.append(tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
					# Multi Layer Preception
					self.mlps.append(tf.keras.Sequential([
						tf.keras.layers.Dense(UNITS_ENCODER * MLP_RATIO, activation=GELU, kernel_initializer=INIT_GLOROT_UNIFORM, use_bias=False),
						tf.keras.layers.Dropout(MLP_DROPOUT_RATIO),
						tf.keras.layers.Dense(UNITS_ENCODER, kernel_initializer=INIT_HE_UNIFORM, use_bias=False), # Can change 
					]))
						# Optional Projection to Decoder Dimension
					if UNITS_ENCODER != UNITS_DECODER:
						self.dense_out = tf.keras.layers.Dense(UNITS_DECODER, kernel_initializer=INIT_GLOROT_UNIFORM, use_bias=False)
						self.apply_dense_out = True
					else:
						self.apply_dense_out = False
						
			def call(self, x, x_inp, training=False):
				#Attention mask to ignore missing frames
				attention_mask = tf.where(tf.math.reduce_sum(x_inp, axis=[2]) == 0.0, 0.0, 1.0)
				attention_mask = tf.expand_dims(attention_mask, axis=1)
				attention_mask = tf.repeat(attention_mask, repeats=N_TARGET_FRAMES, axis=1)
			   # Iterate input over transformer blocks
				for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
					x = ln_1(x + mha(x, x, x, attention_mask=attention_mask))
					x = ln_2(x + mlp(x))
					
				# Optional Projection to Decoder Dimension
				if self.apply_dense_out:
					x = self.dense_out(x)
			
				return x

		# Decoder

		def get_causal_attention_mask(B):
			# My version of the mask AM
			ones = tf.ones((N_TARGET_FRAMES, N_TARGET_FRAMES))
			mask = tf.linalg.band_part(ones, 0, -1)  
			mask = tf.transpose(mask)
			mask = tf.expand_dims(mask, axis=0)
			mask = tf.tile(mask, [B, 1, 1])
			mask = tf.cast(mask, tf.float32)
			return mask

		get_causal_attention_mask(2)


		# Decoder based on multiple transformer blocks
		class Decoder(tf.keras.Model):
			def __init__(self, num_blocks):
				super(Decoder, self).__init__(name='decoder')
				self.num_blocks = num_blocks
				self.supports_masking = True
			
			def build(self, input_shape):
				# Positional Embedding, initialized with zeros
				self.positional_embedding = tf.Variable(
					initial_value=tf.zeros([N_TARGET_FRAMES, UNITS_DECODER], dtype=tf.float32),
					trainable=True,
					name='embedding_positional_encoder',
				)
				# Character Embedding
				self.char_emb = tf.keras.layers.Embedding(N_UNIQUE_CHARACTERS, UNITS_DECODER, embeddings_initializer=INIT_ZEROS)
				# Positional Encoder MHA
				self.pos_emb_mha = MultiHeadAttention(UNITS_DECODER, NUM_HEADS, MHA_DROPOUT_RATIO)
				self.pos_emb_ln = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
				# First Layer Normalisation
				self.ln_1s = []
				self.mhas = []
				self.ln_2s = []
				self.mlps = []
				# Make Transformer Blocks
				for i in range(self.num_blocks):
					# First Layer Normalisation
					self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
					# Multi Head Attention
					self.mhas.append(MultiHeadAttention(UNITS_DECODER, NUM_HEADS, MHA_DROPOUT_RATIO))
					# Second Layer Normalisation
					self.ln_2s.append(tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
					# Multi Layer Perception
					self.mlps.append(tf.keras.Sequential([
						tf.keras.layers.Dense(UNITS_DECODER * MLP_RATIO, activation=GELU, kernel_initializer=INIT_GLOROT_UNIFORM, use_bias=False),
						tf.keras.layers.Dropout(MLP_DROPOUT_RATIO),
						tf.keras.layers.Dense(UNITS_DECODER, kernel_initializer=INIT_HE_UNIFORM, use_bias=False),
					]))
						

			def get_causal_attention_mask(self, B):
				# My version of the mask AM
				ones = tf.ones((N_TARGET_FRAMES, N_TARGET_FRAMES))
				mask = tf.linalg.band_part(ones, 0, -1)  
				mask = tf.transpose(mask)
				mask = tf.expand_dims(mask, axis=0)
				mask = tf.tile(mask, [B, 1, 1])
				mask = tf.cast(mask, tf.float32)
				return mask
			
					
			def call(self, encoder_outputs, phrase, training=False):
				# Batch Size
				B = tf.shape(encoder_outputs)[0]
				# Cast to INT32
				phrase = tf.cast(phrase, tf.int32)
				# Prepend SOS Token
				phrase = tf.pad(phrase, [[0,0], [1,0]], constant_values=SOS_TOKEN, name='prepend_sos_token')
				# Pad With PAD Token
				phrase = tf.pad(phrase, [[0,0], [0,N_TARGET_FRAMES-MAX_PHRASE_LENGTH-1]], constant_values=PAD_TOKEN, name='append_pad_token')
				# Causal Mask
				causal_mask = self.get_causal_attention_mask(B)
				# Positional Embedding
				x = self.positional_embedding + self.char_emb(phrase)
				# Causal Attention
				x = self.pos_emb_ln(x + self.pos_emb_mha(x, x, x, attention_mask=causal_mask))
				# Iterate input over causal_masktransformer blocks
				for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
					x = ln_1(x + mha(x, encoder_outputs, encoder_outputs, attention_mask=causal_mask))
					x = ln_2(x + mlp(x))
				# Slice 31 Characters
				x = tf.slice(x, [0, 0, 0], [-1, MAX_PHRASE_LENGTH, -1])
			
				return x

		# Non Pad/SOS/EOS Token Accuracy

		# TopK accuracy for multi dimensional output
		class TopKAccuracy(tf.keras.metrics.Metric):
			def __init__(self, k, **kwargs):
				super(TopKAccuracy, self).__init__(name=f'top{k}acc', **kwargs)
				self.top_k_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)

			def update_state(self, y_true, y_pred, sample_weight=None):
				y_true = tf.reshape(y_true, [-1])
				y_pred = tf.reshape(y_pred, [-1, N_UNIQUE_CHARACTERS])
				character_idxs = tf.where(y_true < N_UNIQUE_CHARACTERS0)
				y_true = tf.gather(y_true, character_idxs, axis=0)
				y_pred = tf.gather(y_pred, character_idxs, axis=0)
				self.top_k_acc.update_state(y_true, y_pred)

			def result(self):
				return self.top_k_acc.result()
			
			def reset_state(self):
				self.top_k_acc.reset_state()

			

		# Sparse Categorical Crossentropy With Label Smoothing

		# source:: https://stackoverflow.com/questions/60689185/label-smoothing-for-sparse-categorical-crossentropy
		def scce_with_ls(y_true, y_pred):
			# Filter Pad Tokens
			idxs = tf.where(y_true != PAD_TOKEN)
			y_true = tf.gather_nd(y_true, idxs)
			y_pred = tf.gather_nd(y_pred, idxs)
			# One Hot Encode Sparsely Encoded Target Sign
			y_true = tf.cast(y_true, tf.int32)
			y_true = tf.one_hot(y_true, N_UNIQUE_CHARACTERS, axis=1)
			# Categorical Crossentropy with native label smoothing support
			loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.25, from_logits=True)
			loss = tf.math.reduce_mean(loss)
			return loss



		# This gives equal weight to all charcters except for Pad token during loss.

		# Create Initial Loss Weights All Set To 1
		loss_weights = tf.ones([N_UNIQUE_CHARACTERS], dtype=tf.float32)
		# Set Loss Weight Of Pad Token To 0
		loss_weights = tf.tensor_scatter_nd_update(loss_weights, [[PAD_TOKEN]], [0])
			

		# Model

		def get_model():
			# Inputs
			frames_inp = tf.keras.layers.Input([N_TARGET_FRAMES, N_COLS], dtype=tf.float32, name='frames')
			phrase_inp = tf.keras.layers.Input([MAX_PHRASE_LENGTH], dtype=tf.int32, name='phrase')
			# Frames
			x = frames_inp
			
			# Masking 
			x = tf.keras.layers.Masking(mask_value=0.0, input_shape=(N_TARGET_FRAMES, N_COLS))(x)
			
			# Embedding
			x = Embedding()(x)
			
			# Encoder Transformer Blocks
			x = Encoder(NUM_BLOCKS_ENCODER)(x, frames_inp)
			
			# Decoder
			x = Decoder(NUM_BLOCKS_DECODER)(x, phrase_inp)
			
			# Classifier
			x = tf.keras.Sequential([
				# Dropout
				tf.keras.layers.Dropout(CLASSIFIER_DROPOUT_RATIO),
				# Output Neurons
				tf.keras.layers.Dense(N_UNIQUE_CHARACTERS, activation=tf.keras.activations.linear, kernel_initializer=INIT_HE_UNIFORM, use_bias=False),
			], name='classifier')(x)
			
			outputs = x
			
			# Create Tensorflow Model
			model = tf.keras.models.Model(inputs=[frames_inp, phrase_inp], outputs=outputs)
			
			#optimizer
			optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # We should try different optimizers with learning rate scheduler
			
			# Categorical Crossentropy Loss With Label Smoothing
			loss = scce_with_ls
			
			metrics = [
				TopKAccuracy(1),
				TopKAccuracy(5),
			]
			
			model.compile(
				loss=loss,
				optimizer=optimizer,
				metrics=metrics,
				loss_weights=loss_weights,
			)
			
			return model


		for k, v in X_batch.items():
			print(f'{k}: {v.shape}')

		tf.keras.backend.clear_session()

		model = get_model()

		print(model.summary())

		tf.keras.utils.plot_model(model, show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)

		N_EPOCHS=20


		 history = model.fit(
					x=train_dataset,
					steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
					epochs=N_EPOCHS,
					validation_data=val_dataset,
					validation_steps=N_VAL_STEPS_PER_EPOCH,
					verbose = VERBOSE,
				)

		tf.saved_model.save(model, 'Transformer_20_epoch.h5')
			


		model.save('Transformer_20_2_epoch.h5')
		   

		# Save history
		train_loss = history.history['loss']
		train_top1acc = history.history['top1acc']
		train_top5acc = history.history['top5acc']

		val_loss = history.history['val_loss']
		val_top1acc = history.history['val_top1acc']
		val_top5acc = history.history['val_top5acc']
			
		epochs = range(1, len(val_loss) + 1)

		# Plotting Loss
		plt.figure(figsize=(6,6))
		plt.plot(epochs, train_loss, 'b', label='Train Loss')
		plt.plot(epochs, val_loss, 'r', label='Validation Loss')
		plt.title('Loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.tight_layout()
		plt.show();

		# Plotting Accuracy
		plt.figure(figsize=(6,6))
		plt.plot(epochs, train_top1acc, 'b', label='Train Top 1 Accuracy')
		plt.plot(epochs, val_top1acc, 'r', label='Validation Top 1 Accuracy')
		plt.title('Top 1 Accuracy')
		plt.xlabel('Epochs')
		plt.ylabel('Top 1 Accuracy')
		plt.legend()
		plt.tight_layout()
		plt.show();

		plt.figure(figsize=(6,6))
		plt.plot(epochs, train_top5acc, 'b', label='Train Top 5 Accuracy')
		plt.plot(epochs, val_top5acc, 'r', label='Validation Top 5 Accuracy')
		plt.title('Top 5 Accuracy')
		plt.xlabel('Epochs')
		plt.ylabel('Top 5 Accuracy')
		plt.legend()
		plt.tight_layout()
		plt.show();


		def get_model():
			# Inputs
			frames_inp = tf.keras.layers.Input([N_TARGET_FRAMES, N_COLS], dtype=tf.float32, name='frames')
			phrase_inp = tf.keras.layers.Input([MAX_PHRASE_LENGTH], dtype=tf.int32, name='phrase')
			# Frames
			x = frames_inp
			
			# Masking 
			x = tf.keras.layers.Masking(mask_value=0.0, input_shape=(N_TARGET_FRAMES, N_COLS))(x)
			
			# Embedding
			x = Embedding()(x)
			
			# Encoder Transformer Blocks
			x = Encoder(NUM_BLOCKS_ENCODER)(x, frames_inp)
			
			# Decoder
			x = Decoder(NUM_BLOCKS_DECODER)(x, phrase_inp)
			
			# Classifier
			x = tf.keras.Sequential([
				# Dropout
				tf.keras.layers.Dropout(CLASSIFIER_DROPOUT_RATIO),
				# Output Neurons
				tf.keras.layers.Dense(N_UNIQUE_CHARACTERS, activation=tf.keras.activations.linear, kernel_initializer=INIT_HE_UNIFORM, use_bias=False),
			], name='classifier')(x)
			
			outputs = x
			
			# Create Tensorflow Model
			model = tf.keras.models.Model(inputs=[frames_inp, phrase_inp], outputs=outputs)
			
			#optimizer
			optimizer = tf.keras.optimizers.Adam() # We should try different optimizers with learning rate scheduler
			
			# Categorical Crossentropy Loss With Label Smoothing
			loss = scce_with_ls
			
			metrics = [
				TopKAccuracy(1),
				TopKAccuracy(5),
			]
			
			 # Adam Optimizer 
			optimizer = tfa.optimizers.RectifiedAdam(sma_threshold=4)
			optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=5)
			
			model.compile(
				loss=loss,
				optimizer=optimizer,
				metrics=metrics,
				loss_weights=loss_weights,
				
			)
				

		# Custom callback to update weight decay with learning rate
		class WeightDecayCallback(tf.keras.callbacks.Callback):
			def __init__(self, wd_ratio=WD_RATIO):
				self.step_counter = 0
				self.wd_ratio = wd_ratio
			
			def on_epoch_begin(self, epoch, logs=None):
				model.optimizer.weight_decay = model.optimizer.learning_rate * self.wd_ratio
				print(f'learning rate: {model.optimizer.learning_rate.numpy():.2e}, weight decay: {model.optimizer.weight_decay.numpy():.2e}')


		def lrfn(current_step, num_warmup_steps, lr_max, num_cycles=0.50, num_training_steps=N_EPOCHS):
			
			if current_step < num_warmup_steps:
				if WARMUP_METHOD == 'log':
					return lr_max * 0.10 ** (num_warmup_steps - current_step)
				else:
					return lr_max * 2 ** -(num_warmup_steps - current_step)
			else:
				progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

				return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max


if __name__ == "__main__":
    eda_preprocess = EDA_Preprocess()
    eda_preprocess.run()
	training = Training()
	training.main()
	inference = Inference()
	inference.main()



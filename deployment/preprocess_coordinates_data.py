import pandas as pd
import numpy as np
import tensorflow as tf
import json

# Convert the variables to the correct data type
# Load the variables from the JSON file
json_file_path = "variables.json"
with open(json_file_path, 'r') as json_file:
    variables_dict = json.load(json_file)

# Lips Landmark Face Ids
LIPS_LANDMARK_IDXS = variables_dict['LIPS_LANDMARK_IDXS']
N_TARGET_FRAMES = variables_dict['N_TARGET_FRAMES']
N_DIMS0 = variables_dict['N_TARGET_FRAMES']

# Read data from a CSV file
df = pd.read_csv('landmarks.csv')

def get_idxs(df, words_pos, words_neg=[], ret_names=True, idxs_pos=None):
    """
    Given a DataFrame and a list of words, this function will find all the column names
    that contain all the words in 'words_pos' and none of the words in 'words_neg'.

    Parameters:
        df (pandas.DataFrame): Dataframe to search for column names
        words_pos (list of str): List of words that column names should contain
        words_neg (list of str, optional): List of words that column names should not contain. Default is empty list.
        ret_names (bool, optional): Whether to return column names. Default is True.
        idxs_pos (list of int, optional): Column indices to search within. Default is None, which means search all columns.

    Returns:
        idxs (np.array): Column indices where column names meet the criteria
        names (np.array): Column names that meet the criteria. Only returned if 'ret_names' is True.
    """
    idxs = []
    names = []
    for w in words_pos:
        for col_idx, col in enumerate(df.columns):
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


# Get the indices of columns of interest
LEFT_HAND_IDXS0, LEFT_HAND_NAMES0 = get_idxs(df, ['left_hand'], ['z'])
RIGHT_HAND_IDXS0, RIGHT_HAND_NAMES0 = get_idxs(df, ['right_hand'], ['z'])
LIPS_IDXS0, LIPS_NAMES0 = get_idxs(df, ['face'], ['z'], idxs_pos=LIPS_LANDMARK_IDXS)
COLUMNS0 = np.concatenate((LEFT_HAND_NAMES0, RIGHT_HAND_NAMES0, LIPS_NAMES0))
N_COLS0 = len(COLUMNS0)
N_COLS = N_COLS0


class PreprocessLayerNonNaN(tf.keras.layers.Layer):
    """
    This is a custom layer in Keras that replaces NaN values in the input tensor with 0.
    """

    def __init__(self):
        super(PreprocessLayerNonNaN, self).__init__()

    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, N_COLS0], dtype=tf.float32),),
    )
    def call(self, data0):
        """
        This method is called when the layer instance is called with some inputs.

        Parameters:
            data0 (Tensor): Input tensor

        Returns:
            data (Tensor): Output tensor with the same shape as the input, but with NaN values replaced with 0
        """
        # Fill NaN Values With 0
        data = tf.where(tf.math.is_nan(data0), 0.0, data0)

        # Hacky
        data = data[None]

        # Empty Hand Frame Filtering
        hands = tf.slice(data, [0, 0, 0], [-1, -1, 84])
        hands = tf.abs(hands)
        mask = tf.reduce_sum(hands, axis=2)
        mask = tf.not_equal(mask, 0)
        data = data[mask][None]
        data = tf.squeeze(data, axis=[0])

        return data


class PreprocessLayer(tf.keras.layers.Layer):
    """
    This is a custom layer in Keras that pre-processes the input data in a specific way,
    which includes filling NaN values with 0, filtering empty frames and resizing frames.
    """

    def __init__(self):
        super(PreprocessLayer, self).__init__()

    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, None, N_COLS0], dtype=tf.float32),),
    )
    def call(self, data0, resize=True):
        """
        This method is called when the layer instance is called with some inputs.

        Parameters:
            data0 (Tensor): Input tensor
            resize (bool, optional): Whether to resize the frames. Default is True.

        Returns:
            data (Tensor): Output tensor after pre-processing
        """
        # Fill NaN Values With 0
        data = tf.where(tf.math.is_nan(data0), 0.0, data0)

        # Empty Hand Frame Filtering
        hands = tf.slice(data, [0, 0, 0], [-1, -1, 84])
        hands = tf.abs(hands)
        mask = tf.reduce_sum(hands, axis=2)
        mask = tf.not_equal(mask, 0)
        data = data[mask][None]

        # Pad Zeros
        N_FRAMES = len(data[0])
        if N_FRAMES < N_TARGET_FRAMES:
            data = tf.concat((
                data,
                tf.zeros([1, N_TARGET_FRAMES - N_FRAMES, N_COLS], dtype=tf.float32)
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


df = df[COLUMNS0]  # select only columns of interest equal to N_COLS0
hand_tracking_sequence = df.values.reshape(1, -1, N_COLS0)  # reshape after converting DataFrame to numpy array

preprocess_layer_instance = PreprocessLayer()  # instantiate PreprocessLayer class
processed_sequence = preprocess_layer_instance(hand_tracking_sequence)  # call instance with data

# print(f'input sequence shape: {hand_tracking_sequence.shape}')
# print(f'processed sequence shape: {processed_sequence.shape}')

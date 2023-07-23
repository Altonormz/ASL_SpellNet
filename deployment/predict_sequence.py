import json
import numpy as np
import tensorflow as tf

# Convert the variables to the correct data type
# Load the variables from the JSON file
json_file_path = "variables.json"
with open(json_file_path, 'r') as json_file:
    variables_dict = json.load(json_file)

MAX_PHRASE_LENGTH = variables_dict['MAX_PHRASE_LENGTH']
PAD_TOKEN = variables_dict['PAD_TOKEN']
N_UNIQUE_CHARACTERS = variables_dict['N_UNIQUE_CHARACTERS']

# Read Character to Ordinal Encoding Mapping
with open('character_to_prediction_index.json') as json_file:
    ORD2CHAR = json.load(json_file)



# Output Predictions to string
def outputs2phrase(outputs, ORD2CHAR):
    """
    Convert output sequence to a human-readable phrase by mapping
    each output to a corresponding character.

    Parameters:
    outputs (np.array): A sequence of model output,
                        can be 1D (sequence of character IDs)
                        or 2D (sequence of one-hot encodings).
    ORD2CHAR (dict): A mapping from character IDs to characters.

    Returns:
    str: The converted phrase.
    """
    ORD2CHAR = {int(k): v for k, v in ORD2CHAR.items()} # Convert keys to integers
    if outputs.ndim == 2:
        outputs = np.argmax(outputs, axis=1)
    return ''.join([ORD2CHAR.get(s, '') for s in outputs])


@tf.function()
def predict_phrase(batch_frames, model):
    """
    Use a pre-trained model to predict a phrase from a batch of frame sequences.

    Parameters:
    batch_frames (np.array): A batch of frame sequences.
    model (tf.keras.Model): The pre-trained model to use for prediction.

    Returns:
    tf.Tensor: One-hot encoding of the predicted phrase.
    """
    batch_frames = tf.convert_to_tensor(batch_frames)
    phrase = tf.fill([batch_frames.shape[0], MAX_PHRASE_LENGTH], PAD_TOKEN)
    phrase = tf.cast(phrase, tf.int32)  # Cast phrase to int32 initially
    for idx in tf.range(MAX_PHRASE_LENGTH):
        # Predict Next Token
        outputs = model({
            'frames': batch_frames,
            'phrase': phrase,
        })

        phrase = tf.where(
            tf.range(MAX_PHRASE_LENGTH)[None, :] < idx + 1,
            tf.argmax(outputs, axis=-1, output_type=tf.int32),
            phrase,
        )
    # one-hot encode the outputs
    outputs_one_hot = tf.one_hot(phrase, depth=N_UNIQUE_CHARACTERS)
    return outputs_one_hot

# # Assuming sequence is your array of shape (128, 164)
# sequence = processed_sequence._shape(1, *processed_sequence.shape)  # reshapes sequence to (1, 128, 164)
#
# # Now you can feed sequence to your prediction function
# pred_phrase_one_hot = predict_phrase(sequence)
#
# # Convert the one-hot encoded prediction to a string
# # Remember the output is one-hot encoded so we need to convert it to integers first
# pred_phrase = outputs2phrase(tf.argmax(pred_phrase_one_hot, axis=-1).numpy())
#
# print(pred_phrase)

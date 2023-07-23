import numpy as np
import os
import json
from tensorflow.math import argmax
import gradio as gr
from tensorflow.keras.models import model_from_json
import video_with_landmarks
import video_to_landmark_coordinates
import preprocess_coordinates_data
from load_model import Embedding, Encoder, Decoder, LandmarkEmbedding, EncoderTransformerBlock, MultiHeadAttention, \
    DecoderTransformerBlock
import predict_sequence

# Load the character to prediction index dictionary
character_to_prediction = 'character_to_prediction_index.json'
with open(character_to_prediction) as json_file:
    ORD2CHAR = json.load(json_file)

# Load the variables from the JSON file
json_file_path = "variables.json"
with open(json_file_path, 'r') as json_file:
    variables_dict = json.load(json_file)

# Load the model architecture from the JSON file
json_file = open('model_architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Import lips landmark indices
LIPS_LANDMARK_IDXS = np.array(variables_dict['LIPS_LANDMARK_IDXS'])

custom_objects = {'Embedding': Embedding,
                  'Encoder': Encoder,
                  'Decoder': Decoder,
                  'LandmarkEmbedding': LandmarkEmbedding,
                  'EncoderTransformerBlock': EncoderTransformerBlock,
                  'MultiHeadAttention': MultiHeadAttention,
                  'DecoderTransformerBlock': DecoderTransformerBlock}


def process_and_print_sequence(df):
    """
    Process the input DataFrame using specified data processing steps and print the shapes of the sequences.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing tracking data.

    Returns:
    processed_sequence (np.ndarray): Processed sequence as a NumPy array.
    """
    LEFT_HAND_IDXS0, LEFT_HAND_NAMES0 = preprocess_coordinates_data.get_idxs(df, ['left_hand'], ['z'])
    RIGHT_HAND_IDXS0, RIGHT_HAND_NAMES0 = preprocess_coordinates_data.get_idxs(df, ['right_hand'], ['z'])
    LIPS_IDXS0, LIPS_NAMES0 = preprocess_coordinates_data.get_idxs(df, ['face'], ['z'], idxs_pos=LIPS_LANDMARK_IDXS)
    COLUMNS0 = np.concatenate((LEFT_HAND_NAMES0, RIGHT_HAND_NAMES0, LIPS_NAMES0))
    N_COLS0 = len(COLUMNS0)

    df = df[COLUMNS0]  # select only columns of interest equal to N_COLS0
    all_tracking_sequence = df.values.reshape(1, -1, N_COLS0).astype(
        np.float32)  # reshape after converting DataFrame to numpy array
    preprocess_layer_instance = preprocess_coordinates_data.PreprocessLayer()  # instantiate PreprocessLayer class
    processed_sequence = preprocess_layer_instance(all_tracking_sequence)  # call instance with data

    print(f'input sequence shape: {all_tracking_sequence.shape}')
    print(f'processed sequence shape: {processed_sequence.shape}')

    return processed_sequence


def predict_final_sequence(processed_sequence, model):
    """
        This function makes a prediction on a given sequence using a pre-trained model.

        The sequence is expanded along the 0th dimension to account for batch size.
        The prediction is made using the `predict_phrase` function, which should return a one-hot encoded prediction.
        This one-hot encoded prediction is then converted into index values using argmax.
        Finally, these index values are converted into a string representation using the `outputs2phrase` function.

        Args:
        processed_sequence (numpy array): An array representing the sequence to make a prediction on.
                                          This should be of shape (128,164).
        model (tensorflow.python.keras.engine.training.Model): The pre-trained model to use for making predictions.

        Returns:
        final_prediction (str): The final prediction made by the model, represented as a string.
        """
    # change shape to (1,128,164)
    sequence = np.expand_dims(processed_sequence, axis=0)  # change shape to (1,128,164)

    # Convert the one-hot encoded prediction to a string
    predicted_phrase_one_hot = predict_sequence.predict_phrase(sequence, model)
    predicted_phrase_one_hot = predicted_phrase_one_hot[0]  # Remove the batch dimension
    predicted_phrase = argmax(predicted_phrase_one_hot, axis=-1).numpy()  # Convert one-hot encoding to index values
    print(predicted_phrase)
    final_prediction = predict_sequence.outputs2phrase(predicted_phrase, ORD2CHAR)
    return final_prediction


def video_identity(video):
    """
    Processes a video, extracts landmarks, feeds them to a pre-trained model, and makes a prediction.

    The processing pipeline consists of the following steps:
    1. Process the video with landmarks.
    2. Extract landmarks coordinates and save them into a DataFrame.
    3. Preprocess the landmarks.
    4. Load a pre-trained model.
    5. Feed the preprocessed landmarks to the model and get a prediction.

    Parameters:
    video (str): Path to the video file.

    Returns:
    tuple: The path to the processed video with landmarks and the predicted outcome.
    """
    # 1. load video and process it with landmarks
    original_video_path = video
    output_path = "video_landmarks.mp4"
    video_with_landmarks.process_video_with_landmarks(original_video_path, output_path)

    # 2. extract landmarks coordinates
    df = video_to_landmark_coordinates.video_to_landmarks(output_path,
                                                          video_to_landmark_coordinates.generate_column_names())
    # Save the DataFrame to a CSV file
    # df.to_csv('landmarks.csv', index=False)

    # 3. preprocess landmarks
    # Read data from a CSV file
    # df = pd.read_csv('landmarks2.csv')
    # df.drop(['sequence_id'],axis = 1, inplace=True)
    processed_sequence = process_and_print_sequence(df)

    # 4. load model
    # load model architecture from JSON file
    model = model_from_json(loaded_model_json, custom_objects=custom_objects)

    # load weights into the new model
    model.load_weights("model.h5")

    # loaded_model.summary(expand_nested=True, show_trainable=True, )

    # 5. predict
    prediction = predict_final_sequence(processed_sequence, model)
    print(prediction)

    return output_path, prediction


iface = gr.Interface(video_identity,
                     gr.inputs.Video(label="Upload your video", source="upload"),  # Adding a label to the input
                     [gr.outputs.Video(label="Processed video"), gr.outputs.Textbox(label="Predicted Outcome")],
                     # Adding labels to the outputs
                     title="spellNET",  # Adding a title
                     description="This application analyzes your video input to interpret American Sign Language (ASL) gestures corresponding to letters, numbers, and other signs. The output consists of the original video enhanced with overlaid landmarks that represent key points of ASL gestures, along with the predicted decoded ASL sequence expressed in textual form.",
                     # Adding a description
                     theme="abidlabs/pakistan",  # Changing the theme
                     examples=[[os.path.join(os.path.dirname(__file__), "yoni-phone.mp4")],
                               [os.path.join(os.path.dirname(__file__), "videoplayback.mp4")]],
                     cache_examples=True)

if __name__ == "__main__":
    iface.launch(share=True)

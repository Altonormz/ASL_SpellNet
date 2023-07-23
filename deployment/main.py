import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import model_from_json

import video_with_landmarks
import video_to_landmark_coordinates
import preprocess_coordinates_data
from load_model import Embedding, Encoder, Decoder, LandmarkEmbedding, EncoderTransformerBlock, MultiHeadAttention, \
    DecoderTransformerBlock
import predict_sequence

# import Gradio_inference

# Load the character to prediction index dictionary
character_to_prediction = 'character_to_prediction_index.json'
with open(character_to_prediction) as json_file:
    ORD2CHAR = json.load(json_file)

# Load the variables from the JSON file
json_file_path = "variables.json"
with open(json_file_path, 'r') as json_file:
    variables_dict = json.load(json_file)

# Import lips landmark indices
LIPS_LANDMARK_IDXS = np.array(variables_dict['LIPS_LANDMARK_IDXS'])

custom_objects = {'Embedding': Embedding,
                  'Encoder': Encoder,
                  'Decoder': Decoder,
                  'LandmarkEmbedding': LandmarkEmbedding,
                  'EncoderTransformerBlock': EncoderTransformerBlock,
                  'MultiHeadAttention': MultiHeadAttention,
                  'DecoderTransformerBlock': DecoderTransformerBlock}


def main():
    # 1. load video and process it with landmarks
    original_video_path = "the ASL ALPHABET in 15 seconds - American Sign Language.mp4"
    output_path = "ASL_with_landmarks.mp4"
    video_with_landmarks.process_video_with_landmarks(original_video_path, output_path)

    # 2. extract landmarks coordinates
    df = video_to_landmark_coordinates.video_to_landmarks(output_path,
                                                          video_to_landmark_coordinates.generate_column_names())
    # Save the DataFrame to a CSV file
    # df.to_csv('landmarks.csv', index=False)

    # 3. preprocess landmarks
    # Read data from a CSV file
    # df = pd.read_csv('landmarks.csv')

    # Get the indices of columns of interest
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

    # 4. load model
    json_file = open('model_architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load model architecture from JSON file
    model = model_from_json(loaded_model_json, custom_objects=custom_objects)

    # load weights into the new model
    model.load_weights("model.h5")

    # loaded_model.summary(expand_nested=True, show_trainable=True, )

    # 5. predict
    sequence = np.expand_dims(processed_sequence, axis=0)  # change shape to (1,128,164)

    # Convert the one-hot encoded prediction to a string
    predicted_phrase_one_hot = predict_sequence.predict_phrase(sequence, model)
    # Assuming the output of predict_phrase is stored in 'outputs'
    predicted_phrase_one_hot = predicted_phrase_one_hot[0]  # Remove the batch dimension
    predicted_phrase = tf.argmax(predicted_phrase_one_hot, axis=-1).numpy()  # Convert one-hot encoding to index values
    print(predicted_phrase)

    true_phrase = predict_sequence.outputs2phrase(predicted_phrase, ORD2CHAR)
    print(true_phrase)


# 6. Gradio
if __name__ == "__main__":
    main()

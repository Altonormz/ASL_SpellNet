import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

video_path = "fingerspelling_yonatan.mp4"

def generate_column_names():
    columns = ['frame']

    # face columns
    for coordinate in ['x', 'y']:
        for i in range(468):  # Mediapipe face mesh contains 468 landmarks
            columns.append(f'{coordinate}_face_{i}')

    # hands columns
    for hand in ['left_hand', 'right_hand']:
        for coordinate in ['x', 'y']:
            for i in range(21):  # Mediapipe hand model contains 21 landmarks
                columns.append(f'{coordinate}_{hand}_{i}')

    return columns

def video_to_landmarks(video_path, columns):
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(video_path)
    df = pd.DataFrame(columns=columns)

    with mp_face_mesh.FaceMesh() as face_mesh, mp_hands.Hands(max_num_hands=2) as hands:
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(rgb_frame)
            results_hands = hands.process(rgb_frame)

            # Initialize frame dictionary with NaNs
            frame_data = {column: np.NaN for column in columns}
            frame_data['frame'] = frame_count

            # Process face landmarks
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    for i, landmark in enumerate(face_landmarks.landmark):
                        frame_data[f'x_face_{i}'] = landmark.x
                        frame_data[f'y_face_{i}'] = landmark.y

            # Process hand landmarks
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x:
                        hand_type = 'left_hand'
                    else:
                        hand_type = 'right_hand'

                    for i, landmark in enumerate(hand_landmarks.landmark):
                        frame_data[f'x_{hand_type}_{i}'] = landmark.x
                        frame_data[f'y_{hand_type}_{i}'] = landmark.y

            df = df._append(frame_data, ignore_index=True)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return df

df = video_to_landmarks(video_path, generate_column_names())

# Save the DataFrame to a CSV file
df.to_csv('landmarks.csv', index=False)

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

# Open video file
video = cv2.VideoCapture('video.mp4')

frame_count = 0
frames_data = []

with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands, mp_face.FaceMesh(static_image_mode=True, min_detection_confidence=0.5) as face_mesh:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Convert the image from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        frame.flags.writeable = False

        # Recognize hands
        hands_result = hands.process(frame)

        # Recognize face
        face_result = face_mesh.process(frame)

        frame_data = {'frame': frame_count, 'hands': [], 'face': []}

        # Extract hands data
        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    frame_data['hands'].append({
                        'id': id,
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z
                    })

        # Extract face data
        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                for id, lm in enumerate(face_landmarks.landmark):
                    frame_data['face'].append({
                        'id': id,
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z
                    })

        frames_data.append(frame_data)
        frame_count += 1

video.release()

# Now frames_data list contains face and hand landmarks for each frame.
# You can save it to a file or use as is.

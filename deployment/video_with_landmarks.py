import cv2
import mediapipe as mp

def process_video_with_landmarks(video_path, output_path, scale_percent=100):
    """
    Process a video to identify and draw landmarks on faces and hands.

    Parameters:
    video_path (str): The path to the input video file.
    output_path (str): The path to the output video file.
    scale_percent (int, optional): The percentage of the original size. Default is 100.
    """
    # MediaPipe solutions
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the scale dimensions
    width = int(width * scale_percent / 100)
    height = int(height * scale_percent / 100)

    # Define the output video file
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    with mp_face_mesh.FaceMesh() as face_mesh, mp_hands.Hands() as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Resize the frame
            frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process face landmarks
            results_face = face_mesh.process(rgb_frame)
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                    )

            # Process hand landmarks
            results_hands = hands.process(rgb_frame)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x:
                        landmark_color = (255, 0, 0)  # Left hand (Blue)
                    else:
                        landmark_color = (0, 0, 255)  # Right hand (Red)

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=landmark_color, thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=landmark_color, thickness=1)
                    )

            # Write the annotated frame to the output video
            out.write(frame)

            # If 'q' is pressed on the keyboard, exit this loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Close the video file
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Now you can call the function with your video path
# process_video_with_landmarks("the ASL ALPHABET in 15 seconds - American Sign Language.mp4","ASL_landmarks.mp4")

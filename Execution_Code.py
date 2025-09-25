# Import required libraries
import cv2  # OpenCV is used for capturing video from the webcam and displaying output.
import mediapipe as mp  # A library from Google used to detect hand landmarks.
import numpy as np  # For numerical operations on landmarks.
from tensorflow.keras.models import load_model  # Used to load the pre-trained ASL hand landmarks model.
import pyttsx3  # Text-to-speech library to convert recognized ASL letters into speech.

# Disable OneDNN optimizations (for performance tuning)
TF_ENABLE_ONEDNN_OPTS = 0

# Load the pre-trained ASL hand landmarks model
model = load_model(r'asl_hand_landmarks_model.h5')

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Enable dynamic video feed processing.
    max_num_hands=1,  # Detect only one hand at a time.
    min_detection_confidence=0.5  # Minimum confidence threshold for hand detection.
)
mp_drawing = mp.solutions.drawing_utils  # Utility for drawing hand landmarks on the frame.

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define mapping of class indices to ASL letters and signs
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'delete', 'space', 'nothing'
]

# Initialize variables for sentence formation
sentence = []  # List to store recognized letters forming a sentence.
last_predicted_letter = ''  # Track the last predicted letter to avoid repetition.

# Open webcam for video capture
cap = cv2.VideoCapture(0)  # Capture video from the default webcam.

# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the webcam.
    if not ret:  # Exit the loop if the frame cannot be read.
        break

    # Convert the video frame to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks in the RGB frame
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:  # Check if any hand landmarks are detected.
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the detected hand landmarks on the video frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Extract landmarks as a flat array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            landmarks = np.expand_dims(landmarks, axis=0)  # Add batch dimension.

            # Predict the class label using the trained model
            prediction = model.predict(landmarks)
            predicted_class = np.argmax(prediction)  # Get the index of the highest probability class.
            predicted_letter = class_names[predicted_class]  # Map the index to the ASL character.

            # Avoid repeating letters and handle special cases
            if predicted_letter != last_predicted_letter:
                if predicted_letter == 'W' and sentence:
                    # Read the full sentence when 'W' is detected
                    full_sentence = "".join(sentence).replace('  ', ' ').strip()
                    engine.say(f"The sentence is: {full_sentence}")
                    engine.runAndWait()
                    sentence = []  # Clear the sentence after reading it.
                elif predicted_letter == 'space':
                    sentence.append(' ')
                    engine.say("space")
                    engine.runAndWait()
                else:
                    # Add the letter to the sentence
                    sentence.append(predicted_letter)
                    engine.say(predicted_letter)
                    engine.runAndWait()

                # Update the last predicted letter
                last_predicted_letter = predicted_letter

    # Display the current sentence on the video frame
    cv2.putText(
        frame,
        "Sentence: " + "".join(sentence),
        (10, 30),  # Position on the frame.
        cv2.FONT_HERSHEY_SIMPLEX,  # Font type.
        1,  # Font scale.
        (255, 0, 0),  # Font color (blue in BGR).
        2  # Font thickness.
    )

    # Show the video frame with recognized sentence
    cv2.imshow('ASL Sign Language Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(2000) & 0xFF == ord('q'):
        break

# Release resources
cap.release()  # Release the webcam.
cv2.destroyAllWindows()  # Close all OpenCV windows.

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('D:\Sign_language_Detection_notRLT\sign_language_model.h5')

# Define labels corresponding to the output classes (adjust according to your dataset)
sign_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z']  # Modify based on your dataset

# Function to preprocess frames before feeding them into the model
def preprocess_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28 (the input shape for the model)
    gray_resized = cv2.resize(gray, (28, 28))
    # Reshape the image to fit the model's input shape (28x28x1)
    gray_resized = gray_resized.reshape(1, 28, 28, 1)
    # Normalize pixel values to [0, 1]
    gray_resized = gray_resized.astype('float32') / 255.0
    return gray_resized

# Set up buffer for smoothing predictions
prediction_buffer = []

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Perform prediction
    prediction = model.predict(processed_frame)
    
    # Get the predicted class and the probability (confidence)
    predicted_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction)

    # Add a confidence threshold to filter out low-confidence predictions
    if confidence > 0.75:  # Tune this threshold if needed
        prediction_buffer.append(predicted_class[0])

    # Use a smoothing technique by considering the most frequent prediction in the buffer
    if len(prediction_buffer) > 10:  # Adjust buffer size as needed
        most_frequent_sign = max(set(prediction_buffer), key=prediction_buffer.count)
        predicted_sign = sign_labels[most_frequent_sign]
        prediction_buffer = []  # Reset buffer
    else:
        predicted_sign = "..."  # Show nothing if the buffer is not filled

    # Display the prediction on the frame
    cv2.putText(frame, f'Sign: {predicted_sign} | Confidence: {confidence:.2f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Real-time Sign Language Prediction', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()

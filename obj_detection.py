import cv2
import numpy as np
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Define the class labels for the COCO dataset
CLASSES = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
           "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
           "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
           "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
           "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
           "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
           "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
           "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def detect_objects(frame):
    # Load the pre-trained model for object detection
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

    # Resize the frame to have a maximum width of 400 pixels
    frame = cv2.resize(frame, (640, 480))

    # Prepare the frame for object detection by converting it to a blob
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (640, 480), 127.5)

    # Pass the blob through the network to detect objects
    net.setInput(blob)
    detections = net.forward()

    # List to store detected objects
    detected_objects = []

    # Loop over the detections and draw boxes around detected objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            object_name = CLASSES[class_id]
            box = detections[0, 0, i, 3:7] * np.array([400, 400, 400, 400])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            detected_objects.append(object_name)

    # Convert the list of detected objects into a single string
    detected_objects_str = ', '.join(detected_objects)

    return frame, detected_objects_str

def text_to_speech_from_video(video_url):
    video = cv2.VideoCapture(video_url)
    while True:
        ret, frame = video.read()
        if not ret:
            break

        processed_frame, detected_objects_str = detect_objects(frame)

        # Display the resulting frame
        cv2.imshow('Frame', processed_frame)

        # Speak the detected objects
        engine.say(detected_objects_str)
        engine.runAndWait()

        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_url = "https://192.168.1.67:8080/video"  # Example URL for video stream
    text_to_speech_from_video(video_url)

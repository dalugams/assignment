from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2
import numpy as np

app = FastAPI()

# Load the object detection model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Function to perform object detection
def detect_objects(frame):
    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the blob as input to the network
    net.setInput(blob)

    # Run forward pass to get detections
    detections = net.forward()

    # Process the detections
    detected_objects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Adjust confidence threshold as needed
            class_id = int(detections[0, 0, i, 1])
            CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                       "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                       "sofa", "train", "tvmonitor","humans","non-humans"]
            label = CLASSES[class_id]

            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            detected_objects.append({"label": label, "confidence": float(confidence), "box": [startX, startY, endX, endY]})
    return detected_objects

# Function to capture video frames
def get_frame():
    cap = cv2.VideoCapture(0)  # Access camera with ID 0
    while True:
        ret, frame = cap.read()  # Read frame from camera
        if not ret:
            break
        detected_objects = detect_objects(frame)
        for obj in detected_objects:
            label = obj['label']
            confidence = obj['confidence']
            (startX, startY, endX, endY) = obj['box']
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, "{}: {:.2f}%".format(label, confidence * 100), (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', frame)  # Encode frame to JPEG
        if not ret:
            break
        yield jpeg.tobytes()  # Return frame as bytes
    cap.release()

# WebSocket endpoint to stream video frames
@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    for frame in get_frame():
        await websocket.send_bytes(frame)

# HTML endpoint to serve the client page
@app.get("/")
async def root():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Object Detection</title>
    </head>
    <body>
        <h1>Real-time Object Detection</h1>
        <img id="video_feed" style="width:100%;" />
        <script>
            var ws = new WebSocket("ws://" + window.location.host + "/video_feed");
            var video = document.getElementById('video_feed');
            ws.onmessage = function(event) {
                video.src = "data:image/jpeg;base64," + event.data;
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

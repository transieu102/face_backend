from fastapi import FastAPI
from fastapi import WebSocket
from fastapi.responses import HTMLResponse
import cv2
from modules.detector import Detector
from modules.features_extractor import Extractor
import numpy as np
import base64
detector = Detector()
extractor = Extractor()
process_skip = 1
processed = 0
app = FastAPI()
# WebSocket route to handle frame transmission
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global processed
    await websocket.accept()
    while True:
        frame_data = await websocket.receive_text()
        # Decode base64 frame data to image
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        # Process the frame (example: convert to grayscale)
        # processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not processed % process_skip:
            face = detector.detect(frame)
            print(processed,face)
            if face[0] is not None:
                x, y, x2, y2 = [int(i) for i in face[0][0]]
                # face_img = frame[y:y2, x:x2]
                # vector = extractor.extract(face_img)
                # print(vector.shape)
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        # Encode the processed frame as base64
        _, processed_frame_data = cv2.imencode('.jpg', frame)
        processed+=1
        processed_frame_base64 = base64.b64encode(processed_frame_data).decode('utf-8')
        await websocket.send_text(processed_frame_base64)

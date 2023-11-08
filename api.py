from fastapi import FastAPI
from fastapi import WebSocket
from fastapi.responses import HTMLResponse
import cv2
from modules.detector import Detector
from modules.features_extractor import Extractor
import numpy as np
import base64
import asyncio
import dlib
from modules.landmark import LandmarkDetector
from modules.faiss import Faiss
from fastapi.middleware.cors import CORSMiddleware
from src.anti_spoof_predict import AntiSpoof
import os
from typing import Dict
detector = Detector()
extractor = Extractor()
anti = AntiSpoof()
anti2 = AntiSpoof(r'weights\anti_spoof_models\2.7_80x80_MiniFASNetV2.pth')
landmark_detector = LandmarkDetector()
if not os.path.exists('./database/data_auto.index'):
    faiss = Faiss()
else:
    faiss = Faiss(path_to_index = './database/data_auto.index')
# print(faiss.user)
process_skip = 1
processed = 0
infer_flag = 0
name_to_add = ''

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Create an async generator function to continuously capture frames
async def infer():
    global infer_flag
    cap = cv2.VideoCapture(0)  # Replace 0 with the camera index or a video file path
    real = 0
    while cap.isOpened() and infer_flag:
        ret, frame = cap.read()
        frame = cv2.resize(frame,(360,240))
        if not ret:
            break
        face = detector.detect(frame)
        label = ''
        
        # print(processed,face)
        if face[0] is not None:
            x, y, x2, y2 = [int(i) for i in face[0][0]]
            if  (x>0 and y>0 and x2 < frame.shape[1] and y2 < frame.shape[0]):
                label=''
                face_img = frame[y:y2, x:x2]
                isReal = np.argmax(anti.predict(face_img)+anti2.predict(face_img)) >= 1
                if isReal:
                    if real < 5:
                        real+=1
                else:
                    if real > -5:
                        real -= 1
                if real < 0:
                    label = 'Fake '
                else:
                    label = ''
                cv2.imwrite('test.jpg',face_img)
                if landmark_detector.well_face(face_img):
                    vector = extractor.extract(face_img)
                    label += faiss.search(vector,10)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # print(vector.shape)
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        # Convert the frame to a base64 encoded JPEG image
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer).decode('utf-8')

        yield frame_bytes

        await asyncio.sleep(0)
@app.post('/infer_flag')
async def infer_toggle(data: Dict[str,str]):
    global infer_flag
    if data['flag'] =='1':
        infer_flag = 1
    else:
        infer_flag = 0
    print(infer_flag)
@app.websocket("/infer")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async for frame in infer():
        await websocket.send_text(frame)

    await websocket.close()

async def add_mode(num_of_samples = 50):
    sample = 0
    cap = cv2.VideoCapture(0)  # Replace 0 with the camera index or a video file path
    while cap.isOpened() and sample<num_of_samples:
        ret, frame = cap.read()
        frame = cv2.resize(frame,(360,240))
        if not ret:
            break
        face = detector.detect(frame)
        if face[0] is not None:
            x, y, x2, y2 = [int(i) for i in face[0][0]]
            if  (x>0 and y>0 and x2 < frame.shape[1] and y2 < frame.shape[0]):
                face_img = frame[y:y2, x:x2]
                if landmark_detector.well_face(face_img) and len(name_to_add):
                    vector = extractor.extract(face_img)
                    faiss.add(vector, name_to_add)
                    sample+=1
                    yield str(int(sample*100/num_of_samples))
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        # Convert the frame to a base64 encoded JPEG image
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer).decode('utf-8')

        yield frame_bytes

        await asyncio.sleep(0)
    if sample == num_of_samples:
        faiss.save()
@app.websocket("/add")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async for frame in add_mode():
        await websocket.send_text(frame)
    await websocket.close()
@app.post('/name')
async def name(data: Dict[str, str]):
    # print(data)
    global name_to_add
    n = data['name']
    name_to_add = n
    if name_to_add in faiss.user:
        return False 
    else:
        return True
    # print(name_to_add)
    
@app.get('/get_user')
async def get_user():
    return faiss.user
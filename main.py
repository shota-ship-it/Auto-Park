from flask import Flask, redirect, url_for, render_template, request, Response
import cv2
import torch
import pickle
import numpy as np
# from flask import jsonify
from flask_socketio import SocketIO, emit

### WSGI Application
app=Flask(__name__)
socketio = SocketIO(app)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

with open('parkingpoints_4.pkl', 'rb') as f:
    parking_slots = pickle.load(f)
    

camera=cv2.VideoCapture('parking.mp4')

def detect_parking_slots(frame):
    
    frame = cv2.resize(frame, (1250, 700))
    results = model(frame)
    occupied_list = []
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d=(row['name'])
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        if 'car' in d:
            for i, slot in enumerate(parking_slots):
                results = cv2.pointPolygonTest(np.array(slot, np.int32), (cx, cy), False)
                if results >= 0:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                    cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
                    occupied_list.append(i)
    for i, slot in enumerate(parking_slots):
        if i not in occupied_list:
            cv2.line(frame, slot[0], slot[1], color=(0,255,0), thickness=2)
            cv2.line(frame, slot[1], slot[2], color=(0,255,0), thickness=2)
            cv2.line(frame, slot[2], slot[3], color=(0,255,0), thickness=2)
            cv2.line(frame, slot[3], slot[0], color=(0,255,0), thickness=2)
            cv2.line(frame, slot[3], slot[1], color=(0,255,0), thickness=2)
            cv2.line(frame, slot[0], slot[2], color=(0,255,0), thickness=2)
    global a
    a = len(parking_slots) - len(occupied_list)
    
    cv2.putText(frame, "Free", (56,35), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
    cv2.putText(frame, "Occupied", (56,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),3)
    cv2.putText(frame, str(a), (149, 35), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
    cv2.putText(frame, str(len(occupied_list)), (230,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),3)
    socketio.emit('update_data', {'total_slots': len(parking_slots), 'occupied_slots': len(occupied_list), 'unoccupied_slots': a})
    return frame

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame = detect_parking_slots(frame)
        
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#decorator
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')


# @app.route('/data')
# def data():
#     frame = detect_parking_slots()
#     return jsonify({'total_slots': len(parking_slots), 'occupied_slots': len(occupied_list), 'unoccupied_slots': a})

# @socketio.on('connect')
# def test_connect():
#     while True:
#         global a
#         variable1_value = str(a) 
#         global occupied_list
#         variable2_value = str(len(occupied_list))  
#         emit('variable update', {'variable1': variable1_value, 'variable2': variable2_value})


if __name__ == '__main__':
    app.run(debug=True)
 
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
import pickle
import matplotlib.pyplot as plt

# from app import 

points=[]
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)            
    
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('parking.mp4')
count=0


area = [(38, 428), (13, 511), (358, 492), (761, 424), (692, 367)]


# print(parking_slots)
parking_slots = [
# [(52, 512), (22, 601), (91, 590), (115, 510)],
# [(91, 590), (115, 510), (184, 505), (164, 598)],
# [(164, 598), (184, 505), (253, 501), (262, 591)],
# [(262, 591), (253, 501), (328, 496), (336, 579)],
# [(336, 579), (328, 496), (402, 490), (412, 579)],
# [(412, 579), (402, 490), (475, 483), (507, 567)],
# [(507, 567), (475, 483), (550, 479), (583, 558)],
# [(583, 558), (550, 479), (617, 469), (667, 545)],
# [(667, 545), (617, 469), (684, 463), (743, 534)],
# [(743, 534), (684, 463), (744, 453), (809, 522)],
# [(809, 522), (744, 453), (800, 445), (875, 508)],
# [(875, 508), (800, 445), (858, 420), (937, 488)]
]
with open('parkingpoints_4.pkl', 'rb') as f:
    parking_slots = pickle.load(f)

print(parking_slots[0])
while True:
    ret,frame=cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame=cv2.resize(frame,(1250,700))

    results=model(frame)
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
    # print(len(list))
    a = len(parking_slots) - len(occupied_list)
    
    cv2.putText(frame, "Free", (56,35), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),3)
    cv2.putText(frame, "Occupied", (56,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),3)
    cv2.putText(frame, str(a), (149, 35), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),3)
    cv2.putText(frame, str(len(occupied_list)), (230,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),3)
    frame=cv2.resize(frame,(1250,700))
    cv2.imshow("FRAME",frame)
    cv2.setMouseCallback("FRAME",POINTS)
    
    time.sleep(0.1)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()


from __future__ import division
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
import pickle 
from moviepy.editor import VideoFileClip

def show_images(images, cmap=None):
   cols = 2
   rows = (len(images)+1)//cols
   plt.figure(figsize=(15, 12))
   for i, image in enumerate(images):
       plt.subplot(rows, cols, i+1)
       cmap = 'gray' if len(image.shape)==2 else cmap
       plt.imshow(image, cmap=cmap)
       plt.xticks([])
       plt.yticks([])
   plt.tight_layout(pad=0, h_pad=0, w_pad=0)
   plt.show()
   res = plt.show()
   return res
 
def capture_initial_frame(video_path, save_path,target_width=1250, target_height=700):
   cap = cv2.VideoCapture(video_path)
 
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
 
   if not cap.isOpened():
       print("Error: Couldn't open the video.")
       return
 
   ret, frame = cap.read()
   if not ret:
       print("Error: Couldn't read the first frame.")
       cap.release()
       return
 
 
   cv2.imwrite(save_path, frame)
   cap.release()
 
def select_rgb_white_yellow(image):
   lower = np.uint8([120, 120, 120])
   upper = np.uint8([255, 255, 255])
   white_mask = cv2.inRange(image, lower, upper)
   lower = np.uint8([190, 190,   0])
   upper = np.uint8([255, 255, 255])
   yellow_mask = cv2.inRange(image, lower, upper)
   mask = cv2.bitwise_or(white_mask, yellow_mask)
   masked = cv2.bitwise_and(image, image, mask = mask)
   return masked

def convert_gray_scale(image):
   return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def detect_edges(image, low_threshold=200, high_threshold=500):
   return cv2.Canny(image, low_threshold, high_threshold)

def filter_region(image, vertices):
   mask = np.zeros_like(image)
   if len(mask.shape)==2:
       cv2.fillPoly(mask, vertices, 255)
   else:
       cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
   return cv2.bitwise_and(image, mask)

def select_region(image):
   rows, cols = image.shape[:2]
   pt_1  = [cols*0.00062, rows*0.722]
   pt_2 = [cols*0.409, rows*0.64704]
   pt_3 = [cols*0.920, rows*0.543]
   pt_4 = [cols*0.9666, rows*0.616]
   pt_5 = [cols*0.4421, rows*0.8408]
   pt_6 = [cols*0.0041, rows*0.902]
   vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
   return filter_region(image, vertices)

def hough_lines(image):
   return cv2.HoughLinesP(image,rho = 1,theta = 1*np.pi/180,threshold = 65,minLineLength = 100,maxLineGap = 25)

def trim_coords(points):
   result = []
   i = 0
   n = len(points)
   points.sort()
   while i < n:
       x1, y1 = points[i]
       if i + 1 < n:
           x2, y2 = points[i + 1]
           if abs(x1 - x2) <= 60 and abs(y1 - y2) <= 80:
               result.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
               i += 2
           else:
               result.append((x1, y1))
               i += 1
       else:
           result.append((x1, y1))
           i += 1
   return result

def draw_points(image, coordinates):
    for i,point in enumerate(coordinates,1):
       cv2.circle(image, point, radius=5,color=[255,255,255],thickness=-1)

def draw_lines(image, lines, points1, points2,color=[0, 255, 0], thickness=2, make_copy=True):
   if make_copy:
       image = np.copy(image)
   cleaned = []
   for i,line in enumerate(lines):
       for x1,y1,x2,y2 in line:
           if abs(y2-y1) <=1000 and abs(x2-x1)>=1:
            cleaned.append((x1,y1,x2,y2))
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
   draw_points(image, points1)
   draw_points(image, points2)
   return image

cwd = os.getcwd()
capture_initial_frame("parking1.mp4", "data/img.jpg")
test_images = [plt.imread(path) for path in glob.glob('data/*.jpg')]

white_yellow_images = list(map(select_rgb_white_yellow, test_images))

gray_images = list(map(convert_gray_scale, white_yellow_images))

edge_images = list(map(lambda image: detect_edges(image), gray_images))

roi_images = list(map(select_region, edge_images))

list_of_lines = list(map(hough_lines, roi_images)) 

coordinates = []
for _ in list_of_lines[0]:
   for line in _:
       coordinates.append((line[0], line[1]))
       coordinates.append((line[2], line[3]))
coordinates.sort()
 
for _ in range(100):
   coordinates = trim_coords(points=coordinates)
new_coordinates = []
 
 
for point in coordinates:
   new_coordinates.append((point[1], point[0]))
new_coordinates.sort()
 
 
for _ in range(100):
   new_coordinates = trim_coords(points=new_coordinates)
 
temp = []
 
 
for point in new_coordinates:
   temp.append((point[1], point[0]))
 
coordinates = temp
coordinates.sort()
 
new_coordinates = []
for i in range(len(coordinates)):
    point = coordinates[i]
    x = 125 * point[0]
    x /= 192
    y = 70 * point[1]
    y /= 108
    new_coordinates.append((int(x),int(y)))

coordinates = new_coordinates

for point in coordinates:
   print(point)
 
upper_points=[]
lower_points=[]
 
lower_points.append(coordinates[0])
upper_points.append(coordinates[1])
 
for i in range(2, len(coordinates)):
    x = coordinates[i][0]
    y = coordinates[i][1]

    (x1, y1) = lower_points[-1]
    (x2, y2) = upper_points[-1]

    avg = (y1 + y2) / 2

    if(y <= avg):
        lower_points.append((coordinates[i][0], coordinates[i][1]))
    else:
        upper_points.append((coordinates[i][0], coordinates[i][1]))

width = (upper_points[1][0] - upper_points[0][0])


rectangle_coordinates = []
for i in range(len(upper_points) - 1):
    if(i + 1 < len(lower_points) - 1):
        temp = []
        temp.append(upper_points[i])
        temp.append(lower_points[i])
        temp.append(lower_points[i + 1])
        temp.append(upper_points[i + 1])
        print(temp)
        rectangle_coordinates.append(temp)

with open('parkingpoints.pkl', 'wb') as f:
    pickle.dump(rectangle_coordinates, f)

line_images = []
for image, lines in zip(test_images, list_of_lines):
   line_images.append(draw_lines(image, lines, lower_points, upper_points))

 
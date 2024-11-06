# TITLE OF THE PROJECT
AUTO PARKING DETCTION.

#AIM OF THE PROJECT
Automated car parking system aims to address the challenges associated with finding parking spaces in crowded areas
such as shopping malls, hospitals, and city centers.<br>


# Introduction
These days detecting vacant slots takes lot of time and manual work is needed. This Project
'Auto parking detection' refers to the use of technology and algorithms to automatically detect and 
map parking spaces in a parking lot or area.
The goal is to streamline the parking process, improve efficiency, and enhance the overall parking experience for drivers.

#Scope of the Project
Automated car parking system aims to address the challenges associated with finding parking spaces in crowded areas,
such as shopping malls, hospitals, and city centers.
By automating the process of parking space detection, these systems can save time for drivers and reduce congestion in parking lots.<br>


#Fastenal Usecase:
Finding parking spaces in large parking areas and detecting the vehicles.

#Techstack and technology used:
1.Python<br>
2.Computer Vision<br>
3.Image Processing Techniques<br>
4.Yolov5, Yolov8 (Object detection)

#Libraries used
1.OpenCV<br>
2.PyTorch<br>
3.Ultralytics<br>
4.Numpy<br>
5.Pickle<br>
6.matplotlib


#Approaches towards the Project
-> Using Computer Vision for Parking Lot detection and Yolov5 for vehicle detection on parking Space.<br>
-> Training the model with labelled data using Deep Learning for detection of vacant spaces on the entire Parking Lot.


#Discussion and Implementation of 1st Approach.

Parking Lots are identified Using Computer Vision and Image Processing Techniques.<br>

Functions involved:<br><br><br>
1.Capture the initial frame from the video(Parking1.mp4)<br>
2.Apply White Yellow Mask on the frame to indicate white and yellow parking lines.<br>
3.Convert into Grey Scale Image to minimize the pixel counts and for easy computations<br>
4.Detect Edges using Canny Edge detection imported from open CV<br>
5.Apply Region of interest to eliminate unwanted area from the frame. ROI selects and apply on the frame so that it process only selected portion of frame.<br>
6.Lines can be identified through Hough Line Transformation and co ordinates of the lines are stored in the list_of_lines <br>
7.To eliminate the near co ordinates we trim the points using cluster so that we can identify the unique line segement points<br>
8.Finally we can draw the lines from these trimmed co ordinates so that we can collect the co ordinates of each rectangle(i.e Parking Space)<br>
9.Code can be accessed in parkingSlotDetector.py<br>
10.Coordinates are stored in parkingpoints_4.pkl file so that it can use in detecting the vehicle in that parking space.
vehicle on the parking space are identified through Yolov5 <br><br>

Functions involved:<br><br><br>
1.Setting up the environment<br>
2.Loading the YOLO model<br>
3.Capturing the Video<br>
4.Loading Parking Slot Positions<br>
5.Object Detection with YOLO<br>
6.Analyzing Occupied Parking Slots<br>
7.Visualizing Results<br>
8.Exiting the Application


Vehicles on the Parking space are identified through pixel Comparison<br><br>

Functions involved:<br><br><br>


1.Converting the initial  frame to a binary image after processing it to gray scale image , adaptive threshold and dilation<br>
2.It results in output image so that consists of minimal pixels which is easy to compute.<br>
3.For each parking slot, the ratio of the number of non zero pixels to the total number of pixels is calculated and
if it is lesser than a certain threshold, then it is vacant else occupied.<br>



#Discussion and Implementation of 2nd Approach.

1.We have used Yolov8 nano model to train with custom dataset.<br>
2.We aquired the PK Lot dataset which comprises of nearly 12K images as we are low on computing power and
 GPU, we narrowed down the dataset to 600 images for training and 120 images for validation.<br>
3.We have trained the yolo model with our custom dataset and using the trained model we predicted the results.<br>
4.There is no false predictions. Due to training and GPU limitations, some occupied spaces are not detected.<br>
5.The trained model and code can be accessed in parking.ipynb


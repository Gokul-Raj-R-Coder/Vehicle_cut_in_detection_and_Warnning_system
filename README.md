# **Vehicle_cut_in_detection_and_Warnning_system**
 This repository contains a project for real-time vehicle cut-in detection and collision warning system using YOLOv8 for object detection and SORT for object tracking. The system is designed to process dashcam video footage to detect vehicles cutting into the lane, calculate their relative speed, and estimate time-to-collision (TTC). If a potential collision is detected, the system issues a warning.

**FEATURES**

    -Real-time vehicle detection using YOLOv8

    -Object tracking with SORT

    -Calculation of relative speed and time-to-collision (TTC)

    -Collision warning system



**PREREQUISITES**

    -Python 3.x

    -OpenCV

    -NumPy

    -SciPy

    -CVZone

    -Ultralytics

    -SORT



**TEAM**

    -Govindraj R
    -Gokul Raj R
    -Karthik R
    -Sreya Krishnan
    -Sanjay Krishna K


**CHALLENGES FACED**

1. We where unable to find a good video dataset with satisfying cases
        -So we created a custom video dataset using iphone 13 pro by placing it on the dash board of a car.

2. Accuracy of the distance and relative velocity calculation
        -By refining the equations for the calculations we were able to achieve a good accuracy.For further implementations hardware can be used to find the distance and relative velocity accurately.

3. Pixel to real world distance conversion.
        -By doing proper camera calibration we can find the pixel to real world distance. 

**NOTES**

    -The region of interest has to be customised according to your video dimensions


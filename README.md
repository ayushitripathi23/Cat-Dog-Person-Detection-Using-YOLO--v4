# Cat-Dog-Person-Detection-Using-YOLO--v4

**OBJECT DETECTION USING YOLOv4 MODULE**

PROBLEM STATEMENT
To Detect Whether object in a image or video is  cat, dog or human.

INTRODUCTION
YOLO is an abbreviation for the term ‘You Only Look Once’. This is an algorithm that detects and recognizes various objects in a picture (in real-time). Object detection in YOLO is done as a regression problem and provides the class probabilities of the detected images.
YOLO algorithm employs convolutional neural networks (CNN) to detect objects in real-time. As the name suggests, the algorithm requires only a single forward propagation through a neural network to detect objects.
This means that prediction in the entire image is done in a single algorithm run. The CNN is used to predict various class probabilities and bounding boxes simultaneously.
The YOLO algorithm consists of various variants. Some of the common ones include tiny YOLO ,YOLOv3 and YOLOv4.

VARIOUS APPROACHES
1.	R-CNN Family
2.	Single Shot MultiBox Detector
3.	YOLO

Until now, we saw some very famous and well performing architectures for Object detection. All these algorithms solved some problems mentioned in the Challenges but fail on the most important one — Speed for real-time object detection
The biggest problem with the R-CNN family of networks is their speed — they were incredibly slow, obtaining only 5 FPS on a GPU.
YOLO algorithm gives a much better performance on all the parameters we discussed along with a high fps for real-time usage. YOLO algorithm is an algorithm based on regression, instead of selecting the interesting part of an Image, it predicts classes and bounding boxes for the whole image in one run of the Algorithm.
Why YOLO?
As a single-stage detector, YOLO performs classification and bounding box regression in one step, making it much faster than most convolutional neural networks. For example, YOLO object detection is more than 1000x faster than R-CNN and 100x faster than Fast R-CNN.


**PS: Downloads the weights through the drive link attached -** https://drive.google.com/file/d/1-2Gb32XrBWlOVi6HW-VE4MbeDul0yDmN/view?usp=sharing

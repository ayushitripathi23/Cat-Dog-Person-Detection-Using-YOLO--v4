import cv2
import numpy as np


# In[2]:


class ObjectDetector():
    def __init__(self):
        # Load Yolo
        self.net = cv2.dnn.readNet(r"C:\Users\AyushiTripathi\PycharmProjects\CDH\NewWT\yolov4_train_last_latest_cdh.weights",                   r"C:\Users\AyushiTripathi\PycharmProjects\CDH\NewWT\yolov4_test.cfg")
        self.classes = ["Cat","Dog","Person"]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i- 1] for i in self.net.getUnconnectedOutLayers()]
        
        
    def returnNet(self):
        return self.net
    
    def returnOutputLayers(self, img):
        height, width, channels = img.shape
        
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        #net=p.returnNet()
        #passing blob image to yolo algo work
        self.net.setInput(blob) 
        outs = self.net.forward(self.output_layers)
        
        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
      
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.returnClasses(class_ids[i]))
                #Draw rectangle around boxes.'2' is the width of box
                cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,255), 2)
                cv2.putText(img, label, (x, y + 30), font, 3, (255,0,0), 3)
                
        return img, outs
    
    def returnClasses(self,i=0):
        return self.classes[i]

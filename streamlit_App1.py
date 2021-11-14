#!/usr/bin/env python
# coding: utf-8

# In[2]:


import asyncio


# In[3]:


import logging


# In[4]:


import queue


# In[5]:


import threading


# In[6]:


import urllib.request


# In[7]:


from pathlib import Path


# In[8]:


from typing import List, NamedTuple


# In[9]:


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


# In[11]:


import av


# In[12]:


import cv2


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


import numpy as np


# In[17]:


import pydub


# In[18]:


import streamlit as st


# In[21]:


from aiortc.contrib.media import MediaPlayer


# In[22]:


import time


# In[23]:


import pandas as pd


# In[24]:


from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)


# In[26]:


# HERE = Path(__file__).parent


# In[27]:


logger = logging.getLogger(__name__)


# In[28]:


st.set_page_config(page_title="Object Detection", page_icon="🤖")


# In[29]:


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": True,
    },
)


# In[32]:


def main():
    st.title("Real time Object Detection WebApp")
    st.subheader("Using YOLOv4 ")
    
    option = st.selectbox('Please Select the Configuration file', 
                         (r"C:\Users\AyushiTripathi\PycharmProjects\CDH\NewWT\yolov4_test.cfg",))
    
    
    option = st.selectbox('Please Select the Weight file',
                          (r"C:\Users\AyushiTripathi\PycharmProjects\CDH\NewWT\yolov4_train_last_latest_cdh.weights",))
    with st.spinner('Wait for the Weights and Configuration files to load'):
        time.sleep(3)
    
    st.success('Done!')
    st.info("Please wait for 30-40 seconds for the webcam to load with the dependencies")
    
    app_object_detection()
    st.error('Please allow access to camera and microphone in order for this to work')
    st.warning('The object detection model might varies due to the server speed and internet speed')
    
    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")
    
Conf_threshold = 0.4
NMS_threshold = 0.4


# Colours
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]


# empty list
class_name = []



COCO = r"C:\Users\AyushiTripathi\PycharmProjects\CDH\NewWT\classes.names"

# for reading all the datasets from the coco.names file into the array
with open(COCO, 'rt') as f:
    class_name = f.read().rstrip('\n').split('\n')




model_config_file = r"C:\Users\AyushiTripathi\PycharmProjects\CDH\NewWT\yolov4_test.cfg"
model_weight = r"C:\Users\AyushiTripathi\PycharmProjects\CDH\NewWT\yolov4_train_last_latest_cdh.weights"


# darknet files
net = cv2.dnn.readNet(model_weight, model_config_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load Model
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


def app_object_detection():
    
    class Video(VideoProcessorBase):
        
         def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                image = frame.to_ndarray(format="bgr24")
                
                classes, scores, boxes = model.detect(image, Conf_threshold, NMS_threshold)
                
                for (classid, score, box) in zip(classes, scores, boxes):
                    color = COLORS[int(classid) % len(COLORS)]
                    label = "%s : %f" % (class_name[classid[0]], score)
                    cv2.rectangle(image, box, color, 1)
                    cv2.putText(image, label, (box[0], box[1]-10),cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
                return av.VideoFrame.from_ndarray(image, format="bgr24")
        
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=Video,
        async_processing=True,)

                

    


# In[37]:


if __name__ == "__main__":
     import os
     DEBUG = os.environ.get("DEBUG", "false").lower() not in [
        "false", "no", "0"]
    
     logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,)
        
     logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)
     st_webrtc_logger = logging.getLogger("streamlit_webrtc")
     st_webrtc_logger.setLevel(logging.DEBUG)

     fsevents_logger = logging.getLogger("fsevents")
     fsevents_logger.setLevel(logging.WARNING)

     main()
      

    


# In[ ]:





# In[ ]:





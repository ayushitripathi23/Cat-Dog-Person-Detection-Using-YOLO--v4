#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
from PIL import Image 
import cv2 
import numpy as np
import time 
import tempfile
from pathlib import Path
import CDyolo4
import streamlit_App1

# In[ ]:

def welcome():
    

    st.title('Task')
    

    
    background =Image.open(r'C:\Users\AyushiTripathi\PycharmProjects\face_detection_app\FinalObject_blurr\FinalObject_blurr\eye.jpg')
    st.image(background, width=690)
    
 

def object_detection():

    st.header("Object detection: Cat, Dog and Person")
    selected_box = st.sidebar.selectbox('Choose Media Type', ('Image', 'Video'))
    if selected_box == 'Image':
        ImageDetection()
    if selected_box == 'Video':
        st.write("Select Video:")
        video1()
def ImageDetection():
   
    img_file = st.file_uploader("Choose an Image", type=['jpg', 'jpeg', 'jfif', 'png'])
    
    if img_file is not None:
        img= np.array(Image.open(img_file))
        imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        place_h = st.columns(2)
        place_h[0].image(img)
        
        #detector=OBC.ObjectDetector(basket=["Cat", "Dog","Person"])
        #detector=OBC.ObjectDetector
        detector=CDyolo4.ObjectDetector()
        
        #imgRGB, Meta=detector.getOutputs(imgRGB)
        imgRGB ,_ = detector.returnOutputLayers(imgRGB)
        imgRGB= cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB)
        
        place_h[1].image(imgRGB)
        #df = pd.json_normalize(Meta)
        #st.write("Objects found:")
        #st.dataframe(df)

def video1():
   
    st.header("Object detection in Video files")
    cap= st.file_uploader("Choose a video...", type=["mp4"])
    Ptime=0
    detector=CDyolo4.ObjectDetector()
        
    if cap is not None:
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(cap.read())
        
        vf = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        st1frame = st.empty()
                    
        while vf.isOpened():
            ret, frame = vf.read() 
            Ctime=time.time() 
            
            if not ret:
                
                print("Can't receive frame (stream end?). Exiting ...")
                break
                          
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            orgImg=frame.copy() 
            
            #frame, _=detector.getOutputs(frame)
            frame ,_ = detector.returnOutputLayers(frame)
                        
            fps= 1.0/max(Ctime-Ptime, 0.001)
            Ptime=Ctime

            cv2.putText(frame, f"FPS:{int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
            frame = cv2.resize(frame, None, fx=0.8, fy=0.8)
            pil_img = Image.fromarray(frame)
            pil_img1 = Image.fromarray(orgImg)
            #imgRGB= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(pil_img)
            st1frame.image(pil_img1)

    
    
# In[ ]:

def main():
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Object Detection','Real Time Object Detection')
    )
    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'Object Detection':
        object_detection()
    if selected_box == 'Real Time Object Detection':
        streamlit_App1.main()

if __name__ == "__main__":
    main()




# In[ ]:





# In[ ]:





# In[ ]:





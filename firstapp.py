#create newapp
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import time
import glob
import random
import re
import os
import tempfile
import ssl
import math

st.write('Loading saved model') 

#Load Saved Model
PATH=os.getcwd()
K_I_512D=tf.keras.models.load_model(PATH +'/saved_model/5sec_AR_kineticsweightsonly_noflatten_moredense.h5')
#K_I_aug_20E= tf.keras.models.load_model(PATH+'/saved_model/augumented_5sec_AR_kinetics+ImageNetweightsonly.h5')
#K_3DL= tf.keras.models.load_model(PATH+'/saved_model/5sec_AR_kineticsweightsonly_noflatten_moredense.h5')
# Check its architecture

st.write('All loaded') 

vidup= st.file_uploader('Upload Your Video')
def vid_read(vid):
    vid_feed=vid.read()
    return vid_feed

@st.cache
def video_test(vidup):

    def crop_center_square(frame):
        (y, x) = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = x // 2 - min_dim // 2
        start_y = y // 2 - min_dim // 2
        return frame[start_y:start_y + min_dim, start_x:start_x
                     + min_dim]

    max_frames = 150
    vidhold = tempfile.NamedTemporaryFile(delete=False)
    vidhold.write(vid_read(vidup))
    
    cap = cv2.VideoCapture(vidhold.name)
    #cap = cv2.VideoCapture(video_paths)
    #frames = []
    frames = np.zeros(shape=(max_frames, 224, 224, 3))
    i =0
    try:
        while True:
            (ret, frame) = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, (224, 224))
            frame = frame[:, :, [2, 1, 0]]

            frames[i]=frame
            i+=1
            if i==150:
                break
#                 frames.append(frame)

#                 if len(frames) == max_frames:
#                     break
    finally:
        cap.release()
    #yield np.array(frames) / 255.0, Label
    return (tf.constant(frames, dtype=tf.float32)[tf.newaxis, ...])/ 255.0



def to_gif(images):
    #Code to see the videos that are loaded in Gif format. pass in the output of generator. The np.array only
    converted_images =np.clip(images.numpy()[0]* 255, 0, 255).astype(np.uint8)
    #gifnum=len([f for f in glob.glob(PATH+"**/*.gif", recursive=True)])+1
    imageio.mimsave('./animation.gif', converted_images, fps=25)
    #IPython.display.Image('animation.gif')
    return embed.embed_file('./animation.gif')
    #![SegmentLocal](191px-Seven_segment_display-animation.gif "segment")

if vidup is not None:
    upload_details={'video_type':vidup.type,'video_name':vidup.name}
    st.write(upload_details)

    
    st.write('About running Prediction')   
    labels=['on_feet', 'active', 'rest', 'escape', 'crawling']
    clip=video_test(vidup) #FUNCTION TO crop and preprocess video
    output1= K_I_512D.predict(clip)
    #print(output1)

    kin_RGB_k_Iresult=output1[0]
    #print(kin_RGB_k_Iresult)
    probabilities1 = tf.nn.softmax(kin_RGB_k_Iresult)
    #print(kin_RGB_k_Iresultresult)
    st.write("The Kinetics and Imagenet model topmost three predictions are:")
    for i in np.argsort(probabilities1)[::-1][:3]:
        st.write(f"  {labels[i]:}: {probabilities1[i] * 100:5.2f}%")
        #st.image(to_gif(clip))
    #video_byte = vidup.read()
    st.video(vid_read(vidup))#video_byte)

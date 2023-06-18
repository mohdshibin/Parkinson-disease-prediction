import streamlit as st
import cv2 
import numpy as np 
from streamlit_autorefresh import st_autorefresh

# import tensorflow as tf
# import time
# import os
# from patchify import patchify, unpatchify
# import matplotlib.pyplot as plt
# from PIL import Image
# import keras

import pickle as pkl

import parselmouth


from utils import measurePitch
from utils import quantify_image

def main():
    models()
    
    
def models():
    st.title("PARKINSON’S DISEASE PREDICTION: A MULTI-MODAL MACHINE LEARNING APPROACH")

    image = st.file_uploader('Upload the image below', type=['png', ])
    audio = st.file_uploader('Upload the audio file below', type=['wav', 'mp3'])

    predict_button = st.button('Predict')

    if predict_button:
        if image is not None and audio is not None:
            #read image
            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            #read audio
            audio_bytes = audio.read()
            # Save the audio file locally
            file_path = "uploaded_audio.wav"
            with open(file_path, "wb") as file:
                file.write(audio_bytes)
            sound = parselmouth.Sound(file_path)
            predictionParkinson(img,sound)
        else:
            st.text('Please upload the File')
    
def get_audio_model():
    model = pkl.load(open('models/audio_model.pkl', 'rb'))
    return model

def get_image_model():
    return pkl.load(open('models/drawing_model.pkl', 'rb'))



def predictionParkinson(image,sound):
    state = st.text('\n Please wait while processing.....')
    progress_bar = st.progress(0)
    audio_model = get_audio_model()
    image_model = get_image_model()    
    # logic here
    (meanF0, maxf0, minf0, stdevF0, hnr, nhr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(sound, 75, 500, "Hertz")
    audio_input = [meanF0, maxf0, minf0, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, nhr, hnr]
    audio_input = list(map(float, audio_input))

    image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image , (200,200))
    image =cv2.threshold(image, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(image)
    print("Features is : ")
    print(features)
    inp_list = []
    inp_list.append(features)
    image_input = np.array(inp_list)

    # print("============================")
    # print(audio_input)
    # print(type(audio_input))
    # print("============================")

    audio_result = audio_model.predict([audio_input, ])
    # st.write("Audio Model Result")
    # st.write(audio_result)


    
    print("============================")
    print(image_input)
    print(type(image_input))
    print("============================")
    image_result = image_model.predict(image_input)
    # logic end
    # st.write("Image Model Result")
    if(image_result==0):
        st.write("Result : You don't have Parkinson!")
    else:
        st.write("Result : You have Parkinson!")

    

    progress_bar.progress(100)
    state.text('\n Completed!')
    # state.text('\n ' + str(audio_result))
    progress_bar.empty()
    
if __name__ == "__main__":
    main()   

#d:/Academics/ImageDenoising/envDenoising/Scripts/activate.bat
#streamlit run app.py
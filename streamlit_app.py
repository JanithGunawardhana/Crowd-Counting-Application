from random import randint
import tempfile
import time
import cv2
import urllib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
from streamlit.server.server import Server
from crowd_count_prediction import get_image_prediction
from session_state import SessionState
from fpdf import FPDF
import base64

def trigger_rerun():
    session_infos = Server.get_current()._session_info_by_id.values() 
    for session_info in session_infos:
        this_session = session_info.session
    this_session.request_rerun(None)

def ProcessFrames(vf, stop): 

    try:
        num_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vf.get(cv2.CAP_PROP_FPS)) 
        print('Total number of frames to be processed:', num_frames,
        '\nFrame rate (frames per second):', fps)
    except:
        print('We cannot determine number of frames and FPS!')


    frame_counter = 0
    _stop = stop.button("Stop Processing")
    bar = st.progress(frame_counter)
    col1, col2 = st.columns(2)
    with col1:
        fps_meas_txt = st.empty()
        real_frame = st.empty()
    with col2:
        crowd_count_txt = st.empty()
        density_map_frame = st.empty()
    start = time.time()

    while vf.isOpened():
        # if frame is read correctly ret is True
        ret, frame = vf.read()
        if _stop:
            break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        
        color_converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with st.spinner('Processing...'):    
            crowd_count, density_map = get_image_prediction(color_converted_frame)
        density_map_show = None
        density_map_show = cv2.normalize(density_map, density_map_show, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        density_map_show = cv2.applyColorMap(density_map_show, cv2.COLORMAP_JET)
        density_map_final = cv2.cvtColor(density_map_show, cv2.COLOR_RGB2BGR)

        end = time.time()

        frame_counter += 1
        fps_measurement = frame_counter/(end - start)
        fps_meas_txt.markdown(f'**Frames per second:** {fps_measurement:.2f}')
        crowd_count_txt.markdown(f'**Crowd Count:** {crowd_count}')
        bar.progress(frame_counter/num_frames)

        real_frame.image(frame)
        density_map_frame.image(density_map_final)

def main():
    st.set_page_config(
        page_title="Two Counters: Crowd Density Monitoring",
        page_icon="movie_camera",
        layout="wide",
    )
    st.sidebar.image("./Images/logo_2.png", use_column_width=True)
    st.sidebar.title("Settings")
    app_mode = st.sidebar.selectbox("Choose App Mode",
        ["Project Details", "Proposed Solution", "Image Based Crowd Counting", "Video Based Crowd Counting", "Live Video Crowd Counting"])
    expander_faq = st.sidebar.expander("More about our project")
    expander_faq.write("Hi there! If you have any questions about our project, or simply want to check out the source code, please visit our [github repository](https://github.com/JanithGunawardhana/Crowd-Counting-Application)")

    if app_mode == "Project Details":
        st.title("Crowd Density Monitoring")

        st.subheader('Introduction')
        st.write('* **Crowd Density Monitoring** means keeping track of people count in an identified scene and taking necessary actions to manage the people count.')
        st.write("* Automated crowd density monitoring is a modern-day emerging research area.")
        st.write("* Since neural networks can extract meaningful features from a given image, CNN-based methods are currently very popular and proving to be highly accurate and flexible in the task of automated crowd counting.")

        st.subheader('Motivation')
        st.write('* Automated Crowd Density Monitoring has drawn increased attention in computer vision due to the prevailing contagious outbreaks like Covid-19.')
        st.write('* Cross Domain Crowd Counting (CDCC) assures an application to be used in any domain which helps for effective automated crowd density monitoring.')
        st.write('* Serving in many real time applications such as traffic controlling systems, security management systems, and disaster management systems.')

        st.subheader('Problem Statement')
        st.write('* Many related previous studies carried in CDCC focused on a variety of ways to propose solutions to reduce domain gap between model training synthetic data and model testing real data in order to increase the accuracy in crowd counting.')
        st.write('* Due to this, crowd counting model architecture did not receive sufficient attention and many used simple Spatial Fully Convolutional Neural Network (SFCN) architecture.')
        st.write('* Therefore, the problem that we came up with: **_" How to improve the accuracy of CDCC while addressing domain shift by using an improved model architecture than usual SFCN architecture."_**')
    elif app_mode == "Proposed Solution":
        st.title("Crowd Density Monitoring")
        st.write("Multi Layered Deep Neural Network for Feature Extraction in Cross Domain Crowd Counting")

        st.subheader('Proposed CNN based Model Architecture')
        st.write('Proposed multi layered model architecture consists of two components as,')
        st.write("1. **_Frontend network_** – A VGG 16 classifier and five feature generation levels as P5, P4, P3, P2, and P1 for predicting density maps in high quality with multi-scale feature representation.")
        st.write("2. **_Backend network_** - Five multi-module branches and a dilation convolution network for scale variation feature extractions to increase the accuracy in crowd count estimations while maintaining the resolution and high quality of generated density maps.")
        image = Image.open('./Images/model.jpeg')
        st.image(image, caption='~~ Proposed CNN based Multi Layered Model Architecture ~~', use_column_width=True)
        
    elif app_mode == "Image Based Crowd Counting":
        st.title("Image Based Crowd Counting")
        st.write("Select any image and get corresponding density map with crowd count")
        uploaded_file = st.file_uploader("Choose an image...")
        col1, col2 = st.columns(2)
        if uploaded_file is not None:
            image = Image.open(uploaded_file)	
            img_array = np.array(image)
            color_converted_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            with col1:
                st.image(uploaded_file, caption='Input Image', use_column_width=True)
            with col2:
                with st.spinner('Processing...'):    
                    crowd_count, density_map = get_image_prediction(color_converted_image)
                density_map_show = None
                density_map_show = cv2.normalize(density_map, density_map_show, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                density_map_show = cv2.applyColorMap(density_map_show, cv2.COLORMAP_JET)
                st.image(density_map_show, caption='Density Map', use_column_width=True, channels='BGR')
            st.success('Crowd Count: ' + str(crowd_count))
            # pdf = FPDF()
            # pdf.add_page()
            # pdf.set_font('Arial', 'B', 16)
            # pdf.cell(40, 10, 'Crowd Count: '+ str(crowd_count))
            # pdf.output('tuto1.pdf')


    elif app_mode == "Video Based Crowd Counting":
        st.title("Video Based Crowd Counting")
        st.write("Select any video and get corresponding density map feed with crowd counts")
        state = SessionState.get(upload_key = None, enabled = True, start = False, conf = 70, nms = 50, run = False)
        upload = st.empty()
        start_button = st.empty()
        stop_button = st.empty()
        with upload:
            f = st.file_uploader('Upload Video file (mpeg/mp4 format)', key = state.upload_key)
        if f is not None:
            tfile  = tempfile.NamedTemporaryFile(delete = True)
            tfile.write(f.read())

            upload.empty()
            vf = cv2.VideoCapture(tfile.name)

            if not state.run:
                start = start_button.button("Start Processing")
                state.start = start
            
            if state.start:
                start_button.empty()
                #state.upload_key = str(randint(1000, int(1e6)))
                state.enabled = False
                if state.run:
                    tfile.close()
                    f.close()
                    state.upload_key = str(randint(1000, int(1e6)))
                    state.enabled = True
                    state.run = False
                    ProcessFrames(vf, stop_button)
                else:
                    state.run = True
                    trigger_rerun()
    
    elif app_mode == "Live Video Crowd Counting":
        st.title("Live Video Crowd Counting")
        st.write("Enter the public url of the camera to get the video feed and get corresponding density map feed with crowd counts.")

        if 'video_session_input_state' not in st.session_state:
            st.session_state['video_session_input_state'] = True
            st.session_state['url'] = ''

        if st.session_state['video_session_input_state']:
            form = st.form(key='url-form')
            url = form.text_input('Enter the url of the camera')
            submit = form.form_submit_button('Submit')
            if submit:
                if url == '':
                    st.error("Invalid url")
                else:
                    st.session_state['video_session_input_state'] = False
                    st.session_state['url'] = url

        if not st.session_state['video_session_input_state']:
            vcap = cv2.VideoCapture(st.session_state['url'])
            frame_counter = 0
            stop_process = st.button('Stop Processing')
            col1, col2 = st.columns(2)
            with col1:
                fps_meas_txt = st.empty()
                real_frame = st.empty()
            with col2:
                crowd_count_txt = st.empty()
                density_map_frame = st.empty()
            start = time.time()

            while(True):
                ret, frame = vcap.read()
                if stop_process:
                    st.session_state['video_session_input_state'] = True
                    st.session_state['url'] = ''
                    break
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                if frame is not None:
                    color_converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with st.spinner('Processing...'):    
                        crowd_count, density_map = get_image_prediction(color_converted_frame)
                    density_map_show = None
                    density_map_show = cv2.normalize(density_map, density_map_show, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    density_map_show = cv2.applyColorMap(density_map_show, cv2.COLORMAP_JET)
                    density_map_final = cv2.cvtColor(density_map_show, cv2.COLOR_RGB2BGR)
                    end = time.time()

                    frame_counter += 1
                    fps_measurement = frame_counter/(end - start)
                    fps_meas_txt.markdown(f'**Frames per second:** {fps_measurement:.2f}')
                    crowd_count_txt.markdown(f'**Crowd Count:** {crowd_count}')

                    real_frame.image(frame)
                    density_map_frame.image(density_map_final)
                else:
                    break


if __name__ == "__main__":
    main()
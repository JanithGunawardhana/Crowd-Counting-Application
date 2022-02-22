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

#@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/JanithGunawardhana/Crowd-Counting-Application/main/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

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
    _stop = stop.button("stop")
    crowd_count_txt = st.empty()
    fps_meas_txt = st.empty()
    bar = st.progress(frame_counter)
    real_frame = st.empty()
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
        #st.image(frame, caption='Input Image', use_column_width=True)
        #st.image(density_map_show, caption='Density Map', use_column_width=True, channels='BGR')
        #st.success('Crowd Count: ' + str(crowd_count))

        # labels, current_boxes, confidences = obj_detector.ForwardPassOutput(frame)
        # frame = tc.drawBoxes(frame, labels, current_boxes, confidences) 
        # new_car_count = tracker.TrackCars(current_boxes)
        # new_car_count_txt.markdown(f'**Total car count:** {new_car_count}')

        end = time.time()

        frame_counter += 1
        fps_measurement = frame_counter/(end - start)
        fps_meas_txt.markdown(f'**Frames per second:** {fps_measurement:.2f}')
        crowd_count_txt.markdown(f'**Crowd Count:** {crowd_count}')
        bar.progress(frame_counter/num_frames)

        # col1, col2 = st.columns(2)
        # col1.header("Original")
        # col1.image(frame, use_column_width=True)

        # col2.header("Grayscale")
        # col2.image(density_map_final, use_column_width=True)

        real_frame.image(frame, width = 500)
        density_map_frame.image(density_map_final, width = 500)

def main():
    st.sidebar.title("Settings")

    app_mode = st.sidebar.selectbox("Choose App Mode",
        ["Project Details", "Image Based Crowd Counting", "Video Based Crowd Counting", "Live Video Crowd Counting"])

    if app_mode == "Project Details":
        st.title("Crowd Density Monitoring")
        st.write("Multi Layered Deep Neural Network for Feature Extraction in Cross Domain Crowd Counting")
        readme_text = st.markdown(get_file_content_as_string("README.md"))

    elif app_mode == "Image Based Crowd Counting":
        st.title("Image Based Crowd Counting")
        st.write("Select any image and get corresponding density map with crowd count")
        uploaded_file = st.file_uploader("Choose an image...")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)	
            img_array = np.array(image)
            color_converted_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            st.image(uploaded_file, caption='Input Image', use_column_width=True)
            with st.spinner('Processing...'):    
                crowd_count, density_map = get_image_prediction(color_converted_image)
            density_map_show = None
            density_map_show = cv2.normalize(density_map, density_map_show, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            density_map_show = cv2.applyColorMap(density_map_show, cv2.COLORMAP_JET)
            st.image(density_map_show, caption='Density Map', use_column_width=True, channels='BGR')
            st.success('Crowd Count: ' + str(crowd_count))


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
                start = start_button.button("start")
                state.start = start
            
            if state.start:
                start_button.empty()
                #state.upload_key = str(randint(1000, int(1e6)))
                state.enabled = False
                print("Test1")
                if state.run:
                    print("Test2")
                    tfile.close()
                    f.close()
                    state.upload_key = str(randint(1000, int(1e6)))
                    state.enabled = True
                    state.run = False
                    ProcessFrames(vf, stop_button)
                else:
                    print("Test3")
                    state.run = True
                    trigger_rerun()
    # elif app_mode == "Live Video Crowd Counting":

if __name__ == "__main__":
    main()
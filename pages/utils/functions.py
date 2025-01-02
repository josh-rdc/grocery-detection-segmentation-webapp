import io
import time

import cv2
import av

import PIL

import streamlit as st


def handle_image_inference(st, model, conf, iou, selected_ind, button_text):
    source_img = st.sidebar.file_uploader("Choose an image file...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    col1, col2 = st.columns(2)

    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        with col1:
            st.image(source_img, caption="Uploaded Image", use_container_width=True)

        with col2:
            if st.sidebar.button(button_text):
                res = model(uploaded_image, conf=conf, iou=iou, classes=selected_ind)
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image', use_container_width=True)


# def handle_video_inference(st, model, conf, iou, selected_ind, button_text, fps_display):
#     source_vid = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])
#     if source_vid and st.sidebar.button(button_text):
#         try:
#             video_bytes = source_vid.read()
#             video_container = av.open(io.BytesIO(video_bytes))
            
#             stop_button = st.button("Stop")
#             col1, col2, col3 = st.columns([1, 3, 1])  
#             st_frame = col2.empty()
            
#             for frame in video_container.decode(video=0):
#                 frame = frame.to_ndarray(format="bgr24")
#                 prev_time = time.time()
#                 results = model(frame, conf=conf, iou=iou, classes=selected_ind)
#                 annotated_frame = results[0].plot()
#                 curr_time = time.time()
#                 fps = 1 / (curr_time - prev_time)
#                 st_frame.image(annotated_frame, channels="BGR")
#                 fps_display.metric("FPS", f"{fps:.2f}")
#                 if stop_button:
#                     break
#         except Exception as e:
#             st.sidebar.error("Error loading video: " + str(e))
import cv2
import av
import io
import time
import streamlit as st
import numpy as np
from tempfile import TemporaryDirectory

def handle_video_inference(st, model, conf, iou, selected_ind, button_text, fps_display):
    source_vid = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])
    
    if source_vid and st.sidebar.button(button_text):
        try:
            # Read video file
            video_bytes = source_vid.read()
            video_container = av.open(io.BytesIO(video_bytes))
            
            # Temporary directory to save the output
            with TemporaryDirectory() as tmp_dir:
                output_path = f"{tmp_dir}/output_video.mp4"
                
                # Get video properties
                video_stream = video_container.streams.video[0]
                width, height = video_stream.width, video_stream.height
                fps = float(video_stream.average_rate)
                
                # Video writer setup
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                stop_button = st.button("Stop")
                col1, col2, col3 = st.columns([1, 3, 1])
                st_frame = col2.empty()
                
                for frame in video_container.decode(video=0):
                    frame = frame.to_ndarray(format="bgr24")
                    prev_time = time.time()
                    
                    # Perform inference
                    results = model(frame, conf=conf, iou=iou, classes=selected_ind)
                    annotated_frame = results[0].plot()
                    
                    # Save annotated frame to video
                    out.write(annotated_frame)
                    
                    # Stream the frame
                    curr_time = time.time()
                    fps_value = 1 / (curr_time - prev_time)
                    st_frame.image(annotated_frame, channels="BGR")
                    fps_display.metric("FPS", f"{fps_value:.2f}")
                    
                    if stop_button:
                        break
                
                # Release video writer
                out.release()
                
                # Provide the output video as a downloadable file
                with open(output_path, "rb") as file:
                    st.sidebar.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
                
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))




def handle_webcam_inference(st, model, conf, iou, selected_ind, fps_display):
    st.sidebar.info("Press 'Start' to start the webcam feed.")

    if st.sidebar.button("Start"):
        
        videocapture = cv2.VideoCapture(0)

        if not videocapture.isOpened():
            st.error("Could not open webcam.")
        stop_button = st.button("Stop")

        col1, col2, col3 = st.columns([1, 3, 1])  
        ann_frame = col2.empty()

        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                st.warning("Failed to read frame from webcam.")
                break

            prev_time = time.time()
  
            results = model(frame, conf=conf, iou=iou, classes=selected_ind)
            annotated_frame = results[0].plot()
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)

            ann_frame.image(annotated_frame, channels="BGR")
            fps_display.metric("FPS", f"{fps:.2f}")
            if stop_button:
                break

        videocapture.release()
        cv2.destroyAllWindows()
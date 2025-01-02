import io
import time

import cv2
import av

import PIL

import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, RTCConfiguration

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

# Custom Video Transformer for processing video frames
class VideoProcessor(VideoTransformerBase):
    def __init__(self, model, conf, iou, selected_ind):
        self.model = model
        self.conf = conf
        self.iou = iou
        self.selected_ind = selected_ind

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to BGR

        # flip image for better visualization
        img = cv2.flip(img, 1)

        prev_time = time.time()

        # Perform inference using the model
        results = self.model(img, conf=self.conf, iou=self.iou, classes=self.selected_ind)
        annotated_frame = results[0].plot()

        curr_time = time.time()
        fps = 0.3 / (curr_time - prev_time)

        # Display FPS on the frame with a styled background
        fps_text = f"FPS: {fps:.2f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x, text_y = 10, 30  # Top-left corner of the text
        box_coords = ((text_x - 5, text_y - 20), (text_x + text_size[0] + 55, text_y + 5))

        # Draw a filled rectangle with slight opacity
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], (0, 0, 0), -1)  # Black background
        alpha = 0.5  # Opacity factor
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

        # Overlay the FPS text
        cv2.putText(
            annotated_frame,
            fps_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,  # Font scale (smaller size)
            (255, 255, 255),  # White text color
            1,  # Thickness
            cv2.LINE_AA,
        )
        return annotated_frame
    
def handle_webcam_inference(st, model, conf, iou, selected_ind, fps_display):
    st.sidebar.info("Press 'Start' to start the webcam feed.")

    # Configure ICE servers
    rtc_config = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", 
                      "stun:stun1.l.google.com:19302"]}
        ]
    })

    # Start WebRTC streamer
    webrtc_streamer(
        key="example",
        video_transformer_factory=lambda: VideoProcessor(model, conf, iou, selected_ind),
        rtc_configuration=rtc_config,
    )

# def handle_webcam_inference(st, model, conf, iou, selected_ind, fps_display):
#     st.sidebar.info("Press 'Start' to start the webcam feed.")

#     if st.sidebar.button("Start"):
        
#         videocapture = cv2.VideoCapture(0)

#         if not videocapture.isOpened():
#             st.error("Could not open webcam.")
#         stop_button = st.button("Stop")

#         col1, col2, col3 = st.columns([1, 3, 1])  
#         ann_frame = col2.empty()

#         while videocapture.isOpened():
#             success, frame = videocapture.read()
#             if not success:
#                 st.warning("Failed to read frame from webcam.")
#                 break

#             prev_time = time.time()
  
#             results = model(frame, conf=conf, iou=iou, classes=selected_ind)
#             annotated_frame = results[0].plot()
            
#             curr_time = time.time()
#             fps = 1 / (curr_time - prev_time)

#             ann_frame.image(annotated_frame, channels="BGR")
#             fps_display.metric("FPS", f"{fps:.2f}")
#             if stop_button:
#                 break

#         videocapture.release()
#         cv2.destroyAllWindows()
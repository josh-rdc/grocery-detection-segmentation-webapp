import io
import time
import os

import cv2
import av

import numpy as np
import time
import tempfile
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image

from urllib.parse import urlparse
from validators import url as is_valid_url

import streamlit as st
from pages.utils.remote_functions import remote_inference
from pages.utils.remote_functions_sio import remote_inference_sio

import asyncio
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

def handle_image_inference_remote(st, model, conf, iou, button_text):
    """
    Streamlit function to handle remote image inference.
    """

    source_img = st.sidebar.file_uploader("Choose an image file...", type=("jpg", "jpeg", "png", "bmp", "webp"))
    col1, col2 = st.columns(2)

    # st.markdown("Perform segmentation and detection based on selected configuration.")
    
    if source_img:
        uploaded_image = Image.open(source_img)
        uploaded_image = np.array(uploaded_image)
        uploaded_image_cv = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR)

        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        with col2:
            if st.sidebar.button(button_text):

                prediction_image = remote_inference(uploaded_image_cv, model, conf, iou)
                # prediction_image = remote_inference_sio(uploaded_image_cv, model, conf, iou)

                # Convert BGR (OpenCV default) to RGB
                if isinstance(prediction_image, np.ndarray):
                    prediction_image = prediction_image[:, :, ::-1]  # BGR to RGB

                st.image(prediction_image, caption="Predictions", use_container_width=True)

def handle_video_inference_remote(st, model, conf, iou, button_text, fps_display):
    source_vid = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])
    if source_vid and st.sidebar.button(button_text):
        try:
            video_bytes = source_vid.read()
            video_container = av.open(io.BytesIO(video_bytes))
            
            stop_button = st.button("Stop")
            col1, col2, col3 = st.columns([1, 3, 1])  
            st_frame = col2.empty()
            
            for frame in video_container.decode(video=0):
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.to_ndarray(format="bgr24")    
                # frame = cv2.resize(frame, (640, 480))

                prev_time = time.time()
                
                prediction_image = remote_inference(frame, model, conf, iou)
                # prediction_image = remote_inference_sio(frame, model, conf, iou)

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)

                st_frame.image(prediction_image, channels="BGR")

                fps_display.metric("FPS", f"{fps:.2f}")
                if stop_button:
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def handle_webcam_inference_remote(st, model, conf, iou, fps_display):
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
            
            # reduce frame size to speed up inference
            # frame = cv2.resize(frame, (640, 480))
            
            prediction_image = remote_inference(frame, model, conf, iou)
            # prediction_image = asyncio.run(remote_inference_sio(frame, model, conf, iou))


            curr_time = time.time()
            fps = 2 / (curr_time - prev_time)

            # Display the annotated frame and FPS
            ann_frame.image(prediction_image, channels="BGR")
            fps_display.metric("FPS", f"{fps:.2f}")

            if stop_button:
                break

        videocapture.release()
        cv2.destroyAllWindows()


#################
# from threading import Thread, Lock
# import queue

# class WebcamStream :
#     def __init__(self, stream_id=0):
#         self.stream_id = stream_id   # default is 0 for primary camera 
        
#         # opening video capture stream 
#         self.vcap      = cv2.VideoCapture(self.stream_id)
#         if self.vcap.isOpened() is False :
#             print("[Exiting]: Error accessing webcam stream.")
#             exit(0)
#         fps_input_stream = int(self.vcap.get(5))
#         print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))
            
#         # reading a single frame from vcap stream for initializing 
#         self.grabbed , self.frame = self.vcap.read()
#         if self.grabbed is False :
#             print('[Exiting] No more frames to read')
#             exit(0)             # self.stopped is set to False when frames are being read from self.vcap stream 
#         self.stopped = True     # reference to the thread for reading next available frame from input stream 
#         self.t = Thread(target=self.update, args=())
#         self.t.daemon = True    # daemon threads keep running in the background while the program is executing 
        
#     # method for starting the thread for grabbing next available frame in input stream 
#     def start(self):
#         self.stopped = False
#         self.t.start()          # method for reading next frame 
#     def update(self):
#         while True :
#             if self.stopped is True :
#                 break
#             self.grabbed , self.frame = self.vcap.read()
#             if self.grabbed is False :
#                 print('[Exiting] No more frames to read')
#                 self.stopped = True
#                 break 
#         self.vcap.release()     # method for returning latest read frame 
#     def read(self):
#         return self.frame       # method called to stop reading frames 
#     def stop(self):
#         self.stopped = True     # initializing and starting multi-threaded webcam capture input stream 
#         self.t.join()           # Wait for the thread to finish

# import numpy as np
# import queue
# import time
# from threading import Thread, Lock

# class FrameWrapper:
#     """Custom wrapper to store a frame and metadata."""
#     def __init__(self, frame, processed=False):
#         self.frame = frame
#         self.processed = processed

# def handle_webcam_inference_remote(st, model, conf, iou, fps_display):
#     st.sidebar.info("Press 'Start' to start the webcam feed.")

#     if st.sidebar.button("Start"):
#         # Initialize Webcam Stream
#         webcam_stream = WebcamStream(stream_id=0)
#         webcam_stream.start()

#         # Queue for frame communication between threads
#         result_queue = queue.Queue(maxsize=1)  # Single frame queue

#         stop_flag = False  # Flag to signal stopping threads
#         lock = Lock()  # To synchronize access to shared variables

#         def inference_thread(frame):
#             """Inference thread function to process the frame."""
#             annotated_frame = remote_inference(frame.frame, model, conf, iou)
#             # Return annotated frame in a wrapped object
#             result_queue.put(FrameWrapper(annotated_frame, processed=True))

#         col1, col2, col3 = st.columns([1, 3, 1])
#         ann_frame = col2.empty()
#         stop_button = st.button("Stop")

#         frame_count = 0
#         fps_start_time = time.time()

#         while not stop_flag:
#             frame = webcam_stream.read()

#             if frame is not None:
#                 # Wrap the frame in FrameWrapper
#                 frame_wrapper = FrameWrapper(frame, processed=False)

#                 # Only process every 2nd frame (can be adjusted)
#                 if frame_count % 5 == 0:
#                     # Start inference for the current frame in a new thread
#                     Thread(target=inference_thread, args=(frame_wrapper,)).start()
#                 else:
#                     # If inference is skipped, just put the unprocessed frame in the queue
#                     try:
#                         result_queue.put_nowait(frame_wrapper)
#                     except queue.Full:
#                         pass  # Skip if queue is full

#                 # Try to get the annotated frame from the result queue
#                 if not result_queue.empty():
#                     annotated_frame_wrapper = result_queue.get()

#                     # Check if the frame was processed (annotated)
#                     if annotated_frame_wrapper.processed:
#                         ann_frame.image(annotated_frame_wrapper.frame, channels="BGR")

#                         # After displaying the annotated frame, flush the queue
#                         while not result_queue.empty():
#                             result_queue.get()

#                     # Display the skipped frame if it's unprocessed
#                     elif not annotated_frame_wrapper.processed:
#                         ann_frame.image(annotated_frame_wrapper.frame, channels="BGR")

#                 # Calculate and display FPS
#                 frame_count += 1
#                 end_time = time.time()
#                 elapsed_time = end_time - fps_start_time
#                 fps = 1 / (elapsed_time + 1e-6)
#                 fps_display.metric("FPS", f"{fps:.2f}")

#             if stop_button:
#                 with lock:
#                     stop_flag = True

#         # Cleanup
#         webcam_stream.stop()
#         webcam_stream.vcap.release()

#         cv2.destroyAllWindows()




############
# import cv2
# import time
# import threading
# import streamlit as st

# # Global flag to control thread shutdown
# stop_flag = threading.Event()

# def handle_webcam_inference_remote(st, model, conf, iou, fps_display):
#     st.sidebar.info("Press 'Start' to start the webcam feed.")

#     if st.sidebar.button("Start"):
#         videocapture = cv2.VideoCapture(1)
#         if not videocapture.isOpened():
#             st.error("Could not open webcam.")
#             return
#         stop_button = st.button("Stop")

#         col1, col2, col3 = st.columns([1, 3, 1])  
#         ann_frame = col2.empty()

#         # Shared state between threads
#         frame = None
#         prediction_image = None
#         frame_lock = threading.Lock()
        
#         # Worker function to process webcam feed and inference
#         def webcam_worker():
#             nonlocal frame
#             frame_counter = 0
#             while not stop_flag.is_set() and videocapture.isOpened():
#                 success, new_frame = videocapture.read()
#                 if not success:
#                     st.warning("Failed to read frame from webcam.")
#                     break
                
#                 with frame_lock:
#                     frame = new_frame
                
#                 frame_counter += 1
#                 if frame_counter % 3 == 0:  # Skip frames to reduce load
#                     time.sleep(0.1)  # Adjust to match desired frame rate

#             videocapture.release()

#         # Worker function to run inference
#         def inference_worker():
#             nonlocal prediction_image
#             while not stop_flag.is_set():
#                 if frame is not None:
#                     with frame_lock:
#                         # Resize frame to speed up inference
#                         frame_resized = cv2.resize(frame, (640, 480))  # Resize to lower resolution
#                         prediction_image = remote_inference(frame_resized, model, conf, iou)
                
#                 time.sleep(0.1)  # Sleep to avoid high CPU usage while waiting for the frame

#         # Create threads for webcam feed and inference
#         webcam_thread = threading.Thread(target=webcam_worker, daemon=True)
#         inference_thread = threading.Thread(target=inference_worker, daemon=True)

#         # Start threads
#         webcam_thread.start()
#         inference_thread.start()

#         prev_time = time.time()

#         # Main loop to display frames
#         while videocapture.isOpened():
#             if prediction_image is not None:
#                 curr_time = time.time()
#                 fps = 1 / (curr_time - prev_time)  # FPS calculation
#                 prev_time = curr_time

#                 # Display the annotated frame and FPS
#                 ann_frame.image(prediction_image, channels="BGR", use_container_width=True)
#                 fps_display.metric("FPS", f"{fps:.2f}")

#             # Check if stop button is pressed
#             if stop_button:
#                 stop_flag.set()  # Signal all threads to stop
#                 break

#         # Wait for threads to finish before releasing the capture
#         webcam_thread.join()
#         inference_thread.join()

#         cv2.destroyAllWindows()











        
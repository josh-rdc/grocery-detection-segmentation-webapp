import io
import time
import os
import cv2
import av
import pandas as pd

from PIL import Image
import torch

from pathlib import Path
import streamlit as st
from ultralytics import YOLO

from pages.utils.functions import handle_image_inference
from pages.utils.functions import handle_video_inference
from pages.utils.functions import handle_webcam_inference

# from pages.utils.functions_client import handle_image_inference_remote
# from pages.utils.functions_client import handle_video_inference_remote
# from pages.utils.functions_client import handle_webcam_inference_remote

import settings as settings  

def detection_and_segmentation_app():
    """Performs real-time object detection on image and video input 
    using Streamlit web application."""

    # Hide main menu style
    menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""

    # Subtitle of streamlit application
    sub_title_cfg = """<div><h3 style="color:#FFFFFF; text-align:center; 
                    font-family: 'Archivo', sans-serif; margin-top:-25px; margin-bottom:50px;">
                    🛒 Common Grocery Items Detection and Segmentation 🛒 </h3>
                    </div>"""

    # Set html page configuration
    st.set_page_config(page_title="Perform Segmentation and Detection", page_icon="🔎", layout='wide', )

    # st.title("Inference")

    # Append the custom HTML
    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    
    # st.markdown(sub_title_cfg, unsafe_allow_html=True)

    # # Add ultralytics logo in sidebar
    # with st.sidebar:
    #     logo = "logo_upd_and_ai.png"
    #     st.image(logo, width=300)    

    st.sidebar.title("User Configuration")
    
    # Add dropdown menu for detection or segmentation model selection
    selected_task = st.sidebar.selectbox("Task", ["App Guide", "Detection", "Segmentation"])
    
    # Define the model options based on the task
    if selected_task == "App Guide":
        app_guide = st.container(border=True)
        with app_guide:
            col1, col2 = st.columns([1, 1.7])
            with col1:
                app_guide_df = pd.read_csv(settings.app_guide_path)
                st.dataframe(app_guide_df)

            with col2:
                st.markdown("""
                            <style>
                                .custom-text {
                                    font-size: 15px;
                                    line-height: 1.5;
                                }
                                .custom-text ul {
                                    margin-left: 20px;
                                }
                                .custom-text li {
                                    font-size: 14px;
                                }
                                .highlight {
                                    background-color: #FFE5E2;
                                    color: black;
                                    padding: 2px 5px;
                                    border-radius: 5px;
                                }
                            </style>
                            <div class="custom-text">
                                <span class="highlight">Confidence threshold</span> determines the minimum confidence score a detection must have to be considered valid.
                                    <li><b>Higher confidence threshold reduces false positives (FP)</b> by filtering out low-confidence detections.</li>
                                    <li><b>Lower confidence threshold captures more subtle true positives (TP)</b> objects, but increases the risk of false positives (FP).</li>
                                <span class="highlight">Intersection Over Union (IoU) threshold</span> is set for Non-Maximum Suppression (NMS) which helps in reducing duplicate detections.
                                    <li><b>Higher IoU</b> requires bounding boxes to overlap more significantly to be considered duplicates, <b>might result in retaining more false positives (FP).</b></li>
                                    <li><b>Lower IoU</b> allows bounding boxes with less overlap to be suppressed, might result in <b>retaining more true positives (TP).</b></li>
                            </div>
                            """, unsafe_allow_html=True)
        
        app_detail = st.container(border=True)
        with app_detail:
            st.subheader("Inference Implementation")
            st.image(settings.fastapi_path, width=1000)

            st.markdown("""
                - **HTTP** follow a request-response model, where the client sends a request to the server and waits for a response.
                - :red-background[**WebSocket**] is a protocol that enables two-way communication between a client and a server over a single TCP connection. 
                    * Implemented using the [FastAPI framework](https://unfoldai.com/fastapi-and-websockets/).
                        
                        """)
            
        sample_test = st.container(border=True)

        with sample_test:
            # List all images in the folder
            images = [os.path.join(settings.image_folder, img) for img in os.listdir(settings.image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
            image_expander = st.expander("Sample Tests")
            with image_expander:
                # Display the images in a grid format
                cols = st.columns(len(images))  # Create a column for each image
                for col, img_path in zip(cols, images):
                    with col:
                        # Open the image and display it
                        image = Image.open(img_path)

                        if image.width > image.height:
                            image = image.rotate(270, expand=True)
                            
                        st.image(image, use_container_width=True)

            # Display the videos in the Streamlit app
            video_expander = st.expander("Sample Videos")
            with video_expander:
                for video_name, video_path in settings.videos.items():
                    st.markdown(video_name)
                    with open(video_path, "rb") as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes, format="mp4", )
            
            
    elif selected_task == "Segmentation":
        # Extract model names without the directory path, and make them uppercase
        model_options = [Path(model).stem.split()[-2].upper() for model in settings.segmentation_models]
        model_paths = settings.segmentation_models
        button_text = "Perform Segmentation"
        
    elif selected_task == "Detection":
        # Extract model names without the directory path, and make them uppercase
        model_options = [Path(model).stem.split()[-2].upper() for model in settings.detection_models]
        model_paths = settings.detection_models
        button_text = "Perform Detection"

    if selected_task != "App Guide":
        st.markdown(f":red-background[{button_text}] based on selected configuration.")

        # Create dropdown for selecting model
        selected_model_name = st.sidebar.selectbox("Select Model", model_options)
        # print("selected_model_name: ", selected_model_name)

        # Map the selected model name back to its corresponding path
        selected_model_path = model_paths[model_options.index(selected_model_name)]

        # Load the selected model
        model = YOLO(Path(selected_model_path))

        # Check if GPU is available and set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # print(f"Device: {device}")

        # Add video source selection dropdown
        source = st.sidebar.selectbox(
            "Source",
            ("image", "video", "webcam", ),
        )
        
        # Add "All" option
        class_names = list(model.names.values())  # Convert dictionary to list of class names
        class_names.sort()  
        class_options = ["All"] + class_names

        # Multiselect box with class names and get indices of selected classes
        selected_classes = st.sidebar.multiselect("Classes", class_options, default=["All"])

        # If "Select All" is selected, automatically select all class names
        if "All" in selected_classes:
            selected_classes = class_names  # Select all classes
        
        selected_ind = [class_names.index(option) for option in selected_classes]
        # Ensure selected_options is a list
        if not isinstance(selected_ind, list):
            selected_ind = [selected_ind]

        conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.60, 0.05))
        iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.60, 0.05))

        fps_display = st.sidebar.empty()  # Placeholder for FPS display

        # Inference handlers
        if source == "image":
            handle_image_inference(st, model, conf, iou, selected_ind, button_text)
            # handle_image_inference_remote(st, selected_model_name, conf, iou, button_text)
        elif source == "video":
            handle_video_inference(st, model, conf, iou, selected_ind, button_text, fps_display)
            # handle_video_inference_remote(st, selected_model_name, conf, iou, button_text, fps_display)
        elif source == "webcam":
            # handle_webcam_inference(st, model, conf, iou, selected_ind, fps_display)
            handle_webcam_inference_remote(st, selected_model_name, conf, iou, fps_display)


        # Clear CUDA memory
        torch.cuda.empty_cache()

        # Destroy window
        cv2.destroyAllWindows()

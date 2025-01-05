import streamlit as st 
import pandas as pd
import os
import random
from PIL import Image
import cv2

import settings as settings 
from pages.utils.layout_home import read_yolo_annotation
from pages.utils.layout_home import overlay_annotations

def main():
    st.set_page_config(page_title="Home", page_icon="🏠", layout='wide', )

    # Add name at the bottom of the sidebar
    # Add empty space using markdown (custom height)
    st.sidebar.markdown("<br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    
    # Add name and LinkedIn link at the bottom
    st.sidebar.markdown("""
    By: 
                        
    Joshua R. Dela Cruz
                        
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[![LinkedIn](https://img.icons8.com/fluency/20/000000/linkedin.png)](https://www.linkedin.com/in/joshreyesdelacruz/)
    [![Github](https://img.icons8.com/fluency/20/000000/github.png)](https://github.com/josh-rdc/DeepLearningModels_Machine_Exercises)
                        """)


    # Display the logo and text in columns
    col1, col2, col3 = st.columns([1.5, 2 , 0.5])  # Adjust the width ratio if needed

    # Left column for the logo
    with col1:
        st.image(settings.logo_path, width=300)

    # Right column for the text
    with col2:
        st.write(
            """
            <div style="font-size: 16px;">
            Machine Exercise for the course AI 231 - Deep Learning Models<br>    
            Master of Engineering in Artificial Intelligence<br>
            College of Engineering<br>
            University of the Philippines Diliman
            </div>
            """, unsafe_allow_html=True
        )

    # Main title of streamlit application
    main_title_cfg = """<div><h2 style="color:#BF0603; text-align:center; 
                    font-family: 'monospace', monospace; margin-top:15px; margin-bottom:-10px;">
                    🛒  GROCERY ITEMS DETECTION AND SEGMENTATION APP 🛒 </h2>
                    </div>"""
    
    st.markdown(main_title_cfg, unsafe_allow_html=True)

    st.subheader(":red[Description]")
    description_card = st.container(border=True)
    ### App Description
    description_card.write(
        """
        - Web application for detecting and segmenting unique common grocery items using deep learning models trained from images captured under varying conditions. 
        - Check grocery item details and explore sample images with annotations for each class on the section below.
        - Refer to the **Model Cards** for :red-background[model details, training parameters and validation metrics]. 
        - Select **Inference** on the sidebar to :red-background[test the models on images, videos or via live webcam stream].
        """
    )

    
    dataset_count_df = pd.read_csv(settings.dataset_count_path)
    # Calculate totals
    total_train_images = dataset_count_df["train images"].sum()
    total_val_images = dataset_count_df["val images"].sum()
    total_images = dataset_count_df["total images"].sum()
    unique_classes = dataset_count_df["class"].nunique()

    st.subheader(":red[Dataset]")
    # st.markdown("### Dataset")
    tab1, tab2 = st.tabs(["Details", "Explore",])

    with tab1:
        dataset_card1 = st.container(border=True)
        with dataset_card1:
            # st.write("### Dataset Distribution")
            # Previous total
            previous_total = 8449

            st.write(f"""
                    **Unique Classes:** {unique_classes}  
                    **Training Images:** {total_train_images}  
                    **Validation Images:** {total_val_images}  
                    **Total Images:** {total_images} (increase of {((total_images - previous_total) / previous_total) * 100:.2f}%)
                    """)
            st.dataframe(dataset_count_df)
            st.markdown('''
                    *Note: :red-background[Unseen class variant] (different brand, flavor, packaging, unit of measurement) **may NOT be detected** by the model!*
                    ''')

    with tab2:
        dataset_card2 = st.container(border=True)
        with dataset_card2:
            # Dropdown to select a class
            class_names = dataset_count_df["class"].values.tolist()
            selected_class = st.selectbox("Viewing sample images from:", options=class_names)

            # Get the corresponding class ID
            class_id = class_names.index(selected_class)
            class_prefix = f"{class_id + 1:02d}"  # Format as two digits 

            # print("class id:", class_prefix)

            # Filter images by file name prefix
            val_images = [f for f in os.listdir(settings.validations_image_folder) if f.endswith(('.jpeg', '.png', '.jpg'))]
            selected_images = [img for img in val_images if img.startswith(class_prefix)]

            # print('val image folder:', settings.validations_image_folder)
            # print("selected images:", len(selected_images))
            
            # val_labels = [f for f in os.listdir(settings.validations_labels_folder) if f.endswith('.txt')]
            # selected_labels = [txt for txt in val_labels if txt.startswith(class_prefix)]

            # print('val labels folder:', settings.validations_labels_folder)
            # print("selected labels:", len(selected_labels))

            if not selected_images:
                st.warning(f"No images found for the selected class: {selected_class}.")
            else:
                # Display random 12 images
                displayed_images = random.sample(selected_images, min(12, len(selected_images)))
                # displayed_images = selected_images[:12]
                rows = []
                for i, image_name in enumerate(displayed_images):
                    image_path = os.path.join(settings.validations_image_folder, image_name)

                    annotation_name = image_name.replace('.jpeg', '.txt').replace('.png', '.txt').replace('.jpg', '.txt')
                    annotation_path = os.path.join(settings.validations_labels_folder, annotation_name)

                    annotations = read_yolo_annotation(annotation_path)
                    annotated_image = overlay_annotations(image_path, annotations, class_names)
                    
                    # Convert annotated image to RGB for displaying with PIL
                    pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                    rows.append(pil_image)

                # Display images in a 3x4 grid (4 images per row)
                cols = st.columns(4)
                for i, img in enumerate(rows):
                    with cols[i % 4]:
                        st.image(img, use_container_width =True) # use_column_width - deprecated


if __name__ == "__main__":
    main()


# Run with:
# streamlit run 🏠_Home.py
# Image Segmentation Web App

Web application that performs object detection and segmentation on common grocery items using [U-Net](https://arxiv.org/abs/1505.04597) models trained on A100 GPU. It uses a  model trained on [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) and is deployed using [Streamlit](https://streamlit.io/).

## Demo

The deployed version of this project can be accessed at [Hugging Face Spaces](https://grocery-detection-segmentation-webapp.streamlit.app/). Sample images are shown below:
![Segmentation on a sample image](readme_images/image.png)

## Installing Locally

To run this project locally, please follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/soumya-prabha-maiti/image-segmentation-web-app
   ```

2. Navigate to the project folder:

   ```
   cd image-segmentation-web-app
   ```

3. Install the required libraries:

   ```
   pip install -r requirements.txt
   ```

4. Run the application:

   ```
   python app.py
   ```

5. Access the application in your web browser at the specified port.

## Dataset

The dataset containes 24 classes of common grocery items found in the Philippines. These images are manually collected and annotated by the Class of UP Diliman, AI 231 (AY 2024-2024)
The [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) contains 37 categories of pets with roughly 200 images for each category. The images have a large variation in scale, pose and lighting. All images have an associated ground truth annotation of breed, head ROI, and pixel level trimap segmentation. Here the dataset was obtained using [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet).

## Model

The segmentation model uses the UNET architecture. The basic architecture of the UNET model is shown below:
![UNET Architecture](readme_images/unet.png)
The UNET model consists of an encoder and a decoder. The encoder is a series of convolutional layers that extract features from the input image. The decoder is a series of transposed convolutional layers that upsample the features to the original image size. Skip connections are used to connect the encoder and decoder layers. The skip connections concatenate the feature maps from the encoder to the corresponding feature maps in the decoder. This helps the decoder to recover the spatial information lost during the encoding process.

The detailed architecture of the UNET model used in this project is shown below:

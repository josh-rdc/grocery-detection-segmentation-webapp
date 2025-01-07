import os

# Base Path
base_path = os.path.dirname(__file__)

## Assets Folder
assets_folder = os.path.join(base_path, 'assets')
logo_path = os.path.join(assets_folder, 'logo_upd_and_ai.png')
fastapi_path = os.path.join(assets_folder, 'fastapi-websockets.png')

## Trained Model Paths
model_base_path = os.path.join(base_path, 'trained models')

## Validations Folder
validations_image_folder = os.path.join(base_path, 'validations/images')
validations_labels_folder = os.path.join(base_path, 'validations/labels')

### Revision 0 Models
### Model paths for detection
detection_models = [
    os.path.join(model_base_path, "yolo11n T.pt"), # YOLO11N
    # os.path.join(model_base_path, "yolo11m T.pt"), # YOLO11M
    # os.path.join(model_base_path, "yolo11x T.pt"), # YOLO11X
]

### Model paths for segmentation
segmentation_models = [
    os.path.join(model_base_path, "yolo11n-seg T.pt"), # YOLO11N-SEG
    # os.path.join(model_base_path, "yolo11m-seg T.pt"), # YOLO11M-SEG
    # os.path.join(model_base_path, "yolo11x-seg T.pt"), # YOLO11X-SEG
    # os.path.join(model_base_path, "yolo11x2-seg T.pt"), # YOLO11X2-SEG
]

### Revision 1 Models
# ### Model paths for detection
# detection_models = [
#     os.path.join(model_base_path, "RTDTR-x T0.pt"), # RTDTR-X
#     os.path.join(model_base_path, "yolo11x-640F T0.pt"), # YOLO11X Pixel 640 - Freeze Backbone
#     os.path.join(model_base_path, "yolo11x-640NF T0.pt"), # YOLO11X Pixel 640 - No Freeze Backbone
#     os.path.join(model_base_path, "yolo11x-1280F T0.pt"), # YOLO11X Pixel 1280 - Freeze Backbone
# ]

# ### Model paths for segmentation
# segmentation_models = [
#     os.path.join(model_base_path, "yolo11x-seg-640F T0.pt"), # YOLO11X-SEG Pixel 640 - Freeze Backbone
#     os.path.join(model_base_path, "yolo11x-seg-640NF T0.pt"), # YOLO11X-SEG Pixel 640 - No Freeze Backbone
#     os.path.join(model_base_path, "yolo11x-seg-1280F T0.pt"), # YOLO11X-SEG Pixel 1280 - Freeze Backbone
# ]

## Results Folder
result_base_path = os.path.join(base_path, 'results')
training_results_R0 = os.path.join(base_path, 'results/R0')
training_results_R1 = os.path.join(base_path, 'results/R1')

### csv Files
dataset_count_path = os.path.join(result_base_path, 'dataset.csv')
general_param_path = os.path.join(result_base_path, 'general parameters.csv')
training_path = os.path.join(result_base_path, 'training results.csv')
training_path_1 = os.path.join(result_base_path, 'training results R1.csv')
app_guide_path = os.path.join(result_base_path, 'app guide.csv')

### Folder Paths for R0
### YOLO11N 
yolo11n_path = os.path.join(training_results_R0, 'YOLO11N')
yolo11n_loss_path = os.path.join(yolo11n_path, 'results.csv')
yolo11n_conf_path = os.path.join(yolo11n_path, 'conf_matrix.png')
yolo11n_val_path = os.path.join(yolo11n_path, 'val_sample.jpg')

### YOLO11N-SEG 
yolo11ns_path = os.path.join(training_results_R0, 'YOLO11N SEG')
yolo11ns_loss_path = os.path.join(yolo11ns_path, 'results.csv')
yolo11ns_conf_path = os.path.join(yolo11ns_path, 'conf_matrix.png')
yolo11ns_val_path = os.path.join(yolo11ns_path, 'val_sample.jpg')

#### YOLO11M 
yolo11m_path = os.path.join(training_results_R0, 'YOLO11M')
yolo11m_loss_path = os.path.join(yolo11m_path, 'results.csv')
yolo11m_conf_path = os.path.join(yolo11m_path, 'conf_matrix.png')
yolo11m_val_path = os.path.join(yolo11m_path, 'val_sample.jpg')

### YOLO11M-SEG 
yolo11ms_path = os.path.join(training_results_R0, 'YOLO11M SEG')
yolo11ms_loss_path = os.path.join(yolo11ms_path, 'results.csv')
yolo11ms_conf_path = os.path.join(yolo11ms_path, 'conf_matrix.png')
yolo11ms_val_path = os.path.join(yolo11ms_path, 'val_sample.jpg')

#### YOLO11X
yolo11x_path = os.path.join(training_results_R0, 'YOLO11X')
yolo11x_loss_path = os.path.join(yolo11x_path, 'results.csv')
yolo11x_conf_path = os.path.join(yolo11x_path, 'conf_matrix.png')
yolo11x_val_path = os.path.join(yolo11x_path, 'val_sample.jpg')

### YOLO11X-SEG 
yolo11xs_path = os.path.join(training_results_R0, 'YOLO11X SEG')
yolo11xs_loss_path = os.path.join(yolo11xs_path, 'results.csv')
yolo11xs_conf_path = os.path.join(yolo11xs_path, 'conf_matrix.png')
yolo11xs_val_path = os.path.join(yolo11xs_path, 'val_sample.jpg')

### Folder Paths for R1
### RTDTR-X
rtdtr_x_path = os.path.join(training_results_R1, 'RTDTR-X')
rtdtr_x_loss_path = os.path.join(rtdtr_x_path, 'results.csv')
rtdtr_x_conf_path = os.path.join(rtdtr_x_path, 'conf_matrix.png')
rtdtr_x_val_path = os.path.join(rtdtr_x_path, 'val_sample.jpg')

### YOLO11X-640F
yolo11x_640F_path = os.path.join(training_results_R1, 'YOLO11X-640F')
yolo11x_640F_loss_path = os.path.join(yolo11x_640F_path, 'results.csv')
yolo11x_640F_conf_path = os.path.join(yolo11x_640F_path, 'conf_matrix.png')
yolo11x_640F_val_path = os.path.join(yolo11x_640F_path, 'val_sample.jpg')

### YOLO11X-640NF
yolo11x_640NF_path = os.path.join(training_results_R1, 'YOLO11X-640NF')
yolo11x_640NF_loss_path = os.path.join(yolo11x_640NF_path, 'results.csv')
yolo11x_640NF_conf_path = os.path.join(yolo11x_640NF_path, 'conf_matrix.png')
yolo11x_640NF_val_path = os.path.join(yolo11x_640NF_path, 'val_sample.jpg')

### YOLO11X-1280F
yolo11x_1280F_path = os.path.join(training_results_R1, 'YOLO11X-1280F')
yolo11x_1280F_loss_path = os.path.join(yolo11x_1280F_path, 'results.csv')
yolo11x_1280F_conf_path = os.path.join(yolo11x_1280F_path, 'conf_matrix.png')
yolo11x_1280F_val_path = os.path.join(yolo11x_1280F_path, 'val_sample.jpg')

### YOLO11X-SEG-640F
yolo11x_seg_640F_path = os.path.join(training_results_R1, 'YOLO11X-SEG-640F')
yolo11x_seg_640F_loss_path = os.path.join(yolo11x_seg_640F_path, 'results.csv')
yolo11x_seg_640F_conf_path = os.path.join(yolo11x_seg_640F_path, 'conf_matrix.png')
yolo11x_seg_640F_val_path = os.path.join(yolo11x_seg_640F_path, 'val_sample.jpg')

### YOLO11X-SEG-640NF
yolo11x_seg_640NF_path = os.path.join(training_results_R1, 'YOLO11X-SEG-640NF')
yolo11x_seg_640NF_loss_path = os.path.join(yolo11x_seg_640NF_path, 'results.csv')
yolo11x_seg_640NF_conf_path = os.path.join(yolo11x_seg_640NF_path, 'conf_matrix.png')
yolo11x_seg_640NF_val_path = os.path.join(yolo11x_seg_640NF_path, 'val_sample.jpg')

### YOLO11X-SEG-1280F
yolo11x_seg_1280F_path = os.path.join(training_results_R1, 'YOLO11X-SEG-1280F')
yolo11x_seg_1280F_loss_path = os.path.join(yolo11x_seg_1280F_path, 'results.csv')
yolo11x_seg_1280F_conf_path = os.path.join(yolo11x_seg_1280F_path, 'conf_matrix.png')
yolo11x_seg_1280F_val_path = os.path.join(yolo11x_seg_1280F_path, 'val_sample.jpg')


### Test Data
image_folder = "validations/tested_images"

video_folder = "validations"
videos = {
    "Detection Test": f"{video_folder}/detection_testR0.mp4",
    "Segmentation Test": f"{video_folder}/segmentation_testR0.mp4",
    # "Live Test": f"{video_folder}/live_test.mp4",
}
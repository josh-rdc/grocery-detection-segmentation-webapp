import streamlit as st

import pandas as pd

import settings as settings
from pages.utils.layout_model import plot_epoch_vs_loss

def main():
    st.set_page_config(page_title="Model Card", page_icon="📃", layout='wide', )
    # st.title("Model Cards")
    # st.markdown("Summary of model specifications and training result.")

    # Add empty space using markdown (custom height)
    st.sidebar.markdown("<br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    # Add name and LinkedIn link at the bottom
    st.sidebar.markdown("""
    Reference: 
    
    Jocher, G., & Qiu, J. (2024). Ultralytics YOLO11 (11.0.0) [Computer software]. https://github.com/ultralytics/ultralytics  
                        
                        """)

    general_params_df = pd.read_csv(settings.general_param_path)
    training_pdf = pd.read_csv(settings.training_path)
    training_pdf_1 = pd.read_csv(settings.training_path_1)

    st.subheader(":red[Training and General Parameters]")
    genparam_container = st.container(border=True)
    training_code = '''
                    from ultralytics import YOLO

                    model = YOLO(model_name)                    # Initialize model, automatically downloads latest pretrained weights
                    results = model.train(training_parameters)  # Train model, requires yaml file for dataset directory and class labels 
                    '''

    with genparam_container:
        st.code(training_code, language='python', wrap_lines=True)
        st.table(general_params_df)
        st.markdown(
        '''\* *Specific augmentation strategy variable keywords can be found in the
        [ultralytics mode/train](https://docs.ultralytics.com/modes/train/#train-settings).*          
        \* **Image size equal to 1280 was also tested in Revision 1.*       

        ''')
                    

    st.subheader(":red[Metrics and Validation Results]")
    training_container = st.container(border=True)
    with training_container:
        training_results_R0 = st.expander('Revision 0 Results', expanded=False)
        with training_results_R0:
            st.dataframe(training_pdf)

        # training_results_R1 = st.expander('Revision 1 Results', expanded=False)
        # with training_results_R1:
        st.markdown('Revision 1 Results')   
        st.dataframe(training_pdf_1)

        
        st.markdown('''
                    Note: 
                    - ALL Models are evaluated on the validation set using :red-background[minimum confidence and Iou threshold of 0.60].
                    - ↑ means higher values are better, ↓ means lower values are better.
                    - *(b)* corresponds to box (detection) metrics while *(m)* corresponds to mask (segmentation) metrics.
                    - *F* denotes freezed backbone, *NF* denotes non-freezed backbone.
                    - For reference, 
                    [pytorch MaskRCNN ResNet50 FPN V2 pre-trained model](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html#torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights) 
                    has 46.4M parameters. 
                    ''')
    
    revision0_expander = st.expander('Revision 0 Training', expanded=False)

    with revision0_expander:
        tab1n, tab2m, tab3x, tab1n_s, tab2m_s, tab3x_s = st.tabs(["YOLO11N", "YOLO11M", "YOLO11X", 
                                                                    "YOLO11N-SEG", "YOLO11M-SEG", "YOLO11X-SEG", 
                                                                    ])
        
        yolo11n_loss_df = pd.read_csv(settings.yolo11n_loss_path)
        yolo11ns_loss_df = pd.read_csv(settings.yolo11ns_loss_path)

        yolo11m_loss_df = pd.read_csv(settings.yolo11m_loss_path)
        yolo11ms_loss_df = pd.read_csv(settings.yolo11ms_loss_path)
        
        yolo11x_loss_df = pd.read_csv(settings.yolo11x_loss_path)
        yolo11xs_loss_df = pd.read_csv(settings.yolo11xs_loss_path)
        
        with tab1n:
                tabcol1, tabcol2 = st.columns([1, 1])
                with tabcol1:
                    container = st.container(border=True, )
                    with container: 
                        plot_epoch_vs_loss("YOLO11N", yolo11n_loss_df, 'train/box_loss', 'val/box_loss', 'Box')
                        st.markdown('---')
                        plot_epoch_vs_loss("YOLO11N", yolo11n_loss_df, 'train/cls_loss', 'val/cls_loss', 'Class')
                        st.markdown('---')
                        plot_epoch_vs_loss("YOLO11N", yolo11n_loss_df, 'train/dfl_loss', 'val/dfl_loss', 'DFL')
                with tabcol2:
                    st.image(settings.yolo11n_conf_path, use_container_width =True)
                    st.image(settings.yolo11n_val_path, use_container_width =True)
                        
        with tab2m:
                tabcol1, tabcol2 = st.columns([1, 1])
                with tabcol1:
                    container = st.container(border=True, )
                    with container: 
                        plot_epoch_vs_loss("YOLO11M", yolo11m_loss_df, 'train/box_loss', 'val/box_loss', 'Box')
                        st.markdown('---')
                        plot_epoch_vs_loss("YOLO11M", yolo11m_loss_df, 'train/cls_loss', 'val/cls_loss', 'Class')
                        st.markdown('---')
                        plot_epoch_vs_loss("YOLO11M", yolo11m_loss_df, 'train/dfl_loss', 'val/dfl_loss', 'DFL')
                with tabcol2:
                        st.image(settings.yolo11m_conf_path, use_container_width =True)
                        st.image(settings.yolo11m_val_path, use_container_width =True)
            

        with tab3x:
                tabcol1, tabcol2 = st.columns([1, 1])
                with tabcol1:
                    container = st.container(border=True, )
                    with container: 
                        plot_epoch_vs_loss("YOLO11X", yolo11x_loss_df, 'train/box_loss', 'val/box_loss', 'Box')
                        st.markdown('---')
                        plot_epoch_vs_loss("YOLO11X", yolo11x_loss_df, 'train/cls_loss', 'val/cls_loss', 'Class')
                        st.markdown('---')
                        plot_epoch_vs_loss("YOLO11X", yolo11x_loss_df, 'train/dfl_loss', 'val/dfl_loss', 'DFL')
                with tabcol2:
                        st.image(settings.yolo11x_conf_path, use_container_width =True)
                        st.image(settings.yolo11x_val_path, use_container_width =True)

        with tab1n_s:
                tabcol1, tabcol2 = st.columns([1, 1])
                with tabcol1:
                    container = st.container(border=True, )
                    with container: 
                        plot_epoch_vs_loss("YOLO11N-SEG", yolo11ns_loss_df, 'train/box_loss', 'val/box_loss', 'Box')
                        st.markdown('---')
                        plot_epoch_vs_loss("YOLO11N-SEG", yolo11ns_loss_df, 'train/seg_loss', 'val/seg_loss', 'Mask')
                        st.markdown('---')
                        plot_epoch_vs_loss("YOLO11N-SEG", yolo11ns_loss_df, 'train/cls_loss', 'val/cls_loss', 'Class')
                        # st.markdown('---')
                        # plot_epoch_vs_loss("YOLO11N-SEG", yolo11ns_loss_df, 'train/dfl_loss', 'val/dfl_loss', 'DFL')
                        
                with tabcol2:
                        st.image(settings.yolo11ns_conf_path, use_container_width =True)
                        st.image(settings.yolo11ns_val_path, use_container_width =True)

        with tab2m_s:
                tabcol1, tabcol2 = st.columns([1, 1])
                with tabcol1:
                    container = st.container(border=True, )
                    with container: 
                        plot_epoch_vs_loss("YOLO11M-SEG", yolo11ms_loss_df, 'train/box_loss', 'val/box_loss', 'Box')
                        st.markdown('---')
                        plot_epoch_vs_loss("YOLO11M-SEG", yolo11ms_loss_df, 'train/seg_loss', 'val/seg_loss', 'Mask')
                        st.markdown('---')
                        plot_epoch_vs_loss("YOLO11M-SEG", yolo11ms_loss_df, 'train/cls_loss', 'val/cls_loss', 'Class')
                        # st.markdown('---')
                        # plot_epoch_vs_loss("YOLO11M-SEG", yolo11ms_loss_df, 'train/dfl_loss', 'val/dfl_loss', 'DFL')
                        
                with tabcol2:
                        st.image(settings.yolo11ms_conf_path, use_container_width =True)
                        st.image(settings.yolo11ms_val_path, use_container_width =True)

        with tab3x_s:
                tabcol1, tabcol2 = st.columns([1, 1])
                with tabcol1:
                    container = st.container(border=True, )
                    with container: 
                        plot_epoch_vs_loss("YOLO11X-SEG", yolo11xs_loss_df, 'train/box_loss', 'val/box_loss', 'Box')
                        st.markdown('---')
                        plot_epoch_vs_loss("YOLO11X-SEG", yolo11xs_loss_df, 'train/seg_loss', 'val/seg_loss', 'Mask')
                        st.markdown('---')
                        plot_epoch_vs_loss("YOLO11X-SEG", yolo11xs_loss_df, 'train/cls_loss', 'val/cls_loss', 'Class')
                        # st.markdown('---')
                        # plot_epoch_vs_loss("YOLO11X-SEG", yolo11xs_loss_df, 'train/dfl_loss', 'val/dfl_loss', 'DFL')
                        
                with tabcol2:
                        st.image(settings.yolo11xs_conf_path, use_container_width =True)
                        st.image(settings.yolo11xs_val_path, use_container_width =True)

    
    st.markdown("### Revision 1 Training")
    tab1_640NF, tab2_640F, tab3_1280F, tab4_RTDETR, tab5s_640NF, tab6s_640F, tab7s_1280F = st.tabs(["YOLO11X-640NF", 
                                                                                                    "YOLO11X-640F",
                                                                                                    "YOLO11X-1280F",
                                                                                                    "RTDTR-X",
                                                                                                    "YOLO11X-SEG-640NF",
                                                                                                    "YOLO11X-SEG-640F",
                                                                                                    "YOLO11X-SEG-1280F"
                                                                    ])
                                                                    
    yolo11x_640NF_loss_df = pd.read_csv(settings.yolo11x_640NF_loss_path)
    yolo11x_640F_loss_df = pd.read_csv(settings.yolo11x_640F_loss_path)
    yolo11x_1280F_loss_df = pd.read_csv(settings.yolo11x_1280F_loss_path)
    rtdtr_x_loss_df = pd.read_csv(settings.rtdtr_x_loss_path)
    yolo11x_seg_640NF_loss_df = pd.read_csv(settings.yolo11x_seg_640NF_loss_path)
    yolo11x_seg_640F_loss_df = pd.read_csv(settings.yolo11x_seg_640F_loss_path)
    yolo11x_seg_1280F_loss_df = pd.read_csv(settings.yolo11x_seg_1280F_loss_path)

    with tab1_640NF:
        tabcol1, tabcol2 = st.columns([1, 1])
        with tabcol1:
            container = st.container(border=True, )
            with container: 
                plot_epoch_vs_loss("YOLO11X-640NF", yolo11x_640NF_loss_df, 'train/box_loss', 'val/box_loss', 'Box')
                st.markdown('---')
                plot_epoch_vs_loss("YOLO11X-640NF", yolo11x_640NF_loss_df, 'train/cls_loss', 'val/cls_loss', 'Class')
                st.markdown('---')
                plot_epoch_vs_loss("YOLO11X-640NF", yolo11x_640NF_loss_df, 'train/dfl_loss', 'val/dfl_loss', 'DFL')
        with tabcol2:
            st.image(settings.yolo11x_640NF_conf_path, use_container_width =True)
            st.image(settings.yolo11x_640NF_val_path, use_container_width =True)
    
    with tab2_640F:
        tabcol1, tabcol2 = st.columns([1, 1])
        with tabcol1:
            container = st.container(border=True, )
            with container: 
                plot_epoch_vs_loss("YOLO11X-640F", yolo11x_640F_loss_df, 'train/box_loss', 'val/box_loss', 'Box')
                st.markdown('---')
                plot_epoch_vs_loss("YOLO11X-640F", yolo11x_640F_loss_df, 'train/cls_loss', 'val/cls_loss', 'Class')
                st.markdown('---')
                plot_epoch_vs_loss("YOLO11X-640F", yolo11x_640F_loss_df, 'train/dfl_loss', 'val/dfl_loss', 'DFL')
        with tabcol2:
            st.image(settings.yolo11x_640F_conf_path, use_container_width =True)
            st.image(settings.yolo11x_640F_val_path, use_container_width =True)
    
    with tab3_1280F:
        tabcol1, tabcol2 = st.columns([1, 1])
        with tabcol1:
            container = st.container(border=True, )
            with container: 
                plot_epoch_vs_loss("YOLO11X-1280F", yolo11x_1280F_loss_df, 'train/box_loss', 'val/box_loss', 'Box')
                st.markdown('---')
                plot_epoch_vs_loss("YOLO11X-1280F", yolo11x_1280F_loss_df, 'train/cls_loss', 'val/cls_loss', 'Class')
                st.markdown('---')
                plot_epoch_vs_loss("YOLO11X-1280F", yolo11x_1280F_loss_df, 'train/dfl_loss', 'val/dfl_loss', 'DFL')
        with tabcol2:
            st.image(settings.yolo11x_1280F_conf_path, use_container_width =True)
            st.image(settings.yolo11x_1280F_val_path, use_container_width =True)

    with tab4_RTDETR:
        tabcol1, tabcol2 = st.columns([1, 1])
        with tabcol1:
            container = st.container(border=True, )
            with container: 
                plot_epoch_vs_loss("RTDTR-X", rtdtr_x_loss_df, 'train/box_loss', 'val/box_loss', 'Box')
                st.markdown('---')
                plot_epoch_vs_loss("RTDTR-X", rtdtr_x_loss_df, 'train/cls_loss', 'val/cls_loss', 'Class')
                st.markdown('---')
                plot_epoch_vs_loss("RTDTR-X", rtdtr_x_loss_df, 'train/l1_loss', 'val/l1_loss', 'L1')
        with tabcol2:
            st.image(settings.rtdtr_x_conf_path, use_container_width =True)
            st.image(settings.rtdtr_x_val_path, use_container_width =True)

    with tab5s_640NF:
        tabcol1, tabcol2 = st.columns([1, 1])
        with tabcol1:
            container = st.container(border=True, )
            with container: 
                plot_epoch_vs_loss("YOLO11X-SEG-640NF", yolo11x_seg_640NF_loss_df, 'train/box_loss', 'val/box_loss', 'Box')
                st.markdown('---')
                plot_epoch_vs_loss("YOLO11X-SEG-640NF", yolo11x_seg_640NF_loss_df, 'train/seg_loss', 'val/seg_loss', 'Mask')
                st.markdown('---')
                plot_epoch_vs_loss("YOLO11X-SEG-640NF", yolo11x_seg_640NF_loss_df, 'train/cls_loss', 'val/cls_loss', 'Class')
                # st.markdown('---')
                # plot_epoch_vs_loss("YOLO11X-SEG-640NF", yolo11x_seg_640NF_loss_df, 'train/dfl_loss', 'val/dfl_loss', 'DFL')
        with tabcol2:
            st.image(settings.yolo11x_seg_640NF_conf_path, use_container_width =True)
            st.image(settings.yolo11x_seg_640NF_val_path, use_container_width =True)

    with tab6s_640F:
        tabcol1, tabcol2 = st.columns([1, 1])
        with tabcol1:
            container = st.container(border=True, )
            with container: 
                plot_epoch_vs_loss("YOLO11X-SEG-640F", yolo11x_seg_640F_loss_df, 'train/box_loss', 'val/box_loss', 'Box')
                st.markdown('---')
                plot_epoch_vs_loss("YOLO11X-SEG-640F", yolo11x_seg_640F_loss_df, 'train/seg_loss', 'val/seg_loss', 'Mask')
                st.markdown('---')
                plot_epoch_vs_loss("YOLO11X-SEG-640F", yolo11x_seg_640F_loss_df, 'train/cls_loss', 'val/cls_loss', 'Class')
                # st.markdown('---')
                # plot_epoch_vs_loss("YOLO11X-SEG-640F", yolo11x_seg_640F_loss_df, 'train/dfl_loss', 'val/dfl_loss', 'DFL')
        with tabcol2:
            st.image(settings.yolo11x_seg_640F_conf_path, use_container_width =True)
            st.image(settings.yolo11x_seg_640F_val_path, use_container_width =True)

    with tab7s_1280F:
        tabcol1, tabcol2 = st.columns([1, 1])
        with tabcol1:
            container = st.container(border=True, )
            with container: 
                plot_epoch_vs_loss("YOLO11X-SEG-1280F", yolo11x_seg_1280F_loss_df, 'train/box_loss', 'val/box_loss', 'Box')
                st.markdown('---')
                plot_epoch_vs_loss("YOLO11X-SEG-1280F", yolo11x_seg_1280F_loss_df, 'train/seg_loss', 'val/seg_loss', 'Mask')
                st.markdown('---')
                plot_epoch_vs_loss("YOLO11X-SEG-1280F", yolo11x_seg_1280F_loss_df, 'train/cls_loss', 'val/cls_loss', 'Class')
                # st.markdown('---')
                # plot_epoch_vs_loss("YOLO11X-SEG-1280F", yolo11x_seg_1280F_loss_df, 'train/dfl_loss', 'val/dfl_loss', 'DFL')
        with tabcol2:
            st.image(settings.yolo11x_seg_1280F_conf_path, use_container_width =True)
            st.image(settings.yolo11x_seg_1280F_val_path, use_container_width =True)

    
         
                

if __name__ == "__main__":
    main()

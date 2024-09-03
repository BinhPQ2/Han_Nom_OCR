import streamlit as st
import cv2
import numpy as np
from recognition.ppocr.tools.infer.predict_mine import main as predict_characters
from ultralytics import YOLOv10
import tempfile
import os
import shutil

# Define the arguments for your OCR processing
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

class CropSaver:
    def __init__(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path

    def save_crop(self, result):
        conf_s = result.boxes.conf.detach().cpu().numpy()
        xyxy_s = result.boxes.xyxy.detach().cpu().numpy()
        idx_s = [i for i in range(len(conf_s))]
        img_name, img_extension = os.path.splitext(os.path.basename(result.path))
        img = result.orig_img

        height, width = img.shape[:2]

        height_extend_ratio = 0.05
        width_extend_ratio = 0.1

        for idx, conf, xyxy in zip(idx_s, conf_s, xyxy_s):
            x0, y0, x1, y1 = xyxy.tolist()
            x0 = int(max(0, x0 - width_extend_ratio * abs(x1 - x0)))
            x1 = int(min(width, x1 + width_extend_ratio * abs(x1 - x0)))
            y0 = int(max(0, y0 - height_extend_ratio * abs(y1 - y0)))
            y1 = int(min(height, y1 + height_extend_ratio * abs(y1 - y0)))
            crop_image = img[y0:y1, x0:x1]
            name = f"{img_name}_{x0}{img_extension}"

            cv2.imwrite(os.path.join(self.save_path, name), crop_image)

class ModelRunner:
    def __init__(self, detection_model_path):
        self.model = YOLOv10(detection_model_path)

    def run_inference(self, input_image_folder, save_path):
        results = self.model.predict(source=input_image_folder, conf=0.5, stream=True, save_txt=True)
        crop_saver = CropSaver(save_path)
        for result in results:
            crop_saver.save_crop(result)

class ImageProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)  # Ensure output folder exists

    def rotate_images(self):
        for filename in os.listdir(self.input_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.input_folder, filename)
                img = cv2.imread(img_path)
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                output_path = os.path.join(self.output_folder, filename)
                cv2.imwrite(output_path, rotated_img)

def get_image_parts(path):
    filename = os.path.basename(path)  # Get the filename from the path
    # Split by underscore to get "A" and "B" parts
    A, B_with_ext = filename.rsplit('_', 1)
    B = B_with_ext.split('.')[0]  # Remove the file extension
    return A, int(B)

# Define the Streamlit application
def main():
    st.title("Han Nom Text Recognition")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_image_path = temp_file.name
        
        st.image(temp_image_path, caption='Uploaded Image', use_column_width=True)

        if st.button("Process"):
            temp_input_dir = tempfile.mkdtemp()
            temp_output_dir = tempfile.mkdtemp()

            temp_image_dest = os.path.join(temp_input_dir, os.path.basename(temp_image_path))
            os.rename(temp_image_path, temp_image_dest)

            # Run model inference on the original images
            detection_model_path = './weight/detection_yolov10.pt'
            model_runner = ModelRunner(detection_model_path)
            model_runner.run_inference(temp_input_dir, temp_output_dir)

            # Rotate images
            image_processor = ImageProcessor(temp_output_dir, temp_output_dir)
            image_processor.rotate_images()

            # Define arguments for OCR
            args = Args(
                use_gpu=True,
                use_xpu=False,
                use_npu=False,
                use_mlu=False,
                ir_optim=True,
                use_tensorrt=False,
                min_subgraph_size=15,
                precision='fp32',
                gpu_mem=500,
                gpu_id=0,
                image_dir=temp_output_dir,
                page_num=0,
                det_algorithm='DB',
                det_model_dir=None,
                det_limit_side_len=960,
                det_limit_type='max',
                det_box_type='quad',
                det_db_thresh=0.3,
                det_db_box_thresh=0.6,
                det_db_unclip_ratio=1.5,
                max_batch_size=10,
                use_dilation=False,
                det_db_score_mode='fast',
                det_east_score_thresh=0.8,
                det_east_cover_thresh=0.1,
                det_east_nms_thresh=0.2,
                det_sast_score_thresh=0.5,
                det_sast_nms_thresh=0.2,
                det_pse_thresh=0,
                det_pse_box_thresh=0.85,
                det_pse_min_area=16,
                det_pse_scale=1,
                scales=[8, 16, 32],
                alpha=1.0,
                beta=1.0,
                fourier_degree=5,
                rec_algorithm='SVTR_LCNet',
                rec_model_dir='./weight/recognition_PPOCRLABEL',
                rec_image_inverse=True,
                rec_image_shape='3, 48, 320',
                rec_batch_num=6,
                max_text_length=25,
                rec_char_dict_path='/content/recognition/ppocr/utils/ppocr_keys_v1.txt',
                use_space_char=True,
                vis_font_path='./doc/fonts/simfang.ttf',
                drop_score=0.5,
                e2e_algorithm='PGNet',
                e2e_model_dir=None,
                e2e_limit_side_len=768,
                e2e_limit_type='max',
                e2e_pgnet_score_thresh=0.5,
                e2e_char_dict_path='./ppocr/utils/ic15_dict.txt',
                e2e_pgnet_valid_set='totaltext',
                e2e_pgnet_mode='fast',
                use_angle_cls=False,
                cls_model_dir=None,
                cls_image_shape='3, 48, 192',
                label_list=['0', '180'],
                cls_batch_num=6,
                cls_thresh=0.9,
                enable_mkldnn=False,
                cpu_threads=10,
                use_pdserving=False,
                warmup=False,
                sr_model_dir=None,
                sr_image_shape='3, 32, 128',
                sr_batch_num=1,
                draw_img_save_dir='./inference_results',
                save_crop_res=False,
                crop_res_save_dir='./output',
                use_mp=False,
                total_process_num=1,
                process_id=0,
                benchmark=False,
                save_log_path='./log_output/',
                show_log=True,
                use_onnx=False,
                return_word_box=False
            )

            results = predict_characters(args)

            print(f"results: {results}")

            # Sorting the dictionary
            sorted_result = dict(sorted(results.items(), key=lambda x: get_image_parts(x[0])))

            st.write("OCR Results:")
            for label in sorted_result.values():
              if len(label)==0:
                st.write("Can't detect any characters")
              else:
                st.write(f"{label}")

            # Clean up temporary files
            os.remove(temp_image_dest)
            shutil.rmtree(temp_input_dir)
            shutil.rmtree(temp_output_dir)

if __name__ == "__main__":
    main()
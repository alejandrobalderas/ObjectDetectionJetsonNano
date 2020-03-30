import argparse
import os

from yolo import YOLO
import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np


class Predictor:
    """"
    Class for predicting and detecting objects

    General
    ------
    This class has the purpose to use a tiny_yolo model to detect objects

    Attributes
    ------
    model_path: path to the weight file for the model which should be used for inference

    anchors_path: Path to the anchors to be used by the tiny_yolo model

    classes_path: Path to the classes for which the model was trained on

    image: Path to the image to be used for inference in the object detection

    folder: Path to the folder with images to do inference on

    video: Path to the video to do inference on

    video_stream: If the value of 1 is given to this argument the video of a connected webcam will be used for inference

    save_results: If the value is True the predicted images will be stored in the results folder


    Methods
    ------
    predict(image) : returns an image with the bounding box on the object, the values of the bounding boxes and the time of inference for one image

    predict_image(image): returns the image after a forward pass on the detection

    predict_boxes(image): returns the values of the bounding boxes on an image

    show_prediction(image): shows the image with the bounding boxes.

    predict_folder(folder_path): calls on the predict function for all the images in the containing folder

    predict_video(capture_value): makes an inference on every frame of a given video
    """

    _defaults = {
        "model_path": "model_data/tiny_yolo_trained.h5",
        "anchors_path": "model_data/tiny_yolo_anchors.txt",
        "classes_path": "model_data/classes.txt",
        "results_path": "results/",
        "save_images": False,
        "log_inference": False
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.yolo = YOLO(model_path=self.model_path,
                         anchors_path=self.anchors_path, classes_path=self.classes_path)

    def set_save_flag(self, save_images):
        self.save_images = save_images

    def predict(self, image):
        image_pil = Image.fromarray(image)
        image_inference, bbox, inference_time = self.yolo.detect_image(
            image_pil)
        return np.asarray(image_inference), bbox, inference_time

    def predict_boxes(self, image):
        _, detections, _ = self.predict(image)
        return detections

    def predict_image(self, image):
        pred_image, _, _ = self.predict(image)
        return pred_image

    def predict_boxes_from_path(self, image_path):
        image = cv2.imread(image_path)
        return self.predict_boxes(image)

    def show_prediction(self, image):
        pred_img = self.predict_image(image)
        cv2.imshow("Result", pred_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if self.save_images:
            cv2.imwrite(os.path.join(
                os.path.abspath(self.results_path), "prediction"), pred_img)

    def predict_folder(self, folder_path):
        for subdir, dirs, files in os.walk(folder_path):
            for file in files:
                image_path = os.path.join(subdir, file)
                image = cv2.imread(image_path)

                if self.save_results:
                    pred_img = self.predict_image(image)
                    cv2.imwrite(os.path.join(
                        os.path.abspath(self.results_path), file), pred_img)
                else:
                    self.show_prediction(image)
                if self.log_inference:
                    print("log inference")

    def predict_video(self, capture_val):
        vid = cv2.VideoCapture(capture_val)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        while True:
            return_value, frame = vid.read()
            if return_value:
                pred_image, bbox, inference_time = self.predict(frame)
                result = np.asarray(pred_image)
                cv2.imshow("result", result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    def close_session(self):
        self.yolo.sess.close()


def string_to_bool(text):
    if text == 'False' or text == 'false':
        return False
    else:
        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Command Line parser for Predictor class')
    parser.add_argument('--model_path', type=str, default='model_data/trained_weights_final.h5',
                        help='Path to the weight file for the model which should be used for inference')
    parser.add_argument('--image', type=str,
                        help='Path to the image to be predicted')
    parser.add_argument("--folder", nargs="?", type=str,
                        help="Path to the folder with images to do inference on")
    parser.add_argument("--video", type=str,
                        help="Path to the video to detect.")
    parser.add_argument("--video_stream", type=int, nargs="?",
                        help="A video stream of the webcam will be used for the detection. To activate the stream give the argument a value of 1")
    parser.add_argument("--save_results", type=str, default='False',
                        help="If True images are saved to results for visualization")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    predictor = Predictor(**vars(args))
    predictor.set_save_flag(string_to_bool(str(args.save_results)))
    if args.image:
        image_path = os.path.abspath(args.image)
        image = cv2.imread(image_path)
        predictor.show_prediction(image)
    elif args.folder:
        predictor.predict_folder(args.folder)
    elif args.video:
        predictor.predict_video(args.video)
    elif args.video_stream == 1:
        predictor.predict_video(0)

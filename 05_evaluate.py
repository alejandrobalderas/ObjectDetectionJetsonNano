from utils import BoundingBox
from utils import BoundingBoxes
from utils.utils import BBFormat, BBType, CoordinatesType
from utils import Evaluator as MetricEvaluator
import argparse
predictor_module = __import__('04_predict')


class Evaluator:

    """
    Class for Evaluation

    General
    ------
    Class for evaluation of the trained weights on the test set. Reads the ground truth labels from test.txt 
    and creates BoundingBox objects from that (see https://github.com/rafaelpadilla/Object-Detection-Metrics for Details). Afterwards it performs 
    inference on the test images using the model with trained weights and creates BoundingBox objects from the detections.
    Finally it calculates the PascalVoc metric for each class for a given IOU using the Evaluator script from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    For an interpretation of the metric see the Readme.

    Attributes
    ------
    test_annotation_path : path to test.txt 

    image_paths : list of extracted image paths (from test.txt)

    predictor : Instance of predictor class used for making detections on test set (class for making inference - see class doc for details)

    allBoundingBoxes : list of all bounding box objects (detections and ground truth)

    iou_threshold : intersection over union threshold from which detected bounding box counts as a correct detection in the calculation of the PVOC metric

    classes_path : path to classes.txt 

    classes_dict : dict for the classNames -> classId mapping

    Methods
    ------
    load_ground_truth() : loads ground truth labels from test.txt and saves it to the BoundingBoxes list

    make_inference() : calls the predictor object in order to make inference on the test images and saves the detections to the BoundingBoxes list

    evaluate() : orchestrates the different necessary steps 

    calculate_metrics() : calculates PVOC metric (AP@IOU) on the given BoundingBoxes list and prints out the results

    """

    def __init__(self, test_annotations_path, predictor, iou_threshold):
        self.test_annotations_path = test_annotations_path
        self.image_paths = []
        self.predictor = predictor
        self.allBoundingBoxes = BoundingBoxes.BoundingBoxes()
        self.iou_threshold = iou_threshold

    def set_classes_path(self, classes_path):
        self.classes_path = classes_path
        self.class_dict = {}

    def read_classes(self):
        with open(self.classes_path, 'r') as classes_file:
            classes_string = classes_file.read()
        classes = classes_string.split()
        counter = 0
        for label_class in classes:
            self.class_dict[label_class] = counter
            counter += 1

    def retrieve_class_by_id(self, id):
        for key, value in self.class_dict.items():
            if value == id:
                return key

    def load_ground_truth(self):
        with open(self.test_annotations_path, 'r') as annotation_file:
            annotation_string = annotation_file.read()
        annotations = annotation_string.split('\n')[:-1]
        counter = 0
        for annotation in annotations:
            annotation_list = annotation.split(' ')
            path = annotation_list[0]
            boxes_strings = annotation_list[1:]
            image_name = 'image_'+str(counter)
            counter += 1
            self.image_paths.append((path, image_name))
            for box in boxes_strings:
                box_list = box.split(',')
                xmin = int(box_list[0])
                ymin = int(box_list[1])
                xmax = int(box_list[2])
                ymax = int(box_list[3])
                id_class = int(box_list[4])
                bbox = BoundingBox.BoundingBox(image_name, id_class, xmin, ymin, xmax-xmin, ymax-ymin, CoordinatesType.Absolute,
                                               None, BBType.GroundTruth, format=BBFormat.XYWH)
                self.allBoundingBoxes.addBoundingBox(bbox)

    def make_inference(self):
        for image_path, image_name in self.image_paths:
            print(image_name)
            # ToDo check sizes of predictions
            detections = self.predictor.predict_boxes_from_path(image_path)
            for detection in detections:
                xmin, ymin, xmax, ymax, class_name, score = detection
                id_class = self.class_dict[class_name]
                bbox = BoundingBox.BoundingBox(image_name, id_class, xmin, ymin, xmax-xmin, ymax-ymin, CoordinatesType.Absolute,
                                               None, BBType.Detected, score, format=BBFormat.XYWH)
                self.allBoundingBoxes.addBoundingBox(bbox)

    def calculate_metrics(self):
        metric_evaluator = MetricEvaluator.Evaluator()
        metricsPerClass = metric_evaluator.GetPascalVOCMetrics(
            self.allBoundingBoxes, IOUThreshold=self.iou_threshold)
        print("\n--- Results ---\n")
        print("AP@"+str(self.iou_threshold)+" per class:\n")

        for mc in metricsPerClass:
            class_id = mc['class']
            class_name = self.retrieve_class_by_id(class_id)
            average_precision = mc['AP']
            print(class_name+": "+"{0:.3f}".format(average_precision*100)+"%")

    def evaluate(self):
        self.read_classes()
        self.load_ground_truth()
        self.make_inference()
        self.calculate_metrics()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Command Line parser for Evaluator class')
    parser.add_argument('--weight_file_path', type=str, default='model_data/trained_weights_final.h5',
                        help='path to the weight file for the model which should be used for inference')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='path to the weight file for the model which should be used for inference')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    predictor = predictor_module.Predictor(model_path=args.weight_file_path)
    evaluator = Evaluator('test.txt', predictor, args.iou_threshold)
    evaluator.set_classes_path('./model_data/classes.txt')
    evaluator.evaluate()

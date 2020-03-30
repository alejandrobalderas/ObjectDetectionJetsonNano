import xml.etree.ElementTree as ET
import os
import argparse


class Preprocessor:

    """
    Class for preprocessing

    General
    ------
    This class has the purpose to read in data where the labels are in VOC format (for example after labeling them with Boobs) 
    and retrieving two files named train.txt and test.txt. These files specify the paths to the images and the corresponding ground truth boxes
    all in one file (for train and test separate files). These files can be processed by the train and evaluation script. The train test split in this case is expected
    to be done by the user, providing four paths to this script (train-images,train-labels,test-images,test-labels). The train.txt and test.txt are finally saved 
    to the root of this repo. The following parts of the pipeline also expect the files to lie in the root of the repo.

    Attributes
    ------
    train_image_path : path where training images are located (for each label file the corresponding image is expected to extist - file name is retrieved from the label xml file)

    train_label_path : path where training xml files (in Voc format) are located

    test_image_path : path where test images are located (for each label file the corresponding image is expected to extist - file name is retrieved from the label xml file)

    test_label_path : path where test xml files (in Voc format) are located

    classes_path : path to classes.txt (is necessary to write class ids in train.txt and test.txt)

    class_dict : mapping of class names to corresponding ids in the order given in classes.txt and starting by id=0 

    Methods
    ------
    read_classes() : reads classes from classes.txt

    process(): orchestrates the different methods in the right order

    read_train_labels() : reads all .xml files in train_labels_path, parses the xml`s and saves the labels and the image paths in lists

    read_test_labels() : reads all .xml files in test_labels_path, parses the xml`s and saves the labels and the image paths in lists

    parse_xml(): parses a label .xml files (in Voc format) and retrieves image_path and list of boxes where each box has the following format: box=[xmin, ymin, xmax, ymax, object_id]

    create_yolo_file(): creates train.txt and test.txt whereby the rows of these files look as follows: image_path xmin,ymin,xmax,ymax,object_id ... xmin,ymin,xmax,ymax,object_id

    """

    def __init__(self, train_image_path, train_label_path, test_image_path, test_label_path):
        self.train_image_path = train_image_path
        self.train_label_path = train_label_path
        self.test_image_path = test_image_path
        self.test_label_path = test_label_path
        self.train_label_list = []
        self.test_label_list = []

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

    def read_train_labels(self):
        label_file_list = [self.train_label_path+"/"+file_name for file_name in os.listdir(
            self.train_label_path) if file_name.endswith('.xml')]
        for label_file in label_file_list:
            image_file_name, boxes = self.parse_xml(label_file)
            if image_file_name in os.listdir(self.train_image_path):
                full_image_path = self.train_image_path+"/"+image_file_name
                self.train_label_list.append(
                    (full_image_path, boxes))
            elif image_file_name.replace("JPG", "jpg") in os.listdir(self.train_image_path):
                print("Replace JPG with jpg")
                new_image_file_name = image_file_name.replace("JPG", "jpg")
                full_image_path = self.train_image_path+"/"+new_image_file_name
                self.train_label_list.append(
                    (full_image_path, boxes))

    def read_test_labels(self):
        label_file_list = [self.test_label_path+"/" + file_name for file_name in os.listdir(
            self.test_label_path) if file_name.endswith('.xml')]
        for label_file in label_file_list:
            image_file_name, boxes = self.parse_xml(label_file)
            if image_file_name in os.listdir(self.test_image_path):
                full_image_path = self.test_image_path+"/"+image_file_name
                self.test_label_list.append(
                    (full_image_path, boxes))
            elif image_file_name.replace("JPG", "jpg") in os.listdir(self.test_image_path):
                print("Replace JPG with jpg")
                new_image_file_name = image_file_name.replace("JPG", "jpg")
                full_image_path = self.test_image_path+"/"+new_image_file_name
                self.test_label_list.append(
                    (full_image_path, boxes))

    def create_yolo_file(self, train):
        main_string = ''
        if train:
            output_file_name = 'train.txt'
            label_list = self.train_label_list
        else:
            output_file_name = 'test.txt'
            label_list = self.test_label_list

        for label in label_list:
            full_image_path, boxes = label
            single_line = full_image_path
            for box in boxes:
                single_line += ' '+','.join(str(e) for e in box)
            main_string += single_line + '\n'

        with open(output_file_name, 'w+') as yolo_file:
            yolo_file.write(main_string)

    def parse_xml(self, label_file):
        tree = ET.parse(label_file)
        root = tree.getroot()
        image_file_name = root.find('filename').text
        database = root.find('source').find('database').text
        # if not database == 'Unknown':
        #     image_file_name = database + "_data_"+image_file_name
        objects_xml = root.findall('object')
        boxes = []
        for object_xml in objects_xml:
            name = object_xml.find('name').text
            object_id = self.class_dict[name]
            bounding_box_xml = object_xml.find('bndbox')
            xmin = int(bounding_box_xml.find('xmin').text)
            xmax = int(bounding_box_xml.find('xmax').text)
            ymin = int(bounding_box_xml.find('ymin').text)
            ymax = int(bounding_box_xml.find('ymax').text)
            box = [xmin, ymin, xmax, ymax, object_id]
            boxes.append(box)
        print(image_file_name)
        print(boxes)
        return image_file_name, boxes

    def process(self):
        self.read_classes()
        self.read_train_labels()
        self.read_test_labels()
        # Creates yolo files for train and test data
        self.create_yolo_file(True)
        self.create_yolo_file(False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Command Line parser for Preprocessor class')
    parser.add_argument('--train_image_path', type=str,
                        help='path where training images are located')
    parser.add_argument('--test_image_path', type=str,
                        help='path where test images are located')
    parser.add_argument('--test_label_path', type=str,
                        help='path where test labels are located')
    parser.add_argument('--train_label_path', type=str,
                        help='path where train labels are located')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_label_path = os.path.abspath(args.train_label_path)
    test_label_path = os.path.abspath(args.test_label_path)
    train_image_path = os.path.abspath(args.train_image_path)
    test_image_path = os.path.abspath(args.test_image_path)
    processor = Preprocessor(
        train_image_path, train_label_path, test_image_path, test_label_path)
    processor.set_classes_path('./model_data/classes.txt')
    processor.process()

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
import argparse
import os

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf


class Trainer:
    """
    Main class for training

    General
    ------
    This class is used for training the tiny yolo v3 model on a given training set (given in train.txt) on pretrained weights 
    with the classes specified in classes.txt. It has a warm up phase where only the first two layers are trained in order to not destroy the back layers 
    when training on new classes, which is the case here. The second phase is training on all weights. The script spits out a trained weight file (.h5) in the save_path.

    Attributes 
    ------
    num_epochs : number of epochs (warm_up + normal training)

    batch_size : batch_size in Adam

    val_split : percentage of data (in train.txt) that is used as validation data while training (for example to decide when to stop training)

    warm_up_epochs : number of epochs that is used for warm up phase (phase where only the first layers are trained and the backbone is fixed)

    learning_rate : learning rate for the normal training phase

    warm_up_learning_rate : learning rate for the warm up training phase

    input_shape : input shape of the yolo model (should not be changed!!)

    annotation_path : path to train.txt file

    log_path : path where (not final) weights are saved while training as well as tensorboard files

    class_path : path to classes.txt which defines number and names of the classes used

    anchor_path : path to tiny_yolo_anchors.txt file which is used to define anchors in the yolo model

    weight_path : path to the pretrained weights (.h5) for the tiny yolo model which was first converted from Darknet weights

    save_path : path where the final weights should be saved to 

    model : keras model object holding tiny yolo v3

    Methods
    ------

    train() : main training method

    get_classes() : retrieves the different classes form the classes.txt

    get_anchors() : retrieves anchors from anchors file

    create_model() : builds keras model and produces model object

    data_generator() : python generator to iteratively retrieve data loaded from train.txt and image path to the fit methods of keras

    """

    # Default values for Data Augmentation function

    _default_data_augmentation = {
        "jitter": 0.3,
        "hue": .1,
        "sat": 1.5,
        "val": 1.5,
        "scale_min": 0.5,
        "scale_max": 1.5,
        "flip_image": True
    }

    def __init__(self, num_epochs, batch_size, val_split):
        # Adjustable parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.val_split = val_split

        self.__dict__.update(self._default_data_augmentation)

        # Fix parameters - not recommended to change these
        self.warm_up_epochs = 10
        self.warm_up_learning_rate = 1e-3
        self.input_shape = (416, 416)
        self.learning_rate = 1e-4

    def set_main_paths(self, annotation_path, log_path, classes_path, anchor_path, weight_path, save_path):
        self.annotation_path = annotation_path
        self.log_path = log_path
        self.classes_path = classes_path
        self.anchor_path = anchor_path
        self.weight_path = weight_path
        self.save_path = save_path

    def train(self):
        class_names = self.get_classes(self.classes_path)
        num_classes = len(class_names)
        anchors = self.get_anchors(self.anchor_path)
        input_shape = self.input_shape
        self.create_model(input_shape, anchors, num_classes, freeze_body=2)
        logging = TensorBoard(log_dir=self.log_path)
        checkpoint = ModelCheckpoint(os.path.join(self.log_path, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                     monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=30, verbose=1)
        val_split = self.val_split
        with open(self.annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines)*val_split)
        num_train = len(lines) - num_val

        # First epochs with frozen backbone num_epochs-10 epochs normal training
        # Train with frozen layers first, to get a stable loss.
        if True:
            self.model.compile(optimizer=Adam(
                lr=self.warm_up_learning_rate), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(
                num_train, num_val, self.batch_size))
            self.model.fit_generator(self.data_generator_wrapper(lines[:num_train], input_shape, anchors, num_classes),
                                     steps_per_epoch=max(
                                         1, num_train//self.batch_size),
                                     validation_data=self.data_generator_wrapper(
                lines[num_train:], input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//self.batch_size),
                epochs=self.warm_up_epochs,
                initial_epoch=0,
                callbacks=[logging, checkpoint])

            self.model.save_weights(os.path.join(
                self.log_path, 'trained_weights_stage_1.h5'))

        # Unfreeze and continue training, to fine-tune.
        if True:
            for i in range(len(self.model.layers)):
                self.model.layers[i].trainable = True
            # recompile to apply the change
            self.model.compile(optimizer=Adam(lr=self.learning_rate),
                               loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            print('Unfreeze all of the layers.')
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(
                num_train, num_val, self.batch_size))
            self.model.fit_generator(self.data_generator_wrapper(lines[:num_train], input_shape, anchors, num_classes),
                                     steps_per_epoch=max(
                                         1, num_train//self.batch_size),
                                     validation_data=self.data_generator_wrapper(
                lines[num_train:], input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//self.batch_size),
                epochs=self.num_epochs,
                initial_epoch=self.warm_up_epochs,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
            self.model.save_weights(os.path.join(
                self.save_path, 'trained_weights_final.h5'))

    def get_classes(self, classes_path):
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self, anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def create_model(self, input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2):
        K.clear_session()
        config = tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
        set_session(tf.compat.v1.Session(config=config))
        image_input = Input(shape=(None, None, 3))
        h, w = input_shape
        num_anchors = len(anchors)
        y_true = [Input(shape=(h//{0: 32, 1: 16}[l], w//{0: 32, 1: 16}[l],
                               num_anchors//2, num_classes+5)) for l in range(2)]
        model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
        print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(
            num_anchors, num_classes))

        if load_pretrained:
            model_body.load_weights(
                self.weight_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(self.weight_path))
            if freeze_body in [1, 2]:
                num = (20, len(model_body.layers)-2)[freeze_body-1]
                for i in range(num):
                    model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(
                    num, len(model_body.layers)))

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', arguments={
                            'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})([*model_body.output, *y_true])
        self.model = Model([model_body.input, *y_true], model_loss)

    def data_generator(self, annotation_lines, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(self.batch_size):
                if i == 0:
                    np.random.shuffle(annotation_lines)
                image, box = get_random_data(
                    annotation_lines[i], input_shape,
                    random=True,
                    scale_min=self.scale_min,
                    scale_max=self.scale_max,
                    jitter=self.jitter,
                    hue=self.hue,
                    sat=self.sat,
                    val=self.val,
                    flip_image=self.flip_image)
                image_data.append(image)
                box_data.append(box)
                i = (i+1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(
                box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(self.batch_size)

    def data_generator_wrapper(self, annotation_lines, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n == 0 or self.batch_size <= 0:
            return None
        return self.data_generator(annotation_lines, input_shape, anchors, num_classes)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Command Line parser for Preprocessor class')
    parser.add_argument('--weight_path', type=str, default='model_data/tiny_yolo_weights.h5',
                        help='path to the .h5 file with pretrained weights')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='number of epochs (warmup + normal training)')
    parser.add_argument('--batch_size', default=32,  type=int,
                        help='batch_size in adam optimizer')
    parser.add_argument('--val_split', default=0.1, type=float,
                        help='Split portion of train-val split')
    parser.add_argument('--save_path', type=str, default='model_data/',
                        help='Path where final weight file should be saved')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    annotation_path = 'train.txt'
    log_path = os.path.abspath('logs/')
    save_path = os.path.abspath(args.save_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    classes_path = 'model_data/classes.txt'
    anchor_path = 'model_data/tiny_yolo_anchors.txt'
    weight_path = args.weight_path
    trainer = Trainer(args.num_epochs, args.batch_size, args.val_split)
    trainer.set_main_paths(annotation_path, log_path,
                           classes_path, anchor_path, weight_path, save_path)
    trainer.train()

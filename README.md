## Description

This repo contains all the necessary files to get started building, training and deploying an object detection model either on a computer or a Jetson Nano for inference on the edge.

## Getting Started

Please make sure that you have git installed in your computer/jetson nano and clone this repo with the following command:

The repo has the following structure:
There are 6 main scripts which defines the workflow in the repo as well as `yolo.py` script that hold the constructor for the model and three folders with different utility functions:

```bash
├── 01_preprocess.py
├── 02_convert.py
├── 03_train.py
├── 04_predict.py
├── 05_evaluate.py
├── 06_postprocess.py
├── yolo.py
├── README.md
├── requirements.txt
├── model_data
├── yolo3
└── utils
```

The workflow on how to run the python scripts will be described in the next section Running the code. The folder `model_data` holds model specific configuration files like `classes.txt` where the classes are defined, `tiny_yolo_anchors.txt` where the anchors for yolo are defined, and the weight files (`h5`) of the model while executing the pipeline. The `utils` folder holds the code for the metrics and the `yolo3` folder holds the code defining the model (`Tiny YoloV3`). The `yolo.py` script is used for inference and hold

### Installation

#### Setup Laptop

We recommend to have a separate virtual environment (using Python 3.5.5, which was used on data science vm) to run this scripts. Please follow this links if you want more information on how to use `venv` or `conda`.

1. venv: https://docs.python.org/3/tutorial/venv.html
2. conda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Activate your virtual environment and run the following code to install all the required dependencies

```python
pip install -r requirements.txt
```

#### Setup Jetson Nano

Please follow the following steps to configure your Jetson Nano. Open a terminal and introduce the following commands.

**Note:** This installation can take a couple of hours to complete

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt install nano git cmake python3-pip libatlas-base-dev gfortran libhdf5-serial-dev hdf5-tools python3-matplotlib
​
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
​
pip3 install virtualenvwrapper
echo 'export WORKON_HOME=$HOME/.venvs' >> ~/.bashrc
echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' >> ~/.bashrc
echo 'export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/bin/virtualenv' >> ~/.bashrc
echo 'source ~/.local/bin/virtualenvwrapper.sh' >> ~/.bashrc
mkvirtualenv enbw-bs-poc
workon enbw-bs-poc
​
pip install numpy pillow
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.14.0+nv19.7
pip install keras

```

Once you have successfully installed all the needed packages and dependencies your Jetson Nano is ready to run the scripts for object detection.

### Running the code:

### Step 1 - Preprocess:

As a first step we need to preprocess all the images and labels to train the model. You will need to organize the data in four folders. Two folders for labels in VOC Format (train and test) and two folders for the corresponding images (train and test). An example for the train data:

```bash
├── TRAIN_IMAGE_PATH
    ├── file1.jpg
    └── file2.jpg

├── TRAIN_LABEL_PATH
    ├── file1.xml
    └── file2.xml
```

From the root folder of this repo execute the following command:

```bash
python 01_preprocess.py --train_image_path=TRAIN_IMAGE_DIR --train_label_path=TRAIN_LABEL_PATH --test_image_path=TEST_IMAGE_PATH --test_label_path=TEST_LABEL_PATH
```

This script collects the labels and images from the corresponding folders and outputs `train.txt` and `test.txt` which are readable by the upcoming train and evaluate scripts.

### Step 2 - Convert Darknet to Keras:

**Note:** This step can be skipped if you already have a trained modelled to be used for inference.

This step converts the pre-trained weights of the tiny-yolo3 model and converts it into a keras `.h5` format.

```bash
wget -P $(pwd)/model_data https://pjreddie.com/media/files/yolov3-tiny.weights
python 02_convert.py model_data/yolov3-tiny.cfg model_data/yolov3-tiny.weights model_data/tiny_yolo_weights.h5
```

s

### Step 3 - Train:

Presuming you already have pre-trained weights in `.h5` format for the tiny yolo model you can start the training with the following command:

Please make sure to change the arguments in the function call.

```bash
python 03_train.py --weight_path=WEIGHT_FILE_PATH --num_epochs=NUM_EPOCHS --batch_size=BATCH_SIZE --val_split=VAL_SPLIT
```

This function takes 4 arguments as inputs

1. weight_path - path to the weights file as .h5
2. num_epochs - number of epochs to train
3. batch_size - batch size in the Adam optimizer
4. val_split - proportion of the training data which should be used as validation data while training (for example `--val_split=0.1`)

The model will be saved in the `model_data` folder as `tiny_yolo_trained.h5`

### Step 4 - Predict:

This script can be used to detect objects using an arbitrary tiny-yolo model. This script has multiple functionalities that can be called in the arguments

1. image: Path to the image to be detected
2. folder: Path to a folder with multiple images to be detected
3. video: Path to a video for which a detection will be made
4. video_stream: if the value 1 is passed a detection using the webcam will be started

Additionally to this parameters the boolean `save_results` flag can be passed if you wish to store the detected images or videos to visualize the results

```bash
python 04_predict.py --folder=`FOLDER_PATH`
```

### Step 5 - Evaluate Model:

In order to get a performance metric on the test data, the following script can be executed

```bash
python 05_evaluate.py --iou_threshold=IOU_THRESHOLD
```

where `IOU_THRESHOLD` describes the overlap a detection must have with the ground truth in order to count as detected (default is 0.3). This script reads the labels and image paths from the `test.txt` file and uses the trained weights to make inference in order to compare detections and labels and calculate the AP metric. The output looks as follows:

```bash
--- Results ---

AP@0.3 per class:

Class1: 84.091%
Class2: 72.119%
```

#### How to interpret these results:

An AP@0.3 of 84.091% for the class _Class1_ is approximatly equal to the precision of the detections ( number of correctly detected _Class1_ objects divided through all detections with class Class1). A bounding box is counted as a detection if the Intersection-Over-Union (IOU) is greater than 0.3 with the ground truth bounding box. The AP is only approximatly equal to the precision as it also considers the scores as well as the false positives of the detections. In fact it is the integral over the Precision-Recall Curve. For details we refer to the following medium article: https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

## How Train A Model On A New Data Set:

To train a new model on a new data set you just have to follow a few steps.

1. Organize the new data into a train and test group. The train set should have a big variability in the data which means in this case many pictures from different angles and different backgrounds. Make sure that "very similar" pictures in the test group are not contained in a training set.
2. Adjust the `./model_data/classes.txt` file if the classes for the model have changed.
3. Run Step 1 to preprocess the labels
4. Run Step 2 (optional). Run this step if you do not have the original tiny_yolo.h5 model in your `./model_data` folder
5. Run Step 3 to train the model
6. Run Step 5 to evaluate and test how good it performs.

## Extras

### Data Augmentation

Included in this repo under `./yolo3/utils.py` you will find the function `get_random_data`. This function returns an image as well as the bounding box for the object to detect. If the argument `random=True` is passed to the function an augmentation process will be started. This four steps will be executed:

1. Resizing / Scaling
2. Image Flipping
3. Color Shifting - changes in the HSV values
4. Box Correction

If `random=True` a modified copy of the original image will be returned. For every image in the training set this function will generate new random parameters to modify the image and make it unique. This function does not create new images for the model to train instead it changes the current image. In this way we can be sure that every epoch a "new version" of the image will be used to train.

If the value `random=False` is passed then no data augmentation will be executed

**Parameters:**

- jitter: Distorts the image. In simple words it squeezes the image together
- scale_min: min value for the scaling
- scale_max: max value for scaling
- hue: Adjust the hue value of the image.
- sat: Adjust the saturation of the image. If set to 1 the image will remain unchanged
- val: Adjust the value of the image. If set to 1 the image will remain unchanged
- flip_image: With a probability of 50% the image will be flipped on the vertical axis.

This parameters are set as default in the `03_train.py` script and are passed on to the function to train the images.

Scaling will be based on a random number between `scale_min` and `scale_max` to scale the image.

In the following snippet you can take a glance at how this function creates new values to modify the image.

```python
# distort image
hue = rand(-hue, hue)
sat = rand(1, sat) if rand() < .5 else 1/rand(1, sat)
val = rand(1, val) if rand() < .5 else 1/rand(1, val)
x = rgb_to_hsv(np.array(image)/255.)
x[..., 0] += hue
#... # Other operations
image_data = hsv_to_rgb(x)
```

Please see this link to understand more about the HSV Color Model
https://en.wikipedia.org/wiki/HSL_and_HSV

### Anchor generation

Please refer to the [Github repo](https://github.com/AlexeyAB/darknet#how-to-improve-object-detection).

Only if you are an **expert** in neural detection networks - recalculate anchors for your dataset for `width` and `height` from cfg-file:
`darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416`
then set the same 9 `anchors` in each of 3 `[yolo]`-layers in your cfg-file. But you should change indexes of anchors `masks=` for each [yolo]-layer, so that 1st-[yolo]-layer has anchors larger than 60x60, 2nd larger than 30x30, 3rd remaining. Also you should change the `filters=(classes + 5)*<number of mask>` before each [yolo]-layer. If many of the calculated anchors do not fit under the appropriate layers - then just try using all the default anchors.

Tiny yolo would have 6 clusters.

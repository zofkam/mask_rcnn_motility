"""
Mask R-CNN
Train on worm images, color splash images and calculate hatch and development test statistics.

Licensed under the MIT License (see LICENSE for details)
Written by Martin Zofka

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 worm.py train --dataset=/path/to/worm/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 worm.py train --dataset=/path/to/worm/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 worm.py train --dataset=/path/to/worm/dataset --weights=imagenet

    # Run detection for an image
    python3 worm.py detect --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Run detection for an image using the last weights you trained
    python3 worm.py detect --weights=last --image=<URL or path to file>

    # inspect the results of the run on a dashboard with the metrics
    tensorboard --logdir ../Mask_RCNN/logs/worm20191111T1914/

"""

if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from collections import OrderedDict
import csv
# import imgaug
from skimage.exposure import match_histograms
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class WormConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "worm"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + (egg/L2/L3)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50#30 #10
    #VALIDATION_STEPS = 1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6 #0.75

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # original (32, 64, 128, 256, 512)

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 400

    # Maximum number of ground truth instances to use in one image (reduces the number of objects taken from the image)
    MAX_GT_INSTANCES = 400

    # Number of ROIs per image to feed to classifier/mask heads
    TRAIN_ROIS_PER_IMAGE = 800

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.8

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.5

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Image mean (RGB)
    MEAN_PIXEL = np.array([164.6, 164.6, 164.6])

############################################################
#  Dataset
############################################################


class WormDataset(utils.Dataset):

    def __init__(self, class_map=None):

        super().__init__(class_map)
        # add translation mapping
        self.translate_value = OrderedDict()

    def load_worm(self, dataset_dir, subset, reference_image=None):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have 3 classes to add.
        self.add_class("worm", 1, "egg")
        self.add_class("worm", 2, "L2")
        self.add_class("worm", 3, "L3")

        # add all the values for mapping
        for record in self.class_info:
            self.translate_value[record['name']] = record['id']

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "annotation_file.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                objects = [s['region_attributes'] for s in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                objects = [s['region_attributes'] for s in a['regions']]

            class_ids = [int(self.translate_value[n['Hco_stage']]) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)

            # If grayscale. Convert to RGB for consistency.
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
            # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                image = image[..., :3]

            if reference_image is not None:
                image = match_histograms(image, reference_image, multichannel=True)

            height, width = image.shape[:2]

            self.add_image(
                "worm",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_ids=class_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "worm":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):

            if p['name'] == 'rect':
                p['all_points_x'] = [p['x'], p['x'] + p['width'], p['x'] + p['width'], p['x']]
                p['all_points_y'] = [p['y'], p['y'], p['y'] + p['height'], p['y'] + p['height']]

            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.array(info['class_ids'], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "worm":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, augmentation=None, reference_image=None):
    """
    Train the model.

    :param model:
    :param augmentation:
    :return:
    """
    epochs = 30

    # Training dataset.
    dataset_train = WormDataset()
    dataset_train.load_worm(args.dataset, "train", reference_image=reference_image)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = WormDataset()
    dataset_val.load_worm(args.dataset, "val", reference_image=reference_image)
    dataset_val.prepare()

    # cycle for
    for i in range(1, epochs):

        # augmenter to modify the images prior to training to avoid overfitting
        augmenter = iaa.SomeOf(2, [
             iaa.Fliplr(0.5),
             iaa.Flipud(0.5),
             iaa.ContrastNormalization((0.75, 1.5)),
             iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=False),
             # set per_channel to False else the color changes from grey to rgb shades
             iaa.Multiply((0.8, 1.2), per_channel=False)
         ], random_order=True,
            random_state=i)

        # The training schedule should be modified based on the tensorboard results
        # it starts with training only the heads for 30 epochs, later we train other layers as well
        # if re-training the worm model and keeping the same categories to detect then we can retrain the
        # all layers directly and lower the learning rate
        print("Training network heads: {}".format(i))
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=i,
                    layers='heads', #'4+', #'all',
                    augmentation=augmenter)


def export_result(out_path,
                  data,
                  data_keys):
    """
    write results to csv file

    :param out_path: path to the file with exported values
    :type out_path: str
    :param data: dictionary containing the extracted values
    :type data: list
    :param data_keys: list of keys used for the header in the export
    :type data_keys: list
    :return:
    """
    with open(out_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_keys)
        writer.writeheader()
        for record in data:
            writer.writerow(record)


def calculate_additional_fields(counts):
    """
    Calculate additional fractions based on the provided values

    :param counts:
    :type dict
    :return:
    """
    # LDT (larval development test) = (egg + L2) / (egg + L2 +L3) * 100
    if (counts.get('egg', 0)+counts.get('L2', 0)+counts.get('L3', 0)) > 0:
        counts['LDT'] = "{0:.2f}".format(
            (counts.get('egg', 0) + counts.get('L2', 0)) /
            (counts.get('egg', 0) + counts.get('L2', 0) + counts.get('L3', 0)) * 100)

    if (counts.get('egg', 0) + counts.get('L2', 0)) > 0:
        # Egg hatch test
        counts['EHT'] = "{0:.2f}".format((counts.get('egg', 0) / (counts.get('egg', 0) + counts.get('L2', 0))) * 100)

    return counts


def detect_files_in_folder(model,
                           object_classes,
                           in_folder,
                           out_folder,
                           save_object_count=False):
    """
    Wrapper for detect_and_color_splash, to process multiple files sequentially

    :param model:
    :param object_classes:
    :param in_folder: path to the folder containing the images to run detection on
    :param out_folder: path to the folder for the output of the detection
    :param save_object_count:
    :return:
    """
    assert out_folder

    output_counts = []

    for entry in sorted(os.scandir(in_folder), key=lambda e: e.name):
        if entry.name.endswith(".jpg") and entry.is_file():
            # detect the objects for the image, save the coloured image and return counts
            file_counts = detect_and_color_splash(model=model,
                                                  object_classes=object_classes,
                                                  image_path=os.path.join(in_folder, entry.name),
                                                  out_folder=out_folder)
            # add the filename
            file_counts['file_name'] = entry.name

            # calculate additional fields
            file_counts = calculate_additional_fields(file_counts)

            # add to list of values
            output_counts.append(file_counts)

    if save_object_count:
        # remove background
        object_classes.remove('BG')

        # add the filename to the keys
        object_classes.append('file_name')

        # add calculated field
        object_classes.append('LDT')
        object_classes.append('EHT')

        export_result(out_path=os.path.join(out_folder, 'object_counts.csv'),
                      data=output_counts,
                      data_keys=object_classes)


def create_colors_per_class_instance(detected_objects):
    """
    Based on the detected classes assign the same color to each class ID and return a list of tuples for
    the display_instances function

    :param detected_objects:
    :return: list of tuples containing the color
    :rtype: list
    """

    # TODO: rewrite to enable more classes if necessary
    red = (1, 0, 0)
    green = (0, 1, 0)
    blue = (0, 0, 1)

    all_colors = {1: red,
                  2: green,
                  3: blue}

    return [all_colors[class_id] for class_id in detected_objects['class_ids']]


def detect_and_color_splash(model,
                            object_classes,
                            out_folder,
                            image_path=None):
    """
    Method to run detection on a video or image and store the output (currently for images only)

    :param model:
    :param object_classes:
    :param out_folder:
    :param image_path:
    :return:
    """
    assert out_folder
    assert image_path

    # Run model detection and generate the color splash effect
    print("Running on {}".format(image_path))
    # Read image
    image = skimage.io.imread(image_path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    # Detect objects
    r = model.detect([image], verbose=1)[0]

    predicted_colors = create_colors_per_class_instance(detected_objects=r)

    # Save image with masks
    visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'],
        # object classes gives me a list of class names
        object_classes, r['scores'],
        show_bbox=False, show_mask=False,
        colors=predicted_colors,
        title="Predictions",
        figsize=(15, 15))
    file_name = "{0}/{1}{2}.png".format(out_folder, image_path.split('/')[-1].split('.')[0], '_predicted')
    plt.savefig(file_name)
    print("Saved to ", file_name)
    # get the number of instances in each image
    values, counts = np.unique(r['class_ids'], return_counts=True)

    values = [object_classes[element] for element in list(values)]

    value_counts = dict(zip(values, list(counts)))

    print("Total counts: {0}".format(value_counts))

    return dict(zip(values, list(counts)))


############################################################
#  Training
############################################################

if __name__ == '__main__':

    # model doesn't save names of classes, so we provide it here again
    # the ORDER is crucial!!!
    CLASS_LIST = ['BG', 'egg', 'L2', 'L3']
    # get the reference image path to match contrast and brightness
    REFERENCE_PATH = '/path/to/file'
    reference_image = None # skimage.io.imread(REFERENCE_PATH)

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect worms.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/worm/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--in_folder', required=False,
                        metavar="path to folder containing images",
                        help='Path of images to run the detection on')
    parser.add_argument('--out_folder', required=False,
                        metavar="path to folder to store detection",
                        help='Path to folder to store detection')
    parser.add_argument('--save_counts', required=False,
                        metavar="save csv with output",
                        help='save csv with output',
                        nargs='?',
                        const=True,
                        default=False)
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image or args.in_folder,\
               "Provide --image or  --in_folder to apply detection"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = WormConfig()
    else:
        class InferenceConfig(WormConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":

        train(model, reference_image=reference_image)
    elif args.command == "detect":
        assert args.out_folder, "Provide path for storing images with detected objects"
        if args.in_folder:
            detect_files_in_folder(model,
                                   object_classes=CLASS_LIST,
                                   in_folder=args.in_folder,
                                   out_folder=args.out_folder,
                                   save_object_count=args.save_counts)

        else:
            detect_and_color_splash(model,
                                    object_classes=CLASS_LIST,
                                    out_folder=args.out_folder,
                                    image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))

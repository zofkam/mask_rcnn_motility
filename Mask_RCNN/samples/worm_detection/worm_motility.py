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
from skimage.exposure import match_histograms
import pandas as pd
import random
import colorsys
from skimage.measure import regionprops
import trackpy as tp
import cv2
import time
#import psutil
import sparse

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
    """
    Configuration for training on the worm  dataset.
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
    STEPS_PER_EPOCH = 50  # 30 #10
    # VALIDATION_STEPS = 1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6  # 0.75

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


############################################################
#  Dataset
############################################################


class WormMotility(object):

    def __init__(self, model_path, config, log_path):
        """
        Initialize model object and load the required model weights

        :param model_path:
        :param config:
        :param log_path:
        """
        # initialize model instance
        self.model = modellib.MaskRCNN(mode="inference", config=config,
                                       model_dir=log_path)
        # load weights
        self.model.load_weights(model_path, by_name=True)

    def process_image(self,
                      image_path,
                      image=None):
        """
        Process a single image/frame

        :return:
        """
        features = []

        if image is None:
            # Read image
            image = skimage.io.imread(image_path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        # Detect objects
        r = self.model.detect([image], verbose=1)[0]

        #r['masks'] = r['masks'].astype(int)

        # cycle over all instance
        for i in range(r['rois'].shape[0]):

            current_mask = r['masks'][:, :, i]
            centroid = np.argwhere(current_mask == 1).sum(0) / (current_mask == 1).sum()

            features.append({'index': i,
                             'x': centroid[0],
                             'y': centroid[1]})

        df_features = pd.DataFrame(features)

        return df_features, r

    @staticmethod
    def link_across_frames(features):
        """
        DataFrame containig the summary statistics for the instances across the individual frames

        :param features:
        :return:
        """
        print('Creating links for detections across frames')

        return tp.link_df(features,
                          search_range=100,
                          neighbor_strategy='KDTree',
                          memory=5)

    def process_motility(self, frame_detections, linked_df):
        """
        Compare all adjacent frames for 2 particles and calculate the IoU

        :param frame_detections:
        :param linked_df:
        :return:
        """

        for worm in linked_df['particle'].unique():
            restricted_df = linked_df.loc[linked_df['particle'] == worm,]

            for i in range(1, max(restricted_df['frame'])+1):
                try:
                    index_t1 = restricted_df.loc[restricted_df['frame'] == i-1, 'index'].values[0]
                    index_t2 = restricted_df.loc[restricted_df['frame'] == i, 'index'].values[0]
                    # todense should be removed if we are no longer passing in sparse arrays
                    result = self.calculate_overlap(x=frame_detections[i-1]['masks'].todense()[:, :, index_t1],
                                                    y=frame_detections[i]['masks'].todense()[:, :, index_t2])
                    linked_df.loc[(linked_df['particle'] == worm) & (linked_df['frame'] == i), 'IoU'] = result
                    print('particle {0} - frame {1}: {2}'.format(worm, i, result))
                except:
                    print('skipping particle {0} - frame {1}'.format(worm, i))
                    pass

        return linked_df

    @staticmethod
    def calculate_overlap(x, y, use_union=True):
        """
        Return the percentage of an overlap between 2 numpy arrays either use percentage to x
        or use an intersect over union

        :param x:
        :param y:
        :param use_union:
        :return:
        """
        assert list(np.unique(x)) == [0, 1]
        assert list(np.unique(y)) == [0, 1]
        assert x.shape == y.shape

        z = x & y
        union = x | y
        overlap = z.sum()

        if use_union:
            return overlap / union.sum()
        else:
            return overlap / x.sum()

    @staticmethod
    def color_image_with_ids(image_path,
                             linked_data,
                             detections,
                             out_path,
                             image=None):
        """
        Color the image with the detections

        :param image_path:
        :param linked_data:
        :param detections:
        :param out_path:
        :param image
        :return:
        """

        if image is None:
            # Read image
            image = skimage.io.imread(image_path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        # only required for sparse matrix
        if type(detections['masks']) == sparse._coo.core.COO:
            detections['masks'] = detections['masks'].todense()

        predicted_colors = create_colors_per_class_instance(detected_objects=detections)

        # Save image with masks
        visualize.display_instances(
            image, detections['rois'], detections['masks'], detections['class_ids'],
            # object classes gives me a list of class names
            ['BG', 'egg', 'L2', 'L3'], detections['scores'],
            show_bbox=False, show_mask=False,
            colors=predicted_colors,
            title="Predictions",
            captions=list(linked_data['particle'].astype(str)),
            figsize=(22, 22))
        file_name = "{0}/{1}{2}.jpg".format(out_path, image_path.split('/')[-1].split('.')[0], '_predicted_with_id')

        plt.savefig(file_name)
        print("Saved to ", file_name)

        # the original value in the dict is converted to dense and causes a memory increase, del to remove
        # as we no longer need the value
        del detections['masks']

        return file_name

    @staticmethod
    def calculate_mask_area(x, y):
        """
        calculate the area of the mask given the set of x and y points

        :param x:
        :type x: np.array
        :param y:
        :type y: np.array
        :return:
        """

        # copied from https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @staticmethod
    def random_colors(n, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / n, 1, brightness) for i in range(n)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    @staticmethod
    def get_region_info(region, index):
        """
        Return required columns based on the regionprop

        :param region:
        :return:
        """
        # major_axis_length, minor_axis_length, filled_area, bbox_area

        # Compute features
        feat_dict = {'y': region.centroid[0],
                     'x': region.centroid[1],
                     'index': index
                     }

        return feat_dict

    def process_video(self, video_path, out_folder):
        """
        Processes the video, creates an output video with detected and linked objects. Also creates 2 output csv files.
        The links csv file contains the IoU values per particle and frame

        :param video_path: Input video path (needs to be .avi format)
        :param out_folder: Folder where to store the outputs of the motility detection
        :return:
        """

        # create empty dataframe
        frames_statistics = pd.DataFrame()

        frame_detections = {}

        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "{0}_processed.avi".format(video_path.split('/')[-1].split('.')[0])

        vwriter = cv2.VideoWriter(os.path.join(out_folder, file_name),
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  20,# fps,
                                  (2200, 2200))# (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)

            # Read next image
            success, image = vcapture.read()

            if count > 0:

                if success:
                    # OpenCV returns images as BGR, convert to RGB
                    image = image[..., ::-1]

                    # process a given frame
                    region_statistics, frame_detection = self.process_image(image_path=None,
                                                                            image=image)

                    # add identification of the frame
                    region_statistics['frame'] = count
                    # add the statistics
                    frames_statistics = frames_statistics.append(region_statistics, ignore_index=True)
                    # convert to sparse matrix
                    frame_detection['masks'] = sparse.COO(frame_detection['masks'])

                    # add the detections
                    frame_detections[count] = frame_detection

            count += 1

            if count == 10:
                success = False

        # link worm instance across frames
        frames_statistics = self.link_across_frames(frames_statistics)

        frames_statistics.sort_index(inplace=True)

        # initialize IoU with default value = -1
        frames_statistics['IoU'] = -1

        #
        frames_statistics = self.process_motility(frame_detections=frame_detections,
                                                  linked_df=frames_statistics)

        frames_statistics.to_csv(os.path.join(out_folder,
                                              '{0}_links.csv'.format(video_path.split('/')[-1].split('.')[0])),
                                 index=False)

        grouped_frames = frames_statistics.groupby('particle').agg({'frame': ['size', 'min']}).reset_index()

        grouped_frames.columns = ['particle', 'total_frames', 'first_frame']

        # only used as guidance for manual counting so that the same ids are used
        # grouped_frames.to_csv(os.path.join(out_folder,
        #                                   '{0}.csv'.format(video_path.split('/')[-1].split('.')[0])),
        #                      index=False)

        # Video capture
        vcapture2 = cv2.VideoCapture(video_path)
        success = True
        count = 0
        while success:

            print("saving frame: ", count)

            # Read next image
            success, image = vcapture2.read()

            if count > 0:

                if success:
                    # OpenCV returns images as BGR, convert to RGB
                    image = image[..., ::-1]

                    marked_image_path = self.color_image_with_ids(
                        # TODO: remove this temporary path which is only used for saving an image to be reloaded again
                        image_path=video_path,
                        linked_data=frames_statistics.loc[frames_statistics['frame'] == count,],
                        detections=frame_detections[count],
                        out_path=out_folder,
                        image=image)

                    # Read image
                    image_out = skimage.io.imread(marked_image_path)

                    # RGB -> BGR to save image to video
                    image_out = image_out[..., ::-1]
                    # Add image to video writer
                    vwriter.write(image_out)

            count += 1

            if count == 10:
                success = False

        vwriter.release()


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
        figsize=(20, 20))
    file_name = "{0}/{1}{2}.png".format(out_folder, image_path.split('/')[-1].split('.')[0], '_predicted')
    plt.savefig(file_name)
    print("Saved to ", file_name)
    # get the number of instances in each image
    values, counts = np.unique(r['class_ids'], return_counts=True)

    values = [object_classes[element] for element in list(values)]

    value_counts = dict(zip(values, list(counts)))

    print("Total counts: {0}".format(value_counts))

    return dict(zip(values, list(counts)))


if __name__ == '__main__':

    # model doesn't save names of classes, so we provide it here again
    # the ORDER is crucial!!!
    CLASS_LIST = ['BG', 'egg', 'L2', 'L3']

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to evaluate worm motility.')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--in_file', required=True,
                        metavar="path to input video file",
                        help='Path to input video file')
    parser.add_argument('--out_folder', required=True,
                        metavar="path to folder to store motility results",
                        help='Path to folder to store motility results')

    args = parser.parse_args()

    start_time = time.time()

    motmodel = WormMotility(model_path=args.weights,
                            config=WormConfig(),
                            log_path=DEFAULT_LOGS_DIR)

    if args.in_file:
        motmodel.process_video(video_path=args.in_file,
                               out_folder=args.out_folder)

    end_time = time.time()

    print('duration in seconds: {0}'.format(end_time-start_time))
# General

The model for motility detection is based on the Matterport implementation of Mask R-CNN ([https://github.com/matterport/Mask_RCNN]()). To set up the `python3.7` environment it is necessary to install the packages listed in `requirements.txt` with the specified versions.

For using the repository for motility detection you will require to download the model along with the motility videos from [https://owncloud.cesnet.cz/index.php/s/HHLQ6s42UcowMHo](). 

In case you want to re-train the model you also have to download the annotated images described in the below part `Image annotation`.

## Data

### Model

The model `mask_rcnn_worm.h5` is stored in the repository (`Mask R-CNN Model` folder) and needs to be copied to `Mask_RCNN/logs` in a folder with the `wormYYYYMMDDTHHMM` format, when using the last weights.

### Image annotation

The images have been annotated using VGG Image Annotator version 2.0.8. To use the annotated files download them from the `Annotation Data` folder and place them in the datasets folder of this repository. The annotations consist of two json files, one for validation and one for training and the associated images. This step is only required if you want to use the annotated images for training the model.

### Motility videos

The motility videos are available in the folder `Motility Videos`. They always have a prefix `xxx` which is numeric and contains the motility group defining the percentage of alive worms.

## Mask R-CNN scripts

The `mask_rcnn_worm.h5` model is being used by 2 files. The first `worm.py`, which is used for training the model and detection in images. And `worm_motility.py`, which is used for detecting motility in videos.

### Model training


The input arguments are described in the `worm.py` file. To start training a model use the input argument **train**:


`worm.py train --weights=last --dataset=/path/to/datasets/`

To run the object on an image or folder use the input argument **detect**:

`worm.py detect --weights=/path/to_model/mask_rcnn.h5 --in_folder=/path/to/images --out_folder=/output/path/`

To test that the model works correctly we can use the image `sample.jpg` located in `Mask_RCNN/samples/worm_detection`. The output of running the prediction for comparison is `sample_predicted.png`. 


### Motility

To run the motility script execute:

`worm_motility.py --weights=/path/to_model/mask_rcnn.h5
--in_file=/path/to/file/0z001.avi --out_folder=/output/path/`

The output consists of a csv file containing the detected instances in each frame and an annotated video for visually checking the output.

## WF-NTP

The installation and scripts for WF-NTP, were taken from [https://github.com/koopmanm/WF-NTPv2.0](). Slight modifications have been done to `settings.py` and `multiwormtracker.new.py`, which are in the `WF-NTP` folder of this repository.

## Licensing

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
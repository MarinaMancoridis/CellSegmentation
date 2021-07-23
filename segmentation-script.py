import sys
import argparse
import skimage
from skimage.io import imread
from skimage.exposure import rescale_intensity, equalize_adapthist
import PIL.Image
from PIL import Image
import os
import numpy as np
import deepcell
from deepcell.model_zoo.panopticnet import PanopticNet
from deepcell_toolbox.deep_watershed import deep_watershed
import random # choose random tile in test image
import math

"""
python3 segmentation-script.py --heImageName heImageName --heImageDir heImageDir

Example:
python3 segmentation-script.py --heImageName GSM4284316_P2_ST_rep1.jpeg --heImageDir ./he_samples/

This code takes as an input an H&E slide image tile and employs a state of the art deep learning model to
count the total number of cells within the image and save (1) an inner distances map images, (2) an outer
distance map images, and (3) an instance mask images to the computer's hard drive. It saves predictions
on every 256x256 pixel tile within the input image as if each tile were its own image.
"""

# defining global variables
PIL.Image.MAX_IMAGE_PIXELS = None # either use none or replace this with pixel size of largest image, for very large images

# this function takes as input the path to an H&E image file and runs the pretrained deep learning model on it to
# (1) output the total count of segmented cells and (2) save inner distance, outer distance, and instance mask images
def run_predictions(heImageName, heImageDir):
    print("1. starting preprocessing image step... ")
    im = preprocess_image(heImageName, heImageDir)

    print("2. finished preprocessing the image... starting loading the model")
    prediction_model = loadModel()

    print("3. finished loading model... starting prediction making")
    test_images_msk, numTiles, numTilesPerCol = makePredictions(prediction_model, im)  

    print("4. finished prediction making... starting mask preparation")
    masks_msk_without_border, masks_msk_with_border = prepareMasks(test_images_msk)

    print("5. finished mask preparation... starting process of saving relevant images to hard drive")
    saveImages(masks_msk_without_border, masks_msk_with_border, test_images_msk, numTiles, numTilesPerCol)

    print("6. finished process of saving relevant images to hard drive")

# this function preprocess the H&E image by reading it as a numpy array and expanding the dimentions of that array
def preprocess_image(heImageName, heImageDir):
    try:
        im = imread(os.path.join(heImageDir, heImageName))
        im = np.expand_dims(im, axis=0)
        return preprocess_he(im)
    except (IOError, SyntaxError, ValueError) as e:
        print('There was an issue preprocessing your file: ', os.path.join(im_directory, filename))

# this helper function applies addition H&E image preprocessing to each image
def preprocess_he(img):
    new_img = []
    for b in img:
        new_b = rescale_intensity(b, out_range='float')
        new_img.append(new_b)
    new_img = np.stack(new_img, axis=0)
    return new_img

# this function creates a DeepCell PanopticNet model and loads it with pretrained H&E image weights
def loadModel():
    sampleShape = (256, 256, 3) # model was trained on 256x256 image tiles w/ 3 channels
    prediction_model = PanopticNet(
        backbone='resnet50',
        norm_method='whole_image',
        num_semantic_classes=[1, 1], # inner distance, outer distance
        input_shape=sampleShape
    )
    model_name = 'pannuc_panopticnet'
    model_path = '{}.h5'.format(model_name)
    prediction_model.load_weights(model_path, by_name=True)
    return prediction_model

# this function takes in the model and image, tiles the image in 256x256 pixel squares, and runs the tiles through the model
def makePredictions(prediction_model, im):
    y = im
    # the last row / col of image tiles overlap the one before the last if the og. image side lengths are not divisible by 256
    numTilesPerCol = math.ceil(np.array(im).squeeze().shape[1] / 256.0) 

    reshaped_im = deepcell.utils.data_utils.reshape_matrix(im, y, reshape_size=256)
    test_images_msk = prediction_model.predict(reshaped_im[0])
    return test_images_msk, reshaped_im[0].shape[0], numTilesPerCol

# this function prepares the segmentation masks from the predictions made by the trained model
def prepareMasks(test_images_msk):
    masks_msk_without_border = deep_watershed(
        test_images_msk,
        min_distance=10,
        detection_threshold=0.1,
        distance_threshold=0.01,
        exclude_border=True,
        small_objects_threshold=0)

    masks_msk_with_border = deep_watershed(
        test_images_msk,
        min_distance=10,
        detection_threshold=0.1,
        distance_threshold=0.01,
        exclude_border=False,
        small_objects_threshold=0)
    return masks_msk_without_border, masks_msk_with_border

# this function saves all 256x256 pixel tiles in input image to the computer's harddrive, and prints/returns total number of cells counted
# across all tiles. it alternates saving tiles with and without counting cells on the border to ensure no cell overcounting
def saveImages(masks_msk_without_border, masks_msk_with_border, test_images_msk, numTiles, numTilesPerCol):
    totalCount = 0
    for i in range (0, numTiles):
        # index = random.randint(0, numTiles) # use this and remove loop if you want to run predictions on single, random tile within image
        index = i
        inner_distance_msk = test_images_msk[0]
        outer_distance_msk = test_images_msk[1]
        borderIncluded = True

        # alternates including border and not including border in tiles to ensure no cell overcounting
        if (math.floor(i / numTilesPerCol) % 2) == 0: # if in an odd row
            if ((i % numTilesPerCol) % 2) == 0: # if in an odd col
                borderIncluded = True
            else: # if in an even col
                borderIncluded = False
        else: # if in an even row
            if ((i % numTilesPerCol) % 2) == 0: # if in an odd col
                borderIncluded = True
            else: # if in an even col
                borderIncluded = False
        
        # saving inner distance image to computer's hard drive
        path = r'/Users/marinamancoridis/qsure_internship/program_output_images/'
        im = Image.fromarray((inner_distance_msk[index, ..., 0] * 255).astype(np.uint8))
        im_path = os.path.join(path, "inner_distance_images/id%s.jpeg" %index);
        im.save(im_path)

        # saving outer distance image to computer's hard drive
        im = Image.fromarray((outer_distance_msk[index, ..., 0] * 255).astype(np.uint8))
        im_path = os.path.join(path, "outer_distance_images/od%s.jpeg" %index);
        im.save(im_path)

        # saving instance mask image to computer's hard drive
        if (borderIncluded):
            numCellsInTile = np.amax(masks_msk_with_border[index, ...])
            im = Image.fromarray((masks_msk_with_border[index, ...] * 255).astype(np.uint8))
        else:
            numCellsInTile = np.amax(masks_msk_without_border[index, ...])
            im = Image.fromarray((masks_msk_without_border[index, ...] * 255).astype(np.uint8))

        totalCount = totalCount + numCellsInTile
        print("Total number of cells counted in tile %i: %i" % (index, numCellsInTile))
        im_path = os.path.join(path, "instance_mask_images/im%s.jpeg" %index);
        im.save(im_path)

    print("Number of cells in whole image: ", totalCount)
    return(totalCount)

# the main function, consisting of an argument parser and call to the run_predictions function
def main(argv):
    parser = argparse.ArgumentParser('Getting input')

    parser.add_argument('--heImageName', dest = 'heImageName', help = "name of image file", type = str)
    parser.add_argument('--heImageDir', dest = 'heImageDir', help = "path to image directory", type = str)

    args = parser.parse_args()
    heImageName = args.heImageName
    heImageDir = args.heImageDir

    print("Path to input image: " + heImageDir + heImageName)
    run_predictions(heImageName, heImageDir)

if __name__ == "__main__":
    main(sys.argv[1:])

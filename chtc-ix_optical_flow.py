#!/Users/njwheeler/software/miniconda3/bin/python

import argparse
import string
import itertools
import cv2
import numpy as np
# from matplotlib import pyplot as plt
from pathlib import Path
# import struct
import imageio
from PIL import Image
from skimage.filters import threshold_otsu
# import sys
# import pathlib
from scipy import ndimage
from datetime import datetime


def create_plate(plate):
    '''
    Create a list with all the well names for a given plate format.
    '''
    # create a list of the possible well names
    wells = []
    if plate == 96:
        rows = string.ascii_uppercase[:8]
        columns = list(range(1, 13))
        columns = [str(column).zfill(2) for column in columns]
        wells = list(itertools.product(rows, columns))
        wells = [''.join(well) for well in wells]
    print(wells)

    return wells


def organize_arrays(input, output, work_path, plate, frames, reorganize):
    '''
    Create a list of lists where each internal list is all the paths to all the
    images for a given well, and the entire list is the plate.
    '''

    # initialize a list that will contain the arrays/videos
    vid_array = []

    # initialize a list that will contain lists of paths for each well
    plate_paths = []

    for well in plate:
        print("Getting the paths for well {}".format(well))
        # initialize a list that will contain paths to each frame
        well_paths = []
        # append the path pointing to each frame from each well
        for frame in range(1, frames + 1):
            dir = Path.home().joinpath(input)
            name = dir.name.split("_")[0]
            path = dir.joinpath(str(dir), "TimePoint_" +
                                str(frame), name + "_" + well + ".TIF")
            well_paths.append(path)

        # final list of lists with wells and paths
        plate_paths.append(well_paths)

        try:
            # get the dimensions of the images
            first_frame = Image.open(str(well_paths[0]))
            height, width = np.array(first_frame).shape

            # empty array with the correct shape of the final video
            well_array = np.zeros((frames, height, width))
            counter = 0
            print("Reading images for well {}".format(well))
            for frame in well_paths:
                image = Image.open(str(frame))
                matrix = np.array(image)
                well_array[counter] = matrix

                if reorganize:
                    counter_str = str(counter).zfill(2)
                    dir = Path.home().joinpath(work_path)
                    name = Path.home().joinpath(input)
                    name = name.name.split("_")[0]
                    outpath = dir.joinpath(name + "_" + well + "_" + counter_str + ".tiff")
                    cv2.imwrite(str(outpath), matrix)

            vid_array.append(well_array)

            counter += 1

        except FileNotFoundError:
            print("Well {} not found. Moving to next well.".format(well))
            counter += 1

    return vid_array


def dense_flow(well, array, input, output):

    start_time = datetime.now()
    print("Starting optical flow analysis on {}.".format(well))

    length, width, height = array.shape

    vid_array = np.zeros((length, height, width))

    # initialize emtpy array of video length minus one (or, the length of the dense flow output)
    all_mag = np.zeros((length - 1, height, width))
    count = 0
    frame1 = array[count]

    while(1):
        if count < length - 1:
            frame1 = array[count].astype('uint16')
            frame1 = frame1.astype('uint8')
            # frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2 = array[count + 1].astype('uint16')
            frame2 = frame2.astype('uint8')
            # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            vid_array[count + 1] = frame2

            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3,
                                                15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # replace proper frame with the magnitude of the flow between prvs and next frames
            all_mag[count] = mag
            count += 1

        else:
            break

    sum = np.sum(all_mag, axis=0)
    total_sum = np.sum(sum)

    # filename = well + '_sum' + '.png'
    dir = Path.home().joinpath(output)
    name = Path.home().joinpath(input)
    name = name.name.split("_")[0]
    outpath = dir.joinpath(name + "_" + well + '_sum' + ".tiff")

    # write to png
    # uint_sum = imageio.core.image_as_uint(sum)
    # imageio.imwrite(outpath, sum)
    print(str(outpath))
    cv2.imwrite(str(outpath), sum.astype('uint16'))

    print("Optical flow anlaysis completed. Analysis took {}".format(
        datetime.now() - start_time))

    print(total_sum)
    return vid_array, total_sum


def segment_worms(output, array, well):

    start_time = datetime.now()
    print("Starting normalization calculations for {}.".format(well))

    # numpy equivalent to Fiji z-project max intensity
    # max_intensity = np.amax(array, axis=0)
    med_intesity = np.median(array, axis=0)

    print("Segmenting 5th frame...")
    # denoise (I think erroring on subsampled data, will reincorporate later)
    # dst = cv2.fastNlMeansDenoisingMulti(np.uint8(array), 5, 5, None, 4, 7, 21)
    difference = abs(np.subtract(array, med_intesity))
    blur = ndimage.filters.gaussian_filter(difference, 1.5)
    threshold = threshold_otsu(blur)
    binary = blur > threshold

    # filename = well.stem + '_max' + '.png'
    # max_png = output.joinpath(filename)
    # imageio.imwrite(max_png, max_intensity)
    filename = well + '_med' + '.png'
    med_png = Path.home.joinpath(filename)
    imageio.imwrite(med_png, med_intesity)
    filename = well + '_diff' + '.png'
    diff_png = Path.home.joinpath(filename)
    imageio.imwrite(diff_png, difference)
    filename = well + '_blur' + '.png'
    blur_png = Path.home.joinpath(filename)
    imageio.imwrite(blur_png, blur)
    filename = well + '_binary' + '.png'
    binary_png = Path.home.joinpath(filename)
    imageio.imwrite(binary_png, binary.astype(int))

    print("Calculating normalization factor.")
    mass = np.sum(binary)
    print("Normalization factor calculation completed. Calculation took {}".
          format(datetime.now() - start_time))

    return mass

if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description='To be determined...')

    # required arguments
    parser.add_argument('input_directory',
                        help='A path to a directory containing subdirectories \
                        filled with TIFF files (i.e. Plate_63/TimePoint_1,    \
                        etc.)')

    parser.add_argument('output_directory',
                        help='A path to the output directory.')

    parser.add_argument('work_directory',
                        help='A path to the work directory.')

    parser.add_argument('rows', type=int,
                        help='The number of rows in the imaging plate.')

    parser.add_argument('columns', type=int,
                        help='The number of columns in the imaging plate.')

    parser.add_argument('time_points', type=int,
                        help='The number of frames recorded.')

    parser.add_argument('--reorganize', dest='reorganize', action='store_true',
                        default=False,
                        help='Invoke if you want to save the TIFF files       \
                        organized by well instead of time point (default is   \
                        to not reorganize).')

    args = parser.parse_args()

    # create the plate shape
    plate_format = args.rows * args.columns

    # create a list of all the possible wells in the plate
    plate = create_plate(plate_format)

    # re-organize the input TIFFs so that each well has its own array
    vid_array = organize_arrays(
        args.input_directory,
        args.output_directory,
        args.work_directory,
        plate,
        args.time_points,
        args.reorganize)

    print(vid_array.shape)

    # vid_array, motility = dense_flow(well, well_array, input, output)
    # normalization_factor = segment_worms(output, well_array, well)

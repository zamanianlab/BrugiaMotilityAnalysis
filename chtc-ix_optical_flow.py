#!/Users/njwheeler/software/miniconda3/bin/python

import argparse
import string
import itertools
import cv2
# import csv
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import imageio
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.transform import rescale
from skimage import filters
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

    # initialize a dict that will contain the arrays/videos
    vid_dict = {}

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
                    plate_name = Path.home().joinpath(input)
                    plate_name = plate_name.name.split("_")[0]
                    dir.joinpath(well).mkdir(parents=True, exist_ok=True)
                    outpath = dir.joinpath(well, plate_name + "_" + well + "_" + counter_str + ".tiff")
                    cv2.imwrite(str(outpath), matrix)

                counter += 1

            # add to the dict with the well as the key and the array as the value
            vid_dict[well] = well_array

            # saving as 16 bit AVI not currently working
            # fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            # outvid = dir.joinpath(well, plate_name + "_" + well + ".avi")
            # out = cv2.VideoWriter(str(outvid), fourcc, 4, (height,  width), False)
            # for frame in well_array:
            #     frame = frame.astype('uint8')
            #     out.write(frame)

        except FileNotFoundError:
            print("Well {} not found. Moving to next well.".format(well))
            counter += 1

    return vid_dict


def dense_flow(vid_dict, input, output):
    '''
    Uses Farneback's algorithm to calculate optical flow for each well. To get
    a single motility values, the magnitude of the flow is summed across each
    frame, and then again for the entire array.
    '''

    # initialize a dict that will contain the flow sums and one that will contain images
    sum_dict = {}
    flow_dict = {}

    for well in vid_dict:
        start_time = datetime.now()
        print("Starting optical flow analysis on {}.".format(well))

        array = vid_dict[well]

        length, width, height = array.shape

        # initialize emtpy array of video length minus one (or, the length of the dense flow output)
        all_mag = np.zeros((length - 1, height, width))
        count = 0
        frame1 = array[count]

        while(1):
            if count < length - 1:
                frame1 = array[count].astype('uint16')
                frame2 = array[count + 1].astype('uint16')

                flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3,
                                                    15, 3, 5, 1.2, 0)

                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                frame1 = frame2

                # replace proper frame with the magnitude of the flow between prvs and next frames
                all_mag[count] = mag
                count += 1

            else:
                break

        sum = np.sum(all_mag, axis=0)
        total_sum = np.sum(sum)
        sum_dict[well] = total_sum
        flow_dict[well] = sum

        # filename = well + '_sum' + '.png'
        dir = Path.home().joinpath(output)
        name = Path.home().joinpath(input)
        name = name.name.split("_")[0]
        outpath = dir.joinpath(name + "_" + well + '_flow' + ".tiff")

        # write to png
        print(str(outpath))
        cv2.imwrite(str(outpath), sum.astype('uint16'))

        print("Optical flow anlaysis completed. Analysis took {}".format(
            datetime.now() - start_time))

        print(total_sum)

    return sum_dict, flow_dict


def segment_worms(vid_dict, input, output):
    '''
    Segments worms to use for downstream normalization.
    '''

    # initialize a dict that will contain the flow sums
    area_dict = {}

    for well in vid_dict:
        start_time = datetime.now()
        print("Starting normalization calculation for {}.".format(well))

        array = vid_dict[well]

        print("Segmenting 5th frame...")

        # sobel edge detection
        sobel = filters.sobel(array[4])

        # gaussian blur
        blur = ndimage.filters.gaussian_filter(sobel, 1.5)

        # set threshold, make binary
        threshold = threshold_otsu(blur)
        binary = blur > threshold

        dir = Path.home().joinpath(output)
        name = Path.home().joinpath(input)
        name = name.name.split("_")[0]

        sobel_png = dir.joinpath(name + "_" + well + '_edge' + ".png")
        imageio.imwrite(sobel_png, sobel)

        blur_png = dir.joinpath(name + "_" + well + '_blur' + ".png")
        imageio.imwrite(blur_png, blur)

        bin_png = dir.joinpath(name + "_" + well + '_bin' + ".png")
        imageio.imwrite(bin_png, binary.astype(int))

        print("Calculating normalization factor.")

        # the area is the sum of all the white pixels (1.0)
        area = np.sum(binary)
        area_dict[well] = area
        print("Normalization factor calculation completed. Calculation took {} \
              seconds.".format(datetime.now() - start_time))

    return area_dict


def wrap_up(sum_dict, area_dict, input, output):
    '''
    Takes dictionaries of values and writes them to a CSV.
    '''

    # dicts = [sum_dict, area_dict]
    final_dict = defaultdict(list)
    for d in sum_dict, area_dict:
        for key, value in d.items():
            final_dict[key].append(value)

    dir = Path.home().joinpath(output)
    name = Path.home().joinpath(input)
    name = name.name.split("_")[0]
    outfile = dir.joinpath(name + '_data' + ".csv")

    df = pd.DataFrame(final_dict).transpose()
    df.index.name = 'well'
    df.reset_index(inplace=True)
    df.rename(columns={0: 'optical_flow', 1: 'worm_area'}).to_csv(outfile, index=False)


def thumbnails(dict, rows, cols, input, output):
    '''
    Takes a dict that contains a video, rescales into thumbnails, and pastes
    into the structure of the plate.
    '''

    thumbs = {}

    for well, image in dict.items():
        # rescale the imaging without anti-aliasing
        rescaled = rescale(image, 0.125, anti_aliasing=True)
        # normalize to 0-255
        rescaled_norm = cv2.normalize(src=rescaled, dst=None, alpha=0,
                                      beta=255, norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_8U)
        thumbs[well] = rescaled_norm

    # 0.125 of the 4X ImageXpress image is 256 x 256 pixels
    height = rows * 256
    width = cols * 256

    new_im = Image.new('L', (width, height))

    for well, thumb in thumbs.items():
        # row letters can be converted to integers with ord() and subtracting a constant
        row = int(ord(well[:1]) - 64)
        col = int(well[1:].strip())
        # print(well, row, col)
        new_im.paste(Image.fromarray(thumb), ((col - 1) * 256, (row - 1) * 256))

    dir = Path.home().joinpath(output)
    name = Path.home().joinpath(input)
    name = name.name.split("_")[0]
    outfile = dir.joinpath(name + '_thumbs' + ".png")

    new_im.save(outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='To be determined...')

    # required positional arguments
    parser.add_argument('input_directory',
                        help='A path to a directory containing subdirectories \
                        filled with TIFF files (i.e. 20201118-p01-MZ_172/     \
                        TimePoint_1, etc.)')

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

    # optional flags
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
    vid_dict = organize_arrays(
        args.input_directory,
        args.output_directory,
        args.work_directory,
        plate,
        args.time_points,
        args.reorganize)

    motility, flow_dict = dense_flow(
        vid_dict,
        args.input_directory,
        args.output_directory)

    # normalization_factor = segment_worms(
    #     vid_dict,
    #     args.input_directory,
    #     args.output_directory)

    # wrap_up(
    #     motility,
    #     normalization_factor,
    #     args.input_directory,
    #     args.output_directory)

    thumbnails(
        flow_dict,
        args.rows,
        args.columns,
        args.input_directory,
        args.output_directory)

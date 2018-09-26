# import cv2
import numpy as np
# from scipy import ndimage
from pathlib import Path
import imageio
# import matplotlib.pyplot as plt
# from skimage.filters import threshold_local
# from skimage.filters import threshold_otsu
# from skimage import morphology
from datetime import datetime
import csv
import argparse


def optical_flow(wd, flow):

    start_time = datetime.now()
    print("Started reading optical flow analysis for {}.".format(flow))

    all_mag = imageio.imread(flow)
    all_mag = all_mag .astype('float64')

    # print("all_mag shape is: {}".format(all_mag.shape))
    sum = np.sum(all_mag, axis=0)
    total_sum = np.sum(sum)

    print("Optical flow anlaysis completed. Analysis took {}".format(
        datetime.now() - start_time))

    return total_sum


def segment_worms(wd, segment):

    start_time = datetime.now()
    print("Started reading normalization calculations for {}.".format(segment))

    dilate = imageio.imread(segment)
    print("Calculating normalization factor.")
    dilate = dilate.astype('float64')
    mass = np.sum(dilate)
    print("Normalization factor calculation completed. Calculation took {}".
          format(datetime.now() - start_time))

    return mass


def wrap_up(well, outfile, motility, normalization_factor):

    normalized_motility = motility / normalization_factor
    with open(str(outfile), 'a') as of:
        writer = csv.writer(of, delimiter=',')
        writer.writerow([well, motility,
                         normalization_factor, normalized_motility])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process videos and analyze motility.")
    parser.add_argument('directory', type=str,
                        help='directory of videos to be analyzed')
    args = parser.parse_args()
    dir = args.directory

    wd = Path.home().joinpath(dir)
    outfile = wd.joinpath(str(dir) + wd.name + '.csv')
    with open(str(outfile), 'w') as of:
        writer = csv.writer(of, delimiter=',')
        print("Writing output file to {}".format(of))
        writer.writerow(["Well", "Total.Motility",
                         "Normalization.Factor", "Normalized.Motility"])

    for sum in wd.rglob('*_sum.png'):
        name = sum.name
        well, ext = name.split("_")
        # flow = wd.joinpath(sum)
        motility = optical_flow(wd, sum)
        segment = wd.joinpath(well + "_segment.png")
        normalization_factor = segment_worms(wd, segment)
        wrap_up(well, outfile, motility, normalization_factor)
        print("Video:", well, "Total motility:", motility,
              "Normalization factor:", normalization_factor,
              "Normalized motility:", motility / normalization_factor)

import cv2
import numpy as np
from scipy import ndimage
from pathlib import Path
import imageio
# import matplotlib.pyplot as plt
# from skimage.filters import threshold_local
from skimage.filters import threshold_otsu
from skimage import morphology
from datetime import datetime
import csv
import argparse


def optical_flow(wd, vid):

    start_time = datetime.now()
    print("Starting optical flow analysis on {}.".format(vid))

    cap = cv2.VideoCapture(str(vid))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vid_array = np.zeros((length, height, width))

    # read first frame
    ret, frame1 = cap.read()
    # convert to gray scale
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    vid_array[0] = prvs

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # avi = cv2.VideoWriter('A0003_OF_15.avi', fourcc, 7.0, (1920, 1080))

    # txt = open('A0001_OF_15.txt', 'ab')
    # bin = open('A0001_OF_15.bin', 'ab')
    # npy = open('A0001_OF_15.npy', 'ab')

    # initialize emtpy array of video length minus one (or, the length of the
    # dense flow output)
    all_mag = np.zeros((length - 1, height, width))
    # print(all_mag.shape)
    count = 0

    while(1):
        if count < length - 1:
            ret, frame2 = cap.read()
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            vid_array[count + 1] = next

            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15,
                                                3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # write image to screen (for testing)
            # k = cv2.waitKey(7) & 0xff
            # if k == 27:
            #     break
            # elif k == ord('s'):
            #     cv2.imwrite('opticalfb.png', frame2)
            #     cv2.imwrite('opticalhsv.png', rgb)
            prvs = next

            # replace proper frame with the magnitude of the flow between prvs
            # and next frames
            all_mag[count] = mag
            count += 1
            # for testing
            # if count > 4:
            #     break
        else:
            break

    # print("all_mag shape is: {}".format(all_mag.shape))
    sum = np.sum(all_mag, axis=0)
    total_sum = np.sum(sum)
    # print("shape of sum is: {}".format(sum.shape))
    # print("total movement is: {}".format(total_sum))
    # print("the lowest movement value in sum is: {}".format(np.amin(sum)))
    # print("the maximum movement value in sum is: {}".format(np.amax(sum)))

    filename = vid.stem + '_sum' + '.png'
    sum_png = wd.joinpath(filename)

    # write to png
    # uint_sum = imageio.core.image_as_uint(sum)
    imageio.imwrite(sum_png, sum)

    cap.release()
    cv2.destroyAllWindows()
    print("Optical flow anlaysis completed. Analysis took {}".format(
        datetime.now() - start_time))

    return vid_array, total_sum


def segment_worms(wd, array):

    start_time = datetime.now()
    print("Starting normalization calculations for {}.".format(vid))

    # capture = cv2.VideoCapture(str(vid))
    # length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # # for testing
    # # length = 25
    # width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #
    # array = np.zeros((length, height, width))
    # count = 0
    # print("Reading video...")
    # while(capture.isOpened()):
    #     if count < length:
    #         ret, frame = capture.read()
    #         try:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #             array[count] = frame
    #             count += 1
    #             # for testing
    #             # if count == 25:
    #             #     print("Only testing {} frames".format(count))
    #             #     break
    #         except:
    #             pass
    #     else:
    #         break

    # numpy equivalent to Fiji z-project max intensity
    max_intensity = np.amax(array, axis=0)

    first_frame = array[0]
    print("Segmenting first frame...")
    difference = abs(np.subtract(first_frame, max_intensity))
    blur = ndimage.filters.gaussian_filter(difference, 1.5)
    threshold = threshold_otsu(blur)
    binary = blur > threshold
    struct = ndimage.generate_binary_structure(2, 2)
    dilate = morphology.binary_dilation(binary, struct)

    filename = vid.stem + '_max' + '.png'
    max_png = wd.joinpath(filename)
    imageio.imwrite(max_png, max_intensity)
    filename = vid.stem + '_diff' + '.png'
    diff_png = wd.joinpath(filename)
    imageio.imwrite(diff_png, difference)
    filename = vid.stem + '_blur' + '.png'
    blur_png = wd.joinpath(filename)
    imageio.imwrite(blur_png, blur)
    filename = vid.stem + '_binary' + '.png'
    binary_png = wd.joinpath(filename)
    imageio.imwrite(binary_png, binary)
    filename = vid.stem + '_segment' + '.png'
    segment_png = wd.joinpath(filename)
    imageio.imwrite(segment_png, dilate)

    print("Calculating normalization factor.")
    mass = np.sum(dilate)
    print("Normalization factor calculation completed. Calculation took {}".
          format(datetime.now() - start_time))

    return mass


def wrap_up(vid, outfile, motility, normalization_factor):

    normalized_motility = motility / normalization_factor
    with open(str(outfile), 'a') as of:
        writer = csv.writer(of, delimiter=',')
        writer.writerow([vid.stem, motility,
                         normalization_factor, normalized_motility])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process videos and analyze motility.")
    parser.add_argument('directory', type=str,
                        help='directory of videos to be analyzed')
    args = parser.parse_args()
    dir = args.directory

    wd = Path.home().joinpath('Desktop', 'temp_testing', dir)
    filename = str(dir) + '.csv'
    outfile = wd.joinpath(filename)
    with open(str(outfile), 'w') as of:
        writer = csv.writer(of, delimiter=',')
        writer.writerow(["Well", "Total.Motility",
                         "Normalization.Factor", "Normalized.Motility"])

    for file in wd.rglob('*.avi'):
        vid = wd.joinpath(file)
        vid_array, motility = optical_flow(wd, vid)
        normalization_factor = segment_worms(wd, vid_array)
        wrap_up(vid, outfile, motility, normalization_factor)
        print("Video:", file, "Total motility:", motility,
              "Normalization factor:", normalization_factor,
              "Normalized motility:", motility / normalization_factor)

# TODO: fix so that the videos don't have to be manually converted to grayscale
# <NJW 2018-05-07>

# TODO: better verbosity
# <NJW 2018-05-07>

# TODO: write out segmented image, for troubleshooting
# <NJW 2018-05-07>
# DONE

# TODO: write out final calculations to file
# <NJW 2018-05-07
# DONE

# TODO: merge loops in segment_worms()

# TODO: error control for improperly segmented videos

# TODO: normalize only by first segmented frame, not by average across video

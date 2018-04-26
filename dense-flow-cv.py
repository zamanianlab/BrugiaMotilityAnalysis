import cv2
import numpy as np
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
# from skimage.filters import threshold_otsu


def optical_flow(vid):

    cap = cv2.VideoCapture(str(vid))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # read first frame
    ret, frame1 = cap.read()
    # convert to gray scale
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    avi = cv2.VideoWriter('A0003_OF_15.avi', fourcc, 7.0, (1920, 1080))

    # txt = open('A0001_OF_15.txt', 'ab')
    # bin = open('A0001_OF_15.bin', 'ab')
    # npy = open('A0001_OF_15.npy', 'ab')

    # initialize emtpy array of video length minus one (or, the length of the
    # dense flow output)
    all_mag = np.zeros((length - 1, height, width))
    # print(all_mag.shape)
    count = 0

    while(1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5,
                                            1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        print(
            "Finished with frame {} which has shape: {}".format(
                count, mag.shape))

        # write image to screen (for testing)
        # k = cv2.waitKey(7) & 0xff
        # if k == 27:
        #     break
        # elif k == ord('s'):
        #     cv2.imwrite('opticalfb.png', frame2)
        #     cv2.imwrite('opticalhsv.png', rgb)
        prvs = next
        if count < length - 1:
            # replace proper frame with the magnitude of the flow between prvs
            # and next frames
            all_mag[count] = mag
            count += 1
            # elif count == length - 1:
            if count > 4:
                break

        print("all_mag shape is: {}".format(all_mag.shape))
        sum = np.sum(all_mag, axis=0)
        mean = np.mean(all_mag, axis=0)
        print("shape of sum is: {}".format(sum.shape))
        print("total movement is: {}".format(np.sum(sum)))
        print("the lowest movement value in sum is: {}".format(np.amin(sum)))
        print("the maximum movement value in sum is: {}".format(np.amax(sum)))

        txt = wd.joinpath('A0003_OF_15_sum.txt')
        sum_png = wd.joinpath('A0003_OF_15_sum.png')
        mean_png = wd.joinpath('A0003_OF_15_mean.png')

        # write to text
        with txt.open(mode='wb') as file:
            np.savetxt(file, sum, fmt='%0.2e')

        # write to png
        # uint_sum = imageio.core.image_as_uint(sum)
        imageio.imwrite(sum_png, sum)
        imageio.imwrite(mean_png, mean)

        cap.release()
        cv2.destroyAllWindows()


def segment_worms(vid):
    capture = cv2.VideoCapture(str(vid))
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    array = np.zeros((length, height, width))

    count = 0

    while(capture.isOpened()):
        ret, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if count < length:
            array[count] = frame
            count += 1
        else:
            break

    # numpy equivalent to Fiji z-project max intensity
    max_intensity = np.amax(array, axis=0)

    # calculate difference between each frame and max intensity
    difference_array = np.zeros((length, height, width))
    threshold_array = np.zeros((length, height, width))
    binary_array = np.zeros((length, height, width))
    count = 0
    # loop to perform segmentation for every frame (uncomment when finished
    # testing)
    for frame in array:
        difference = abs(np.subtract(frame, max_intensity))
        difference_array[count] = difference
        block_size = 21
        scikit_threshold = threshold_local(difference, block_size, offset=-160)
        threshold_array[count] = scikit_threshold
        # binary = difference >= threshold
        # binary_array[count] = binary
        count += 1

    # sckit-image method for local thresholding
    # difference = abs(np.subtract(array[0], max_intensity))
    # block_size = 21
    # threshold = threshold_local(difference, block_size, offset=-160)
    # binary = difference >= threshold

    # opencv2 watershed threshold
    for threshold in threshold_array:
        dist = ndimage.distance_transform_edt(threshold)
        localMax = peak_local_max(dist, indices=False, min_distance=20,
                                  labels=threshold)
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-dist, markers, mask=threshold)
        print("[INFO] {} unique segments found".format(
            len(np.unique(labels)) - 1))

    # fig, axes = plt.subplots(nrows=6, figsize=(7, 7))
    # ax = axes.ravel()
    # plt.gray()
    #
    # ax[0].imshow(max_intensity)
    # ax[0].set_title('Max Project')
    #
    # ax[1].imshow(array[0])
    # ax[1].set_title('Frame 0')
    #
    # ax[2].imshow(difference_array[0])
    # ax[2].set_title('Diff')
    #
    # ax[3].imshow(threshold_array[0])
    # ax[3].set_title('skimage Threshold')
    #
    # ax[4].imshow(cv_threshold)
    # ax[4].set_title('CV Threshold')
    #
    # ax[5].imshow(markers[0])
    # ax[5].set_title('markers')
    #
    # for a in ax:
    #     a.axis('off')
    #
    # plt.show()

wd = Path.home().joinpath('Desktop', 'temp_testing')
# A0003b.avi is a AVI from WormViz comverted to grayscale in Fiji
vid = wd.joinpath('A0003b.avi')
# optical_flow(vid)
segment_worms(vid)

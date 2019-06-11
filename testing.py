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

import argparse
import cv2
import numpy as np
import os
import pywt

def non_overlapping_max_pooling(image, kernel_size, padding=False):
    '''
    Non-overlapping pooling on 2D or 3D image.

    :param image: ndarray, input image to pool.
    :param kernel_size: kernel size of int or tuple of 2 in (kH, kW).
    :param padding: bool, pad the input image or not
    :return: the max-pooled image
    '''

    H, W = image.shape[:2]
    kH, kW = kernel_size, kernel_size if isinstance(kernel_size, int) else kernel_size

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    if padding:
        nH = _ceil(H, kH)
        nW = _ceil(W, kW)
        size = (nH * kH, nW * kW) + image.shape[2:]
        image_pad = np.full(size, np.nan)
        image_pad[:H, :W, ...] = image
    else:
        nH = H // kH
        nW = W // kW
        image_pad = image[:H // kH * kH, :W // kW * kW, ...]

    new_shape = (nH, kH, nW, kW) + image.shape[2:]
    return np.nanmax(image_pad.reshape(new_shape), axis=(1, 3))

# An optimized version of
# https://github.com/pedrofrodenas/blur-Detection-Haar-Wavelet/blob/master/blur_wavelet.py
def blur_detect(gray_img, threshold, kernel_size=8):
    row, col = gray_img.shape

    # Crop input image to be divisible by kernel size
    # gray_img = gray_img[0:int(row / kernel_size) * kernel_size, 0:int(col / kernel_size) * kernel_size]
    gray_img = gray_img[0:int(row / 16) * kernel_size, 0:int(col / 16) * 16]

    # Step 1, compute Haar wavelet of input image
    LL1, (LH1, HL1, HH1) = pywt.dwt2(gray_img, "haar")
    # Another application of 2D haar to LL1
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, "haar")
    # Another application of 2D haar to LL2
    LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, "haar")

    # Construct the edge map in each scale Step 2
    E1 = np.sqrt(np.power(LH1, 2) + np.power(HL1, 2) + np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2) + np.power(HL2, 2) + np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2) + np.power(HL3, 2) + np.power(HH3, 2))

    # Perform non-overlapping max pooling for each scale
    Emax1 = non_overlapping_max_pooling(E1, kernel_size, padding=True)
    Emax2 = non_overlapping_max_pooling(E2, kernel_size // 2, padding=True)
    Emax3 = non_overlapping_max_pooling(E3, kernel_size // 4, padding=True)

    # Step 3
    EdgePoint1 = Emax1 > threshold
    EdgePoint2 = Emax2 > threshold
    EdgePoint3 = Emax3 > threshold

    # Rule 1 Edge Pojnts
    EdgePoint = np.logical_or.reduce((EdgePoint1, EdgePoint2,  EdgePoint3))

    # Rule 2 Dirak-Structure or Astep-Structure
    DAstructure = np.logical_and(
        Emax1[EdgePoint] > Emax2[EdgePoint],
        Emax2[EdgePoint] > Emax3[EdgePoint]
    )

    # Rule 3 Roof-Structure or Gstep-Structure
    RGstructure = np.logical_and(
        Emax1[EdgePoint] < Emax2[EdgePoint],
        Emax2[EdgePoint] < Emax3[EdgePoint]
    )

    # Rule 4 Roof-Structure

    RSstructure = np.logical_and(
        Emax2[EdgePoint] > Emax1[EdgePoint],
        Emax2[EdgePoint] > Emax3[EdgePoint]
    )

    # Rule 5 Edge more likely to be in a blurred image
    BlurC = Emax1[EdgePoint][np.logical_or(RGstructure,  RSstructure)] < threshold

    # Step 6
    Per = np.sum(DAstructure) / np.sum(EdgePoint)

    # Step 7
    if (np.sum(RGstructure) + np.sum(RSstructure)) == 0:
        BlurExtent = 1.
    else:
        BlurExtent = np.sum(BlurC) / (np.sum(RGstructure) + np.sum(RSstructure))

    return Per, BlurExtent


def find_images(input_dir):
    extensions = [".jpg", ".png", ".jpeg"]

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                yield os.path.join(root, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run Haar Wavelet blur detection on a folder')
    parser.add_argument('-i', '--input_dir', dest="input_dir", type=str, required=True, help="directory of images")
    parser.add_argument("-t", "--threshold", dest='threshold', type=float, default=35, help="blurry threshold")
    parser.add_argument("-e", "--extent", dest='extent', type=float, default=0.6, help="blurry extent threshold")
    parser.add_argument("-k", "--kernel_size", dest='KernelSize', type=int, default=16,
                        help="The kernel size for max pooling")
    args = parser.parse_args()

    for input_path in find_images(args.input_dir):
        try:
            img = cv2.imread(input_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            per, blurext = blur_detect(gray, args.threshold, args.KernelSize)
            classification = blurext < args.extent
            print(f"{input_path}, Per: {per:.5f}, blur extent: {blurext:.3f}, is blur: {classification}")
        except Exception as e:
            print(e)
            pass

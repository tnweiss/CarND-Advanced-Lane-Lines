import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
NX = 6  # number of chessboard corners in the x direction
NY = 8  # number of chessboard corners in the y direction
DST_OFFSET = 100
SOBEL_MIN = 0
SOBEL_MAX = 180


class ImagePointsNotFoundException(Exception):
    pass


def image_size(image_name=None, image=None):
    if image_name:
        image = cv2.imread(image_name)
    return image.shape[1], image.shape[0]


def grey_img(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def object_image_points(gray_chessboard_img, chessboard_nx, chessboard_ny):
    object_points = np.zeros((chessboard_nx * chessboard_ny, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    found, image_points = cv2.findChessboardCorners(gray_chessboard_img, (chessboard_ny, chessboard_nx), None)

    if not found:
        raise ImagePointsNotFoundException("Image points not found")

    return object_points, image_points


def compute_distortion(calibration_images, chessboard_nx, chessboard_ny):
    object_points_list = []
    image_points_list = []

    for calibration_image in calibration_images:
        print(f'Calibrating with image {calibration_image}')
        image = cv2.imread(calibration_image)

        gray_image = grey_img(image)

        try:
            object_points, image_points = object_image_points(gray_image, chessboard_nx, chessboard_ny)
            object_points_list.append(object_points)
            image_points_list.append(image_points)
            print('\tCalibration Success')
        except ImagePointsNotFoundException:
            print(f'\tCalibration Failed')

    if len(object_points_list) == 0 or len(image_points_list) == 0:
        raise Exception("No Object or image points found")

    img_size = image_size(image_name=calibration_images[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points_list, image_points_list, img_size, None, None)
    return mtx, dist


def undistort_image(image, matrix, distortion):
    undistorted_image = cv2.undistort(image, matrix, distortion, None, matrix)
    return undistorted_image


def birds_eye_transform(image, offset):
    size = image_size(image=image)
    w, h = size

    src = np.float32(
        [
            [w * .3, h * .9],
            [w * .4, h * .61],
            [w * .6, h * .61],
            [w * .7, h * .9]
        ]
    )
    dst = np.float32(
        [
            [w * .2, h - offset],
            [w * .2, offset],
            [w * .8, offset],
            [w * .8, h - offset]
        ]
    )

    m = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(image, m, size, flags=cv2.INTER_LINEAR)


def abs_sobel_thresh(image, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = grey_img(image)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    abs_sobel = None
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def pipeline():
    # load calibration images
    calib_images = glob.glob('camera_cal/*14.jpg')

    # compute the distortion matrix
    mtx, dist = compute_distortion(calib_images, NX, NY)

    # for each image in the test_images apply transformations
    for test_image_name in glob.glob('test_images/*.jpg'):
        # load the image
        test_image = cv2.imread(test_image_name)

        # undistort the image
        undist_test_img = undistort_image(test_image, mtx, dist)

        # get the threshold binary image
        test_img_binary = abs_sobel_thresh(undist_test_img, thresh_min=SOBEL_MIN, thresh_max=SOBEL_MAX)

        # apply perspective transform to get birds eye view

        # detect lane pixels

        plt.imshow(undist_test_img)
        #plt.imshow(test_img_binary, cmap='gray')
        plt.show()


if __name__ == '__main__':
    pipeline()
    #
    # # load calibration images
    # calib_images = glob.glob('camera_cal/*14.jpg')
    #
    # # compute the distortion matrix
    # mtx, dist = compute_distortion(calib_images, NX, NY)
    #
    # # load image
    # test_img = cv2.imread('test_images/test2.jpg')
    #
    # # undistort
    # undist_test_img = undistort_image(test_img, mtx, dist)
    #
    # plt.show(test_img)
    #
    # cv2.imwrite('output_images/testing.jpg', undist_test_img)


for x in range(0, 5):


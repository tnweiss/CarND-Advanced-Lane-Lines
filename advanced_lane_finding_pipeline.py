import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
NX = 6  # number of chessboard corners in the x direction
NY = 8  # number of chessboard corners in the y direction


def image_size(image_name):
    image = cv2.imread(image_name)
    return image.shape[1], image.shape[0]


def grey_img(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def object_image_points(gray_chessboard_img, chessboard_nx, chessboard_ny):
    object_points = np.zeros((chessboard_nx * chessboard_ny, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    found, image_points = cv2.findChessboardCorners(gray_chessboard_img, (chessboard_ny, chessboard_nx), None)

    if not found:
        raise Exception("Image points not found")

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
        except Exception:
            print(f'\tCalibration Failed')

    if len(object_points_list) == 0 or len(image_points_list) == 0:
        raise Exception("No Object or image points found")

    img_size = image_size(calibration_images[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points_list, image_points_list, img_size, None, None)
    return mtx, dist


if __name__ == '__main__':
    calib_images = glob.glob('camera_cal/*1*.jpg')
    mtx, dist = compute_distortion(calib_images, NX, NY)
    print(mtx)
    print(dist)

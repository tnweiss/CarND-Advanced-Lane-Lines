"""
This module provides helper functions for the advanced-lane-finding.ipynb
"""
import numpy as np
import math
import cv2
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def image_size(image_name=None, image=None):
    """
    return the size of an image
    :param image_name: if provided load the image and then return the size
    :param image:if provided return the shape
    :return: return the shape in the form of width, height
    """
    if image_name:
        image = cv2.imread(image_name)
    return image.shape[1], image.shape[0]


class ImagePointsNotFoundException(Exception):
    """
    Used when image points have not been identified
    """


def plot_images(*args, cmap='gray', fig_size=(20, 10), num_cols=2, font_size=16, line_color='red'):
    """
    helper function used to plot images and functions
    :param args:
    :param cmap:
    :param fig_size:
    :param num_cols:
    :param font_size:
    :param line_color:
    :return:
    """
    if len(args) == 0:
        return

    num_rows = math.ceil(len(args) / num_cols)
    f, axs = plt.subplots(num_rows, num_cols, figsize=fig_size)

    if num_rows == 1:
        for i in range(0, len(args)):
            axs[i].imshow(args[i][0], cmap=cmap)
            axs[i].set_title(args[i][1], fontsize=font_size)
            if len(args[i]) == 3:
                axs[i].plot(args[i][2][0], args[i][2][1], color=line_color)
    else:
        for i, img in enumerate(args):
            row = int(i / 2)
            col = (i % 2)

            axs[row][col].imshow(args[i][0], cmap=cmap)
            axs[row][col].set_title(args[i][1], fontsize=font_size)
            if len(args[i]) == 3:
                axs[row][col].plot(args[i][2], color=line_color)


def calculate_object_points(chessboard_nx, chessboard_ny):
    """
    calculate the object points provided the number of chessboard points in the x and y direction
    :param chessboard_nx:
    :param chessboard_ny:
    :return:
    """
    obj_points = np.zeros((chessboard_nx * chessboard_ny, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:chessboard_ny, 0:chessboard_nx].T.reshape(-1, 2)
    return obj_points


def calculate_image_points(gray_chessboard_img, chessboard_nx, chessboard_ny):
    """
    locate the corners of a chessboard on an image
    :param gray_chessboard_img:
    :param chessboard_nx:
    :param chessboard_ny:
    :return:
    """
    found, img_points = cv2.findChessboardCorners(gray_chessboard_img, (chessboard_ny, chessboard_nx), None)

    if not found:
        raise ImagePointsNotFoundException("Image points not found")

    return img_points


def compute_distortion(calibration_images, chessboard_nx, chessboard_ny):
    """
    provided a list of image, and number of corners in the x and y direction compute the distortion coefficients
    :param calibration_images:
    :param chessboard_nx:
    :param chessboard_ny:
    :return:
    """
    object_points_list = []
    image_points_list = []

    for calibration_image in calibration_images:
        image = cv2.imread(calibration_image)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        object_points = calculate_object_points(chessboard_nx, chessboard_ny)
        try:
            image_points_list.append(calculate_image_points(gray_image, chessboard_nx, chessboard_ny))
            object_points_list.append(object_points.copy())
            # print('Success')
        except ImagePointsNotFoundException:
            # print(f'Failed')
            pass

    if len(image_points_list) == 0:
        raise Exception("No Object or image points found")

    img_size = image_size(image_name=calibration_images[0])
    _, mtx, dist, _, _ = cv2.calibrateCamera(object_points_list, image_points_list, img_size, None, None)
    return mtx, dist


def sobel_thresh(img, sobel_kernel, thresh_min, thresh_max, orient='x'):
    """
    Calculate the sobel gradient and only keep pixels within a given threshold
    :param img:
    :param sobel_kernel:
    :param thresh_min:
    :param thresh_max:
    :param orient:
    :return:
    """
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # based on orientation, provide params to calculate sobel accordingly
    abs_sobel = None
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,  sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # apply the thresholds
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # return the image
    return binary_output


def magnitude_threshold(img, sobel_kernel, thresh_min, thresh_max):
    """
    compute the magnitude of each pixel in an image and only keep those that fall within the threshold
    :param img:
    :param sobel_kernel:
    :param thresh_min:
    :param thresh_max:
    :return:
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradient_magnitude)/255
    gradient_magnitude = (gradient_magnitude/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradient_magnitude)
    binary_output[(gradient_magnitude >= thresh_min) & (gradient_magnitude <= thresh_max)] = 1

    return binary_output


def directional_threshold(img, sobel_kernel, thresh_min, thresh_max):
    """
    compute the directional threshold of each pixel and only keep those that fall within a range
    :param img:
    :param sobel_kernel:
    :param thresh_min:
    :param thresh_max:
    :return:
    """
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # find the absolute value off the gradient direction
    abs_gradient_dir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    binary_output = np.zeros_like(abs_gradient_dir)

    # only keep values within a threshold
    binary_output[(abs_gradient_dir >= thresh_min) & (abs_gradient_dir <= thresh_max)] = 1

    # Return the binary image
    return binary_output


def hls_threshold(img, thresh_min, thresh_max):
    """
    tranform the image to the hls color space and only keep the pixels that fall within the threshold
    :param img:
    :param thresh_min:
    :param thresh_max:
    :return:
    """
    # convert image to hls color space and keep the s channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # only save points that fall within the threshold
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh_min) & (s_channel <= thresh_max)] = 1

    return binary_output


def threshold_binary(img, sobel_kernel=15, sobel_thresh_min=20, sobel_thresh_max=200, mag_thresh_min=50,
                     mag_thresh_max=150, dir_thresh_min=0.7, dir_thresh_max=1.2, hls_min_thresh=160,
                     hls_max_thresh=255):
    """
    Calculate all the thresholds provided by this module and combine them into one binary image
    :param img:
    :param sobel_kernel:
    :param sobel_thresh_min:
    :param sobel_thresh_max:
    :param mag_thresh_min:
    :param mag_thresh_max:
    :param dir_thresh_min:
    :param dir_thresh_max:
    :param hls_min_thresh:
    :param hls_max_thresh:
    :return:
    """
    grad_x_binary = sobel_thresh(img, sobel_kernel, sobel_thresh_min, sobel_thresh_max, orient='x')
    grad_y_binary = sobel_thresh(img, sobel_kernel, sobel_thresh_min, sobel_thresh_max, orient='y')
    mag_binary = magnitude_threshold(img, sobel_kernel, mag_thresh_min, mag_thresh_max)
    dir_binary = directional_threshold(img, sobel_kernel, dir_thresh_min, dir_thresh_max)
    hls_binary = hls_threshold(img, hls_min_thresh, hls_max_thresh)

    combined = np.zeros_like(dir_binary)
    combined[((grad_x_binary == 1) & (grad_y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) |
             (hls_binary == 1)] = 1

    return combined


def birds_eye_transform_points(w, h):
    """
    Calculate birds eye transform points, proportional to the image size
    :param w:
    :param h:
    :return:
    """
    src = np.float32(
        [
            [w * .22, h * .97],
            [w * .45, h * .65],
            [w * .575, h * .65],
            [w * .88, h * .97]
        ]
    )
    dst = np.float32(
        [
            [w * .22, h * .97],
            [w * .22, h * .05],
            [w * .87, h * .05],
            [w * .87, h * .97]
        ]
    )

    return src, dst


def birds_eye_transform_matrix(w, h):
    """
    Create the birds eye transform matrix
    :param w:
    :param h:
    :return:
    """
    src, dst = birds_eye_transform_points(w, h)

    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_transform_matrix = cv2.getPerspectiveTransform(dst, src)

    return transform_matrix, inverse_transform_matrix


def matrix_to_polygon(src=None, dst=None, img=None):
    """
    Generate the x and y coordinates given a set of points or image
    :param src:
    :param dst:
    :param img:
    :return:
    """
    if img is not None:
        i_size = image_size(image=img)
        src, dst = birds_eye_transform_points(i_size[0], i_size[1])

    src_polygon_x = Polygon(src).get_xy()[:, 0]
    src_polygon_y = Polygon(src).get_xy()[:, 1]

    dst_polygon_x = Polygon(dst).get_xy()[:, 0]
    dst_polygon_y = Polygon(dst).get_xy()[:, 1]

    return (src_polygon_x, src_polygon_y), (dst_polygon_x, dst_polygon_y)


def birds_eye_transform(image, matrix=None):
    """
    Compute a birds eye transform on an image
    :param image:
    :param matrix:
    :return:
    """
    size = image_size(image=image)

    if matrix is None:
        matrix, _ = birds_eye_transform_matrix(size[0], size[1])

    return cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_LINEAR)


def calculate_hist(img):
    """

    :param img:
    :return:
    """
    # Only use the bottom half because those are most likely to be vertical
    bottom_half = img[img.shape[0] // 2:, :]

    # Sum vertical (axis at 0 will sum)
    histogram = np.sum(bottom_half, axis=0)

    return histogram


def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50, draw_windows=False):
    """

    :param binary_warped:
    :param nwindows:
    :param margin:
    :param minpix:
    :param draw_windows:
    :return:
    """
    out_img = None
    if draw_windows:
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        if draw_windows:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, output_debug=False):
    """

    :param binary_warped:
    :param output_debug:
    :return:
    """
    # Find our lane pixels first
    left_x, left_y, right_x, right_y, out_img = find_lane_pixels(binary_warped, draw_windows=output_debug)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    if output_debug:
        out_img[left_y, left_x] = [255, 0, 0]
        out_img[right_y, right_x] = [0, 0, 255]

    return left_fit, right_fit, out_img


def binary_to_img(binary):
    """

    :param binary:
    :return:
    """
    out_img = np.dstack((binary, binary, binary))
    out_img = out_img * 255
    return out_img


def show_polynomial(img, left_fit, right_fit):
    """

    :param img:
    :param left_fit:
    :param right_fit:
    :return:
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.imshow(img)


def add_lines(img, fit, color_bgr=(0, 255, 0), thickness=9):
    """

    :param img:
    :param fit:
    :param color_bgr:
    :param thickness:
    :return:
    """
    # Generate x and y values for plotting
    y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    try:
        x = fit[0] * y ** 2 + fit[1] * y + fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        x = 1 * y ** 2 + 1 * y

    points = np.dstack((x, y))
    cv2.polylines(img, np.int32([points]), False, color_bgr, thickness)

    return img


def overlay_search_area(left_fit, right_fit, margin=100, img=None, binary_img=None):
    """

    :param left_fit:
    :param right_fit:
    :param margin:
    :param img:
    :param binary_img:
    :return:
    """
    # create the output image and window image used to overlay
    if binary_img is not None:
        out_img = np.dstack((img, img, img)) * 255
    elif img is not None:
        out_img = img
    else:
        raise Exception("One of img or binary_img is required")
    window_img = np.zeros_like(out_img)

    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    try:
        left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fit_x = 1 * plot_y ** 2 + 1 * plot_y
        right_fit_x = 1 * plot_y ** 2 + 1 * plot_y

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x - margin, plot_y]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin,
                                                                    plot_y])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x - margin, plot_y]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin,
                                                                     plot_y])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 1, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 1, 0))

    return cv2.addWeighted(out_img, 1, window_img, .25, 0)


def search_around_poly(binary_warped, left_fit, right_fit, margin=100):
    """

    :param binary_warped:
    :param left_fit:
    :param right_fit:
    :param margin:
    :return:
    """
    # Grab only the activated pixels in the new image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # use the fits from the previously calculated polynomial to only save pixels in the next images margin
    # polynomial coefficients to calculate the curve and find all indices within the lower and upper margin
    left_lane_indices = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
        (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin))
    )
    right_lane_indices = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
        (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin))
    )

    # Use the indices to extract the left/right x/y values
    left_x = nonzerox[left_lane_indices]
    left_y = nonzeroy[left_lane_indices]
    right_x = nonzerox[right_lane_indices]
    right_y = nonzeroy[right_lane_indices]

    if left_x.size == 0 or right_x.size == 0:
        return None, None

    # With the new images nonzero pixels, generate a new fit
    new_left_fit = np.polyfit(left_y, left_x, 2)
    new_right_fit = np.polyfit(right_y, right_x, 2)

    return new_left_fit, new_right_fit


def calculate_curvature(left_fit, right_fit, img_shape, ym_per_pix=30/720, xm_per_pix=3.7/700):
    """

    :param left_fit:
    :param right_fit:
    :param img_shape:
    :param ym_per_pix:
    :param xm_per_pix:
    :return:
    """
    plot_y = np.linspace(0, img_shape[0] - 1, num=img_shape[0])  # to cover same y-range as image
    left_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

    left_x = left_x[::-1]
    right_x = right_x[::-1]

    # Fit a second order polynomial to pixel positions in each fake lane line
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(plot_y*ym_per_pix, left_x*xm_per_pix, 2)
    right_fit_cr = np.polyfit(plot_y*ym_per_pix, right_x*xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(img_shape[0])

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad


def car_offset(left_fit, right_fit, img_shape, xm_per_pix=3.7/700):
    """

    :param left_fit:
    :param right_fit:
    :param img_shape:
    :param xm_per_pix:
    :return:
    """
    mid_img_x = img_shape[1] // 2

    plot_y = np.linspace(0, img_shape[0] - 1, num=img_shape[0])  # to cover same y-range as image
    left_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

    car_pos = (left_x[-1] + right_x[-1]) / 2

    return (mid_img_x - car_pos) * xm_per_pix


def draw_lane(img, left_fit, right_fit, m_inv):
    """

    :param img:
    :param left_fit:
    :param right_fit:
    :param m_inv:
    :return:
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img)

    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    try:
        left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fit_x = 1 * plot_y ** 2 + 1 * plot_y
        right_fit_x = 1 * plot_y ** 2 + 1 * plot_y

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp_zero, np.int_(left_line_pts), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(warp_zero, m_inv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    return cv2.addWeighted(img, 1, new_warp, 0.3, 0)


def add_metrics(img, left_lane_curvature, right_lane_curvature, horizontal_offset):
    """

    :param img:
    :param left_lane_curvature:
    :param right_lane_curvature:
    :param horizontal_offset:
    :return:
    """
    # Display lane curvature
    out_img = img.copy()
    cv2.putText(out_img, f'Left lane line curvature: {left_lane_curvature:.2f} m',
                (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
    cv2.putText(out_img, f'Right lane line curvature: {right_lane_curvature:.2f} m',
                (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
    cv2.putText(out_img, f'Horizontal car offset: {horizontal_offset:.2f} m',
                (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
    return out_img


class ProcessImage:
    def __init__(self, images, chessboard_nx=9, chessboard_ny=6):
        # Make a list of calibration images
        images = glob.glob(images)

        # Calibrate camera
        self.mtx, self.dist = compute_distortion(images, chessboard_nx, chessboard_ny)
        self.left_fit = None
        self.right_fit = None

    def __call__(self, img):
        # Undistort image
        undistorted_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        # get the transform matrix
        matrix, inv_matrix = birds_eye_transform_matrix(img.shape[1], img.shape[0])

        # get the binary image with the default thresholds
        source_threshold_binary = threshold_binary(undistorted_img)

        # transform to get the birds eye view
        birds_eye = birds_eye_transform(source_threshold_binary)

        # if left and right fits are calculated then use them to search
        if self.left_fit is None:
            self.left_fit, self.right_fit, _ = fit_polynomial(birds_eye)
        else:
            self.left_fit, self.right_fit = search_around_poly(source_threshold_binary, self.left_fit,
                                                               self.right_fit)
            if self.left_fit is None:
                self.left_fit, self.right_fit, _ = fit_polynomial(source_threshold_binary)

        left_curvature, right_curvature = calculate_curvature(self.left_fit, self.right_fit, img.shape)
        offset = car_offset(self.left_fit, self.right_fit, img.shape)

        undistorted_img = add_metrics(undistorted_img, left_curvature, right_curvature, offset)
        undistorted_img = draw_lane(undistorted_img, self.left_fit, self.right_fit, inv_matrix)

        return undistorted_img

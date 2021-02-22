## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Undistorted"
[image2]: ./output_images/1-undistorted-calibration1.jpg "Road Transformed"
[image3]: ./output_images/3-combined.jpg "Binary"
[image4]: ./output_images/4-birds-eye-transform.jpg "Birds Eye"
[image5]: ./output_images/8-image-w-metrics.jpg "output"
[video1]: ./output_images/9-project_video_solution.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

## Writeup

### Camera Calibration

#### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

First I compute the camera's distortion by using chessboard images. We can determine the distortion because we know what
a that the squares on the chessboard image should be the same size. We can use cv2's library to determine how distorted 
the camera is based on the differentiating sizes of the squares in the chessboard. One of the major difference's of 
our calibration in this project compared to the lesson was the size of the chessboards. Here we used 9x6 whereas in the 
lessons we used 8x6.

Below is the function used to compute distortion. This code can be found in `advanced_lane_finding.py`

```python
def compute_distortion(calibration_images, chessboard_nx, chessboard_ny):
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
```

Below is an example of one of the distorted chessboards used as a calibration image.

![alt text][image1]

### Pipeline (test images)

#### Provide an example of a distortion-corrected image.

Below is an example of the calibration image above undistorted using the coefficients calculated from the provided 
calibration images

![alt text][image2]


#### Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I used `cv2.Sobel` in my `advanced_lane_finding.py` helper functions to compute different gradients along the x and y 
axis to identify lane lines in an image. We can use these values to also compute the direction of the gradient and isolate
those that point in a direction of interest. We can also filter gradients by magnitude to isolate strong gradients in an image.
We also translate the image to the hls color space to isolate lines in an image, we then keep points that have a saturation
value that falls within our threshold. We can then combine all the binary images to produce the image below.

![alt text][image3]


#### Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In `advanced_lane_finding.py` I have a helper function named `birds_eye_transform` which will take a matrix to distort the provided image.
The matrix is determined by a set of source and destination points. The source points are a set of points on an image
and the destination points are where the source points should be on the image. You can then interchange the source and 
destination points to get a transform and inverse transform matrix for your image.

```python
def birds_eye_transform(image, matrix=None):
    size = image_size(image=image)

    if matrix is None:
        matrix, _ = birds_eye_transform_matrix(size[0], size[1])

    return cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_LINEAR)
```

Below is an example of the road that has gone through a birds eye transform

![alt text][image4]

#### Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In `advanced_lane_finding.py` I have two functions; `find_lane_pixels` and `search_around_poly` which are used to identify lane lines.
find_lane_pixels will split the image into nWindows and use a histogram to determine where the densest portion of pixels
are in that image. It will then save the pixels that fall within the center of the densest area. At the end of this all
the remaining points will be fed into `cv2.polyfit` to create a second order polynomial for the line. 

In search_around_poly we just use the previous fit to filter out pixels and then determine the second order polynomial fit for
for the remaining pixels. 

#### Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In `advanced_lane_finding.py` I have two helper functions, `car_offset` and  `calculate_curvature`,  that help 
determine the curvature and vehicle position. We can determine the vehicle position relative to center of the lane by 
looking at the lane lines relative to the center of the image since we can assume the center of the image is the center
of the car. We can calculate the curvature by scaling the position of the activated pixels and creating a second order
polynomial fit. We can then use this to calculate the radius of the curvature


#### Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Below is the resulting image after the offset, lane lines, and curvatures have been detected and stamped on
to the frame.

![alt text][image5]

### Pipeline (video)

[Project Solution](./output_images/9-project_video_solution.mp4)

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

Determining the gradient and hls thresholds was the toughest part of this project and would be the most likely culprit 
of my pipeline failing. To fix this I think I would need a way to change thresholds based on external conditions. 

I also think roads without clear lines would be an issue, in that case i would have to fall back to determining if the
road is a two-way or 1 way road and either drive in the center of the road or the center of the right half of the road.

import cv2 as cv
from MyDetectionMethods import *
import numpy as np

# load aruco detector parameter
parameters_for_aruco = cv.aruco.DetectorParameters_create()

# load aruco detector dictionary
dictionary_for_aruco = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)

# loading the webcam for video capture
video_feed = cv.VideoCapture(0, cv.CAP_DSHOW)

# resizing the output window
video_feed.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
video_feed.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# check if the webcam is opened correctly
if not video_feed.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # capture video feed frame-by-frame
    ret, video_frame = video_feed.read()

    # creating object for MyDetectionMethods class
    detect_contours = MyDetectionMethods(video_frame)

    # detect the aruco marker
    corners, ids, rejected = cv.aruco.detectMarkers(
        video_frame, dictionary_for_aruco, parameters=parameters_for_aruco
    )

    # check at least one aruco marker is present
    if len(corners) > 0:

        # convert corner values from float to int
        corners_in_int = np.int0(corners)

        # draw square box around the aruco marker
        cv.polylines(video_frame, corners_in_int, True, (0, 255, 0), 4)

        # find perimeter of the aruco marker
        perimeter_of_aruco_marker = cv.arcLength(corners[0], True)

        # find pixel to cm ratio of aruco (4 x 3.6 = 14.4)
        # width and height of aruco marker used for calibration is 3.6cm
        px_to_cm_ratio = perimeter_of_aruco_marker / 14.4

        # get contours from the video feed
        contours = detect_contours.find_contours()

        # Draw objects boundaries with minAreaRect function
        for contour in contours:
            # get the rectangle border
            object_border = cv.minAreaRect(contour)
            (x_center, y_center), (width, height), angle = object_border

            # detect the object borders
            rectangle_box = cv.boxPoints(object_border)
            rectangle_box = np.int0(rectangle_box)

            # get width and height of the object in pixel to cm ratio and convert to mm
            object_width = width / px_to_cm_ratio
            object_height = height / px_to_cm_ratio

            # conditions with a precision of +/- 3mm
            # standard credit card dimension - 85.6 x 53.9mm
            credit_card_condition = (
                8.26 <= object_width <= 8.86 and 5.09 <= object_height <= 5.69
            )

            # standard AA battery dimension - 14.5 x 50.5mm
            aa_battery_condition = (
                1.42 <= object_width <= 1.48 and 5.02 <= object_height <= 5.08
            )

            if credit_card_condition or aa_battery_condition or True:

                # mark the center of the object
                cv.circle(
                    video_frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1
                )

                # draw rectangular box around the object
                cv.polylines(video_frame, [rectangle_box], True, (255, 0, 0), 2)

                # display the width
                cv.putText(
                    video_frame,
                    f"Width {round(object_width, 2)}cm",
                    (int(x_center - 80), int(y_center - 15)),
                    cv.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (0, 0, 255),
                    1,
                )

                # display the height
                cv.putText(
                    video_frame,
                    f"Height {round(object_height, 2)}cm",
                    (int(x_center - 80), int(y_center + 25)),
                    cv.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (0, 0, 255),
                    1,
                )

    # display the video feed
    cv.imshow("Detect Object", video_frame)

    # the program will stop when the key 'q' is pressed
    if cv.waitKey(1) == ord("q"):
        break

video_feed.release()
cv.destroyAllWindows()

"""
    a. The dimension of the aruco marker used for calibration is 3.6cm.
    b. Also the battery used is AA battery - 14.5 x 50.5mm.
    c. Any other objects not within the provided dimension (credit cards and AA battery) with a +/- 3mm precision won't be detected by the code
"""

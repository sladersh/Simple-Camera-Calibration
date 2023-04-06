import cv2 as cv


class MyDetectionMethods:
    def __init__(self, video_frame):
        self.video_frame = video_frame

    def find_contours(self):
        """
        returns the contours in the live video feed
        """
        # convert the video feed to grayscale
        gray_scale = cv.cvtColor(self.video_frame, cv.COLOR_BGR2GRAY)

        # create an adaptive threshold mask for the video feed
        threshold_mask = cv.adaptiveThreshold(
            gray_scale, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 19, 5
        )

        # finding contours in the video feed
        contours, hierarchy = cv.findContours(
            threshold_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
        )

        # empty contours array
        contours_in_feed = []

        # filter smaller objects less than 2000 from contours
        for contour in contours:
            contour_area = cv.contourArea(contour)
            if contour_area > 2000:
                contours_in_feed.append(contour)

        return contours_in_feed

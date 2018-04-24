import numpy as np
import cv2

class ORBExample(object):
    """A simple example to run ORB feature detection on webcam image. """

    def __init__(self):
        # Create an OpenCV window.
        self.window = cv2.namedWindow("ORB Features")

        # Init webcam capture.
        self.cap = cv2.VideoCapture(0)

        # Initialize an ORB class with default params.
        self.orb = cv2.ORB_create()

        # Run the loop.
        self.run()

    def run(self):
        """Loop to display webcam with ORB features."""
        while(True):
            # Read frame from webcam.
            ret, frame = self.cap.read()

            # Find ORB keypoints.
            kp = self.orb.detect(frame)

            # Compute the descriptors with ORB.
            kp, des = self.orb.compute(frame, kp)

            # Draw location of keypoints on webcam frame.
            img = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)

            # Show image with keypoints and wait for 'q' to quit.
            cv2.imshow("ORB Features", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When we break out of loop, clean up.
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    example = ORBExample()

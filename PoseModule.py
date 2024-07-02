import cv2
import mediapipe as mp
import time


class PoseDetector:
    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.smooth_segmentation = smooth_segmentation
        self.enable_segmentation = enable_segmentation
        self.smooth_landmarks = smooth_landmarks
        self.model_complexity = model_complexity
        self.static_image_mode = static_image_mode

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence,
                                     self.min_tracking_confidence)

    def getPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img, draw=True):
        lmList = []

        if self.results.pose_landmarks:
            for i, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(i, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([i, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture('videos/3185733-hd_1920_1080_30fps.mp4')
    pTime = 0

    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1080, 720), interpolation=cv2.INTER_LINEAR)
        img = detector.getPose(img)

        lmList = detector.getPosition(img, draw=False)

        # if len(lmList) != 0:
        #     print(lmList)
        #     cv2.circle(img, (lmList[14][1], lmList[14][2]), 3, (255, 0, 0), cv2.FILLED)
        #     cv2.circle(img, (lmList[13][1], lmList[13][2]), 3, (255, 0, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

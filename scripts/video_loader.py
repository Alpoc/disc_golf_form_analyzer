import cv2
import os

video = cv2.VideoCapture("../videos/2023 Disc Golf Pro Tour Championship ｜ FPO R2F9｜ Tattar, Gannon, Scoggins, Handley ｜ Jomez Disc Golf [Djyk-gaCPWQ].mkv")

while True:
    ret, frame = video.read()

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
import os
import cv2
import src.compare8
import src.compare9
import src.real_time_face_recognition
import sys

# def get_image():
#     capture = cv2.VideoCapture(0)
#
#     if capture.isOpened():
#         pass
#     else:
#         capture.open()
#     name = input("请输入你的名字：")
#     if name == "exit":
#         sys.exit()
#     while True:
#         ret, frame = capture.read()
#         if ret is not None:
#
#            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#            os.makedirs("./data/gump1/{}".format(name))
#            cv2.imwrite("./data/gump1/name/{}.jpg".format(name), image)
#
#            if cv2.waitKey(127) & 0xFF == ord("q"):
#                break
#
#     capture.release()
#     cv2.destroyAllWindows()

if __name__ == '__main__':
    # get_image()
    src.compare8.main()
    src.compare9.main()
    src.real_time_face_recognition.main()
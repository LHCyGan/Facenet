# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""

import cv2
import face


def add_overlays(image, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(image,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                if face.name == 'unknown':
                    cv2.putText(image, face.name, (face_bb[0], face_bb[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                thickness=2, lineType=2)
                else:
                    cv2.putText(image, face.name, (face_bb[0], face_bb[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                thickness=2, lineType=2)


def main():
    image = cv2.imread('../images/1-2.jpg')
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_recognition = face.Recognition()
    faces = face_recognition.identify(frame)
    add_overlays(image, faces)
    cv2.imwrite('../images/show.jpg', image)

if __name__ == '__main__':
    main()

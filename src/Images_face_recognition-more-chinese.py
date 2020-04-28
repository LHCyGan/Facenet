# coding=utf-8

import cv2
import face
import os
import time
import numpy
from PIL import Image, ImageDraw, ImageFont
def add_overlays(image, faces):
    if faces is not None:
        img_PIL = Image.fromarray(frame)
        font = ImageFont.truetype('simsun.ttc', 40)
        # 字体颜色
        fillColor1 = (255, 0, 0)
        fillColor2 = (0, 255, 0)
        draw = ImageDraw.Draw(img_PIL)
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            draw.line([face_bb[0],  face_bb[1], face_bb[2], face_bb[1]], "green")
            draw.line([face_bb[0],  face_bb[1], face_bb[0], face_bb[3]], fill=128)
            draw.line([face_bb[0], face_bb[3], face_bb[2], face_bb[3]], "yellow")
            draw.line([face_bb[2], face_bb[1], face_bb[2], face_bb[3]], "black")
            if face.name is not None:
                if face.name == 'unknown':
                    draw.text((face_bb[0], face_bb[1]), '陌生人', font=font, fill=fillColor2)
                else:
                    draw.text((face_bb[0], face_bb[1]), face.name, font=font, fill=fillColor1)
        frame = numpy.asarray(img_PIL)
        return frame


def main():
    testdata_path = '../images'
    face_recognition = face.Recognition()
    start_time = time.time()
    for images in os.listdir(testdata_path):
        print(images)
        filename = os.path.splitext(os.path.split(images)[1])[0]
        file_path = testdata_path + "/" + images
        image = cv2.imread(file_path)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_recognition.identify(frame)
        image = add_overlays(image, faces)
        cv2.imwrite('../images_result/' + filename + '.jpg', image)
    end_time = time.time()
    spend_time = float('%.2f' % (end_time - start_time))
    print('spend_time:',spend_time)

if __name__ == '__main__':
    main()

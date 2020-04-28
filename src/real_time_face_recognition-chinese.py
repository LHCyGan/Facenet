# coding=utf-8

import time
import cv2
import face
import numpy
from PIL import Image, ImageDraw, ImageFont

def add_overlays(frame, faces, frame_rate):
    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)
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
    # video_capture = cv2.VideoCapture("rtsp://admin:12345678hu@192.168.0.100/Streaming/Channels/1")
    # video_capture = cv2.VideoCapture("rtsp://admin:12345678hu@192.168.0.100:80/h264/ch1/main/av_stream")
    face_recognition = face.Recognition()
    video_capture = cv2.VideoCapture(0)
    start_time = time.time()
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.identify(frame)
        # Check our current fps
        end_time = time.time()
        frame_rate = float('%.2f' % (1 / (end_time - start_time)))
        start_time = time.time()
        frame = add_overlays(frame, faces, frame_rate)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

"""Performs face alignment and calculates L2 distance between the embeddings of images."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import facenet
import align.detect_face

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_preprocess
gpu_memory_fraction = 0.3

def main():
    model = '20190128-123456/3001w-train.pb'
    traindata_path = '../data/gump1'
    dirs = os.listdir(traindata_path)
    feature_files = []
    face_label = []
    face_detection = Detection()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            for dir in dirs:
                for root, dirs, files in os.walk(traindata_path + '/' + dir):
                    for file in files:
                        image_path = traindata_path + '/' + dir + '/' + file
                        print(image_path)
                        images = face_detection.find_faces(image_path)
                        if images is not None:
                            # Run forward pass to calculate embeddings
                            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                            emb = sess.run(embeddings, feed_dict=feed_dict)
                            face_label.append(dir)  # face-label
                            feature_files.append(emb)
                        else:
                            print('no find face')
            write_file = open('20190128-123456/knn_classifier.pkl', 'wb')
            pickle.dump(feature_files, write_file, -1)
            pickle.dump(face_label, write_file, -1)
            write_file.close()
            num_count = 0
            for i in range(len(feature_files)):
                for j in range(len(feature_files)):
                    num = np.dot(feature_files[i], feature_files[j].T)
                    sim = 0.5 + 0.5 * num  # 归一化，，余弦距离
                    # print(face_label[i], ' ', face_label[j], ' ', sim)
                    if sim > 0.82 and sim < 0.99:
                        print(face_label[i],' ',face_label[j],' ',sim)
                        num_count = num_count + 1
                    # if sim < 0.82:
                    #     print(face_label[i],' ',face_label[j],' ',sim)
                    #     num_count = num_count + 1
            print(num_count/2)
            print('total_num:',len(os.listdir(traindata_path)))
            print('align_num:',len(face_label))
            print('End')


class Detection:
    minsize = 40  # minimum size of face
    threshold = [0.8, 0.9, 0.9]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=112, face_crop_margin=0):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)


    def find_faces(self,image_paths):
        img = misc.imread(os.path.expanduser(image_paths), mode='RGB')
        _bbox = None
        _landmark = None
        bounding_boxes, points = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        nrof_faces = bounding_boxes.shape[0]
        img_list = []
        max_Aera = 0
        if nrof_faces > 0:
            if nrof_faces == 1:
                bindex = 0
                _bbox = bounding_boxes[bindex, 0:4]
                _landmark = points[:, bindex].reshape((2, 5)).T
                warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size='112,112')
                # cv2.imwrite('1.jpg',warped)
                prewhitened = facenet.prewhiten(warped)
                img_list.append(prewhitened)
            else:
                for i in range(nrof_faces):
                    _bbox = bounding_boxes[i, 0:4]
                    if _bbox[2]*_bbox[3] > max_Aera:
                        max_Aera = _bbox[2]*_bbox[3]
                        _landmark = points[:, i].reshape((2, 5)).T
                        warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size='112,112')
                prewhitened = facenet.prewhiten(warped)
                img_list.append(prewhitened)
        else:
            return None
        images = np.stack(img_list)
        return images


if __name__ == '__main__':
    main()

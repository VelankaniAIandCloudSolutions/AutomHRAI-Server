from models.yolo import YoloOnnx, YoloTrt
# from distutils.command.config import config
import cv2
import numpy as np
import time
import logging
# from app.models.face_aligner import norm_crop
from models.face_aligner import norm_crop
import onnxruntime as rt

logger = logging.getLogger(__name__)
# import tensorrt as trt
try:
    import pycuda.driver as cuda
    import tensorrt as trt
except Exception as e:
    logger.exception(f'PyCuda drivers not found: {e}')

# from models.mtcnn_face import FaceDetector   # mtcnn
# from models.facenet import TFFaceRecognizer # facenet

from models.yolov5_face import FaceDetector   # yolov5 face
# from models.arcface import TrtFaceRecogniser  # arcface
from models.arcface import FaceRecogniser as ArcFace
from typing import Union


# class FaceRecogniser:
#     def __init__(self, facenet_path: str, person_detection_model_path:str, person_conf:float) -> None:
#         self._load_models(facenet_path, person_detection_model_path, person_conf)
    
#     def _load_models(self, facenet_path: str, person_detection_model_path:str, person_conf:float) -> None:
#         # Load face detection model
#         self.face_detector = FaceDetector()
#         # self.face_detector = FaceDetector('models/frozen_models/yolov5n-face-320.onnx', img_size=320)
        
#         # Load face recognition model
#         self.facenet = TFFaceRecognizer(facenet_path)
        
#         # Load person detection model
#         self.person_detection = YoloOnnx(person_detection_model_path, 320, classes=[0], conf_thresh=person_conf)
        
#     def process(self, images: list) -> Union[list, list, list, list]:
#         all_embeddings = [None] * len(images)
        
#         # Person detection
#         _, person_boxes, person_confs, _, _ = self.person_detection.batch_detect(images)
        
#         # Face detection
#         all_face_images = []
#         all_face_boxes = []
#         for img in images:
#             face_images, face_boxes, face_confs, face_results, face_time = self.face_detector.detect(img)
#             all_face_images.append(face_images)
#             all_face_boxes.append(face_boxes)
            
#         face_images = []
#         for i in range(len(all_face_images)):
#             for j in range(len(all_face_images[i])):
#                 face_images.append(all_face_images[i][j])
#         face_detections_per_image = [len(x) if x is not None else 0 for x in all_face_boxes]
        
#         if len(face_images) > 0:
#             # Face recognition
#             face_embeddings = self.facenet.batch_embeddings(face_images)
#             face_embeddings = [self.facenet.l2_normalize(embedding) for embedding in face_embeddings]

#             # Map embeddings to images
#             index = 0
#             for i in range(len(face_detections_per_image)):
#                 for j in range(face_detections_per_image[i]):

#                     if all_embeddings[i] is None:
#                         all_embeddings[i] = []

#                     all_embeddings[i].append(face_embeddings[index])
#                     index += 1
                
#         return all_face_boxes, all_embeddings, person_boxes, person_confs

####################################################################################################

class FaceRecogniser:
    def __init__(self, arcface_path: str, person_detection_model_path:str, face_detection_model_path:str, person_conf:float) -> None:
        self._load_models(arcface_path, person_detection_model_path, face_detection_model_path, person_conf)
    
    def _load_models(self, arcface_path: str, person_detection_model_path:str, face_detection_model_path:str, person_conf:float) -> None:
        # Load face detection model
        self.face_detector = FaceDetector(face_detection_model_path, img_size=320, conf_thresh=0.7)
        
        # Load face recognition model
        self.arcface = ArcFace(arcface_path)
        
        # Load person detection model
        # if person_detection_model_path.split('.')[-1] == 'engine':
        #     self.person_detection = YoloTrt('person', person_detection_model_path, 320, classes=[0], conf_thresh=person_conf)
        # elif person_detection_model_path.split('.')[-1] == 'onnx':
        #     self.person_detection = YoloOnnx(person_detection_model_path, 320, classes=[0], conf_thresh=person_conf)
        # else:
        #     logger.exception('Invalid person detection model')
        #     exit(0)
        
    def getPos(self, landmarks):
        '''
        Done using triangular method
        0, 1, 2 = x of left eye, right eye, nose respectively
        '''
        points = []
        points.append(landmarks[0][0])
        points.append(landmarks[1][0])
        points.append(landmarks[2][0])

        try:
            
            if len(points) == 4:
                if abs(points[0] - points[2]) / abs(points[1] - points[2]) > 2:
                    return 2
                elif abs(points[1] - points[2]) / abs(points[0] - points[2]) > 2:
                    return 0
                else:
                    return 1
            else:                        
                return 1
            
        except ZeroDivisionError as e:
            return 1
        
    def process(self, images: list) -> Union[list, list, list, list]:
        
        all_embeddings = [None] * len(images)
        
        # Person detection
        # _, person_boxes, person_confs, _, _ = self.person_detection.batch_detect(images)
        # _, person_boxes, person_confs, _, _ = self.person_detection.detect(images[0])
        # person_boxes = [person_boxes]
        # person_confs = [person_confs]
        
        # Face detection
        all_face_images = []
        all_face_boxes = []
        lanndmarks_list = []
        for img in images:
            face_images, face_boxes, face_confs, lanndmarks, face_time = self.face_detector.detect(img)
            all_face_images.append(face_images)
            all_face_boxes.append(face_boxes)
            lanndmarks_list.append(lanndmarks)
        
        face_images = []
        face_landmarks = []
        for i in range(len(all_face_images)):
            for j in range(len(all_face_images[i])):
                face_images.append(all_face_images[i][j])
                face_landmarks.append(lanndmarks_list[i][j])
        face_detections_per_image = [len(x) if x is not None else 0 for x in all_face_boxes]
        
        face_poses = []
        if len(face_images) > 0:
            face_embeddings = []
            face_pos = []
            for i in range(len(face_images)):
                img = face_images[i].copy()
                landmark = face_landmarks[i].astype(int)
                landmark = landmark.reshape(-1, 2)
                pos = self.getPos(landmark)
                face_pos.append(pos)
                
                # TODO: This works propely only if there is 1 stream
                face_img = norm_crop(images[0], landmark)
                
                face_embeddings_temp, process_time = self.arcface.get_embeddings([face_img])
                face_embeddings_temp = self.arcface.l2_normalize(face_embeddings_temp[0])
                face_embeddings.append(face_embeddings_temp)
            face_poses.append(face_pos)

            # Map embeddings to images
            index = 0
            for i in range(len(face_detections_per_image)):
                for j in range(face_detections_per_image[i]):

                    if all_embeddings[i] is None:
                        all_embeddings[i] = []

                    all_embeddings[i].append(face_embeddings[index])
                    index += 1
        else:
            # TODO: This might work only for 1 stream, need to change this
            face_poses.append(None)
                
        return all_face_boxes, all_embeddings, face_poses

        
if __name__ == "__main__":
    import config

    # face_recogniser = FaceRecogniser('./models/frozen_models/facenet_tf.pb','./models/frozen_models/yolov5s.onnx',0.7) # facenet
    # face_recogniser = FaceRecogniser('./models/frozen_models/ms1mv3_r34.onnx', './models/frozen_models/yolov5s.onnx',0.7) # archface
    face_recogniser = FaceRecogniser(arcface_path=config.FACE_RECOGNIZER_MODEL, 
                                              person_detection_model_path=config.PERSON_DETECTION_MODEL,
                                              face_detection_model_path=config.FACE_DETECTION_MODEL,
                                              person_conf=config.PERSON_CONF)
    # face_recogniser = FaceRecogniser('./models/frozen_models/facenet_tf.pb')
    
    img = cv2.imread('./test/face_detector_test.jpg')
    img_copy = img[:, :, ::-1]
    
    images = [img_copy, img_copy]


    face_boxes, face_embeddings, _= face_recogniser.process(images)
    
    for i in range(len(images)):
        for box in face_boxes[i]:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            
        cv2.imshow('frame', img)
        if cv2.waitKey(2500) and 0xff == ord('q'):
            break
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # from yolov5_face import FaceDetector
#     from models.face_aligner import norm_crop
    
#     # import pycuda.autoinit
    
#     face_detector = FaceDetector('./models/frozen_models/yolov5n-face-320.onnx', img_size=320)
#     face_recogniser = FaceRecogniser('./models/frozen_models/ms1mv3_r34.onnx')
    
#     orig_images = []
#     images = []
    
#     # Load images
#     img = cv2.imread('./test/face_detector_test.jpg')
#     img_copy = img[:, :, ::-1]
#     orig_images.append(img)
#     images.append(img_copy)
    
#     # img = cv2.imread('./test_2.jpg')
#     # img_copy = img[:, :, ::-1]
#     # orig_images.append(img)
#     # images.append(img_copy)
    
#     # Process the images
#     for _ in range(10):
#         extracted_images, boxes, confs, landmarks, face_time = face_detector.batch_detect(images)
#         print(face_time, boxes)
    
#     # Draw result
#     for i in range(len(images)):
#         img = orig_images[i]
        
#         for box, landmark in zip(boxes[i], landmarks[i]):
#             x1, y1, x2, y2 = box
            
#             # Draw  detection box
#             c1 = (x1, y1)
#             c2 = (x2, y2)
#             print("C1 :", c1)
#             print("C2 :", c2)
#             cv2.rectangle(img, c1, c2, (0, 200, 0), thickness=2)
            
#             # Draw face landmarks
#             landmark = landmark.astype(int)
#             landmark = landmark.reshape(-1, 2)
#             for j in range(5):
                
#                 cv2.circle(img, tuple(list(landmark[j])), 2, (200, 0, 0), -1)
        
#             # Align face        
#             face_img = norm_crop(img, landmark)
            
#             # Get embedding of face
#             for _ in range(10):
#                 face_embeddings, process_time = face_recogniser.get_embeddings([face_img])
#                 print(face_embeddings.shape, process_time)
            
#             cv2.imshow('face', face_img)
#             cv2.waitKey(1500)
            
#         cv2.imshow('result', img)
#         cv2.waitKey(3500)        

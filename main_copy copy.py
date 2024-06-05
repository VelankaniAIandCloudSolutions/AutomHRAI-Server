from face_recogniser import FaceRecogniser
import logging
from logging.handlers import RotatingFileHandler
import traceback

from typing import Union

import os
import cv2
import sys
import math
import time
import faiss
import random
import datetime
from threading import Thread

import numpy as np

import config


from centroid_tracker import CentroidTracker
from videostream import VideoStream
from database.database_config import Database
from utils.utility_functions import draw_rectangle_with_text, is_centroid_in_roi

from integrations.automhr import Velankani
import faiss


os.makedirs('data/logs/', exist_ok=True)
os.makedirs('data/img/', exist_ok=True)

logging.basicConfig(
    handlers=[RotatingFileHandler('data/logs/app.log', maxBytes=25 * 1024 * 1024, backupCount=3), logging.StreamHandler(sys.stdout)],
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

logger = logging.getLogger(__name__)


class Main:
    def __init__(self, is_file: bool=False) -> None:
        logger.info("Starting Program")
        
        self.is_file = is_file
        
        self.face_recogniser = FaceRecogniser(arcface_path=config.FACE_RECOGNIZER_MODEL, 
                                              person_detection_model_path=config.PERSON_DETECTION_MODEL,
                                              face_detection_model_path=config.FACE_DETECTION_MODEL,
                                              person_conf=config.PERSON_CONF)

        self.db = Database()
        self._load_encodings_from_db()
        self.face_cluster_update_time = datetime.datetime.now()
        
        # TODO: Only for debug
        self.time = datetime.datetime(2021, 12, 16, 0, 0, 0)
        
        # self.velankani = Velankani(self.db, self.face_recogniser)
        # self.velankani.velankani_sync_user()
        
    def _load_encodings_from_db(self) -> None:
        known_encodings, self.name, self.employee_ids, self.blacklisted, self.last_id = self.db.get_all_embeddings()
        
        nlist = 0
        if self.last_id > 0:
            known_encodings = known_encodings.astype('float32')
            nlist = math.ceil(len(self.name) / 25)  # number of clusters
            nlist = min(nlist, 10)
            quantiser = faiss.IndexFlatL2(512)
            self.index = faiss.IndexIVFFlat(quantiser, 512, nlist, faiss.METRIC_L2)
            self.index.train(known_encodings)
            self.index.add(known_encodings)
        logger.info(f'Created {nlist} clusters for {len(self.name)}')
    
    def init_rtsp_streams(self) -> None:
        # Get all streams and read them
        self.streams = {}
        self.centroid_trackers = {}
        self.current_trackers = {}
        for i in range(config.STREAMS):
            logger.info(f"Connecting camera {i}")
            stream = VideoStream(config.STREAM_LINKS[i], self.is_file)
            stream.start()  
            self.streams[config.LANES[i]] = stream
            # TODO: Start from last track id
            self.centroid_trackers[config.LANES[i]] = CentroidTracker(maxDisappeared=config.MAX_DISAPPEARED, nextObjectId=0)
            self.current_trackers[config.LANES[i]] = {}
            
    def update_face_clusters(self):
        if (datetime.datetime.now() - self.face_cluster_update_time).total_seconds() >= 60:
            self.update_cluster()
            self.update_visitor_cluster()
            self.face_cluster_update_time = datetime.datetime.now()
        
    def update_cluster(self):
        if self.last_id > 0:
            new_encodings, new_users, new_employee_ids, blacklisted, self.last_id = self.db.get_all_embeddings(from_id=self.last_id)
            if new_users != []:
                new_encodings = new_encodings.astype('float32')
                self.index.add(new_encodings)
                self.name = self.name + new_users
                self.employee_ids = self.employee_ids + new_employee_ids
                self.blacklisted = self.blacklisted + blacklisted
                logger.info(f'Added new {len(new_users)} users to face cluster')
        else:
            self._load_encodings_from_db()
            
    def get_track_id(self, object_tracker, box: list) -> int:
        _track_id = -1
        startX, startY, endX, endY = box
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        inputCentroid = (cX, cY)
        for track_id, centroid in object_tracker.items():
            # Check centroid is inside the box
            if inputCentroid[0] == centroid[0] and inputCentroid[1] == centroid[1]:
                _track_id = track_id
                break
        return _track_id
        
    def process_images(self, images: list) -> Union[list, list]:
        face_boxes, face_embeddings, face_poses = self.face_recogniser.process(images)
        
        return face_boxes, face_embeddings, face_poses
    
    def match_embedding(self, embedding: list):
        if self.last_id > 0:
            distances, indices = self.index.search(np.array([embedding]), 1)
            prediction = indices[0][0] if distances[0][0] <= config.FACE_RECOGNIZER_THRESH else -1
            
            if prediction != -1:
                name = self.name[prediction]
                employee_id = self.employee_ids[prediction]
            else:
                name = 'Unknown'
                employee_id = None
        else:
            return 'Unknown', None, -1, 999
        return name, employee_id, prediction, distances[0][0]
    
    def check_overlap(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        overlap_percentage = interArea / boxBArea

        # return the intersection over union value
        return overlap_percentage

        
    def map_person_face_boxes(self, person_boxes:list, person_confs:list, face_boxes:list, face_embeddings:list):
        ret_val = []
        
        processed_face_indexes = []
        
        # For each person find corresponding face box and add them to dictionary
        for j, person_box in enumerate(person_boxes):
            flag = True
            for i in range(len(face_boxes)):
                overlap_percentage = self.check_overlap(person_box, face_boxes[i])
                if overlap_percentage > 0.9:
                    ret_val.append({"person_box": person_box.tolist(),
                                    "person_conf": person_confs[j],
                                    "face_box":face_boxes[i],
                                    "face_embedding": face_embeddings[i]})
                    processed_face_indexes.append(i)
                    flag = False
                  
            # Add to dict with face box as None if no face detected  
            if flag:
                ret_val.append({"person_box": person_box.tolist(),
                                "person_conf": person_confs[j],
                                "face_box":[],
                                "face_embedding": []})

        # Add to dict with person box as None if no person detected
        for i in range(len(face_boxes)):
            if i not in processed_face_indexes:
                ret_val.append({"person_box": [],
                                "person_conf": 0.0,
                                "face_box":face_boxes[i],
                                "face_embedding": face_embeddings[i]})
        
        return ret_val
    
    def get_current_frames(self) -> dict:
        image_data = {'frames': [], 'lane_ids': []}
        for i in range(config.STREAMS):
            status, frame = self.streams[config.LANES[i]].read()
            if status:
                image_data['frames'].append(frame)
                image_data['lane_ids'].append(config.LANES[i])
        
        self.time = self.time + datetime.timedelta(0, 0, milliseconds=40)
        
        return image_data
    
    def save_image(self, image, bounding_box, track_id, status):
        # Draw bounding box on image
        # TODO set color for bounding box based on blacklisted/whitelisted
        x1, y1, x2, y2 = bounding_box
        
        if status == 0:
            color = (0, 191, 255)
        elif status == 1:
            color = (0, 200, 0)
        else:
            color = (0, 0, 200)
        draw_rectangle_with_text(image, (x1, y1), (x2, y2), color, thickness=3)
        
        # Save image
        filename = f'{config.DEVICE}_{datetime.datetime.now().timestamp()}_{track_id}.jpg'
        cv2.imwrite(f'data/img/{filename}', image)
        
        # Return image path
        return filename
    
    
    def add_tracker(self, lane_id, track_id):
        if lane_id not in self.current_trackers:
            self.current_trackers[lane_id] = {}
        
        if track_id not in self.current_trackers[lane_id]:
            self.current_trackers[lane_id][track_id] = {}

        self.current_trackers[lane_id][track_id]['count'] = 0
        self.current_trackers[lane_id][track_id]['raw_log_ids'] = []
        self.current_trackers[lane_id][track_id]['face_ids'] = []
        self.current_trackers[lane_id][track_id]['face_names'] = []
        self.current_trackers[lane_id][track_id]['status'] = []
        self.current_trackers[lane_id][track_id]['face_distances'] = []
        self.current_trackers[lane_id][track_id]['face_boxes'] = []
        self.current_trackers[lane_id][track_id]['person_box'] = []
        self.current_trackers[lane_id][track_id]['person_conf'] = 0
        self.current_trackers[lane_id][track_id]['face_poses'] = []
        self.current_trackers[lane_id][track_id]['face_embeddings'] = {0:[], 1:[], 2:[]}
        self.current_trackers[lane_id][track_id]['user_type'] = []
        self.current_trackers[lane_id][track_id]['img_paths'] = []
    

    def sort(self, lane_id, track_id):
        tracker = self.current_trackers[lane_id][track_id]
        
        distances = tracker['face_distances']
        if len(distances) > 0:
            min_distance_index = distances.index(min(distances))
            
            raw_log_id = tracker['raw_log_ids'][min_distance_index]
            user_type = tracker['user_type'][min_distance_index]
            
            if user_type == 0:
                self.db.add_log(raw_log_id, lane_id)
            else:
                if distances[min_distance_index] <= config.FACE_RECOGNIZER_THRESH_VISITOR:
                    visitor_id = self.add_old_visitor(tracker, min_distance_index)
                else:
                    visitor_id = self.add_new_visitor(tracker)

                if visitor_id is not None:
                    self.db.add_visitor_log(visitor_id, raw_log_id, lane_id)
            logger.info(f'Added log {raw_log_id} with distance {distances[min_distance_index]}, usertype:{user_type}')
        

    
    def postprocess_image(self, image, lane_id, face_boxes, face_embeddings, face_poses):
        # current_time = self.time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        # current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        # kafka_data = {"frame_time": current_time,
        #                 "lane_id": lane_id,
        #                 "objects": []}
        
        # if person_boxes is None:
        #     person_boxes = []
        
        # Update tracker
        # TODO: testing with person box, if not working change to face box
        # object_tracker, deregistered_objects = self.centroid_trackers[lane_id].update(person_boxes)
        object_tracker, deregistered_objects = self.centroid_trackers[lane_id].update(face_boxes)
        for deregistered_track_id in deregistered_objects:
            if deregistered_track_id in self.current_trackers[lane_id]:
                self.sort(lane_id, deregistered_track_id)
                del self.current_trackers[lane_id][deregistered_track_id]
                
        # person_face_mapping = self.map_person_face_boxes(person_boxes, person_confs, face_boxes, face_embeddings)
                   
        if face_boxes is not None and face_embeddings is not None: 
            # Iterate over each predictions
            try:
                for face_box, face_embedding, face_pose in zip(face_boxes, face_embeddings, face_poses):
                    # face_box, face_embedding, person_box, person_conf = mapping["face_box"], mapping["face_embedding"], mapping["person_box"], mapping["person_conf"]
                        
                    # Check person centroid is inside ROI
                    if len(config.ROI) == 4 and is_centroid_in_roi(face_box, config.ROI):
                    
                        track_id = self.get_track_id(object_tracker, face_box)
                        if track_id not in self.current_trackers[lane_id]:
                            self.add_tracker(lane_id, track_id)
                    
                        if track_id < 0:
                            return
                        
                        self.current_trackers[lane_id][track_id]['count'] += 1
                        
                        # Match embedding to db embeddings
                        name, distance = ['Unknown', 999.0]
                        if face_box != []:
                            
                            x1, y1, x2, y2 = face_box
                            face_area = (x2-x1) * (y2-y1)
                            if face_area < config.MIN_FACE_AREA:
                                continue
                            
                            name, employee_id, prediction, distance = self.match_embedding(face_embedding)
                            # visitor_id, visitor_prediction, visitor_distance = self.match_visitor_embedding(face_embedding)
                            
                            # if distance <= visitor_distance and prediction != -1:
                            user_type = 0   # Registered user
                            face_id = employee_id
                            face_name = name
                            face_distance = distance
                            # else:
                            #     user_type = 1   # Visitor
                            #     face_id = visitor_id
                            #     face_name = f'{visitor_id}'
                            #     face_distance = visitor_distance

                            self.current_trackers[lane_id][track_id]['user_type'].append(user_type)
                            self.current_trackers[lane_id][track_id]['face_ids'].append(face_id)
                            self.current_trackers[lane_id][track_id]['face_names'].append(face_name)
                            self.current_trackers[lane_id][track_id]['face_distances'].append(face_distance)
                            self.current_trackers[lane_id][track_id]['face_boxes'].append(face_box)
                            self.current_trackers[lane_id][track_id]['face_poses'].append(face_pose)
                            self.current_trackers[lane_id][track_id]['face_embeddings'][face_pose].append(face_embedding)
                            # self.current_trackers[lane_id][track_id]['person_box'] = person_box
                            # self.current_trackers[lane_id][track_id]['person_conf'] = person_conf
                            
                            status = 0
                            if user_type ==0 and prediction > 0:
                                status = 1
                                if self.blacklisted[prediction]:
                                    status = 2
                            self.current_trackers[lane_id][track_id]['status'].append(status)
                            
                            # bgr_img = image[:, :, ::-1]
                            bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            img_path = self.save_image(bgr_img, face_box, track_id, status)
                            self.current_trackers[lane_id][track_id]['img_paths'].append(img_path)
                            
                                                            
                            face_box = [str(i) for i in face_box]
                            # person_box = [str(i) for i in person_box]
                            raw_log_id = self.db.add_raw_log(img_path, 
                                                datetime.datetime.now(),
                                                status,
                                                face_id,
                                                float("{:.2f}".format(face_distance)),
                                                ','.join(face_box),
                                                '',
                                                0.0)
                            
                            self.current_trackers[lane_id][track_id]['raw_log_ids'].append(raw_log_id)
            except:
                traceback.print_exc()
                exit(0)
        
    def process_streams(self) -> None:
        # Get frames from all streams
        image_data = self.get_current_frames()
        if len(image_data['frames']) == 0:
            return
            
        # Process them for face recognition
        images = [img[:, :, ::-1] for img in image_data['frames']]
        # images = [cv2.flip(img, 1) for img in image_data['frames']]
        face_boxes, face_embeddings, face_poses = self.process_images(images)
        
        # Iterate over each images
        for i in range(len(images)):
            lane_id = image_data['lane_ids'][i]
            self.postprocess_image(images[i], lane_id, face_boxes[i], face_embeddings[i], face_poses[i])
                
        return image_data, self.current_trackers
        
        
        
    def add_new_visitor(self, tracker):
        face_features = tracker['face_embeddings']
        if len(face_features[1]) == 0:
            return None

        visitor_id = f'{datetime.datetime.now().timestamp()}'
        
        index = [index for index in range(len(tracker['face_poses'])) if tracker['face_poses'][index]==1]
        index = random.choice(index)
        img_path = tracker['img_paths'][index]
        face_box = tracker['face_boxes'][index]
        
        # Crop face from image and save
        x1, y1, x2, y2 = face_box
        img = cv2.imread(os.path.join('data/img/', img_path))
        img = img[y1:y2, x1:x2]
        face_img_path = os.path.join('data/visitor_reg_img/', f'{visitor_id}.jpg')
        cv2.imwrite(face_img_path, img)        
        
        # Create new visitor and save the embeddings
        self.db.add_visitor(visitor_id, face_img_path)
        for feature in face_features:
            #if feature != 1:
            #    continue
            array_mean = np.mean(face_features[feature], axis=0).tolist()
            if not isinstance(array_mean, list):
                array_mean = np.zeros(512)
            face_features[feature] = array_mean
            
        del face_features[0]
        del face_features[2]
        self.db.add_visitor_embeddings(visitor_id, face_features)
        
        self.update_visitor_cluster()
        
        return visitor_id
    
    def add_old_visitor(self, tracker, index):
        face_id = tracker['face_ids'][index]
        
        visitor_id = face_id
        # Update last detection time
        self.db.update_visitor_last_detection_timestamp(str(visitor_id))
        # TODO: Update embedding if older than a month
        
        img_path = tracker['img_paths'][index]
        face_box = tracker['face_boxes'][index]
        
        # Crop face from image and save
        x1, y1, x2, y2 = face_box
        img = cv2.imread(os.path.join('data/img/', img_path))
        img = img[y1:y2, x1:x2]
        
        os.makedirs(f'data/visitor_images/{visitor_id}', exist_ok=True)
        face_img_path = os.path.join(f'data/visitor_images/{visitor_id}', f'{datetime.datetime.now().timestamp()}.jpg')
        cv2.imwrite(face_img_path, img)        
        
        return visitor_id
       
if __name__ == "__main__":
    
    is_file = False
    app = Main(is_file)
    
    if is_file:
        app.init_video_streams()
    else:
        app.init_rtsp_streams()
        
    if config.RECORD_VIDEO:
        out = cv2.VideoWriter('./data/main_output.avi', 
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              25, (1280, 720))
    
    while True:
        since = time.time()
        app.update_face_clusters()
        image_datas, prediction_results = app.process_streams()
        print(time.time()-since)
        
        if config.SHOW_VIDEO:
            if len(image_datas['frames']) > 0:
                # Get each frame and draw bb
                for i in range(len(image_datas['frames'])):
                    img = image_datas['frames'][i]
                    lane = image_datas['lane_ids'][i]
                    predictions = prediction_results[lane]
                    
                    for track_id, preds in predictions.items():
                        if track_id >= 0:
                            if len(preds['face_boxes']) > 0 and len(preds['face_boxes'][-1]) == 4:
                                x1, y1, x2, y2 = preds['face_boxes'][-1]
                                
                                min_distance_index = preds['face_distances'].index(min(preds['face_distances']))
                                name = preds['face_names'][min_distance_index]
                                status = preds['status'][min_distance_index]
                                user_type = preds['user_type'][min_distance_index]
                                distance = preds['face_distances'][min_distance_index]
                                
                                if status == 2:
                                    text = f'{track_id}: {name} - Blacklisted'
                                else:    
                                    text = f'{track_id}: {name}'
                                
                                if user_type == 1 and distance < config.FACE_RECOGNIZER_THRESH_VISITOR:
                                    color = (200, 0, 0)
                                elif status == 0:
                                    color = (0, 191, 255)
                                elif status == 1:
                                    color = (0, 200, 0)
                                else:
                                    color = (0, 0, 200)
                                draw_rectangle_with_text(img, (x1, y1), (x2, y2), color, text=text)
                                
                                # if preds['face_boxes'][-1] != []:
                                #     x1, y1, x2, y2 = preds['face_boxes'][-1]
                                #     draw_rectangle_with_text(img, (x1, y1), (x2, y2), color)
                    
                    img = cv2.resize(img, (1280, 720))
                    
                    if config.RECORD_VIDEO:
                        out.write(img)
                    
                    cv2.imshow(f'{lane}', img)
                    
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

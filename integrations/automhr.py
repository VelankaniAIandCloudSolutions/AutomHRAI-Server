import cv2
from database.database_config import Database
from face_recogniser import FaceRecogniser

import config
import os
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
import config

class Velankani():

    def __init__(self, db:Database = None, fr: FaceRecogniser = None) -> None:
        self.db : Database = db
        self.fr: FaceRecogniser = fr
        
    def get_embeddings(self, image):
        _, face_embeddings, face_poses = self.fr.process([image])
       
        return face_embeddings[0], face_poses[0]
    
    def velankani_sync_user(self, full_path=None, directory_name=None):
        
            
            # existing_user = self.db.query_user(directory_name)
            # if existing_user:
            #     # Assuming you have a method to update users
            #     # employee_id, name, mobile, email, embedding_dict
            #     # self.db.edit_user(directory_name, emb_img)
            #     # print(f"Updated user {directory_name}")
            #     pass
            # else:
            #     new_id = self.db.get_next_user_id()
            
            
            
            if full_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                
                directory_name = os.path.basename(directory_name)                
                data = directory_name.split("--")
                emp_name = data[0]
                emp_mobile = data[1]
                emp_email = data[2]
                existing_user = self.db.query_user(emp_name)
                    
                    
                if not existing_user:
                    new_id = self.db.get_next_user_id() 
                    
                    emb_img = {}
                    img_array = cv2.imread(full_path)
                    emb_img[new_id] = self.get_embeddings(img_array)[0][0]
                
                    self.db.add_user(employee_id=new_id, 
                                     name=emp_name, 
                                     mobile=emp_mobile, 
                                     email=emp_email, 
                                     embedding_dict=emb_img, 
                                     allowed=True, 
                                     blacklist=False)
                    
                    print(f"Added new user {directory_name} with employee ID {new_id}")
                    
    
# class DirectoryWatcher:
#     def __init__(self, sync_func, path):
#         self.event_handler = FileSystemEventHandler()
#         self.event_handler.on_created = self.on_created
#         self.sync_func = sync_func
#         self.path = path
#         self.observer = Observer()

#     def on_created(self, event):
#         if event.is_directory:
#             return
#         if event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#             directory_name = os.path.basename(os.path.dirname(event.src_path))
#             self.sync_func(event.src_path, directory_name)

#     def start(self):
#         self.observer.schedule(self.event_handler, self.path, recursive=True)
#         self.observer.start()
#         try:
#             while True:
#                 pass  # or sleep
#         finally:
#             self.observer.stop()
#             self.observer.join()
    
if __name__ == "__main__":
    db = Database()
    fr = FaceRecogniser(config.FACE_RECOGNIZER_MODEL, 
                        config.PERSON_DETECTION_MODEL, 
                        config.FACE_DETECTION_MODEL, 
                        config.PERSON_CONF)
    v = Velankani(db, fr)
    # v.velankani_sync_user()
    # watcher = DirectoryWatcher(v.velankani_sync_user, "/home/heet/workspace/fr/data/users_data")
    # watcher.start()
    
# if __name__ == "__main__":
#     db = Database()
#     fr = FaceRecogniser(arcface_path=config.FACE_RECOGNIZER_MODEL, 
#                                               person_detection_model_path=config.PERSON_DETECTION_MODEL,
#                                               face_detection_model_path=config.FACE_DETECTION_MODEL,
#                                               person_conf=config.PERSON_CONF)
#     v = Velankani(db, fr)
    
#     v.velankani_sync_user()
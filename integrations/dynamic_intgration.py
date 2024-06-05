import os
from database.database_config import Database
from face_recogniser import FaceRecogniser
import config
from PIL import Image
import numpy as np
from datetime import datetime as current_time

class Velankani:

    def __init__(self, db: Database = None, fr: FaceRecogniser = None) -> None:
        self.db: Database = db
        self.fr: FaceRecogniser = fr
        
    def get_embeddings(self, image_data):
        _, face_embeddings, face_poses = self.fr.process([image_data])
        
        if face_embeddings:
            return face_embeddings[0], face_poses[0]
        else:
            return None, None
    
    def sync_user_from_directory(self, directory_path):
        # print(directory_path)
        username, user_id = directory_path.split('/')[-1].split('--')
        
        emb_img = {}
        for i, image_filename in enumerate(os.listdir(directory_path)):
            
            img_path = os.path.join(directory_path, image_filename)
            image_open = Image.open(img_path).convert('RGB')
            img_array = np.array(image_open)
            
            if len(self.get_embeddings(img_array)[0][0]) > 0:
                emb_img[i] = self.get_embeddings(img_array)[0][0]
        
        self.db.add_user(employee_id=user_id, name=username, mobile=None, email=None, embedding_dict=emb_img, allowed=True, blacklist=False)
        print(username, "added!")


    def add_new_user(self):
        
        user_in_database = self.db.get_all_users()        
        missing_in_database = [data for data in os.listdir(config.USER_DATA_PATH) if data not in user_in_database]
        
        if len(missing_in_database) > 0:
            for name in missing_in_database:
                full_path = os.path.join(os.getcwd(), os.path.join(config.USER_DATA_PATH, name))
                self.sync_user_from_directory(full_path)
        else:
            print(current_time.now(), "database is up to date!")
    

if __name__ == "__main__":
    db = Database()
    fr = FaceRecogniser(config.FACE_RECOGNIZER_MODEL, 
                        config.PERSON_DETECTION_MODEL, 
                        config.FACE_DETECTION_MODEL, 
                        config.PERSON_CONF)
    
    velankani = Velankani(db, fr)
    velankani.add_new_user()
    # velankani.sync_user_from_directory("/home/heet/workspace/fr/data/users_data/kadir--VISL064343")
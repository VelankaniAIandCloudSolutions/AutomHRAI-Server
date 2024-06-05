from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.sql import func
from passlib.hash import sha256_crypt
from database.database_init import *
# from database_init import *
# from storage import Storage

# import register_face as rf
import numpy as np
import traceback
import datetime

class Database:
    
    def __init__(self):
        session = sessionmaker(bind=engine)
        self.Session = scoped_session(session)
        if self.is_info_empty():
            self.add_info(datetime.datetime(2001, 1, 1, 1, 1, 1, 1), True, True, False, True)
        

    def is_info_empty(self):
        session = self.Session()
        result = session.query(Info).first()
        if result is None:
            return True
        return False
    
    def add_info(self, last_sync_time, automhr, mask_detection, face_id, temperature):
        session = self.Session()
        try:
            info = Info(last_sync_time=last_sync_time, automhr=automhr, mask_detection=mask_detection, face_id=face_id, temperature=temperature)
            session.add(info)
            session.commit()
        except Exception as e:
            traceback.print_exc()
            session.rollback()
        finally:
            session.close()
            
    def set_lastsync_time(self, last_sync_time):
        session = self.Session()
        try:
            result = session.query(Info).get(1)
            result.last_sync_time = last_sync_time
            session.commit()
        except Exception as e:
            session.rollback()
        session.close()
    
    
    def query_user(self, name):
        session = self.Session()
        # This queries for a single user by their name
        return session.query(Users).filter_by(name=name).first()

    def get_next_user_id(self):
        session = self.Session()
        # This finds the maximum user ID and adds one to it to generate a new ID
        max_id = session.query(func.max(Users.employee_id)).scalar()
        return int(max_id) + 1 if max_id else 1

    
    # def `add_embeddings`(self, employee_id, user_id, embedding_dict):
    #     session = self.Session()
    #     try:
    #         for key, value in embedding_dict.items():
    #             emb_string = ','.join(str(n) for n in value)
    #             column_string = ','.join(['vector{}'.format(i) for i in range(512)])
    #             insert_string = f"INSERT INTO embeddings (embedding_class, employee_id, {column_string}) VALUES ({key}, {user_id}, {emb_string})"
    #             engine.execute(insert_string)
                
    #     except Exception as e:
    #         traceback.print_exc()
    #         session.rollback()
    #     finally:
    #         session.close()        
        
    
    def add_user(self, employee_id, name, mobile, email, embedding_dict, allowed=True, blacklist=False):
        session = self.Session()
        try:
            user = Users(name=name, employee_id=employee_id, email=email, mobile=mobile, allowed=allowed, blacklist=blacklist)
            session.add(user)
            session.commit()
            user_id = user.user_id

            for key, value in embedding_dict.items():
                # emb = Embeddings(embedding=str(value), embedding_class=key, user_id=user_id)
                # session.add(emb)
                # session.commit()
                emb_string = ','.join(str(n) for n in value)
                column_string = ','.join(['vector{}'.format(i) for i in range(512)])
                insert_string = f"INSERT INTO embeddings (embedding_class, user_id, {column_string}) VALUES ({key}, {user_id}, {emb_string})"
                engine.execute(insert_string)

        except Exception as e:
            traceback.print_exc()
            session.rollback()
        finally:
            session.close()

    def user_access(self, employee_id):
        session = self.Session()

        if session.query(Users).filter(Users.employee_id == employee_id).count() > 0:
            session.close()
            return True
        else:
            session.close()
            return False

            
    def edit_user(self, employee_id, name, mobile, email, embedding_dict, allowed=True, blacklist=False):
        session = self.Session()
        try:
            result = session.query(Users).filter(Users.employee_id == employee_id).first()
            if result is not None:
                result.name = name
                result.mobile = mobile
                result.email = email
                result.allowed = allowed
                result.blacklist = blacklist
                session.commit()

                for key, value in embedding_dict.items():
                    emb_string = ','.join(f'vector{i}={n}' for i, n in enumerate(value))
                    insert_string = f"UPDATE embeddings SET {emb_string} WHERE embedding_class={key} and user_id={result.user_id}"
                    engine.execute(insert_string)
            else:
                self.add_user(employee_id, name, mobile, email, embedding_dict)
        except Exception as e:
            traceback.print_exc()
            session.rollback()
        finally:
            session.close()

    def delete_user(self, employee_id):
        session = self.Session()
        try:
            # print(employee_id)
            user_result = session.query(Users).filter(Users.employee_id == employee_id).first()
            # print(user_result.user_id)
            session.query(Embeddings).filter(Embeddings.user_id == user_result.user_id).delete()

            # session.delete(user_result)
            session.query(Users).filter(Users.employee_id == employee_id).delete()
            session.commit()
        except Exception as e:
            traceback.print_exc()
            session.rollback()
        finally:
            session.close()
    
     
    def get_all_embeddings(self, from_id=0):
        session = self.Session()
        result = engine.execute(f'SELECT * FROM embeddings WHERE embedding_id > {from_id}')

        all_embeddings = []
        users = []
        employee_ids = []
        email_ids = []
        blacklisted = []
        last_id = from_id
        for row in result:
            user_id = row[2]
            user_result = session.query(Users).get(user_id)
            
            if (user_result.allowed):
                emb = list(row[3:])
                # if emb != 'nan':
                #     emb = ast.literal_eval(emb)
                all_embeddings.append(emb)
                users.append(user_result.name)
                employee_ids.append(user_result.employee_id)
                blacklisted.append(user_result.blacklist)
                email_ids.append(user_result.email)
                last_id = row[0]

        session.close()
        
        return np.array(all_embeddings), users, employee_ids, blacklisted, last_id, email_ids
    
    
    def add_raw_log(self, image_path, timestamp, status, face_id, face_distance, face_box, person_box, person_conf):
        session = self.Session()

        raw_log_id = -1
        try:
            raw_log = RawLogs(image_path=image_path,
                              timestamp = timestamp,
                              status=status,
                              face_id=face_id, 
                              face_distance=face_distance,
                              face_box=face_box,
                              person_box=person_box,
                              person_conf=person_conf)
            session.add(raw_log)
            session.commit()
            
            raw_log_id = raw_log.id
        except Exception as e:
            # logger.exception('Failed to add raw log to Logs table')
            session.rollback()
        finally:
            session.close()
            
        return raw_log_id
    
    
    def add_log(self, raw_log_id, lane_id):
        session = self.Session()

        log_id = -1
        try:
            log = Logs(raw_log_id=raw_log_id,
                       lane_id=lane_id,
                       sync_status=0,
                       upload_status=0)
            session.add(log)
            session.commit()
            
            log_id = log.id
        except Exception as e:
            # logger.exception('Failed to add raw log to Logs table')
            session.rollback()
        finally:
            session.close()
            
        return log_id
    
    def get_all_users(self):
        session = self.Session()
        try:
            user_data = []
            user_result = engine.execute(f'SELECT * FROM Users')
            for data in user_result:
                # print(data.employee_id)
                # print(data.name)
                data_to_add = data.name+"--"+data.employee_id
                user_data.append(data_to_add)
                
            return user_data
        
        except Exception as e:
            traceback.print_exc()
            session.rollback()
            
        finally:
            session.close()

if __name__ == "__main__":
    db = Database()
    # db.add
    # print(db.get_all_users())
    # db.delete_user("VAC00013")
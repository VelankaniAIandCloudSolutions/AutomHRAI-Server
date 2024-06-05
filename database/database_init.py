from sqlalchemy import Column, Integer, Boolean, DateTime, ForeignKey, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mssql import NVARCHAR
from sqlalchemy import create_engine
import datetime

Base = declarative_base()


class RawLogs(Base):
    __tablename__ = 'raw_logs'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True)
    image_path = Column(String)
    timestamp = Column(DateTime)
    status = Column(Integer)        # 0: None, 1: Whitelisted, 2:Blacklisted
    
    face_id = Column(String)
    face_distance = Column(Float)
    face_box = Column(String)
    
    person_box = Column(String)
    person_conf = Column(Float)
    
    lane_id = Column(Integer)

class Users(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True)
    # automhr_id = Column(Integer)
    employee_id = Column(String)
    name = Column(String)
    mobile = Column(String)
    email = Column(String)
    blacklist = Column(Boolean)

    allowed = Column(Boolean)
    
    

class Logs(Base):
    __tablename__ = 'logs'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True)
    raw_log_id = Column(Integer, ForeignKey('raw_logs.id'))
    
    lane_id = Column(Integer)
    
    sync_status = Column(Integer, default=0)
    upload_status = Column(Integer, default=0)

    
# abot :
class Info(Base):
    __tablename__ = 'info'
    info_id = Column(Integer, primary_key=True)
    last_sync_time = Column(DateTime, default=datetime.datetime(2001, 1, 1, 1, 1, 1, 1))
    
    automhr = Column(Boolean, default=True)
    mask_detection = Column(Boolean, default=True)
    face_id = Column(Boolean, default=False)
    temperature = Column(Boolean, default=True)
    

# Create embeddings table
attr_dict = {'__tablename__': 'embeddings',
             'embedding_id': Column(Integer, primary_key=True),
             'embedding_class': Column(Integer),
             'user_id': Column(Integer, ForeignKey('users.employee_id'))
             }

for i in range(0, 512):
    attr_dict['vector{}'.format(i)] = Column(Float)

Embeddings = type('embeddings', (Base,), attr_dict)



engine = create_engine('sqlite:///data/database.db', echo=False)
Base.metadata.create_all(engine)    

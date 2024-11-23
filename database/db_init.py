import os
import uuid
from sqlalchemy import create_engine, inspect, Column, Integer, String, TIMESTAMP, Numeric, MetaData, text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DB_URL")
DB_NAME = os.getenv("DB_NAME")
FULL_DATABASE_URL = f"{DB_URL}/{DB_NAME}"

base_engine = create_engine(DB_URL)
engine = create_engine(FULL_DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class ImageInfo(Base):
    __tablename__ = 'image_info'
    id_img_pk = Column(Integer, primary_key=True)
    file_name = Column(String, nullable=False)
    processed_at = Column(TIMESTAMP, nullable=False)
    cow_count = Column(Integer, nullable=False)

class CowDetails(Base):
    __tablename__ = 'cow_details'
    id_detected_cow_pk = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    id_img_fk = Column(Integer, ForeignKey('image_info.id_img_pk'), nullable=False)
    cow_centroid = Column(String, nullable=False)
    prediction_accuracy = Column(Numeric(3, 2), nullable=False)
    posture = Column(String(20))

def create_database():
    with base_engine.connect() as connection:
        result = connection.execute(text(f"SELECT 1 FROM pg_database WHERE datname='{DB_NAME}'"))
        exists = result.scalar() is not None

        if not exists:
            print(f"Creando la base de datos '{DB_NAME}'...")
            connection.execute(text("COMMIT"))
            connection.execute(text(f"CREATE DATABASE {DB_NAME}"))
        else:
            print(f"La base de datos '{DB_NAME}' ya existe.")

def enable_pgcrypto_extension():
    with engine.connect() as connection:
        print("Activando la extensión pgcrypto...")
        connection.execute(text("CREATE EXTENSION pgcrypto"))

def create_tables():
    metadata = MetaData()
    metadata.reflect(bind=engine)

    if 'image_info' not in metadata.tables:
        print("Creando la tabla 'image_info'...")
        ImageInfo.__table__.create(bind=engine)
    else:
        print("La tabla 'image_info' ya existe.")

    if 'cow_details' not in metadata.tables:
        print("Creando la tabla 'cow_details'...")
        CowDetails.__table__.create(bind=engine)
    else:
        print("La tabla 'cow_details' ya existe.")

if __name__ == "__main__":
    create_database()
    enable_pgcrypto_extension()
    create_tables()

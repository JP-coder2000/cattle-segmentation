import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from database.db_init import ImageInfo, CowDetails

load_dotenv()
DATABASE_URL = os.getenv("DB_FULL_URL")

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def add_image_info(proccesed_at, cow_count, file_name):
    session = Session()
    try:
        new_image = ImageInfo(file_name=file_name, proccesed_at=proccesed_at, cow_count=cow_count)
        session.add(new_image)
        session.commit()
        return new_image.id_img_pk
    except Exception as e:
        session.rollback()
        print(f"Error al añadir el registro de imagen: {e}")
    finally:
        session.close()

def add_cow_detail(id_img_fk, centroid, accuracy, posture):
    session = Session()
    try:
        new_cow = CowDetails(
            id_img_fk=id_img_fk,
            cow_centroid=f"({round(centroid[0], 2)}, {round(centroid[1], 2)})",
            prediction_accuracy=accuracy,
            posture=posture
        )
        session.add(new_cow)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error al añadir el detalle de vaca: {e}")
    finally:
        session.close()

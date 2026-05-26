from sqlalchemy import create_engine, Column, Integer, Float, Text, DateTime, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/Real_estate"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class RealEstateObject(Base):
    __tablename__ = 'real_estate_objects'

    id = Column(Integer, primary_key=True, index=True)
    floor = Column(Integer)
    total_area = Column(Float)
    living_area = Column(Float)
    kitchen_area = Column(Float)
    year = Column(Integer)
    flat_status = Column(Integer)
    metro_minutes = Column(Integer)
    description = Column(Text)

    prices = relationship("Price", back_populates="real_estate_object")
    predictions = relationship("Prediction", back_populates="real_estate_object")

class Price(Base):
    __tablename__ = 'prices'

    id = Column(Integer, primary_key=True, index=True)
    object_id = Column(Integer, ForeignKey('real_estate_objects.id'))
    price_value = Column(Float, nullable=True) # Фактическая цена (пока неизвестна при прогнозе)

    real_estate_object = relationship("RealEstateObject", back_populates="prices")

class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, index=True)
    object_id = Column(Integer, ForeignKey('real_estate_objects.id'))
    model_name = Column(String)
    predicted_price = Column(Float)

    real_estate_object = relationship("RealEstateObject", back_populates="predictions")

def init_db():
    Base.metadata.create_all(bind=engine)
    
print("База данных создана успешно!")
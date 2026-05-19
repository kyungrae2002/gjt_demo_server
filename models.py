from sqlalchemy import Column, Integer, String, DateTime, func
from db import Base


class Product(Base):
    __tablename__ = "products"

    id         = Column(Integer, primary_key=True, index=True)
    제품이름   = Column(String(255), nullable=False)
    자산번호   = Column(String(255), nullable=False)
    필요인원수 = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

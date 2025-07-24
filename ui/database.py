import pandas as pd
from sqlalchemy import create_engine

def load_data_from_db():
    try:
        # Kết nối MySQL qua SQLAlchemy
        engine = create_engine("mysql+pymysql://root:lequocmysql@localhost/data_mining_project")  # Thay root và password
        query = "SELECT * FROM students"
        data = pd.read_sql(query, engine)
        return data, "Đã tải dữ liệu từ MySQL!"
    except Exception as e:
        return None, f"Lỗi: {str(e)}"

def load_data_from_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data, "Đã tải dữ liệu từ CSV!"
    except Exception as e:
        return None, f"Lỗi: {str(e)}"
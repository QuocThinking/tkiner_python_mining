from database import load_data_from_db as db_load, load_data_from_csv as csv_load
from tkinter import filedialog

def load_data_from_db():
    return db_load()

def load_data_from_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        return csv_load(file_path)
    return None, "Không chọn file CSV!"
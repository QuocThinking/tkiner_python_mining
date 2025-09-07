from tkinter import filedialog
import pandas as pd

def load_data_from_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            data = pd.read_csv(file_path)
            return data, None
        except Exception as e:
            return None, f"Lỗi khi đọc CSV: {str(e)}"
    return None, "Không chọn file CSV!"
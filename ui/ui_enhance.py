import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttkb
from ttkbootstrap.tooltip import ToolTip

from table_manager import setup_treeview

def setup_enhanced_ui(self):
    control_frame = ttkb.Frame(self.main_frame, padding=10)
    control_frame.grid(row=2, column=0, sticky="ew")  # Nhường row 0,1 cho status và info
    control_frame.columnconfigure(0, weight=1)
    control_frame.columnconfigure(1, weight=1)
    control_frame.columnconfigure(2, weight=1)

    self.load_db_button = ttkb.Button(
        control_frame,
        text="Tải từ DB",
        command=self.load_data_from_db,
        bootstyle="primary",
        width=15
    )
    self.load_db_button.grid(row=0, column=0, padx=5, pady=5)
    ToolTip(self.load_db_button, text="Tải dữ liệu từ cơ sở dữ liệu")
    self.load_db_button.bind("<Enter>", lambda e: self.load_db_button.configure(bootstyle="primary-outline"))
    self.load_db_button.bind("<Leave>", lambda e: self.load_db_button.configure(bootstyle="primary"))

    self.load_csv_button = ttkb.Button(
        control_frame,
        text="Tải từ CSV",
        command=self.load_data_from_csv,
        bootstyle="secondary",
        width=15
    )
    self.load_csv_button.grid(row=0, column=1, padx=5, pady=5)
    ToolTip(self.load_csv_button, text="Tải dữ liệu từ file CSV")
    self.load_csv_button.bind("<Enter>", lambda e: self.load_csv_button.configure(bootstyle="secondary-outline"))
    self.load_csv_button.bind("<Leave>", lambda e: self.load_csv_button.configure(bootstyle="secondary"))

    algo_frame = ttkb.Frame(control_frame)
    algo_frame.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
    ttkb.Label(algo_frame, text="Thuật toán:", font=("Helvetica", 10, "bold"), bootstyle="light").pack(side="left")
    algo_options = ["Naive Bayes", "KNN", "K-Means"]
    algo_combo = ttkb.Combobox(
        algo_frame,
        textvariable=self.algo_var,
        values=algo_options,
        state="readonly",
        width=15,
        bootstyle="info"
    )
    algo_combo.pack(side="left", padx=5)
    ToolTip(algo_combo, text="Chọn thuật toán để chạy")

    k_frame = ttkb.Frame(control_frame)
    k_frame.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
    ttkb.Label(k_frame, text="K (KNN):", font=("Helvetica", 10, "bold"), bootstyle="light").pack(side="left")
    k_entry = ttkb.Entry(k_frame, textvariable=self.k_var, width=5, bootstyle="info")
    k_entry.pack(side="left", padx=5)
    ToolTip(k_entry, text="Số lượng láng giềng cho KNN")

    self.run_button = ttkb.Button(
        control_frame,
        text="Chạy thuật toán",
        command=self.run_algorithm,
        bootstyle="success",
        width=15
    )
    self.run_button.grid(row=0, column=4, padx=5, pady=5)
    ToolTip(self.run_button, text="Chạy thuật toán đã chọn")
    self.run_button.bind("<Enter>", lambda e: self.run_button.configure(bootstyle="success-outline"))
    self.run_button.bind("<Leave>", lambda e: self.run_button.configure(bootstyle="success"))

    self.back_button = ttkb.Button(
        control_frame,
        text="Quay lại",
        command=self.display_data,
        bootstyle="danger",
        width=15
    )
    self.back_button.grid(row=0, column=5, padx=5, pady=5)
    ToolTip(self.back_button, text="Hiển thị lại dữ liệu gốc")
    self.back_button.bind("<Enter>", lambda e: self.back_button.configure(bootstyle="danger-outline"))
    self.back_button.bind("<Leave>", lambda e: self.back_button.configure(bootstyle="danger"))

    self.result_tree = setup_treeview(self.main_frame)
    self.result_tree.grid(row=4, column=0, pady=(15, 10), sticky="nsew")
    scrollbar = ttkb.Scrollbar(self.main_frame, orient="vertical", command=self.result_tree.yview, bootstyle="round")
    scrollbar.grid(row=4, column=1, sticky="ns")
    self.result_tree.configure(yscrollcommand=scrollbar.set)
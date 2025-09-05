import tkinter as tk
from tkinter import ttk
import pandas as pd

def setup_treeview(main_frame):
    # Frame cho bảng dữ liệu/kết quả
    result_frame = tk.Frame(main_frame, bg="#e3e8ee")
    result_frame.grid(row=2, column=0, sticky="nsew", pady=10)
    main_frame.rowconfigure(2, weight=1)
    main_frame.columnconfigure(0, weight=1)

    # Style cho Treeview với border
    style = ttk.Style()
    style.configure("Treeview",
                    background="#ffffff",
                    foreground="#2c3e50",
                    rowheight=25,
                    fieldbackground="#ffffff",
                    bordercolor="#2c3e50",
                    borderwidth=1)
    style.configure("Treeview.Heading",
                    background="#3498db",
                    foreground="#2c3e50",
                    font=("Helvetica", 12, "bold"),
                    borderwidth=1)
    style.map("Treeview.Heading",
              background=[("active", "#2980b9")],
              foreground=[("active", "#2c3e50")])

    # Bảng hiển thị dữ liệu/kết quả
    result_tree = ttk.Treeview(
        result_frame,
        show="headings",
        height=12,
        style="Treeview"
    )
    result_tree.grid(row=0, column=0, sticky="nsew")
    result_frame.rowconfigure(0, weight=1)
    result_frame.columnconfigure(0, weight=1)

    # Thanh cuộn cho bảng
    scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=result_tree.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    result_tree.configure(yscrollcommand=scrollbar.set)

    return result_tree

def reset_treeview(result_tree):
    result_tree.delete(*result_tree.get_children())

def display_data(result_tree, data, result_label):
    if data is None:
        result_label.config(text="Dữ liệu không hợp lệ hoặc chưa được tải!")
        return
    
    if 'label' not in data.columns:
        result_label.config(text="CSV thiếu cột 'label'!")
        return
    
    features = [col for col in data.columns if col != 'label']
    columns = ("ID",) + tuple(features) + ("Label",)
    result_tree.configure(columns=columns)
    
    result_tree.heading("ID", text="ID")
    for feature in features:
        result_tree.heading(feature, text=feature.capitalize())
    result_tree.heading("Label", text="Nhãn")
    
    result_tree.column("ID", width=50, anchor="center")
    for feature in features:
        result_tree.column(feature, width=100, anchor="center")
    result_tree.column("Label", width=100, anchor="center")
    
    result_tree.delete(*result_tree.get_children())
    for i, row in data.head(10).iterrows():
        values = (i+1,) + tuple(row[feature] for feature in features) + (row["label"],)
        result_tree.insert("", "end", values=values)
    
    result_label.config(text="Đã hiển thị dữ liệu!")
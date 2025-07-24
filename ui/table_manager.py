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
                    foreground="#2c3e50",  # Màu chữ xanh đậm, cố định
                    font=("Helvetica", 12, "bold"),
                    borderwidth=1)
    style.map("Treeview.Heading",
              background=[("active", "#2980b9")],  # Chỉ đổi nền khi hover
              foreground=[("active", "#2c3e50")])  # Giữ màu chữ

    # Bảng hiển thị dữ liệu/kết quả
    result_tree = ttk.Treeview(
        result_frame,
        columns=("ID", "Math", "Physics", "Chemistry", "Label"),
        show="headings",
        height=12,
        style="Treeview"
    )
    result_tree.heading("ID", text="ID")
    result_tree.heading("Math", text="Toán")
    result_tree.heading("Physics", text="Lý")
    result_tree.heading("Chemistry", text="Hóa")
    result_tree.heading("Label", text="Nhãn")
    result_tree.column("ID", width=50, anchor="center")
    result_tree.column("Math", width=100, anchor="center")
    result_tree.column("Physics", width=100, anchor="center")
    result_tree.column("Chemistry", width=100, anchor="center")
    result_tree.column("Label", width=100, anchor="center")
    result_tree.grid(row=0, column=0, sticky="nsew")
    result_frame.rowconfigure(0, weight=1)
    result_frame.columnconfigure(0, weight=1)

    # Thanh cuộn cho bảng
    scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=result_tree.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    result_tree.configure(yscrollcommand=scrollbar.set)

    return result_tree

def reset_treeview(result_tree):
    # Khôi phục cấu trúc cột gốc với border
    result_tree.configure(columns=("ID", "Math", "Physics", "Chemistry", "Label"), show="headings")
    result_tree.heading("ID", text="ID")
    result_tree.heading("Math", text="Toán")
    result_tree.heading("Physics", text="Lý")
    result_tree.heading("Chemistry", text="Hóa")
    result_tree.heading("Label", text="Nhãn")
    result_tree.column("ID", width=50, anchor="center")
    result_tree.column("Math", width=100, anchor="center")
    result_tree.column("Physics", width=100, anchor="center")
    result_tree.column("Chemistry", width=100, anchor="center")
    result_tree.column("Label", width=100, anchor="center")

def display_data(result_tree, data, result_label):
    reset_treeview(result_tree)
    result_tree.delete(*result_tree.get_children())
    if data is not None and all(col in data.columns for col in ["math_score", "physics_score", "chemistry_score", "label"]):
        for i, row in data.head(10).iterrows():
            result_tree.insert("", "end", values=(
                i+1,
                row["math_score"],
                row["physics_score"],
                row["chemistry_score"],
                row["label"]
            ))
        result_label.config(text="Đã hiển thị dữ liệu!")
    else:
        result_label.config(text="Dữ liệu không hợp lệ hoặc chưa được tải!")
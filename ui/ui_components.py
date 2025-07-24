import tkinter as tk
from tkinter import ttk

def setup_ui(main_frame, algo_var, k_var):
    # Frame chính, không dùng configure mà để padding trong grid
    control_frame = tk.Frame(main_frame, bg="#e3e8ee")
    control_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=20, sticky="ew")
    control_frame.columnconfigure((0, 1), weight=1)  # Căn giữa các cột

    # Tiêu đề
    title_label = tk.Label(
        main_frame,
        text="Công Cụ Khai Phá Dữ Liệu",
        font=("Helvetica", 20, "bold"),
        bg="#e3e8ee",
        fg="#2c3e50"
    )
    title_label.grid(row=0, column=0, columnspan=2, pady=20, sticky="ew")

    # Chọn thuật toán
    algo_label = tk.Label(
        control_frame,
        text="Chọn thuật toán:",
        font=("Helvetica", 12),
        bg="#e3e8ee",
        fg="#2c3e50"
    )
    algo_label.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")

    style = ttk.Style()
    style.configure("TCombobox", 
                   fieldbackground="#ffffff",
                   background="#e3e8ee",
                   foreground="#2c3e50",
                   arrowcolor="#3498db")
    style.map("TCombobox", 
             fieldbackground=[("readonly", "#ffffff"), ("active", "#e3e8ee")],
             selectbackground=[("readonly", "#ffffff")],
             selectforeground=[("readonly", "#2c3e50")])

    algo_combobox = ttk.Combobox(
        control_frame,
        textvariable=algo_var,
        values=["Naive Bayes", "KNN", "K-Means"],
        state="readonly",
        width=20,
        style="TCombobox"
    )
    algo_combobox.set("Naive Bayes")
    algo_combobox.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

    # Nhập tham số K cho KNN
    k_label = tk.Label(
        control_frame,
        text="Nhập K (chỉ cho KNN):",
        font=("Helvetica", 12),
        bg="#e3e8ee",
        fg="#2c3e50"
    )
    k_label.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

    k_entry = tk.Entry(
        control_frame,
        textvariable=k_var,
        width=10,
        font=("Helvetica", 12)
    )
    k_entry.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")

    # Nút tải dữ liệu từ MySQL
    load_db_button = tk.Button(
        control_frame,
        text="Tải Dữ Liệu Từ MySQL",
        font=("Helvetica", 12),
        bg="#3498db",
        fg="white",
        activebackground="#2980b9",
        activeforeground="white",
        borderwidth=4,
        relief="raised"
    )
    load_db_button.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
    load_db_button.bind("<Enter>", lambda e: load_db_button.config(bg="#2980b9"))
    load_db_button.bind("<Leave>", lambda e: load_db_button.config(bg="#3498db"))

    # Nút chọn file CSV
    load_csv_button = tk.Button(
        control_frame,
        text="Chọn File CSV",
        font=("Helvetica", 12),
        bg="#3498db",
        fg="white",
        activebackground="#2980b9",
        activeforeground="white",
        borderwidth=4,
        relief="raised"
    )
    load_csv_button.grid(row=4, column=1, padx=10, pady=10, sticky="ew")
    load_csv_button.bind("<Enter>", lambda e: load_db_button.config(bg="#2980b9"))
    load_csv_button.bind("<Leave>", lambda e: load_db_button.config(bg="#3498db"))

    # Nút chạy thuật toán
    run_button = tk.Button(
        control_frame,
        text="Chạy Thuật Toán",
        font=("Helvetica", 12),
        bg="#3498db",
        fg="white",
        activebackground="#2980b9",
        activeforeground="white",
        borderwidth=4,
        relief="raised"
    )
    run_button.grid(row=5, column=0, padx=10, pady=10, sticky="ew")
    run_button.bind("<Enter>", lambda e: run_button.config(bg="#2980b9"))
    run_button.bind("<Leave>", lambda e: run_button.config(bg="#3498db"))

    # Nút quay lại xem dữ liệu
    back_button = tk.Button(
        control_frame,
        text="Quay Lại Xem Dữ Liệu",
        font=("Helvetica", 12),
        bg="#3498db",
        fg="white",
        activebackground="#2980b9",
        activeforeground="white",
        borderwidth=4,
        relief="raised"
    )
    back_button.grid(row=5, column=1, padx=10, pady=10, sticky="ew")
    back_button.bind("<Enter>", lambda e: back_button.config(bg="#2980b9"))
    back_button.bind("<Leave>", lambda e: back_button.config(bg="#3498db"))

    return load_db_button, load_csv_button, run_button, back_button
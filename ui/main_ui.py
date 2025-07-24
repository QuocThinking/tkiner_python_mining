import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import threading
import time
from data_loader import load_data_from_db, load_data_from_csv
from algorithm_runner import run_algorithm
from ui_components import setup_ui
from table_manager import setup_treeview, display_data, reset_treeview

class AnimatedNotification:
    def __init__(self, parent, message, notification_type="info"):
        self.parent = parent
        self.notification = tk.Toplevel(parent)
        self.notification.overrideredirect(True)
        self.notification.attributes("-topmost", True)
        
        # Thiết lập màu sắc theo loại thông báo
        colors = {
            "info": {"bg": "#3498db", "fg": "white"},
            "success": {"bg": "#2ecc71", "fg": "white"},
            "error": {"bg": "#e74c3c", "fg": "white"},
            "loading": {"bg": "#f39c12", "fg": "white"}
        }
        color = colors.get(notification_type, colors["info"])
        
        self.notification.configure(bg=color["bg"])
        
        # Tạo frame với border radius effect
        frame = tk.Frame(self.notification, bg=color["bg"], padx=20, pady=10)
        frame.pack()
        
        label = tk.Label(
            frame, 
            text=message, 
            font=("Segoe UI", 11, "bold"),
            bg=color["bg"], 
            fg=color["fg"]
        )
        label.pack()
        
        # Định vị thông báo
        self.notification.update_idletasks()
        x = (self.parent.winfo_rootx() + self.parent.winfo_width() // 2 - 
             self.notification.winfo_width() // 2)
        y = self.parent.winfo_rooty() + 50
        self.notification.geometry(f"+{x}+{y}")
        
        # Hiệu ứng fade in/out
        self.animate_notification()
    
    def animate_notification(self):
        # Fade in
        for alpha in np.linspace(0.0, 0.9, 20):
            try:
                self.notification.attributes("-alpha", alpha)
                self.notification.update()
                time.sleep(0.02)
            except:
                return
        
        # Hiển thị 2 giây
        time.sleep(2)
        
        # Fade out
        for alpha in np.linspace(0.9, 0.0, 20):
            try:
                self.notification.attributes("-alpha", alpha)
                self.notification.update()
                time.sleep(0.02)
            except:
                break
        
        try:
            self.notification.destroy()
        except:
            pass

class LoadingSpinner:
    def __init__(self, parent):
        self.parent = parent
        self.loading_window = tk.Toplevel(parent)
        self.loading_window.overrideredirect(True)
        self.loading_window.attributes("-topmost", True)
        self.loading_window.configure(bg="#2c3e50")
        
        # Frame chính
        frame = tk.Frame(self.loading_window, bg="#2c3e50", padx=30, pady=20)
        frame.pack()
        
        # Label
        self.label = tk.Label(
            frame, 
            text="Đang tải dữ liệu...", 
            font=("Segoe UI", 12, "bold"),
            bg="#2c3e50", 
            fg="white"
        )
        self.label.pack(pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(
            frame, 
            mode='indeterminate', 
            length=200,
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress.pack()
        
        # Định vị
        self.loading_window.update_idletasks()
        x = (self.parent.winfo_rootx() + self.parent.winfo_width() // 2 - 
             self.loading_window.winfo_width() // 2)
        y = (self.parent.winfo_rooty() + self.parent.winfo_height() // 2 - 
             self.loading_window.winfo_height() // 2)
        self.loading_window.geometry(f"+{x}+{y}")
        
        self.progress.start(10)
        self.is_running = True
    
    def stop(self):
        self.is_running = False
        self.progress.stop()
        try:
            self.loading_window.destroy()
        except:
            pass

class StatisticsPopup:
    def __init__(self, parent, data):
        self.popup = tk.Toplevel(parent)
        self.popup.title("Thống kê dữ liệu")
        self.popup.geometry("800x600")
        self.popup.configure(bg="#ecf0f1")
        self.popup.transient(parent)
        self.popup.grab_set()
        
        # Style cho popup
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Popup.TFrame", background="#ecf0f1")
        style.configure("Popup.TLabel", background="#ecf0f1", font=("Segoe UI", 10))
        
        # Header
        header_frame = tk.Frame(self.popup, bg="#34495e", height=50)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="📊 Thống kê và Phân tích Dữ liệu", 
            font=("Segoe UI", 16, "bold"),
            bg="#34495e", 
            fg="white"
        )
        title_label.pack(pady=15)
        
        # Notebook cho các tab
        notebook = ttk.Notebook(self.popup, style="Popup.TNotebook")
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: Thống kê cơ bản
        stats_frame = ttk.Frame(notebook, style="Popup.TFrame")
        notebook.add(stats_frame, text="📈 Thống kê tổng quan")
        
        # Tab 2: Biểu đồ
        chart_frame = ttk.Frame(notebook, style="Popup.TFrame")
        notebook.add(chart_frame, text="📊 Biểu đồ")
        
        self.create_statistics_tab(stats_frame, data)
        self.create_charts_tab(chart_frame, data)
        
        # Hiệu ứng xuất hiện
        self.animate_popup_appearance()
    
    def animate_popup_appearance(self):
        # Bắt đầu từ alpha = 0
        self.popup.attributes("-alpha", 0.0)
        
        # Fade in
        for alpha in np.linspace(0.0, 1.0, 25):
            try:
                self.popup.attributes("-alpha", alpha)
                self.popup.update()
                time.sleep(0.01)
            except:
                break
    
    def create_statistics_tab(self, parent, data):
        # Scrollable frame
        canvas = tk.Canvas(parent, bg="#ecf0f1")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#ecf0f1")
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Thống kê
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        
        # Card container
        card_frame = tk.Frame(scrollable_frame, bg="#ecf0f1")
        card_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Thông tin cơ bản
        info_card = self.create_card(card_frame, "📋 Thông tin cơ bản", "#3498db")
        info_text = f"""
Số lượng mẫu: {len(df)}
Số lượng thuộc tính: {len(df.columns)}
Kích thước dữ liệu: {df.memory_usage(deep=True).sum() / 1024:.2f} KB
        """
        tk.Label(info_card, text=info_text, justify="left", bg="white", 
                font=("Segoe UI", 10)).pack(pady=10)
        
        # Thống kê các cột số
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_card = self.create_card(card_frame, "🔢 Thống kê các thuộc tính số", "#2ecc71")
            stats_text = df[numeric_cols].describe().round(2).to_string()
            tk.Text(numeric_card, height=10, font=("Courier", 9), bg="white", 
                   wrap="none").pack(pady=10, padx=10, fill="both")
            numeric_card.children['!text'].insert("1.0", stats_text)
            numeric_card.children['!text'].configure(state="disabled")
        
        # Thống kê các cột categorical
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_card = self.create_card(card_frame, "📝 Thống kê thuộc tính phân loại", "#e74c3c")
            for col in categorical_cols:
                col_frame = tk.Frame(cat_card, bg="white")
                col_frame.pack(fill="x", pady=5, padx=10)
                
                tk.Label(col_frame, text=f"{col}:", font=("Segoe UI", 10, "bold"),
                        bg="white").pack(anchor="w")
                
                value_counts = df[col].value_counts().head(5)
                for value, count in value_counts.items():
                    tk.Label(col_frame, text=f"  • {value}: {count} mẫu",
                            font=("Segoe UI", 9), bg="white").pack(anchor="w")
    
    def create_card(self, parent, title, color):
        card = tk.Frame(parent, bg="white", relief="solid", bd=1)
        card.pack(fill="x", pady=10)
        
        header = tk.Frame(card, bg=color, height=40)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        tk.Label(header, text=title, font=("Segoe UI", 12, "bold"),
                bg=color, fg="white").pack(pady=10)
        
        return card
    
    def create_charts_tab(self, parent, data):
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        
        # Matplotlib figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('#ecf0f1')
        
        # Chart 1: Histogram của thuộc tính số đầu tiên
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols[0]].hist(ax=ax1, bins=20, color="#3498db", alpha=0.7)
            ax1.set_title(f'Phân phối {numeric_cols[0]}', fontsize=12, fontweight='bold')
            ax1.set_facecolor('#ffffff')
        
        # Chart 2: Bar chart của thuộc tính categorical
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            value_counts = df[categorical_cols[0]].value_counts().head(8)
            bars = ax2.bar(range(len(value_counts)), value_counts.values, 
                          color=['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'][:len(value_counts)])
            ax2.set_title(f'Phân phối {categorical_cols[0]}', fontsize=12, fontweight='bold')
            ax2.set_xticks(range(len(value_counts)))
            ax2.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax2.set_facecolor('#ffffff')
        
        # Chart 3: Correlation heatmap (nếu có nhiều thuộc tính số)
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            im = ax3.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            ax3.set_title('Ma trận tương quan', fontsize=12, fontweight='bold')
            ax3.set_xticks(range(len(numeric_cols)))
            ax3.set_yticks(range(len(numeric_cols)))
            ax3.set_xticklabels(numeric_cols, rotation=45, ha='right')
            ax3.set_yticklabels(numeric_cols)
            
            # Thêm giá trị vào heatmap
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                            ha="center", va="center", color="white", fontweight='bold')
        
        # Chart 4: Pie chart
        if len(categorical_cols) > 0 and 'label' in df.columns:
            label_counts = df['label'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(label_counts)))
            wedges, texts, autotexts = ax4.pie(label_counts.values, labels=label_counts.index, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax4.set_title('Phân phối nhãn', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Nhúng vào tkinter
        canvas_widget = FigureCanvasTkAgg(fig, parent)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

class MainUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Dự Án Khai Phá Dữ Liệu - Phiên bản nâng cao")
        self.root.geometry("1000x750")
        self.root.configure(bg="#2c3e50")
        
        # Style configuration
        self.setup_styles()
        
        # Tạo nền gradient hiện đại
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.create_modern_gradient()

        # Frame chính với shadow effect
        self.main_frame = tk.Frame(self.canvas, bg="#ecf0f1", relief="solid", bd=1)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.main_frame.columnconfigure(0, weight=1)

        # Header với logo và title
        self.create_header()

        # Biến
        self.algo_var = tk.StringVar(value="Naive Bayes")
        self.k_var = tk.StringVar(value="3")
        self.data = None

        # Thiết lập giao diện với animation
        self.setup_enhanced_ui()

        # Thiết lập bảng Treeview với style mới
        self.result_tree = self.setup_enhanced_treeview()

        # Nhãn kết quả với animation
        self.result_label = tk.Label(
            self.main_frame,
            text="✨ Chờ tải dữ liệu để bắt đầu phân tích...",
            font=("Segoe UI", 12, "bold"),
            fg="#2c3e50",
            bg="#ecf0f1"
        )
        self.result_label.grid(row=4, column=0, pady=15, sticky="ew")

        # Animation cho UI elements
        self.animate_ui_elements()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom button styles
        style.configure("Modern.TButton",
                       borderwidth=0,
                       relief="flat",
                       padding=(20, 12),
                       font=("Segoe UI", 10, "bold"))
        
        style.map("Modern.TButton",
                 background=[('active', '#3498db'),
                           ('pressed', '#2980b9')])
        
        # Progress bar style
        style.configure("Custom.Horizontal.TProgressbar",
                       background="#3498db",
                       troughcolor="#bdc3c7",
                       borderwidth=0,
                       relief="flat")

    def create_modern_gradient(self):
        height = 750
        width = 1000
        
        # Gradient từ dark blue đến light blue
        for i in range(height):
            ratio = i / height
            r = int(44 + (236 - 44) * ratio)
            g = int(62 + (240 - 62) * ratio)
            b = int(80 + (241 - 80) * ratio)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.canvas.create_line(0, i, width, i, fill=color)

    def create_header(self):
        header_frame = tk.Frame(self.main_frame, bg="#34495e", height=80)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        header_frame.grid_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🤖 KHAI PHÁ DỮ LIỆU THÔNG MINH",
            font=("Segoe UI", 18, "bold"),
            fg="#ecf0f1",
            bg="#34495e"
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            header_frame,
            text="Phân tích dữ liệu với Machine Learning",
            font=("Segoe UI", 10),
            fg="#bdc3c7",
            bg="#34495e"
        )
        subtitle_label.pack()

    def setup_enhanced_ui(self):
        # Control panel với modern design
        control_frame = tk.LabelFrame(
            self.main_frame, 
            text="🎛️ Bảng điều khiển",
            font=("Segoe UI", 12, "bold"),
            fg="#2c3e50",
            bg="#ecf0f1",
            relief="solid",
            bd=1
        )
        control_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        control_frame.columnconfigure((0, 1, 2, 3), weight=1)
        
        # Data loading buttons với icons
        self.load_db_button = tk.Button(
            control_frame,
            text="📊 Tải từ Database",
            font=("Segoe UI", 11, "bold"),
            bg="#3498db",
            fg="white",
            relief="flat",
            bd=0,
            padx=20,
            pady=12,
            cursor="hand2",
            command=self.load_data_from_db_with_animation
        )
        self.load_db_button.grid(row=0, column=0, padx=10, pady=15, sticky="ew")
        
        self.load_csv_button = tk.Button(
            control_frame,
            text="📁 Tải từ CSV",
            font=("Segoe UI", 11, "bold"),
            bg="#2ecc71",
            fg="white",
            relief="flat",
            bd=0,
            padx=20,
            pady=12,
            cursor="hand2",
            command=self.load_data_from_csv_with_animation
        )
        self.load_csv_button.grid(row=0, column=1, padx=10, pady=15, sticky="ew")
        
        # Algorithm selection với dropdown đẹp
        algo_frame = tk.Frame(control_frame, bg="#ecf0f1")
        algo_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)
        
        tk.Label(algo_frame, text="🧠 Thuật toán:", font=("Segoe UI", 10, "bold"),
                bg="#ecf0f1", fg="#2c3e50").pack(side="left", padx=(10, 5))
        
        algo_combo = ttk.Combobox(
            algo_frame,
            textvariable=self.algo_var,
            values=["Naive Bayes", "KNN", "K-Means"],
            state="readonly",
            font=("Segoe UI", 10),
            width=15
        )
        algo_combo.pack(side="left", padx=5)
        
        tk.Label(algo_frame, text="K:", font=("Segoe UI", 10, "bold"),
                bg="#ecf0f1", fg="#2c3e50").pack(side="left", padx=(20, 5))
        
        k_entry = tk.Entry(
            algo_frame,
            textvariable=self.k_var,
            font=("Segoe UI", 10),
            width=5,
            relief="solid",
            bd=1
        )
        k_entry.pack(side="left", padx=5)
        
        # Run button
        self.run_button = tk.Button(
            control_frame,
            text="🚀 Chạy thuật toán",
            font=("Segoe UI", 11, "bold"),
            bg="#e74c3c",
            fg="white",
            relief="flat",
            bd=0,
            padx=20,
            pady=12,
            cursor="hand2",
            command=self.run_algorithm_with_animation
        )
        self.run_button.grid(row=0, column=2, padx=10, pady=15, sticky="ew")
        
        # Back button
        self.back_button = tk.Button(
            control_frame,
            text="🔄 Xem dữ liệu",
            font=("Segoe UI", 11, "bold"),
            bg="#95a5a6",
            fg="white",
            relief="flat",
            bd=0,
            padx=20,
            pady=12,
            cursor="hand2",
            command=self.display_data_with_animation
        )
        self.back_button.grid(row=0, column=3, padx=10, pady=15, sticky="ew")
        
        # Hover effects
        self.add_hover_effects()

    def add_hover_effects(self):
        buttons = [self.load_db_button, self.load_csv_button, self.run_button, self.back_button]
        original_colors = ["#3498db", "#2ecc71", "#e74c3c", "#95a5a6"]
        hover_colors = ["#2980b9", "#27ae60", "#c0392b", "#7f8c8d"]
        
        for btn, orig, hover in zip(buttons, original_colors, hover_colors):
            btn.bind("<Enter>", lambda e, b=btn, h=hover: b.configure(bg=h))
            btn.bind("<Leave>", lambda e, b=btn, o=orig: b.configure(bg=o))

    def setup_enhanced_treeview(self):
        tree_frame = tk.LabelFrame(
            self.main_frame,
            text="📈 Kết quả phân tích",
            font=("Segoe UI", 12, "bold"),
            fg="#2c3e50",
            bg="#ecf0f1",
            relief="solid",
            bd=1
        )
        tree_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=10)
        
        # Treeview với scrollbar
        tree_container = tk.Frame(tree_frame, bg="#ecf0f1")
        tree_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        tree = ttk.Treeview(tree_container, height=12)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_container, orient="vertical", command=tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_container, orient="horizontal", command=tree.xview)
        
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)
        
        return tree

    def animate_ui_elements(self):
        # Animation cho các button
        def animate_button(button, delay=0):
            self.root.after(delay, lambda: self.fade_in_element(button))
        
        buttons = [self.load_db_button, self.load_csv_button, self.run_button, self.back_button]
        for i, btn in enumerate(buttons):
            animate_button(btn, i * 200)

    def fade_in_element(self, element):
        # Simple fade in effect bằng cách thay đổi state
        element.configure(state="normal")

    def show_notification(self, message, notification_type="info"):
        def show():
            AnimatedNotification(self.root, message, notification_type)
        threading.Thread(target=show, daemon=True).start()

    def load_data_from_db_with_animation(self):
        def load():
            spinner = LoadingSpinner(self.root)
            try:
                time.sleep(1)  # Simulate loading time
                self.data, message = load_data_from_db()
                spinner.stop()
                
                if self.data is not None:
                    self.show_notification("✅ Tải dữ liệu từ Database thành công!", "success")
                    self.display_data_with_animation()
                    # Hiển thị popup thống kê
                    self.root.after(500, lambda: StatisticsPopup(self.root, self.data))
                else:
                    self.show_notification("❌ Lỗi tải dữ liệu từ Database!", "error")
            except Exception as e:
                spinner.stop()
                self.show_notification(f"❌ Lỗi: {str(e)}", "error")
        
        threading.Thread(target=load, daemon=True).start()

    def load_data_from_csv_with_animation(self):
        def load():
            spinner = LoadingSpinner(self.root)
            try:
                time.sleep(1)  # Simulate loading time
                self.data, message = load_data_from_csv()
                spinner.stop()
                
                if self.data is not None:
                    self.show_notification("✅ Tải dữ liệu từ CSV thành công!", "success")
                    self.display_data_with_animation()
                    # Hiển thị popup thống kê
                    self.root.after(500, lambda: StatisticsPopup(self.root, self.data))
                else:
                    self.show_notification("❌ Lỗi tải dữ liệu từ CSV!", "error")
            except Exception as e:
                spinner.stop()
                self.show_notification(f"❌ Lỗi: {str(e)}", "error")
        
        threading.Thread(target=load, daemon=True).start()

    def display_data_with_animation(self):
        if self.data is not None:
            display_data(self.result_tree, self.data, self.result_label)
            self.result_label.config(text="📊 Dữ liệu đã được tải và hiển thị!")

    def run_algorithm_with_animation(self):
        def run():
            algo = self.algo_var.get()
            if not algo:
                self.show_notification("⚠️ Vui lòng chọn thuật toán!", "error")
                return
            if self.data is None:
                self.show_notification("⚠️ Vui lòng tải dữ liệu trước!", "error")
                return
            
            spinner = LoadingSpinner(self.root)
            try:
                k = int(self.k_var.get()) if algo == "KNN" else 3
                if k <= 0:
                    raise ValueError("K phải là số dương!")
                
                reset_treeview(self.result_tree)
                result = run_algorithm(self.data, algo, k)
                spinner.stop()
                
                self.result_tree.delete(*self.result_tree.get_children())
                
                if result["confusion_matrix"] and algo != "K-Means":
                    # Hiển thị ma trận nhầm lẫn dạng bảng
                    cm = result["confusion_matrix"]
                    labels = sorted(set(self.data["label"]))
                    columns = [""] + [f"Pred {lbl}" for lbl in labels]
                    self.result_tree.configure(columns=columns, show="headings")
                    self.result_tree.heading("", text="True")
                    for i, lbl in enumerate(labels):
                        self.result_tree.heading(f"Pred {lbl}", text=f"Pred {lbl}")
                        self.result_tree.column(f"Pred {lbl}", width=100, anchor="center")
                    self.result_tree.column("", width=50, anchor="center")
                    for i, row in enumerate(cm):
                        self.result_tree.insert("", "end", values=[labels[i]] + row)
                    
                    self.result_label.config(text=f"✅ {result['text']}\n📊 Ma trận nhầm lẫn:")
                    self.show_notification(f"🎉 Thuật toán {algo} đã chạy thành công!", "success")
                else:
                    # Hiển thị kết quả dạng văn bản (cho K-Means hoặc lỗi)
                    self.result_tree.configure(columns=("Result",), show="headings")
                    self.result_tree.heading("Result", text="Kết quả")
                    self.result_tree.column("Result", width=450, anchor="center")
                    for line in result["text"].split("\n"):
                        self.result_tree.insert("", "end", values=(line,))
                    
                    self.result_label.config(text=f"✅ Đã chạy thuật toán {algo} thành công!")
                    self.show_notification(f"🎉 Thuật toán {algo} đã chạy thành công!", "success")
                
                self.algo_var.set(algo)
                
            except Exception as e:
                spinner.stop()
                self.result_label.config(text=f"❌ Lỗi: {str(e)}")
                self.show_notification(f"❌ Lỗi khi chạy thuật toán: {str(e)}", "error")
        
        threading.Thread(target=run, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainUI(root)
    root.mainloop()
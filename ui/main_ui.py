import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import threading
import time
from data_loader import load_data_from_csv
from algorithm_runner import run_algorithm
from table_manager import display_data, reset_treeview
from pandas.plotting import scatter_matrix

class AnimatedNotification:
    def __init__(self, parent, message, notification_type="info"):
        self.parent = parent
        self.notification = tk.Toplevel(parent)
        self.notification.overrideredirect(True)
        self.notification.attributes("-topmost", True)
        
        colors = {
            "info": {"bg": "#3498db", "fg": "white"},
            "success": {"bg": "#2ecc71", "fg": "white"},
            "error": {"bg": "#e74c3c", "fg": "white"},
            "loading": {"bg": "#f39c12", "fg": "white"}
        }
        color = colors.get(notification_type, colors["info"])
        
        self.notification.configure(bg=color["bg"])
        
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
        
        self.notification.update_idletasks()
        x = (self.parent.winfo_rootx() + self.parent.winfo_width() // 2 - 
             self.notification.winfo_width() // 2)
        y = self.parent.winfo_rooty() + 50
        self.notification.geometry(f"+{x}+{y}")
        
        self.animate_notification()
    
    def animate_notification(self):
        for alpha in np.linspace(0.0, 0.9, 20):
            try:
                self.notification.attributes("-alpha", alpha)
                self.notification.update()
                time.sleep(0.02)
            except:
                return
        
        time.sleep(2)
        
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
        
        frame = tk.Frame(self.loading_window, bg="#2c3e50", padx=30, pady=20)
        frame.pack()
        
        self.label = tk.Label(
            frame, 
            text="Đang xử lý...", 
            font=("Segoe UI", 12, "bold"),
            bg="#2c3e50", 
            fg="white"
        )
        self.label.pack(pady=(0, 10))
        
        self.progress = ttk.Progressbar(
            frame, 
            mode='indeterminate', 
            length=200,
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress.pack()
        
        self.loading_window.update_idletasks()
        x = (self.parent.winfo_rootx() + self.parent.winfo_width() // 2 - 
             self.loading_window.winfo_width() // 2)
        y = (self.parent.winfo_rooty() + self.parent.winfo_height() // 2 - 
             self.loading_window.winfo_height() // 2)
        self.loading_window.geometry(f"+{x}+{y}")
        
        self.progress.start(10)
        self.is_running = True
    
    def stop(self):
        if self.is_running:
            self.is_running = False
            # Dừng progress và hủy cửa sổ trong luồng chính
            self.parent.after(0, self._safe_stop)

    def _safe_stop(self):
        self.progress.stop()
        try:
            self.loading_window.destroy()
        except:
            pass

class StatisticsPopup:
    def __init__(self, parent, data, algo_result=None, algo_name=None):
        self.popup = tk.Toplevel(parent)
        self.popup.title("Thống kê và Kết quả thuật toán")
        self.popup.geometry("1600x1000")  # Tăng kích thước popup để phù hợp
        self.popup.configure(bg="#ecf0f1")
        self.popup.transient(parent)
        self.popup.grab_set()
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Popup.TFrame", background="#ecf0f1")
        style.configure("Popup.TLabel", background="#ecf0f1", font=("Segoe UI", 10))
        style.configure("Popup.TNotebook", background="#ecf0f1")
        
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
        
        notebook = ttk.Notebook(self.popup, style="Popup.TNotebook")
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        stats_frame = ttk.Frame(notebook, style="Popup.TFrame")
        chart_frame = ttk.Frame(notebook, style="Popup.TFrame")
        algo_frame = ttk.Frame(notebook, style="Popup.TFrame")
        
        notebook.add(stats_frame, text="📈 Thống kê tổng quan")
        notebook.add(chart_frame, text="📊 Biểu đồ dữ liệu")
        if algo_result and algo_name:
            notebook.add(algo_frame, text="🧠 Kết quả thuật toán")
        
        self.create_statistics_tab(stats_frame, data)
        self.create_charts_tab(chart_frame, data)
        if algo_result and algo_name:
            self.create_algorithm_results_tab(algo_frame, algo_result, algo_name, data)
        
        self.animate_popup_appearance()
    
    def animate_popup_appearance(self):
        self.popup.attributes("-alpha", 0.0)
        for alpha in np.linspace(0.0, 1.0, 25):
            try:
                self.popup.attributes("-alpha", alpha)
                self.popup.update()
                time.sleep(0.01)
            except:
                break
    
    def create_statistics_tab(self, parent, data):
        canvas = tk.Canvas(parent, bg="#ecf0f1")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#ecf0f1")
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        
        card_frame = tk.Frame(scrollable_frame, bg="#ecf0f1")
        card_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        info_card = self.create_card(card_frame, "📋 Thông tin cơ bản", "#3498db")
        info_text = f"""
        Số lượng mẫu: {len(df)}
        Số lượng thuộc tính: {len(df.columns)}
        Kích thước dữ liệu: {df.memory_usage(deep=True).sum() / 1024:.2f} KB
                """
        tk.Label(info_card, text=info_text, justify="left", bg="white", 
                font=("Segoe UI", 10)).pack(pady=10)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_card = self.create_card(card_frame, "🔢 Thống kê các thuộc tính số", "#2ecc71")
            stats_text = df[numeric_cols].describe().round(2).to_string()
            text_widget = tk.Text(numeric_card, height=10, font=("Courier", 9), bg="white", 
                                wrap="none")
            text_widget.pack(pady=10, padx=10, fill="both")
            text_widget.insert("1.0", stats_text)
            text_widget.configure(state="disabled")
        
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
        try:
            if 'label' not in data.columns:
                raise ValueError("No 'label' column")
            df = pd.DataFrame(data) if isinstance(data, dict) else data
            features = [col for col in df.columns if col != 'label' and pd.api.types.is_numeric_dtype(df[col])]
            if len(features) == 0:
                raise ValueError("No numeric feature columns found")
            labels = sorted(set(df['label']))
            
            # Tạo layout động dựa trên số lượng features
            num_features = len(features)
            num_rows = (num_features + 1) // 2  # 2 biểu đồ mỗi hàng
            fig, axes = plt.subplots(num_rows, 2, figsize=(16, 6 * num_rows))
            fig.patch.set_facecolor('#ecf0f1')
            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            
            # Nếu chỉ có 1 hàng, axes không phải mảng 2D
            if num_rows == 1:
                axes = [axes] if num_features == 1 else [axes]
            
            colors = {'Giỏi': '#2ecc71', 'Trung bình': '#f39c12', 'Yếu': '#e74c3c'}
            
            # Boxplot cho mỗi feature
            for idx, feature in enumerate(features):
                row = idx // 2
                col = idx % 2
                ax = axes[row][col] if num_rows > 1 else axes[0][col]
                
                data_by_label = [df[df['label'] == lbl][feature].values for lbl in labels]
                positions = [i - 0.3 + j * 0.3 for i in range(len(labels)) for j in range(1)]
                ax.boxplot(data_by_label, positions=positions[:len(labels)], widths=0.25, patch_artist=True,
                           boxprops=dict(facecolor=colors[labels[0]], alpha=0.5))
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_title(f'Phân phối {feature.capitalize()} theo Nhãn', fontsize=12, fontweight='bold', color='#2c3e50')
                ax.set_ylabel('Giá trị', fontsize=10)
                ax.set_facecolor('#ffffff')
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # Nếu số feature lẻ, xóa plot thừa
            if num_features % 2 == 1:
                ax = axes[num_rows - 1][1] if num_rows > 1 else axes[0][1]
                ax.axis('off')
            
            # Scatter matrix (thay thế scatter plot cố định)
            scatter_fig = plt.figure(figsize=(10, 10))
            scatter_matrix(df[features], ax=plt.gca(), hist_kwds={'bins': 20}, c=[colors[lbl] for lbl in df['label']], alpha=0.6)
            scatter_fig.patch.set_facecolor('#ecf0f1')
            
            canvas_widget = FigureCanvasTkAgg(fig, parent)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            
            scatter_canvas = FigureCanvasTkAgg(scatter_fig, parent)
            scatter_canvas.draw()
            scatter_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        except Exception as e:
            print(f"Debug: Error in create_charts_tab - {str(e)}")
            import traceback
            traceback.print_exc()

    # Trong create_algorithm_results_tab: Dynamic feature_importance, cluster_centers
    def create_algorithm_results_tab(self, parent, algo_result, algo_name, data):
        try:
            if 'label' not in data.columns:
                raise ValueError("No 'label' column")
            features = [col for col in data.columns if col != 'label']
            df = pd.DataFrame(data) if isinstance(data, dict) else data
            labels = sorted(set(df['label']))
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f'Kết quả {algo_name}', fontsize=14, fontweight='bold')
            
            if algo_result.get("confusion_matrix"):
                cm = np.array(algo_result["confusion_matrix"])
                im = ax1.imshow(cm, cmap='Blues', interpolation='nearest')
                ax1.set_title('Ma trận nhầm lẫn', fontsize=12, fontweight='bold', color='#2c3e50')
                ax1.set_xticks(range(len(labels)))
                ax1.set_yticks(range(len(labels)))
                ax1.set_xticklabels(labels, rotation=45, ha='right')
                ax1.set_yticklabels(labels)
                ax1.set_xlabel('Dự đoán', fontsize=10)
                ax1.set_ylabel('Thực tế', fontsize=10)
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        ax1.text(j, i, f'{cm[i, j]}', ha='center', va='center', 
                                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
                fig.colorbar(im, ax=ax1, label='Số lượng')
                ax1.set_facecolor('#ffffff')
            
            if algo_name == "Decision Tree" and algo_result.get("additional_info", {}).get("feature_importance"):
                importance_dict = algo_result["additional_info"]["feature_importance"]
                ax2.bar(list(importance_dict.keys()), list(importance_dict.values()), color='#3498db', alpha=0.7)
                ax2.set_title('Tầm quan trọng đặc trưng (Gini Index)', fontsize=12, fontweight='bold', color='#2c3e50')
                ax2.set_ylabel('Tầm quan trọng', fontsize=10)
                ax2.set_facecolor('#ffffff')
                ax2.grid(True, linestyle='--', alpha=0.7)
            
            elif algo_name == "K-Means" and algo_result.get("additional_info", {}).get("cluster_centers"):
                centers = np.array(algo_result["additional_info"]["cluster_centers"])
                # Dynamic feature names
                if algo_result.get("reduct_info"):
                    plot_features = [f"PC{i+1}" for i in range(centers.shape[1])]
                else:
                    plot_features = features[:centers.shape[1]]  # Giới hạn nếu reduct
                for i, center in enumerate(centers):
                    ax2.plot(plot_features, center, marker='o', label=f'Cụm {i} ({algo_result["additional_info"]["cluster_label_map"][i]})')
                ax2.set_title('Tâm cụm K-Means', fontsize=12, fontweight='bold', color='#2c3e50')
                ax2.set_ylabel('Điểm', fontsize=10)
                ax2.set_facecolor('#ffffff')
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.7)
            
            else:
                ax2.text(0.5, 0.5, 'Không có biểu đồ bổ sung', ha='center', va='center', fontsize=10)
                ax2.set_facecolor('#ffffff')
            
            plt.tight_layout()
            
            canvas_widget = FigureCanvasTkAgg(fig, parent)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            
            if algo_result.get("classification_report"):
                report_frame = tk.Frame(parent, bg="#ecf0f1")
                report_frame.pack(fill="x", padx=10, pady=10)
                report_text = tk.Text(report_frame, height=6, font=("Segoe UI", 10), bg="white")
                report_text.pack(fill="x")
                report_str = "Classification Report:\n"
                for label in labels:
                    report_str += f"{label}:\n"
                    report_str += f"  Precision: {algo_result['classification_report'][label]['precision']:.2f}\n"
                    report_str += f"  Recall: {algo_result['classification_report'][label]['recall']:.2f}\n"
                    report_str += f"  F1-score: {algo_result['classification_report'][label]['f1-score']:.2f}\n"
                report_text.insert("1.0", report_str)
                report_text.configure(state="disabled")
            
            if algo_result.get("reduct_info"):
                reduct_frame = tk.Frame(parent, bg="#ecf0f1")
                reduct_frame.pack(fill="x", padx=10, pady=10)
                reduct_text = tk.Text(reduct_frame, height=4, font=("Segoe UI", 10), bg="white")
                reduct_text.pack(fill="x")
                reduct_info = algo_result["reduct_info"]
                reduct_str = "Reduct Info:\n"
                reduct_str += f"- Số thành phần chính: {reduct_info['reduced_features'].shape[1]}\n"
                reduct_str += "- Tỷ lệ phương sai: "
                for i, ratio in enumerate(reduct_info['explained_variance_ratio']):
                    reduct_str += f"PC{i+1}: {ratio*100:.1f}% "
                reduct_text.insert("1.0", reduct_str)
                reduct_text.configure(state="disabled")
            else:
                print(f"Debug: No reduct_info available for algo={algo_name}")
        
        except Exception as e:
            print(f"Debug: Error in create_algorithm_results_tab - {str(e)}")
            import traceback
            traceback.print_exc()

class MainUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Dự Án Khai Phá Dữ Liệu - Phiên bản nâng cao")
        self.root.geometry("1400x900")  # Tăng kích thước cửa sổ chính
        self.root.configure(bg="#2c3e50")
        
        self.setup_styles()
        
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.create_modern_gradient()
        
        self.main_frame = tk.Frame(self.canvas, bg="#ecf0f1", relief="solid", bd=1)
        self.main_frame.pack(fill="both", expand=True, padx=30, pady=30)  # Tăng padding
        self.main_frame.columnconfigure(0, weight=1)
        
        self.create_header()
        
        self.algo_var = tk.StringVar(value="Naive Bayes")
        self.k_var = tk.StringVar(value="3")
        self.use_reduct_var = tk.BooleanVar(value=False)  # Tùy chọn reduct
        self.data = None
        self.last_algo_result = None
        self.last_algo_name = None
        
        self.setup_enhanced_ui()
        
        self.result_tree = self.setup_enhanced_treeview()
        
        self.result_label = tk.Label(
            self.main_frame,
            text="✨ Chờ tải dữ liệu để bắt đầu phân tích...",
            font=("Segoe UI", 12, "bold"),
            fg="#2c3e50",
            bg="#ecf0f1",
            wraplength=1100,  # Tăng wraplength để phù hợp
            justify="left"
        )
        self.result_label.grid(row=4, column=0, pady=15, sticky="ew")
        
        self.animate_ui_elements()
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("Modern.TButton",
                       borderwidth=0,
                       relief="flat",
                       padding=(20, 12),
                       font=("Segoe UI", 10, "bold"))
        
        style.map("Modern.TButton",
                 background=[('active', '#3498db'),
                           ('pressed', '#2980b9')])
        
        style.configure("Custom.Horizontal.TProgressbar",
                       background="#3498db",
                       troughcolor="#bdc3c7",
                       borderwidth=0,
                       relief="flat")
        
        style.configure("Popup.TNotebook",
                       background="#ecf0f1",
                       tabmargins=(5, 5, 0, 0))
    
    def create_modern_gradient(self):
        height = 1400  # Cập nhật height theo kích thước mới
        width = 900  # Cập nhật width theo kích thước mới
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
        control_frame = tk.LabelFrame(
            self.main_frame, 
            text="🎛️ Bảng điều khiển",
            font=("Segoe UI", 12, "bold"),
            fg="#2c3e50",
            bg="#ecf0f1",
            relief="solid",
            bd=1
        )
        control_frame.grid(row=1, column=0, sticky="ew", padx=30, pady=15)  # Tăng padding
        control_frame.columnconfigure((0, 1, 2, 3, 4), weight=1)
        
        self.load_db_button = tk.Button(
            control_frame,
            text="📊 Tải từ Database",
            font=("Segoe UI", 11, "bold"),
            bg="#3498db",
            fg="white",
            relief="flat",
            bd=0,
            padx=25,  # Tăng padding
            pady=15,  # Tăng padding
            cursor="hand2",
            command=self.load_data_from_db_with_animation
        )
        self.load_db_button.grid(row=0, column=0, padx=15, pady=15, sticky="ew")
        
        self.load_csv_button = tk.Button(
            control_frame,
            text="📁 Tải từ CSV",
            font=("Segoe UI", 11, "bold"),
            bg="#2ecc71",
            fg="white",
            relief="flat",
            bd=0,
            padx=25,  # Tăng padding
            pady=15,  # Tăng padding
            cursor="hand2",
            command=self.load_data_from_csv_with_animation
        )
        self.load_csv_button.grid(row=0, column=1, padx=15, pady=15, sticky="ew")
        
        algo_frame = tk.Frame(control_frame, bg="#ecf0f1")
        algo_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=15)
        
        tk.Label(algo_frame, text="🧠 Thuật toán:", font=("Segoe UI", 10, "bold"),
                bg="#ecf0f1", fg="#2c3e50").pack(side="left", padx=(15, 5))  # Tăng padding
        algo_combo = ttk.Combobox(
            algo_frame,
            textvariable=self.algo_var,
            values=["Naive Bayes", "KNN", "K-Means", "Decision Tree", "ID3", "Association Rules"],  # Thêm ID3
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
        
        # Thêm tùy chọn reduct
        reduct_check = tk.Checkbutton(
            algo_frame,
            text="Áp dụng Reduct (PCA)",
            variable=self.use_reduct_var,
            font=("Segoe UI", 10),
            bg="#ecf0f1",
            fg="#2c3e50",
            activebackground="#ecf0f1",
            activeforeground="#2c3e50"
        )
        reduct_check.pack(side="left", padx=15)
        
        self.run_button = tk.Button(
            control_frame,
            text="🚀 Chạy thuật toán",
            font=("Segoe UI", 11, "bold"),
            bg="#e74c3c",
            fg="white",
            relief="flat",
            bd=0,
            padx=25,  # Tăng padding
            pady=15,  # Tăng padding
            cursor="hand2",
            command=self.run_algorithm_with_animation
        )
        self.run_button.grid(row=0, column=2, padx=15, pady=15, sticky="ew")
        
        self.back_button = tk.Button(
            control_frame,
            text="🔄 Xem dữ liệu",
            font=("Segoe UI", 11, "bold"),
            bg="#95a5a6",
            fg="white",
            relief="flat",
            bd=0,
            padx=25,  # Tăng padding
            pady=15,  # Tăng padding
            cursor="hand2",
            command=self.display_data_with_animation
        )
        self.back_button.grid(row=0, column=3, padx=15, pady=15, sticky="ew")
        
        self.view_results_button = tk.Button(
            control_frame,
            text="📊 Xem kết quả thuật toán",
            font=("Segoe UI", 11, "bold"),
            bg="#f39c12",
            fg="white",
            relief="flat",
            bd=0,
            padx=25,  # Tăng padding
            pady=15,  # Tăng padding
            cursor="hand2",
            command=self.view_algorithm_results
        )
        self.view_results_button.grid(row=0, column=4, padx=15, pady=15, sticky="ew")
        
        self.add_hover_effects()
    
    def add_hover_effects(self):
        buttons = [self.load_db_button, self.load_csv_button, self.run_button, 
                  self.back_button, self.view_results_button]
        original_colors = ["#3498db", "#2ecc71", "#e74c3c", "#95a5a6", "#f39c12"]
        hover_colors = ["#2980b9", "#27ae60", "#c0392b", "#7f8c8d", "#e67e22"]
        
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
        tree_frame.grid(row=3, column=0, sticky="ew", padx=30, pady=15)  # Tăng padding
        
        tree_container = tk.Frame(tree_frame, bg="#ecf0f1")
        tree_container.pack(fill="both", expand=True, padx=15, pady=15)  # Tăng padding
        
        tree = ttk.Treeview(tree_container, height=15)  # Tăng chiều cao Treeview
        
        v_scrollbar = ttk.Scrollbar(tree_container, orient="vertical", command=tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_container, orient="horizontal", command=tree.xview)
        
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)
        
        return tree
    
    def animate_ui_elements(self):
        def animate_button(button, delay=0):
            self.root.after(delay, lambda: self.fade_in_element(button))
        
        buttons = [self.load_db_button, self.load_csv_button, self.run_button, 
                  self.back_button, self.view_results_button]
        for i, btn in enumerate(buttons):
            animate_button(btn, i * 200)
    
    def fade_in_element(self, element):
        element.configure(state="normal")
    
    def show_notification(self, message, notification_type="info"):
        def show():
            AnimatedNotification(self.root, message, notification_type)
        threading.Thread(target=show, daemon=True).start()
    
    def load_data_from_db_with_animation(self):
        def load():
            spinner = LoadingSpinner(self.root)
            try:
                time.sleep(1)
                self.data, message = load_data_from_db()
                spinner.stop()
                
                if self.data is not None:
                    self.show_notification("✅ Tải dữ liệu từ Database thành công!", "success")
                    self.display_data_with_animation()
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
                time.sleep(1)
                self.data, message = load_data_from_csv()
                spinner.stop()
                
                if self.data is not None:
                    self.show_notification("✅ Tải dữ liệu từ CSV thành công!", "success")
                    self.display_data_with_animation()
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
    
    def view_algorithm_results(self):
        if self.last_algo_result and self.last_algo_name and self.data is not None:
            StatisticsPopup(self.root, self.data, self.last_algo_result, self.last_algo_name)
        else:
            self.show_notification("⚠️ Vui lòng chạy thuật toán trước!", "error")
    
    # Trong MainUI class, cập nhật run_algorithm_with_animation
    def run_algorithm_with_animation(self):
        def run():
            algo = self.algo_var.get()
            if not algo:
                self.show_notification("⚠️ Vui lòng chọn thuật toán!", "error")
                return
            if self.data is None:
                self.show_notification("⚠️ Vui lòng tải dữ liệu trước!", "error")
                return
            
            print(f"Debug: Data received - shape={self.data.shape if self.data is not None else None}, columns={self.data.columns.tolist() if self.data is not None else None}")

            spinner = LoadingSpinner(self.root)
            try:
                k = int(self.k_var.get()) if algo in ["KNN", "K-Means"] else 3
                if k <= 0:
                    raise ValueError("K phải là số dương!")
                
                reset_treeview(self.result_tree)
                result = run_algorithm(self.data, algo, k, use_reduct=self.use_reduct_var.get())
                spinner.stop()
                
                self.last_algo_result = result
                self.last_algo_name = algo
                
                self.result_tree.delete(*self.result_tree.get_children())
                
                result_text = [f"✅ {result['text']}"]
                result_text.append(f"⏱️ Thời gian chạy: {result['execution_time']:.2f}s")
                
                print(f"Debug: Result received - algo={algo}, reduct_info={result['reduct_info']}, rules={result.get('rules')}")
                
                # Hiển thị reduct info nếu có
                if result["reduct_info"] and algo != "Association Rules":
                    reduct_info = result["reduct_info"]
                    if 'explained_variance_ratio' in reduct_info:
                        reduct_text = f"Reduct Info: Giảm xuống {reduct_info['reduced_features'].shape[1]} chiều, "
                        reduct_text += ", ".join([f"PC{i+1}: {ratio*100:.1f}%" for i, ratio in enumerate(reduct_info['explained_variance_ratio'])])
                        result_text.append(reduct_text)
                    else:
                        print(f"Debug: reduct_info exists but missing expected keys: {reduct_info.keys()}")
                
                # Hiển thị kết quả theo thuật toán
                if algo in ["Naive Bayes", "KNN", "Decision Tree", "ID3"]:
                    labels = sorted(set(self.data["label"]))
                    columns = [""] + [f"Pred {lbl}" for lbl in labels]
                    self.result_tree.configure(columns=columns, show="headings")
                    self.result_tree.heading("", text="True")
                    for i, lbl in enumerate(labels):
                        self.result_tree.heading(f"Pred {lbl}", text=f"Pred {lbl}")
                        self.result_tree.column(f"Pred {lbl}", width=100, anchor="center")
                    self.result_tree.column("", width=50, anchor="center")
                    if result.get("confusion_matrix"):
                        for i, row in enumerate(result["confusion_matrix"]):
                            self.result_tree.insert("", "end", values=[labels[i]] + row)
                    
                    report = result["classification_report"]
                    result_text.append("Classification Report:")
                    for label in labels:
                        result_text.append(f"{label}:")
                        result_text.append(f"  Precision: {report[label]['precision']:.2f}")
                        result_text.append(f"  Recall: {report[label]['recall']:.2f}")
                        result_text.append(f"  F1-score: {report[label]['f1-score']:.2f}")
                
                elif algo == "K-Means":
                    features = [col for col in self.data.columns if col != 'label' and pd.api.types.is_numeric_dtype(self.data[col])]
                    self.result_tree.configure(columns=("Result",), show="headings")
                    self.result_tree.heading("Result", text="Kết quả")
                    self.result_tree.column("Result", width=500, anchor="center")
                
                    cluster_info = result["additional_info"]
                    result_text.append("Tâm cụm:")
                    for i, center in enumerate(cluster_info["cluster_centers"]):
                        center_str = ", ".join([f"{feat}: {val:.2f}" for feat, val in zip(features[:len(center)], center)])
                        result_text.append(f"Cụm {i} ({cluster_info['cluster_label_map'][i]}): {center_str}")
                    result_text.append("Phân phối cụm:")
                    for cluster, count in cluster_info["cluster_distribution"].items():
                        result_text.append(f"Cụm {cluster}: {count} mẫu")
                
                elif algo == "Association Rules":
                    self.result_tree.configure(columns=("Rule", "Support", "Confidence", "Lift"), show="headings")
                    self.result_tree.heading("Rule", text="Quy tắc")
                    self.result_tree.heading("Support", text="Hỗ trợ")
                    self.result_tree.heading("Confidence", text="Độ tin cậy")
                    self.result_tree.heading("Lift", text="Lift")
                    self.result_tree.column("Rule", width=300, anchor="center")
                    self.result_tree.column("Confidence", width=100, anchor="center")
                    self.result_tree.column("Support", width=100, anchor="center")
                    self.result_tree.column("Lift", width=100, anchor="center")
                    
                    rules = result["rules"]
                    print(f"Debug: Processing rules - type={type(rules)}, content={rules}")
                    if isinstance(rules, dict) and "antecedents" in rules:
                        for idx in rules["antecedents"].keys():
                            ante = ", ".join(list(rules["antecedents"][idx]))
                            cons = ", ".join(list(rules["consequents"][idx]))
                            rule = f"{ante} => {cons}"
                            support = rules["support"][idx]
                            confidence = rules["confidence"][idx]
                            lift = rules["lift"][idx]
                            self.result_tree.insert("", "end", values=(rule, f"{support:.3f}", f"{confidence:.3f}", f"{lift:.3f}"))
                    else:
                        self.result_tree.insert("", "end", values=("Không có quy tắc nào", "", "", ""))
                
                if algo in ["Decision Tree", "ID3"] and "feature_importance" in result["additional_info"]:
                    result_text.append(f"Tầm quan trọng đặc trưng ({'Gini Index' if algo == 'Decision Tree' else 'Entropy'}):")
                    for feature, importance in result["additional_info"]["feature_importance"].items():
                        result_text.append(f"{feature}: {importance:.2f}")
                
                for line in result_text:
                    self.result_tree.insert("", "end", values=(line,))
                
                self.result_label.config(text="\n".join(result_text))
                self.show_notification(f"🎉 Thuật toán {algo} đã chạy thành công!", "success")
                
                print(f"Debug: Calling StatisticsPopup with algo={algo}, result={result}")
                self.root.after(500, lambda r=result, a=algo: StatisticsPopup(self.root, self.data, r, a))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Debug: Error occurred - {str(e)}")
                spinner.stop()
                self.result_label.config(text=f"❌ Lỗi: {str(e)}")
                self.show_notification(f"❌ Lỗi khi chạy thuật toán: {str(e)}", "error")
        
        threading.Thread(target=run, daemon=True).start()

    # Trong StatisticsPopup class, cập nhật create_algorithm_results_tab
    def create_algorithm_results_tab(self, parent, algo_result, algo_name, data):
        try:
            if 'label' not in data.columns:
                raise ValueError("No 'label' column")
            df = pd.DataFrame(data) if isinstance(data, dict) else data
            features = [col for col in df.columns if col != 'label' and pd.api.types.is_numeric_dtype(df[col])]
            if len(features) == 0:
                raise ValueError("No numeric feature columns found")
            labels = sorted(set(df['label']))
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.patch.set_facecolor('#ecf0f1')
            plt.subplots_adjust(wspace=0.4)
            
            print(f"Debug: Entering create_algorithm_results_tab - algo={algo_name}, reduct_info={algo_result.get('reduct_info')}, rules={algo_result.get('rules')}")
            
            # Vẽ ma trận nhầm lẫn nếu có
            if algo_result.get("confusion_matrix") and algo_name != "K-Means":
                cm = np.array(algo_result["confusion_matrix"])
                im = ax1.imshow(cm, cmap='Blues', aspect='auto')
                ax1.set_title(f'Ma trận nhầm lẫn ({algo_name})', fontsize=12, fontweight='bold', color='#2c3e50')
                ax1.set_xticks(range(len(labels)))
                ax1.set_yticks(range(len(labels)))
                ax1.set_xticklabels(labels, rotation=45, ha='right')
                ax1.set_yticklabels(labels)
                ax1.set_xlabel('Dự đoán', fontsize=10)
                ax1.set_ylabel('Thực tế', fontsize=10)
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        ax1.text(j, i, f'{cm[i, j]}', ha='center', va='center', 
                                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
                fig.colorbar(im, ax=ax1, label='Số lượng')
                ax1.set_facecolor('#ffffff')
            
            # Vẽ biểu đồ bổ sung theo thuật toán
            if algo_name in ["Decision Tree", "ID3"] and algo_result.get("additional_info", {}).get("feature_importance"):
                importance_dict = algo_result["additional_info"]["feature_importance"]
                ax2.bar(list(importance_dict.keys()), list(importance_dict.values()), color='#3498db', alpha=0.7)
                ax2.set_title('Tầm quan trọng đặc trưng (Gini Index)' if algo_name == "Decision Tree" else 'Tầm quan trọng đặc trưng (Entropy)', 
                              fontsize=12, fontweight='bold', color='#2c3e50')
                ax2.set_ylabel('Tầm quan trọng', fontsize=10)
                ax2.set_xticklabels(list(importance_dict.keys()), rotation=45, ha='right')
                ax2.set_facecolor('#ffffff')
                ax2.grid(True, linestyle='--', alpha=0.7)
            
            elif algo_name == "K-Means" and algo_result.get("additional_info", {}).get("cluster_centers"):
                centers = np.array(algo_result["additional_info"]["cluster_centers"])
                # Dynamic feature names
                plot_features = [f"PC{i+1}" for i in range(centers.shape[1])] if algo_result.get("reduct_info") else features[:centers.shape[1]]
                for i, center in enumerate(centers):
                    ax2.plot(plot_features, center, marker='o', label=f'Cụm {i} ({algo_result["additional_info"]["cluster_label_map"][i]})')
                ax2.set_title('Tâm cụm K-Means', fontsize=12, fontweight='bold', color='#2c3e50')
                ax2.set_ylabel('Giá trị', fontsize=10)
                ax2.set_xticklabels(plot_features, rotation=45, ha='right')
                ax2.set_facecolor('#ffffff')
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.7)
            
            elif algo_name == "Association Rules" and algo_result.get("rules"):
                ax2.text(0.5, 0.5, 'Luật kết hợp hiển thị trong bảng', ha='center', va='center', fontsize=10)
                ax2.set_facecolor('#ffffff')
            
            else:
                ax2.text(0.5, 0.5, 'Không có biểu đồ bổ sung', ha='center', va='center', fontsize=10)
                ax2.set_facecolor('#ffffff')
            
            plt.tight_layout()
            
            canvas_widget = FigureCanvasTkAgg(fig, parent)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            
            # Hiển thị classification report nếu có
            if algo_result.get("classification_report"):
                report_frame = tk.Frame(parent, bg="#ecf0f1")
                report_frame.pack(fill="x", padx=10, pady=10)
                report_text = tk.Text(report_frame, height=6, font=("Segoe UI", 10), bg="white")
                report_text.pack(fill="x")
                report_str = "Classification Report:\n"
                for label in labels:
                    report_str += f"{label}:\n"
                    report_str += f"  Precision: {algo_result['classification_report'][label]['precision']:.2f}\n"
                    report_str += f"  Recall: {algo_result['classification_report'][label]['recall']:.2f}\n"
                    report_str += f"  F1-score: {algo_result['classification_report'][label]['f1-score']:.2f}\n"
                report_text.insert("1.0", report_str)
                report_text.configure(state="disabled")
            
            # Hiển thị reduct info nếu có
            if algo_result.get("reduct_info"):
                reduct_frame = tk.Frame(parent, bg="#ecf0f1")
                reduct_frame.pack(fill="x", padx=10, pady=10)
                reduct_text = tk.Text(reduct_frame, height=4, font=("Segoe UI", 10), bg="white")
                reduct_text.pack(fill="x")
                reduct_info = algo_result["reduct_info"]
                reduct_str = "Reduct Info:\n"
                reduct_str += f"- Số thành phần chính: {reduct_info['reduced_features'].shape[1]}\n"
                reduct_str += "- Tỷ lệ phương sai: "
                for i, ratio in enumerate(reduct_info['explained_variance_ratio']):
                    reduct_str += f"PC{i+1}: {ratio*100:.1f}% "
                reduct_text.insert("1.0", reduct_str)
                reduct_text.configure(state="disabled")
            else:
                print(f"Debug: No reduct_info available for algo={algo_name}")
        
        except Exception as e:
            print(f"Debug: Error in create_algorithm_results_tab - {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainUI(root)
    root.mainloop()
import os
import time
import threading

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from tkinter import ttk, filedialog, messagebox, scrolledtext

from convex_layers_naive import NaiveHullBuilder
from convex_layers_tree import HullTree
from geometry import Point

class ConvexLayersGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Convex hull analyzer")
        self.root.geometry("1400x900")

        self.points = []
        self.layers = []
        self.depth_map = {}
        self.max_depth = 0
        self.depth_function = {}
        self.filename = None
        self.execution_time = 0

        self.algorithms = {
            "Жадное построение": NaiveHullBuilder,
            "Дерево оболочек": HullTree,
        }

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')

        self.create_menu()

        self.create_toolbar()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="Данные")
        self.create_data_tab()

        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="Визуализация")
        self.create_visualization_tab()

        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Результаты")
        self.create_results_tab()

        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Статистика")
        self.create_statistics_tab()

        self.status_bar = ttk.Label(self.root, text="Готов к работе", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Открыть...", command=self.load_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Сохранить результаты...", command=self.save_results, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Генерировать точки...", command=self.generate_points)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit, accelerator="Ctrl+Q")

        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Анализ", menu=analysis_menu)
        analysis_menu.add_command(label="Запустить анализ", command=self.run_analysis, accelerator="F5")
        analysis_menu.add_command(label="Сравнить алгоритмы", command=self.compare_algorithms)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Экспорт графиков", command=self.export_plots)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Вид", menu=view_menu)
        view_menu.add_command(label="Показать все слои", command=lambda: self.update_visualization("all"))
        view_menu.add_command(label="Показать по слоям", command=lambda: self.update_visualization("layers"))
        view_menu.add_command(label="Тепловая карта", command=lambda: self.update_visualization("heatmap"))

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self.show_about)

    def create_toolbar(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        ttk.Button(toolbar, text="📂 Открыть", command=self.load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Сохранить", command=self.save_results).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Button(toolbar, text="▶️ Анализ", command=self.run_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📊 Сравнить", command=self.compare_algorithms).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Label(toolbar, text="Алгоритм:").pack(side=tk.LEFT, padx=5)
        self.algorithm_var = tk.StringVar(value=list(self.algorithms.keys())[0])
        self.algorithm_combo = ttk.Combobox(toolbar, textvariable=self.algorithm_var,
                                           values=list(self.algorithms.keys()),
                                           state="readonly", width=25)
        self.algorithm_combo.pack(side=tk.LEFT, padx=2)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(toolbar, variable=self.progress_var,
                                           mode='indeterminate', length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
        
    def create_data_tab(self):
        info_frame = ttk.LabelFrame(self.data_tab, text="Информация о данных")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.info_labels = {}
        info_items = ["Файл:", "Количество точек:", "Диапазон X:", "Диапазон Y:"]
        for i, item in enumerate(info_items):
            ttk.Label(info_frame, text=item).grid(row=i, column=0, sticky=tk.W, padx=10, pady=5)
            self.info_labels[item] = ttk.Label(info_frame, text="-")
            self.info_labels[item].grid(row=i, column=1, sticky=tk.W, padx=10, pady=5)
        
        points_frame = ttk.LabelFrame(self.data_tab, text="Точки")
        points_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        columns = ("№", "X", "Y", "Глубина")
        self.points_tree = ttk.Treeview(points_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.points_tree.heading(col, text=col)
            self.points_tree.column(col, width=100)
        
        vsb = ttk.Scrollbar(points_frame, orient="vertical", command=self.points_tree.yview)
        hsb = ttk.Scrollbar(points_frame, orient="horizontal", command=self.points_tree.xview)
        self.points_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.points_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        points_frame.grid_rowconfigure(0, weight=1)
        points_frame.grid_columnconfigure(0, weight=1)
        
        control_frame = ttk.Frame(self.data_tab)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Загрузить файл", command=self.load_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Генерировать", command=self.generate_points).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Очистить", command=self.clear_data).pack(side=tk.LEFT, padx=5)
        
    def create_visualization_tab(self):
        viz_control = ttk.Frame(self.viz_tab)
        viz_control.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(viz_control, text="Режим:").pack(side=tk.LEFT, padx=5)
        self.viz_mode = tk.StringVar(value="all")
        modes = [("Все слои", "all"), ("По слоям", "layers"), ("Тепловая карта", "heatmap")]
        for text, value in modes:
            ttk.Radiobutton(viz_control, text=text, variable=self.viz_mode, 
                          value=value, command=self.update_visualization).pack(side=tk.LEFT, padx=5)

        self.layer_slider_frame = ttk.Frame(viz_control)
        self.layer_slider_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(self.layer_slider_frame, text="Слой:").pack(side=tk.LEFT)
        self.layer_var = tk.IntVar(value=0)
        self.layer_slider = ttk.Scale(
            self.layer_slider_frame,
            from_=0,
            to=0, 
            variable=self.layer_var,
            orient=tk.HORIZONTAL,
            command=lambda e: self.update_visualization(),
        )
        self.layer_slider.pack(side=tk.LEFT, padx=5)
        self.layer_label = ttk.Label(self.layer_slider_frame, text="0")
        self.layer_label.pack(side=tk.LEFT)

        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        toolbar_frame = ttk.Frame(self.viz_tab)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def create_results_tab(self):
        results_frame = ttk.LabelFrame(self.results_tab, text="Результаты анализа")
        results_frame.pack(fill=tk.X, padx=10, pady=10)

        self.result_labels = {}
        result_items = ["Максимальная глубина M(S):", "Время выполнения:", 
                       "Использованный алгоритм:", "Количество слоев:"]
        for i, item in enumerate(result_items):
            ttk.Label(results_frame, text=item).grid(row=i, column=0, sticky=tk.W, padx=10, pady=5)
            self.result_labels[item] = ttk.Label(results_frame, text="-", font=("Arial", 10, "bold"))
            self.result_labels[item].grid(row=i, column=1, sticky=tk.W, padx=10, pady=5)

        depth_frame = ttk.LabelFrame(self.results_tab, text="Функция глубин F(m)")
        depth_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        columns = ("Глубина m", "Количество точек F(m)", "Процент")
        self.depth_tree = ttk.Treeview(depth_frame, columns=columns, show="headings")

        for col in columns:
            self.depth_tree.heading(col, text=col)
            self.depth_tree.column(col, width=150)

        vsb = ttk.Scrollbar(depth_frame, orient="vertical", command=self.depth_tree.yview)
        self.depth_tree.configure(yscrollcommand=vsb.set)

        self.depth_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        text_frame = ttk.LabelFrame(self.results_tab, text="Подробный отчет")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_text = scrolledtext.ScrolledText(text_frame, height=10, width=80)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_statistics_tab(self):
        self.stats_fig = Figure(figsize=(12, 8))
        self.stats_canvas = FigureCanvasTkAgg(self.stats_fig, master=self.stats_tab)
        self.stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        stats_control = ttk.Frame(self.stats_tab)
        stats_control.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(stats_control, text="Обновить статистику", 
                  command=self.update_statistics).pack(side=tk.LEFT, padx=5)
        ttk.Button(stats_control, text="Экспорт в CSV", 
                  command=self.export_statistics).pack(side=tk.LEFT, padx=5)

    def sort_hull_points(self, points):
        """
        Sort points for correct visualization
        """
        if len(points) <= 2:
            return points

        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)

        def polar_angle(p):
            return np.arctan2(p[1] - cy, p[0] - cx)

        return sorted(points, key=polar_angle)

    def load_file(self):
        filename = filedialog.askopenfilename(
            title="Выберите файл с точками",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            self.filename = filename
            self.points = []

            with open(filename, 'r', encoding='utf-8') as f:
                n = int(f.readline().strip())
                for _ in range(n):
                    line = f.readline().strip()
                    if line:
                        x, y = map(float, line.split())
                        self.points.append((x, y))

            self.update_data_display()
            self.update_status(f"Загружено {len(self.points)} точек из {os.path.basename(filename)}")

            self.visualize_points()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")

    def generate_points(self):
        dialog = GeneratePointsDialog(self.root)
        self.root.wait_window(dialog.dialog)

        if dialog.result:
            n, distribution = dialog.result
            self.points = self.generate_random_points(n, distribution)
            self.filename = f"generated_{distribution}_{n}.txt"
            self.update_data_display()
            self.update_status(f"Сгенерировано {n} точек ({distribution})")
            self.visualize_points()

    def generate_random_points(self, n, distribution):
        np.random.seed(42)
        points = []

        if distribution == "uniform":
            points = [(np.random.uniform(0, 1000), np.random.uniform(0, 1000)) 
                     for _ in range(n)]
        elif distribution == "circle":
            for _ in range(n):
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0, 500) ** 0.5
                x = 500 + r * np.cos(angle)
                y = 500 + r * np.sin(angle)
                points.append((x, y))
        elif distribution == "gaussian":
            points = [(np.random.normal(500, 150), np.random.normal(500, 150)) 
                     for _ in range(n)]
        elif distribution == "clusters":
            n_clusters = 5
            points_per_cluster = n // n_clusters
            for i in range(n_clusters):
                cx = np.random.uniform(100, 900)
                cy = np.random.uniform(100, 900)
                for _ in range(points_per_cluster):
                    x = np.random.normal(cx, 50)
                    y = np.random.normal(cy, 50)
                    points.append((x, y))

        return points

    def run_analysis(self):
        """
        Fork analyzer process
        """
        if not self.points:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        thread = threading.Thread(target=self._run_analysis_thread)
        thread.start()

    def _run_analysis_thread(self):
        try:
            self.progress_bar.start()
            self.update_status("Выполняется анализ...")

            algorithm_name = self.algorithm_var.get()
            algorithm_class = self.algorithms[algorithm_name]
            algo = algorithm_class()

            start_time = time.time()

            layers = algo.compute_layers([Point(pt[0], pt[1]) for pt in self.points])
            self.layers = [[(pt.x, pt.y) for pt in layer] for layer in layers]
            self.max_depth = len(self.layers)
            self.depth_function = {i: len(layer) for i, layer in enumerate(self.layers)}
            self.depth_map = {pt: i for i, layer in enumerate(self.layers) for pt in layer}

            self.execution_time = time.time() - start_time

            self.root.after(0, self._update_after_analysis)

        except Exception as e:
            self.root.after(0, lambda e=e: messagebox.showerror("Ошибка", f"Ошибка анализа:\n{str(e)}"))
        finally:
            self.progress_bar.stop()

    def _update_after_analysis(self):
        self.update_results_display()
        self.update_visualization()
        self.update_statistics()
        self.update_data_display()

        self.layer_slider.configure(to=self.max_depth)

        self.update_status(f"Анализ завершен за {self.execution_time:.4f} сек. M(S)={self.max_depth}")

    def update_data_display(self):
        if self.points:
            x_coords = [p[0] for p in self.points]
            y_coords = [p[1] for p in self.points]

            self.info_labels["Файл:"].config(text=os.path.basename(self.filename) if self.filename else "Сгенерировано")
            self.info_labels["Количество точек:"].config(text=str(len(self.points)))
            self.info_labels["Диапазон X:"].config(text=f"[{min(x_coords):.2f}, {max(x_coords):.2f}]")
            self.info_labels["Диапазон Y:"].config(text=f"[{min(y_coords):.2f}, {max(y_coords):.2f}]")

        self.points_tree.delete(*self.points_tree.get_children())
        for i, point in enumerate(self.points[:1000]):
            depth = self.depth_map.get(point, "-")
            self.points_tree.insert("", "end", values=(i+1, f"{point[0]:.2f}", f"{point[1]:.2f}", depth))

    def update_results_display(self):
        self.result_labels["Максимальная глубина M(S):"].config(text=str(self.max_depth))
        self.result_labels["Время выполнения:"].config(text=f"{self.execution_time:.4f} сек")
        self.result_labels["Использованный алгоритм:"].config(text=self.algorithm_var.get())
        self.result_labels["Количество слоев:"].config(text=str(self.max_depth + 1))

        self.depth_tree.delete(*self.depth_tree.get_children())
        total_points = len(self.points)

        for depth in range(self.max_depth + 1):
            count = self.depth_function.get(depth, 0)
            percentage = (count / total_points * 100) if total_points > 0 else 0
            self.depth_tree.insert("", "end", values=(depth, count, f"{percentage:.2f}%"))

        self.result_text.delete(1.0, tk.END)
        report = self.generate_report()
        self.result_text.insert(1.0, report)

    def visualize_points(self):
        if not self.points:
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        
        ax.scatter(x_coords, y_coords, alpha=0.6, s=20)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Исходные точки ({len(self.points)} точек)")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        self.canvas.draw()
    
    def update_visualization(self, mode=None):
        if mode:
            self.viz_mode.set(mode)
        
        if not self.points or not self.layers:
            self.visualize_points()
            return
        
        mode = self.viz_mode.get()
        self.fig.clear()
        
        if mode == "all":
            self.visualize_all_layers()
        elif mode == "layers":
            self.visualize_single_layer()
        elif mode == "heatmap":
            self.visualize_heatmap()

        self.canvas.draw()

    def visualize_all_layers(self):
        ax = self.fig.add_subplot(111)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.layers)))
        
        for i, (layer, color) in enumerate(zip(self.layers, colors)):
            if len(layer) > 2:
                sorted_layer = self.sort_hull_points(layer)
                poly = Polygon(sorted_layer, alpha=0.3, facecolor=color, 
                             edgecolor=color, linewidth=2)
                ax.add_patch(poly)
                
                layer_x = [p[0] for p in sorted_layer]
                layer_y = [p[1] for p in sorted_layer]
                ax.plot(layer_x + [layer_x[0]], layer_y + [layer_y[0]], 
                       'o-', color=color, markersize=4, label=f'Слой {i}')
            elif len(layer) == 2:
                ax.plot([layer[0][0], layer[1][0]], [layer[0][1], layer[1][1]], 
                       'o-', color=color, markersize=4, label=f'Слой {i}')
            elif len(layer) == 1:
                ax.plot(layer[0][0], layer[0][1], 'o', color=color, 
                       markersize=6, label=f'Слой {i}')
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Все выпуклые слои (M(S) = {self.max_depth})")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        if len(self.layers) <= 10:
            ax.legend(loc='best', fontsize='small')
    
    def visualize_single_layer(self):
        ax = self.fig.add_subplot(111)
        layer_idx = self.layer_var.get()

        self.layer_label.config(text=str(layer_idx))
        
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        ax.scatter(x_coords, y_coords, alpha=0.2, s=10, c='gray', label='Все точки')
        
        if 0 <= layer_idx < len(self.layers):
            for i in range(layer_idx + 1, len(self.layers)):
                inner_layer = self.layers[i]
                if len(inner_layer) > 2:
                    sorted_layer = self.sort_hull_points(inner_layer)
                    poly = Polygon(sorted_layer, alpha=0.1, facecolor='blue', 
                                 edgecolor='blue', linewidth=1)
                    ax.add_patch(poly)
            
            layer = self.layers[layer_idx]
            if len(layer) > 2:
                sorted_layer = self.sort_hull_points(layer)
                poly = Polygon(sorted_layer, alpha=0.5, facecolor='red', 
                             edgecolor='darkred', linewidth=3)
                ax.add_patch(poly)
                
                layer_x = [p[0] for p in sorted_layer]
                layer_y = [p[1] for p in sorted_layer]
                ax.plot(layer_x + [layer_x[0]], layer_y + [layer_y[0]], 
                       'o-', color='red', markersize=8, linewidth=2, label=f'Слой {layer_idx}')
            elif len(layer) == 2:
                ax.plot([layer[0][0], layer[1][0]], [layer[0][1], layer[1][1]], 
                       'o-', color='red', markersize=8, linewidth=2, label=f'Слой {layer_idx}')
            elif len(layer) == 1:
                ax.plot(layer[0][0], layer[0][1], 'o', color='red', 
                       markersize=10, label=f'Слой {layer_idx}')
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Слой {layer_idx} (содержит {self.depth_function.get(layer_idx, 0)} точек)")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(loc='best')
    
    def visualize_heatmap(self):
        ax = self.fig.add_subplot(111)
        
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        depths = [self.depth_map.get(p, 0) for p in self.points]
        
        scatter = ax.scatter(x_coords, y_coords, c=depths, cmap='viridis', 
                           s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Глубина', rotation=270, labelpad=15)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Тепловая карта глубин точек")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    def update_statistics(self):
        if not self.depth_function:
            return
        
        self.stats_fig.clear()
        
        axes = []
        axes.append(self.stats_fig.add_subplot(2, 2, 1))
        axes.append(self.stats_fig.add_subplot(2, 2, 2))
        axes.append(self.stats_fig.add_subplot(2, 2, 3))
        axes.append(self.stats_fig.add_subplot(2, 2, 4))
        
        depths = list(range(self.max_depth + 1))
        counts = [self.depth_function.get(d, 0) for d in depths]
        axes[0].bar(depths, counts, color='steelblue', alpha=0.7)
        axes[0].set_xlabel("Глубина")
        axes[0].set_ylabel("Количество точек")
        axes[0].set_title("Функция глубин F(m)")
        axes[0].grid(True, alpha=0.3)
        
        cumulative = np.cumsum(counts)
        axes[1].plot(depths, cumulative, 'o-', color='green', linewidth=2)
        axes[1].fill_between(depths, cumulative, alpha=0.3, color='green')
        axes[1].set_xlabel("Глубина")
        axes[1].set_ylabel("Накопленное количество")
        axes[1].set_title("Накопительное распределение")
        axes[1].grid(True, alpha=0.3)
        
        if len(depths) <= 10:
            axes[2].pie(counts, labels=[f"Слой {d}" for d in depths], 
                       autopct=lambda p: f'{p:.1f}%' if p > 1 else '')
            axes[2].set_title("Распределение по слоям")
        else:
            percentages = [c/sum(counts)*100 for c in counts]
            axes[2].plot(depths, percentages, 'o-', color='red', linewidth=2)
            axes[2].set_xlabel("Глубина")
            axes[2].set_ylabel("Процент точек (%)")
            axes[2].set_title("Процентное распределение")
            axes[2].grid(True, alpha=0.3)
        
        axes[3].axis('off')
        stats_text = self.calculate_statistics()
        axes[3].text(0.1, 0.9, stats_text, transform=axes[3].transAxes,
                    fontsize=10, verticalalignment='top', family='monospace')
        
        self.stats_fig.tight_layout()
        self.stats_canvas.draw()
    
    def calculate_statistics(self):
        if not self.depth_function:
            return "Нет данных"
        
        depths = []
        for depth, count in self.depth_function.items():
            depths.extend([depth] * count)
        
        if not depths:
            return "Нет данных"
        
        mean_depth = np.mean(depths)
        median_depth = np.median(depths)
        std_depth = np.std(depths)
        
        stats = f"""Статистические показатели:
════════════════════════════
Всего точек:     {len(self.points):>10}
Максимальная глубина: {self.max_depth:>5}
Средняя глубина:  {mean_depth:>9.2f}
Медиана глубины:  {median_depth:>9.2f}
Станд. отклонение: {std_depth:>8.2f}
Время анализа:    {self.execution_time:>8.4f} сек
"""
        return stats

    def generate_report(self):
        report = f"""
{'='*60}
ОТЧЕТ О ВЫПОЛНЕНИИ АНАЛИЗА ВЫПУКЛЫХ СЛОЕВ
{'='*60}

ИСХОДНЫЕ ДАННЫЕ:
----------------
Файл: {self.filename if self.filename else 'Сгенерировано'}
Количество точек: {len(self.points)}
Алгоритм: {self.algorithm_var.get()}

РЕЗУЛЬТАТЫ:
-----------
Максимальная глубина M(S): {self.max_depth}
Время выполнения: {self.execution_time:.6f} секунд
Скорость обработки: {len(self.points)/self.execution_time:.0f} точек/сек

ФУНКЦИЯ ГЛУБИН F(m):
--------------------
"""
        for depth in range(self.max_depth + 1):
            count = self.depth_function.get(depth, 0)
            percentage = count / len(self.points) * 100
            report += f"Глубина {depth:3d}: {count:6d} точек ({percentage:5.2f}%)\n"
        
        report += f"""
СТАТИСТИКА:
-----------
{self.calculate_statistics()}

{'='*60}
Отчет сгенерирован: {time.strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
"""
        return report
    
    def save_results(self):
        if not self.depth_function:
            messagebox.showwarning("Предупреждение", "Нет результатов для сохранения")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Сохранить результаты",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            if filename.endswith('.csv'):
                self.save_results_csv(filename)
            else:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.generate_report())
            
            messagebox.showinfo("Успешно", f"Результаты сохранены в {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
    
    def save_results_csv(self, filename):
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow(["Глубина", "Количество точек", "Процент"])
            
            total = len(self.points)
            for depth in range(self.max_depth + 1):
                count = self.depth_function.get(depth, 0)
                percentage = count / total * 100 if total > 0 else 0
                writer.writerow([depth, count, f"{percentage:.2f}"])
    
    def export_plots(self):
        if not self.layers:
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Экспорт графиков",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                      ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            
            stats_filename = filename.rsplit('.', 1)[0] + "_stats." + filename.rsplit('.', 1)[1]
            self.stats_fig.savefig(stats_filename, dpi=300, bbox_inches='tight')
            
            messagebox.showinfo("Успешно", f"Графики сохранены:\n{os.path.basename(filename)}\n{os.path.basename(stats_filename)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить графики:\n{str(e)}")
    
    def export_statistics(self):
        if not self.depth_function:
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Экспорт статистики",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.save_results_csv(filename)
            messagebox.showinfo("Успешно", f"Статистика экспортирована в {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать статистику:\n{str(e)}")
    
    def compare_algorithms(self):
        if not self.points:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return
        
        CompareAlgorithmsDialog(self.root, self.points, self.algorithms)
    
    def clear_data(self):
        self.points = []
        self.layers = []
        self.depth_map = {}
        self.max_depth = 0
        self.depth_function = {}
        self.filename = None
        self.execution_time = 0
        
        self.points_tree.delete(*self.points_tree.get_children())
        self.depth_tree.delete(*self.depth_tree.get_children())
        self.result_text.delete(1.0, tk.END)
        
        self.fig.clear()
        self.canvas.draw()
        self.stats_fig.clear()
        self.stats_canvas.draw()
        
        for label in self.info_labels.values():
            label.config(text="-")
        for label in self.result_labels.values():
            label.config(text="-")
        
        self.update_status("Данные очищены")
    
    def update_status(self, message):
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def show_about(self):
        about_text = """Программа для вычисления и визуализации
выпуклых слоев множества точек на плоскости.

Реализованные алгоритмы:
• Наивный (использует алгоритм Грехема-Эндрю для
построения каждого слоя выпуклой оболочки)
• Разделяй и властвуй

Автор: Дьяков Илья
2025"""

        messagebox.showinfo("О программе", about_text)


class GeneratePointsDialog:
    def __init__(self, parent):
        self.result = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Генерация точек")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        ttk.Label(self.dialog, text="Количество точек:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.n_var = tk.IntVar(value=100)
        ttk.Spinbox(self.dialog, from_=10, to=100000, textvariable=self.n_var, width=20).grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(self.dialog, text="Распределение:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.dist_var = tk.StringVar(value="uniform")

        distributions = [
            ("Равномерное в квадрате", "uniform"),
            ("В круге", "circle"),
            ("Нормальное", "gaussian"),
            ("Кластеры", "clusters")
        ]

        for i, (text, value) in enumerate(distributions):
            ttk.Radiobutton(self.dialog, text=text, variable=self.dist_var, 
                          value=value).grid(row=i+2, column=0, columnspan=2, padx=20, pady=5, sticky=tk.W)
        
        button_frame = ttk.Frame(self.dialog)
        button_frame.grid(row=10, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Генерировать", command=self.generate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Отмена", command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def generate(self):
        self.result = (self.n_var.get(), self.dist_var.get())
        self.dialog.destroy()


class CompareAlgorithmsDialog:
    def __init__(self, parent, points, algorithms):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Сравнение алгоритмов")
        self.dialog.geometry("800x600")
        self.dialog.transient(parent)
        
        results = {}
        for name, algo_class in algorithms.items():
            algo = algo_class()
            start = time.time()

            layers = algo.compute_layers([Point(pt[0], pt[1]) for pt in points])
            max_depth = len(layers)
            depth_func = {i: len(layer) for i, layer in enumerate(layers)}

            exec_time = time.time() - start
            results[name] = {
                'time': exec_time,
                'max_depth': max_depth,
                'depth_function': depth_func
            }

        frame = ttk.Frame(self.dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        columns = ("Алгоритм", "Время (сек)", "M(S)", "Скорость (точек/сек)")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=180)
        
        for name, res in results.items():
            speed = len(points) / res['time'] if res['time'] > 0 else 0
            tree.insert("", "end", values=(
                name,
                f"{res['time']:.6f}",
                res['max_depth'],
                f"{speed:.0f}"
            ))
        
        tree.pack(fill=tk.BOTH, expand=True)
        
        fig = Figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        
        names = list(results.keys())
        times = [results[n]['time'] for n in names]
        
        bars = ax.bar(range(len(names)), times, color=['blue', 'green'])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_ylabel("Время выполнения (сек)")
        ax.set_title(f"Сравнение производительности ({len(points)} точек)")
        ax.grid(True, alpha=0.3)
        
        for bar, t in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{t:.4f}s', ha='center', va='bottom')

        canvas = FigureCanvasTkAgg(fig, master=self.dialog)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        canvas.draw()
        
        ttk.Button(self.dialog, text="Закрыть", 
                  command=self.dialog.destroy).pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    _ = ConvexLayersGUI(root)
    root.mainloop()

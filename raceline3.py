import sys
import os
import yaml
import json
from datetime import datetime
import numpy as np
import pandas as pd
import glob
from PIL import Image
import networkx as nx
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt as edt
from scipy.interpolate import splprep, splev
import runpy
import subprocess

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QGroupBox, QComboBox, QDoubleSpinBox,
                             QSpinBox, QTabWidget, QFileDialog, QListWidget, 
                             QListWidgetItem, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class WorkerThread(QThread):
    """Worker thread for long-running operations"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result if result else "Operation completed successfully")
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots with Pan/Zoom"""
    def __init__(self, parent=None, width=8, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        
        # Pan/Zoom variables
        self.pressed = False
        self.xpress = None
        self.ypress = None
        self.zoom_scale = 1.1

        # Editing variables
        self.editable_points = None
        self.selected_point_idx = None
        self.drag_callback = None
        
        # Store zoom state
        self.saved_xlim = None
        self.saved_ylim = None

        # Connect events
        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_move)

    def on_scroll(self, event):
        if event.inaxes != self.axes: return
        cur_xlim = self.axes.get_xlim()
        cur_ylim = self.axes.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        
        if event.button == 'up':
            scale_factor = 1 / self.zoom_scale
        elif event.button == 'down':
            scale_factor = self.zoom_scale
        else:
            scale_factor = 1

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.axes.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.axes.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        # Save zoom state
        self.saved_xlim = self.axes.get_xlim()
        self.saved_ylim = self.axes.get_ylim()
        
        self.draw()

    def on_press(self, event):
        if event.inaxes != self.axes: return
        
        # Check for editable points
        if self.editable_points is not None and event.button == 1:
            pts_display = self.axes.transData.transform(self.editable_points)
            event_pt = np.array([event.x, event.y])
            diff = pts_display - event_pt
            dists = np.sqrt(np.sum(diff**2, axis=1))
            closest_idx = np.argmin(dists)
            if dists[closest_idx] < 10: # 10 pixel tolerance
                self.selected_point_idx = closest_idx
                # Save zoom state when dragging starts
                self.saved_xlim = self.axes.get_xlim()
                self.saved_ylim = self.axes.get_ylim()
                return

        if event.button == 1: # Left click
            self.pressed = True
            self.xpress = event.xdata
            self.ypress = event.ydata

    def on_release(self, event):
        self.selected_point_idx = None
        self.pressed = False
        self.xpress = None
        self.ypress = None

    def on_move(self, event):
        if event.inaxes != self.axes: return

        if self.selected_point_idx is not None:
            self.editable_points[self.selected_point_idx] = [event.xdata, event.ydata]
            if self.drag_callback:
                # Preserve zoom state during callback
                self.drag_callback(preserve_zoom=True)
            return

        if self.pressed and self.xpress is not None and self.ypress is not None and event.inaxes == self.axes:
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            
            cur_xlim = self.axes.get_xlim()
            cur_ylim = self.axes.get_ylim()
            
            self.axes.set_xlim(cur_xlim - dx)
            self.axes.set_ylim(cur_ylim - dy)
            self.draw()


class RacelineGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("F1Tenth Raceline Optimization")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize variables
        self.map_name = "blackbox2"
        self.final_data = None
        self.transformed_data = None
        self.raw_map_img = None
        self.map_resolution = None
        self.origin = None
        
        # Editing
        self.edit_control_points = None

        self.history_file = "raceline_history.json"
        self.history_data = []
        self.load_history_data()
        
        self.init_ui()
    
    def load_history_data(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.history_data = json.load(f)
            except:
                self.history_data = []

    def save_history_data(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history_data, f, indent=4)
    
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel - Controls
        left_panel = QVBoxLayout()
        
        # Create Tabs
        self.tabs = QTabWidget()
        
        # --- Tab 1: Optimization ---
        opt_tab = QWidget()
        opt_tab_layout = QVBoxLayout(opt_tab)
        
        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        
        # Map name
        map_layout = QHBoxLayout()
        map_layout.addWidget(QLabel("Map Name:"))
        self.map_name_input = QLineEdit(self.map_name)
        map_layout.addWidget(self.map_name_input)
        params_layout.addLayout(map_layout)
        
        # Threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold:"))
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 1.0)
        self.threshold_input.setSingleStep(0.05)
        self.threshold_input.setValue(0.25)
        thresh_layout.addWidget(self.threshold_input)
        params_layout.addLayout(thresh_layout)
        
        # Spur threshold
        spur_layout = QHBoxLayout()
        spur_layout.addWidget(QLabel("Spur Threshold (m):"))
        self.spur_input = QDoubleSpinBox()
        self.spur_input.setRange(0.0, 10.0)
        self.spur_input.setSingleStep(0.05)
        self.spur_input.setValue(0.25)
        spur_layout.addWidget(self.spur_input)
        params_layout.addLayout(spur_layout)
        
        # Track width margin
        margin_layout = QHBoxLayout()
        margin_layout.addWidget(QLabel("Track Width Margin (m):"))
        self.margin_input = QDoubleSpinBox()
        self.margin_input.setRange(0.0, 5.0)
        self.margin_input.setSingleStep(0.1)
        self.margin_input.setValue(0.2)
        margin_layout.addWidget(self.margin_input)
        params_layout.addLayout(margin_layout)
        
        # Downsample factor
        down_layout = QHBoxLayout()
        down_layout.addWidget(QLabel("Downsample Factor:"))
        self.downsample_input = QSpinBox()
        self.downsample_input.setRange(1, 20)
        self.downsample_input.setValue(1)
        down_layout.addWidget(self.downsample_input)
        params_layout.addLayout(down_layout)
        
        # Optimization type
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(QLabel("Optimization Type:"))
        self.opt_type_combo = QComboBox()
        self.opt_type_combo.addItems(["shortest_path", "mincurv", "mincurv_iqp", "mintime"])
        opt_layout.addWidget(self.opt_type_combo)
        params_layout.addLayout(opt_layout)
        
        params_group.setLayout(params_layout)
        opt_tab_layout.addWidget(params_group)
        
        # Action buttons
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        self.extract_btn = QPushButton("1. Extract Centerline")
        self.extract_btn.clicked.connect(self.extract_centerline)
        actions_layout.addWidget(self.extract_btn)
        
        self.visualize_btn = QPushButton("2. Visualize Centerline")
        self.visualize_btn.clicked.connect(self.visualize_centerline)
        self.visualize_btn.setEnabled(False)
        actions_layout.addWidget(self.visualize_btn)
        
        self.raceline_btn = QPushButton("3. Create Raceline")
        self.raceline_btn.clicked.connect(self.create_raceline)
        self.raceline_btn.setEnabled(False)
        actions_layout.addWidget(self.raceline_btn)
        
        self.visualize_raceline_btn = QPushButton("4. Visualize Raceline")
        self.visualize_raceline_btn.clicked.connect(self.visualize_raceline)
        self.visualize_raceline_btn.setEnabled(False)
        actions_layout.addWidget(self.visualize_raceline_btn)
        
        self.velocity_btn = QPushButton("5. Show Velocity Profile")
        self.velocity_btn.clicked.connect(self.show_velocity_profile)
        self.velocity_btn.setEnabled(False)
        actions_layout.addWidget(self.velocity_btn)
        
        actions_group.setLayout(actions_layout)
        opt_tab_layout.addWidget(actions_group)
        opt_tab_layout.addStretch()
        
        # --- Tab 2: History ---
        hist_tab = QWidget()
        hist_layout = QVBoxLayout(hist_tab)
        
        self.history_list = QListWidget()
        self.update_history_list()
        hist_layout.addWidget(self.history_list)
        
        self.load_hist_btn = QPushButton("Load Selected Raceline")
        self.load_hist_btn.clicked.connect(self.load_history_item)
        hist_layout.addWidget(self.load_hist_btn)
        
        self.delete_hist_btn = QPushButton("Delete Selected")
        self.delete_hist_btn.clicked.connect(self.delete_history_item)
        hist_layout.addWidget(self.delete_hist_btn)
        
        # --- Tab 3: Manual Edit ---
        edit_tab = QWidget()
        edit_layout = QVBoxLayout(edit_tab)
        
        self.load_edit_btn = QPushButton("Load Selected from History")
        self.load_edit_btn.clicked.connect(self.load_for_editing)
        edit_layout.addWidget(self.load_edit_btn)
        
        self.gen_curve_btn = QPushButton("Generate Editable Curve (Every 5th point)")
        self.gen_curve_btn.clicked.connect(self.generate_editable_curve)
        self.gen_curve_btn.setEnabled(False)
        edit_layout.addWidget(self.gen_curve_btn)
        
        self.save_edit_btn = QPushButton("Save Edited Raceline")
        self.save_edit_btn.clicked.connect(self.save_edited_raceline)
        self.save_edit_btn.setEnabled(False)
        edit_layout.addWidget(self.save_edit_btn)
        
        edit_layout.addStretch()

        # Add tabs to left panel
        self.tabs.addTab(opt_tab, "Optimization")
        self.tabs.addTab(hist_tab, "History")
        self.tabs.addTab(edit_tab, "Manual Edit")
        left_panel.addWidget(self.tabs)
        
        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        left_panel.addWidget(log_group)
        
        # Right panel - Visualization
        right_panel = QVBoxLayout()
        self.canvas = MplCanvas(self, width=8, height=8, dpi=100)
        right_panel.addWidget(self.canvas)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)
    
    def log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def extract_centerline(self):
        self.log("Starting centerline extraction...")
        self.extract_btn.setEnabled(False)
        
        worker = WorkerThread(self._extract_centerline_worker)
        worker.finished.connect(self.on_extraction_finished)
        worker.error.connect(self.on_error)
        worker.start()
        self.worker = worker  # Keep reference
    
    def _extract_centerline_worker(self):
        MAP_NAME = self.map_name_input.text()
        THRESHOLD = self.threshold_input.value()
        SPUR_THRESH_M = self.spur_input.value()
        TRACK_WIDTH_MARGIN = self.margin_input.value()
        DOWNSAMPLE_FACTOR = self.downsample_input.value()
        
        # Load map
        png_path = f"maps/{MAP_NAME}.png"
        pgm_path = f"maps/{MAP_NAME}.pgm"
        if os.path.exists(png_path):
            map_img_path = png_path
        elif os.path.exists(pgm_path):
            map_img_path = pgm_path
        else:
            raise FileNotFoundError(f"Neither {png_path} nor {pgm_path} found.")
        
        img_pil = Image.open(map_img_path).convert('L')
        raw_map_img = np.array(img_pil.transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        
        yaml_path = f"maps/{MAP_NAME}.yaml"
        with open(yaml_path, 'r') as f:
            map_metadata = yaml.safe_load(f)
        
        map_resolution = map_metadata['resolution']
        origin = map_metadata['origin']
        
        # Binarize and compute EDT
        binary_map = raw_map_img.copy()
        binary_map[binary_map <= 230.0] = 0
        binary_map[binary_map > 230.0] = 1
        binary_map = binary_map.astype(np.uint8)
        
        dist_transform = edt(binary_map)
        
        # Extract skeleton
        max_dist = dist_transform.max()
        centers_mask = dist_transform > (THRESHOLD * max_dist)
        centerline_bool = skeletonize(centers_mask)
        
        # Build graph
        skel_pts = np.column_stack(np.nonzero(centerline_bool))
        node_mapping = {(int(y), int(x)): idx for idx, (y, x) in enumerate(skel_pts)}
        
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1), (0, 1),
                            (1, -1), (1, 0), (1, 1)]
        edges = []
        for idx, (y, x) in enumerate(skel_pts):
            for dy, dx in neighbor_offsets:
                ny, nx_coord = int(y + dy), int(x + dx)
                if (ny, nx_coord) in node_mapping:
                    jdx = node_mapping[(ny, nx_coord)]
                    if jdx > idx:
                        edges.append((idx, jdx))
        
        G = nx.Graph()
        G.add_nodes_from(range(len(skel_pts)))
        G.add_edges_from(edges)
        
        # Prune spurs
        G_pruned = self._prune_spurs(G, skel_pts, map_resolution, SPUR_THRESH_M)
        
        # Extract main path
        ordered_nodes = self._extract_main_path(G_pruned, skel_pts, map_resolution)
        
        # Convert to world coords
        waypoints_px = np.array([[skel_pts[n][1], skel_pts[n][0]] for n in ordered_nodes], dtype=np.float64)
        track_widths_px = np.array([
            [dist_transform[int(skel_pts[n][0]), int(skel_pts[n][1])],
             dist_transform[int(skel_pts[n][0]), int(skel_pts[n][1])]]
            for n in ordered_nodes
        ], dtype=np.float64)
        
        if DOWNSAMPLE_FACTOR > 1:
            waypoints_px = waypoints_px[::DOWNSAMPLE_FACTOR]
            track_widths_px = track_widths_px[::DOWNSAMPLE_FACTOR]
        
        waypoints_m = waypoints_px * map_resolution
        waypoints_m[:, 0] += origin[0]
        waypoints_m[:, 1] += origin[1]
        
        track_widths_m = track_widths_px * map_resolution
        track_widths_m[:, 0] = np.maximum(track_widths_m[:, 0] - TRACK_WIDTH_MARGIN, 0.0)
        track_widths_m[:, 1] = np.maximum(track_widths_m[:, 1] - TRACK_WIDTH_MARGIN, 0.0)
        
        self.final_data = np.hstack([waypoints_m, track_widths_m])
        
        # Save to CSV
        os.makedirs("inputs/tracks", exist_ok=True)
        out_path = f"inputs/tracks/{MAP_NAME}.csv"
        header = "x_m,y_m,w_tr_right_m,w_tr_left_m"
        np.savetxt(out_path, self.final_data, fmt="%.4f", delimiter=",", header=header)
        
        return f"Saved {self.final_data.shape[0]} centerline points to {out_path}"
    
    def _prune_spurs(self, G_in, skel_pts, resolution, spur_thresh_m):
        Gp = G_in.copy()
        changed = True
        while changed:
            changed = False
            degrees = dict(Gp.degree())
            leaves = [n for n, d in degrees.items() if d == 1]
            to_remove = set()
            for leaf in leaves:
                path_nodes = [leaf]
                curr = leaf
                prev = None
                length_px = 0.0
                while True:
                    nbrs = [n for n in Gp.neighbors(curr) if n != prev]
                    if not nbrs:
                        break
                    next_node = nbrs[0]
                    y1, x1 = skel_pts[curr]
                    y2, x2 = skel_pts[next_node]
                    length_px += np.hypot(y2 - y1, x2 - x1)
                    path_nodes.append(next_node)
                    deg_next = Gp.degree(next_node)
                    if deg_next != 2:
                        break
                    prev, curr = curr, next_node
                length_m = length_px * resolution
                if length_m < spur_thresh_m:
                    to_remove.update(path_nodes)
                    changed = True
            if changed and to_remove:
                Gp.remove_nodes_from(to_remove)
        return Gp
    
    def _extract_main_path(self, Gp, skel_pts, resolution):
        cycles = nx.cycle_basis(Gp)
        if cycles:
            best_cycle = None
            best_len = 0.0
            for cyc in cycles:
                length_px = 0.0
                for i in range(len(cyc)):
                    n1 = cyc[i]
                    n2 = cyc[(i + 1) % len(cyc)]
                    y1, x1 = skel_pts[n1]
                    y2, x2 = skel_pts[n2]
                    length_px += np.hypot(y2 - y1, x2 - x1)
                length_m = length_px * resolution
                if length_m > best_len:
                    best_len = length_m
                    best_cycle = cyc
            ordered = [best_cycle[0]]
            used = {best_cycle[0]}
            curr = best_cycle[0]
            while len(used) < len(best_cycle):
                for nbr in Gp.neighbors(curr):
                    if nbr in best_cycle and nbr not in used:
                        ordered.append(nbr)
                        used.add(nbr)
                        curr = nbr
                        break
            return ordered
        
        degrees = dict(Gp.degree())
        leaves = [n for n, d in degrees.items() if d == 1]
        if len(leaves) < 2:
            return list(nx.nodes(Gp))
        
        def edge_wt(u, v, d):
            y1, x1 = skel_pts[u]
            y2, x2 = skel_pts[v]
            return np.hypot(y2 - y1, x2 - x1)
        
        best_pair = None
        best_dist = 0.0
        for i, leaf in enumerate(leaves):
            dist_dict = nx.single_source_dijkstra_path_length(Gp, leaf, weight=edge_wt)
            for other in leaves[i+1:]:
                d = dist_dict.get(other, np.inf)
                if d > best_dist:
                    best_dist = d
                    best_pair = (leaf, other)
        
        if best_pair is None:
            return list(nx.nodes(Gp))
        
        start_node, end_node = best_pair
        path_nodes = nx.shortest_path(Gp, source=start_node, target=end_node, weight=edge_wt)
        return path_nodes
    
    def on_extraction_finished(self, message):
        self.log(message)
        self.extract_btn.setEnabled(True)
        self.visualize_btn.setEnabled(True)
        self.raceline_btn.setEnabled(True)
    
    def on_error(self, error_msg):
        self.log(f"ERROR: {error_msg}")
        self.extract_btn.setEnabled(True)
    
    def visualize_centerline(self):
        try:
            MAP_NAME = self.map_name_input.text()
            THRESHOLD = self.threshold_input.value()
            
            if os.path.exists(f"maps/{MAP_NAME}.png"):
                map_img_path = f"maps/{MAP_NAME}.png"
            elif os.path.exists(f"maps/{MAP_NAME}.pgm"):
                map_img_path = f"maps/{MAP_NAME}.pgm"
            else:
                raise Exception("Map not found!")
            
            map_yaml_path = f"maps/{MAP_NAME}.yaml"
            img_pil = Image.open(map_img_path).convert('L')
            self.raw_map_img = np.array(img_pil.transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
            
            with open(map_yaml_path, 'r') as yaml_stream:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                self.origin = map_metadata['origin']
            
            # Create filtered map (same as in extraction)
            binary_map = self.raw_map_img.copy()
            binary_map[binary_map <= 230.0] = 0
            binary_map[binary_map > 230.0] = 1
            binary_map = binary_map.astype(np.uint8)
            
            dist_transform = edt(binary_map)
            max_dist = dist_transform.max()
            centers_mask = dist_transform > (THRESHOLD * max_dist)
            
            raw_data = pd.read_csv(f"inputs/tracks/{MAP_NAME}.csv")
            x = raw_data["# x_m"].values
            y = raw_data["y_m"].values
            
            x -= self.origin[0]
            y -= self.origin[1]
            x /= self.map_resolution
            y /= self.map_resolution
            
            self.canvas.axes.clear()
            self.canvas.axes.imshow(centers_mask, cmap="gray", origin="lower", alpha=0.7)
            self.canvas.axes.plot(x, y, color='blue', linewidth=2, label='Centerline')
            self.canvas.axes.set_title(f"Centerline for {MAP_NAME} (Threshold: {THRESHOLD})")
            self.canvas.axes.legend()
            self.canvas.axes.axis('off')
            self.canvas.draw()
            
            self.log("Centerline visualization complete")
        except Exception as e:
            self.log(f"Error visualizing centerline: {str(e)}")
    
    def create_raceline(self):
        self.log("Creating raceline (this may take a while)...")
        self.raceline_btn.setEnabled(False)
        
        worker = WorkerThread(self._create_raceline_worker)
        worker.finished.connect(self.on_raceline_finished)
        worker.error.connect(self.on_error)
        worker.start()
        self.worker = worker
    
    def _create_raceline_worker(self):
        MAP_NAME = self.map_name_input.text()
        OPT_TYPE = self.opt_type_combo.currentText()
        
        result = subprocess.run(
            [sys.executable, "-m", "main_globaltraj_f110", "--map_name", MAP_NAME, "--opt_type", OPT_TYPE],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Raceline creation failed: {result.stderr}")
        
        return "Raceline created successfully"
    
    def on_raceline_finished(self, message):
        self.log(message)
        self.raceline_btn.setEnabled(True)
        self.visualize_raceline_btn.setEnabled(True)
        self.velocity_btn.setEnabled(True)
        
        # Add to history
        self.add_current_to_history()

    def add_current_to_history(self):
        MAP_NAME = self.map_name_input.text()
        OPT_TYPE = self.opt_type_combo.currentText()
        
        # Find latest csv
        try:
            csv_files = glob.glob(f'outputs/{MAP_NAME}/*.csv', recursive=True)
            if not csv_files:
                return
            csv_files = sorted(csv_files)
            latest_file = csv_files[-1]
            
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "map_name": MAP_NAME,
                "opt_type": OPT_TYPE,
                "filepath": os.path.abspath(latest_file)
            }
            
            self.history_data.append(entry)
            self.save_history_data()
            self.update_history_list()
            self.log(f"Run saved to history: {latest_file}")
            
        except Exception as e:
            self.log(f"Failed to save history: {e}")

    def update_history_list(self):
        self.history_list.clear()
        for i, entry in enumerate(reversed(self.history_data)):
            item_text = f"{entry['timestamp']} - {entry['map_name']} ({entry['opt_type']})"
            item = QListWidgetItem(item_text)
            # Store index in original list (since we display reversed)
            original_index = len(self.history_data) - 1 - i
            item.setData(Qt.ItemDataRole.UserRole, original_index)
            self.history_list.addItem(item)

    def load_history_item(self):
        current_item = self.history_list.currentItem()
        if not current_item:
            return
            
        idx = current_item.data(Qt.ItemDataRole.UserRole)
        entry = self.history_data[idx]
        filepath = entry['filepath']
        map_name = entry['map_name']
        
        if not os.path.exists(filepath):
            QMessageBox.warning(self, "Error", f"File not found: {filepath}")
            return
            
        self.log(f"Loading history item: {filepath}")
        self.map_name_input.setText(map_name)
        self.visualize_raceline_from_file(map_name, filepath)

    def delete_history_item(self):
        current_item = self.history_list.currentItem()
        if not current_item:
            return
            
        idx = current_item.data(Qt.ItemDataRole.UserRole)
        del self.history_data[idx]
        self.save_history_data()
        self.update_history_list()

    def visualize_raceline_from_file(self, map_name, filepath):
        try:
            # Load map image first
            if os.path.exists(f"maps/{map_name}.png"):
                map_img_path = f"maps/{map_name}.png"
            elif os.path.exists(f"maps/{map_name}.pgm"):
                map_img_path = f"maps/{map_name}.pgm"
            else:
                self.log(f"Map image for {map_name} not found.")
                return

            map_yaml_path = f"maps/{map_name}.yaml"
            img_pil = Image.open(map_img_path).convert('L')
            self.raw_map_img = np.array(img_pil.transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
            
            with open(map_yaml_path, 'r') as yaml_stream:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                self.origin = map_metadata['origin']

            # Load CSV
            raw_data = pd.read_csv(filepath, header=None, sep=',')
            
            self.transformed_data = raw_data.copy()
            self.transformed_data -= np.array([self.origin[0], self.origin[1], 0])
            self.transformed_data.iloc[:, :2] /= self.map_resolution
            
            self.canvas.axes.clear()
            self.canvas.axes.imshow(self.raw_map_img, cmap='gray', origin='lower')
            self.canvas.axes.plot(self.transformed_data.iloc[:, 0], 
                                 self.transformed_data.iloc[:, 1], 
                                 color='red', linewidth=2)
            self.canvas.axes.set_title(f"Raceline: {map_name} (History)")
            self.canvas.axes.axis('off')
            self.canvas.draw()
            
            self.velocity_btn.setEnabled(True)
            
        except Exception as e:
            self.log(f"Error loading history item: {str(e)}")

    def visualize_raceline(self):
        try:
            MAP_NAME = self.map_name_input.text()
            csv_files = glob.glob(f'outputs/{MAP_NAME}/*.csv', recursive=True)
            csv_files = sorted(csv_files)
            raw_data = pd.read_csv(csv_files[-1], header=None, sep=',')
            
            self.transformed_data = raw_data.copy()
            self.transformed_data -= np.array([self.origin[0], self.origin[1], 0])
            self.transformed_data.iloc[:, :2] /= self.map_resolution
            
            self.canvas.axes.clear()
            self.canvas.axes.imshow(self.raw_map_img, cmap='gray', origin='lower')
            self.canvas.axes.plot(self.transformed_data.iloc[:, 0], 
                                 self.transformed_data.iloc[:, 1], 
                                 color='red', linewidth=2)
            self.canvas.axes.set_title(f"Raceline for {MAP_NAME}")
            self.canvas.axes.axis('off')
            self.canvas.draw()
            
            # Save image
            self.canvas.fig.savefig(f"outputs/{MAP_NAME}_raceline.png", 
                                   bbox_inches='tight', pad_inches=0)
            
            self.log("Raceline visualization complete")
        except Exception as e:
            self.log(f"Error visualizing raceline: {str(e)}")
    
    def load_for_editing(self):
        current_item = self.history_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a raceline in the History tab first.")
            return
            
        idx = current_item.data(Qt.ItemDataRole.UserRole)
        entry = self.history_data[idx]
        filepath = entry['filepath']
        map_name = entry['map_name']
        
        if not os.path.exists(filepath):
            QMessageBox.warning(self, "Error", f"File not found: {filepath}")
            return
            
        self.map_name_input.setText(map_name)
        self.visualize_raceline_from_file(map_name, filepath)
        self.gen_curve_btn.setEnabled(True)
        self.log(f"Loaded {map_name} for editing. Click 'Generate Editable Curve'.")

    def generate_editable_curve(self):
        if self.transformed_data is None:
            return

        points = self.transformed_data.iloc[:, :2].values
        # Downsample: every 5th point
        self.edit_control_points = points[::5].copy()
        
        self.canvas.editable_points = self.edit_control_points
        self.canvas.drag_callback = self.update_spline_visualization
        
        self.update_spline_visualization()
        self.save_edit_btn.setEnabled(True)
        self.log("Editable curve generated. Drag points to edit.")

    def update_spline_visualization(self, preserve_zoom=False):
        if self.edit_control_points is None:
            return
        
        # Save current zoom state if preserving
        if preserve_zoom:
            saved_xlim = self.canvas.axes.get_xlim()
            saved_ylim = self.canvas.axes.get_ylim()
            
        pts = self.edit_control_points.T
        try:
            # per=1 for closed curve
            tck, u = splprep(pts, u=None, s=0.0, per=1) 
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = splev(u_new, tck, der=0)
            
            self.canvas.axes.clear()
            if self.raw_map_img is not None:
                self.canvas.axes.imshow(self.raw_map_img, cmap='gray', origin='lower')
            
            self.canvas.axes.plot(self.edit_control_points[:, 0], self.edit_control_points[:, 1], 'bo', markersize=4, label='Control Points')
            self.canvas.axes.plot(x_new, y_new, 'r-', linewidth=2, label='Edited Raceline')
            
            # Restore zoom state if preserving
            if preserve_zoom:
                self.canvas.axes.set_xlim(saved_xlim)
                self.canvas.axes.set_ylim(saved_ylim)
            
            self.canvas.axes.set_title("Manual Editing")
            self.canvas.axes.legend()
            self.canvas.axes.axis('off')
            self.canvas.draw()
            
        except Exception as e:
            self.log(f"Spline error: {e}")

    def save_edited_raceline(self):
        if self.edit_control_points is None:
            return
            
        pts = self.edit_control_points.T
        try:
            tck, u = splprep(pts, u=None, s=0.0, per=1)
            # Generate points matching original count
            u_new = np.linspace(u.min(), u.max(), len(self.transformed_data)) 
            x_new, y_new = splev(u_new, tck, der=0)
            
            x_m = x_new * self.map_resolution + self.origin[0]
            y_m = y_new * self.map_resolution + self.origin[1]
            
            # Interpolate velocity
            v_orig = self.transformed_data.iloc[:, 2].values
            v_new = np.interp(np.linspace(0, 1, len(x_new)), np.linspace(0, 1, len(v_orig)), v_orig)
            
            map_name = self.map_name_input.text()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/{map_name}/{map_name}_edited_{timestamp}.csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            data_to_save = np.column_stack((x_m, y_m, v_new))
            
            # Pad with zeros if original had more columns
            orig_width = self.transformed_data.shape[1]
            if orig_width > 3:
                extras = np.zeros((len(x_new), orig_width - 3))
                data_to_save = np.column_stack((data_to_save, extras))
                
            np.savetxt(filename, data_to_save, delimiter=",", fmt="%.4f")
            
            self.log(f"Saved edited raceline to {filename}")
            
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "map_name": map_name,
                "opt_type": "manual_edit",
                "filepath": os.path.abspath(filename)
            }
            self.history_data.append(entry)
            self.save_history_data()
            self.update_history_list()
            
        except Exception as e:
            self.log(f"Error saving: {e}")

    def show_velocity_profile(self):
        try:
            MAP_NAME = self.map_name_input.text()
            
            self.canvas.axes.clear()
            scatter = self.canvas.axes.scatter(
                self.transformed_data.iloc[:, 0], 
                self.transformed_data.iloc[:, 1], 
                c=self.transformed_data.iloc[:, 2],
                cmap='viridis'
            )
            self.canvas.axes.imshow(self.raw_map_img, cmap='gray', origin='lower', alpha=0.5)
            cbar = self.canvas.fig.colorbar(scatter, ax=self.canvas.axes, shrink=0.5)
            cbar.set_label('Velocity (m/s)')
            self.canvas.axes.set_title(f"Velocity Profile for {MAP_NAME}")
            self.canvas.axes.axis('off')
            self.canvas.draw()
            
            self.log("Velocity profile visualization complete")
        except Exception as e:
            self.log(f"Error showing velocity profile: {str(e)}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = RacelineGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

"""
Refactored LLaDA GUI Application with modularized OpenGL visualizations.

The visualization functionality has been restructured using an abstract
Visualizer base class that defines common drawing methods (draw_shape,
adjust_color_alpha, render_text, etc.) and an abstract draw() method. Each
visualization (Token Stream, Test Square, Memory Influence Map, Abstract Token Cloud,
Probability Distribution) is implemented as a self-contained subclass of Visualizer.
The QOpenGLWidget subclass now holds a dictionary mapping visualization type
names to Visualizer instances and simply delegates the paint call to the current
visualizer.
"""

import random
import sys
import numpy as np
import torch
from OpenGL.GL import *
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QSettings, QThread
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTextEdit, QPushButton, QLabel, QSpinBox, QComboBox, QGroupBox,
    QCheckBox, QProgressBar, QSplitter, QMessageBox, QGridLayout,
    QScrollArea, QDoubleSpinBox, QRadioButton, QButtonGroup, QStatusBar
)
import matplotlib.pyplot as plt

from core.config import DEFAULT_GENERATION_PARAMS as DEFAULT_PARAMS
from core.llada_worker import LLaDAWorker
from core.performance.enhanced_worker import EnhancedLLaDAWorker
from gui.memory_monitor import MemoryMonitor


# ------------------------------------------------------------------------------
# Abstract Visualizer Base Class and Concrete Implementations
# ------------------------------------------------------------------------------

class Visualizer:
    def __init__(self, parent=None):
        # The parent should be a QOpenGLWidget so that we can query its width/height.
        self.parent = parent
        self.color_scheme_name = "Cool"
        self.animation_time = 0.0
        self.animation_speed = 0.01
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.text_textures = {}
        self.text_coords = {}

    def update_animation(self):
        self.animation_time += self.animation_speed

    def draw(self):
        """Subclasses must implement this method to perform their drawing."""
        raise NotImplementedError("draw() must be implemented by Visualizer subclasses")

    def draw_shape(self, shape_type, x, y, size, color, num_vertices=None):
        vertices = []
        colors = []
        color_rgba = [color.redF(), color.greenF(), color.blueF(), 0.7]
        mode = GL_TRIANGLES

        if shape_type == "Circle":
            num_vertices = num_vertices or 32
            for theta in np.linspace(0, 2 * np.pi, num_vertices, endpoint=False):
                x0 = x + size * np.cos(theta)
                y0 = y + size * np.sin(theta)
                x1 = x + size * np.cos(theta + 2 * np.pi / num_vertices)
                y1 = y + size * np.sin(theta + 2 * np.pi / num_vertices)
                vertices.extend([x, y, 0.0, x0, y0, 0.0, x1, y1, 0.0])
                colors.extend(color_rgba * 3)
        elif shape_type == "Square":
            vertices.extend([x - size, y - size, 0.0,
                             x + size, y - size, 0.0,
                             x + size, y + size, 0.0,
                             x - size, y + size, 0.0])
            colors.extend(color_rgba * 4)
            mode = GL_QUADS
        elif shape_type == "Triangle":
            vertices.extend([x, y + size, 0.0,
                             x - size, y - size, 0.0,
                             x + size, y - size, 0.0])
            colors.extend(color_rgba * 3)
        elif shape_type == "Line":
            vertices.extend([x, y - size, 0.0,
                             x, y + size, 0.0])
            colors.extend(color_rgba * 2)
            mode = GL_LINES
            glLineWidth(2.0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glColorPointer(4, GL_FLOAT, 0, colors)
        glDrawArrays(mode, 0, len(vertices) // 3)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        if mode == GL_LINES:
            glLineWidth(1.0)

    def adjust_color_alpha(self, color, alpha_factor):
        new_alpha = max(0.0, min(1.0, color.alphaF() * alpha_factor))
        return QColor.fromRgbF(color.redF(), color.greenF(), color.blueF(), new_alpha)

    def render_text(self, x, y, text, color, font_size=16, widget_width=None, widget_height=None):
        if not text:
            return
        font = QFont("Arial", font_size)
        metrics = QFontMetrics(font)
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()

        # Use widget dimensions if provided, fallback otherwise.
        if widget_width is None:
            widget_width = self.parent.width() if self.parent else 800
        if widget_height is None:
            widget_height = self.parent.height() if self.parent else 600

        if text not in self.text_textures:
            image = QImage(text_width, text_height, QImage.Format.Format_ARGB32)
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(0, metrics.ascent(), text)
            painter.end()

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            image_data = image.bits().asstring(image.sizeInBytes()) if image.bits() else None
            if image_data:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_width, text_height,
                             0, GL_BGRA, GL_UNSIGNED_BYTE, image_data)
            self.text_textures[text] = texture_id
            self.text_coords[text] = (text_width / (widget_width * self.zoom_level),
                                        text_height / (widget_height * self.zoom_level))

        glBindTexture(GL_TEXTURE_2D, self.text_textures[text])
        w, h = self.text_coords[text]
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x, y)
        glTexCoord2f(1, 1); glVertex2f(x + w, y)
        glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
        glTexCoord2f(0, 0); glVertex2f(x, y + h)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)

    def get_colormap(self, scheme_name="Cool"):
        colormap_options = {
            "Cool": plt.cm.viridis,
            "Warm": plt.cm.magma,
            "GrayScale": plt.cm.gray,
            "Rainbow": plt.cm.rainbow,
            "CoolWarm": plt.cm.coolwarm,
            "Plasma": plt.cm.plasma
        }
        return colormap_options.get(scheme_name, plt.cm.viridis)

    def get_token_color(self, index, total_tokens):
        hue = (index * 360 / total_tokens) % 360 / 360.0
        color_scheme_funcs = {
            "Cool": lambda h: QColor.fromHslF(h * 0.5 + 0.5, 0.8, 0.7),
            "Warm": lambda h: QColor.fromHslF(h * 0.1, 0.7, 0.7),
            "GrayScale": lambda h: QColor.fromRgbF(0.2 + (1.0 - (index / total_tokens) * 0.7),
                                                     0.2 + (1.0 - (index / total_tokens) * 0.7),
                                                     0.2 + (1.0 - (index / total_tokens) * 0.7)),
            "Rainbow": lambda h: QColor.fromHslF(h, 0.9, 0.6),
            "CoolWarm": lambda h: QColor.fromHslF(h * 0.5, 0.7, 0.7),
            "Plasma": lambda h: QColor.fromHslF(h * 0.2, 0.8, 0.6)
        }
        return color_scheme_funcs.get(self.color_scheme_name,
                                      lambda h: QColor.fromHslF(h * 0.5 + 0.5, 0.8, 0.7))(hue)


class TokenStreamVisualizer(Visualizer):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.token_stream_data_strings = []
        self.token_stream_mask_indices = []
        self.token_stream_confidence_scores = []
        self.token_stream_data_mode = "Decoded Tokens"
        self.token_shape_name = "Circle"
        self.token_size = 0.03
        self.token_spacing = 0.07
        self.show_probability_bar = True

    def set_token_stream_data(self, data, masks, confidences, data_mode="Decoded Tokens"):
        self.token_stream_data_strings = data
        self.token_stream_mask_indices = masks
        self.token_stream_confidence_scores = confidences
        self.token_stream_data_mode = data_mode

    def draw(self):
        num_tokens = len(self.token_stream_data_strings)
        if num_tokens == 0:
            return
        spacing = self.token_spacing
        start_x = - (num_tokens - 1) * spacing / 2

        for i in range(num_tokens):
            x = start_x + i * spacing
            y = np.sin(self.animation_time * 2.0 + i * 0.5) * 0.02
            size = self.token_size
            color = self.get_token_color(i, num_tokens)
            shape = self.token_shape_name

            if self.token_stream_mask_indices[i]:
                shape = "Square"
                size *= 0.8
                color = color.darker(150)
            else:
                confidence = self.token_stream_confidence_scores[i]
                size += confidence * 0.01
                color = self.adjust_color_alpha(color, 0.5 + confidence * 0.5)

            text_to_render = (self.token_stream_data_strings[i]
                              if self.token_stream_data_mode == "Decoded Tokens"
                              else str(self.token_stream_data_strings[i]))
            widget_width = self.parent.width() if self.parent else 800
            widget_height = self.parent.height() if self.parent else 600
            self.draw_shape(shape, x - self.token_spacing * 0.3, y - self.token_size * 2, size, color)
            self.render_text(x - self.token_spacing * 0.3, y - self.token_size * 2,
                             text_to_render, QColor(200, 200, 220),
                             widget_width=widget_width, widget_height=widget_height)

            if self.show_probability_bar:
                prob_bar_height = confidence * 0.1
                prob_bar_y_offset = self.token_size * 1.5
                prob_bar_color = color.lighter(130)
                self.draw_shape("Line", x, y + prob_bar_y_offset, prob_bar_height, prob_bar_color)


class TestSquareVisualizer(Visualizer):
    def draw(self):
        self.draw_shape("Square", 0, 0, 0.5, QColor.fromRgbF(1.0, 1.0, 0.0))


class MemoryInfluenceMapVisualizer(Visualizer):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.memory_influence_data = None

    def set_memory_influence_data(self, data):
        self.memory_influence_data = data

    def draw(self):
        grid_resolution = 32
        influence_data = (self.memory_influence_data
                          if self.memory_influence_data is not None
                          else np.random.rand(grid_resolution, grid_resolution))
        cmap = self.get_colormap(self.color_scheme_name)
        normalized_data = (influence_data - influence_data.min()) / (
            influence_data.max() - influence_data.min() + 1e-8)
        colors = cmap(normalized_data.flatten())
        cell_size = 2.0 / grid_resolution
        vertices = []
        vertex_colors = []
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                x = -1.0 + j * cell_size
                y = 1.0 - i * cell_size
                color_tuple = colors[i * grid_resolution + j]
                vertices.extend([x, y, 0.0,
                                 x + cell_size, y, 0.0,
                                 x + cell_size, y - cell_size, 0.0,
                                 x, y - cell_size, 0.0])
                vertex_colors.extend([color_tuple[0], color_tuple[1], color_tuple[2], 1.0] * 4)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glColorPointer(4, GL_FLOAT, 0, vertex_colors)
        glDrawArrays(GL_QUADS, 0, len(vertices) // 3)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)


class AbstractTokenCloudVisualizer(Visualizer):
    def draw(self):
        num_clouds = 50
        for _ in range(num_clouds):
            x, y = random.uniform(-0.9, 0.9), random.uniform(-0.9, 0.9)
            size = random.uniform(0.01, 0.05)
            color = QColor.fromRgbF(*[random.uniform(0.3, 0.7) for _ in range(3)])
            self.draw_shape("Triangle", x, y, size, color)
        widget_width = self.parent.width() if self.parent else 800
        widget_height = self.parent.height() if self.parent else 600
        self.render_text(-0.5, 0, "Token Cloud Active", QColor(255, 255, 255),
                         widget_width=widget_width, widget_height=widget_height)


class ProbabilityDistributionVisualizer(Visualizer):
    def draw(self):
        num_bars = 50
        bar_width = 0.02
        spacing = bar_width * 1.2
        start_x = - (num_bars - 1) * spacing / 2
        probabilities = np.random.rand(num_bars)
        normalized_probs = probabilities / np.max(probabilities) if np.max(probabilities) > 0 else probabilities

        for i in range(num_bars):
            x = start_x + i * spacing
            y_base = -0.9
            bar_height = normalized_probs[i] * 1.8
            color = self.get_token_color(i, num_bars)
            self.draw_shape("Square", x, y_base + bar_height / 2, bar_width / 2, color)


# ------------------------------------------------------------------------------
# QOpenGLWidget subclass delegating drawing to the active Visualizer
# ------------------------------------------------------------------------------

class LladaVis(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.vis_mode = "Token Stream"
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update)
        self.animation_timer.start(20)
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.setMouseTracking(True)
        self.last_mouse_pos = None

        # Initialize visualizers for each type.
        self.visualizers = {
            "Token Stream": TokenStreamVisualizer(self),
            "Test Square": TestSquareVisualizer(self),
            "Memory Influence Map": MemoryInfluenceMapVisualizer(self),
            "Abstract Token Cloud": AbstractTokenCloudVisualizer(self),
            "Probability Distribution": ProbabilityDistributionVisualizer(self)
        }
        self.current_visualizer = self.visualizers[self.vis_mode]

    # --- Setter methods ---
    def set_visualization_type(self, viz_type):
        self.vis_mode = viz_type
        if viz_type in self.visualizers:
            self.current_visualizer = self.visualizers[viz_type]
        self.update()

    def set_color_scheme(self, scheme_name):
        for viz in self.visualizers.values():
            viz.color_scheme_name = scheme_name
        self.update()

    def set_token_stream_data(self, data, masks, confidences, data_mode="Decoded Tokens"):
        if "Token Stream" in self.visualizers:
            self.visualizers["Token Stream"].set_token_stream_data(data, masks, confidences, data_mode)
        self.update()

    def set_memory_influence_data(self, data):
        if "Memory Influence Map" in self.visualizers:
            self.visualizers["Memory Influence Map"].set_memory_influence_data(data)
        self.update()

    def set_token_shape(self, shape_name):
        if "Token Stream" in self.visualizers:
            self.visualizers["Token Stream"].token_shape_name = shape_name
        self.update()

    def set_animation_speed(self, speed):
        for viz in self.visualizers.values():
            viz.animation_speed = speed
        self.update()

    def set_token_size(self, size):
        if "Token Stream" in self.visualizers:
            self.visualizers["Token Stream"].token_size = size
        self.update()

    def set_token_spacing(self, spacing):
        if "Token Stream" in self.visualizers:
            self.visualizers["Token Stream"].token_spacing = spacing
        self.update()

    def set_zoom_level(self, zoom):
        self.zoom_level = zoom
        for viz in self.visualizers.values():
            viz.zoom_level = zoom
        self.update()

    def set_pan(self, pan_x, pan_y):
        self.pan_x = pan_x
        self.pan_y = pan_y
        self.update()

    def set_token_stream_data_mode(self, mode):
        if "Token Stream" in self.visualizers:
            self.visualizers["Token Stream"].token_stream_data_mode = mode
        self.update()

    def set_show_probability_bar(self, show_bar):
        if "Token Stream" in self.visualizers:
            self.visualizers["Token Stream"].show_probability_bar = show_bar
        self.update()

    def initializeGL(self):
        ctx = self.context()
        if not ctx.isValid():
            raise RuntimeError("Failed to create a valid OpenGL context")
        fmt = ctx.format()
        if fmt.majorVersion() < 3:
            print("Warning: OpenGL 3.3+ not supported.")
        fmt.setSwapInterval(1)  # Enable V-Sync
        if not ctx.setFormat(fmt):
            print("Warning: Could not set V-Sync, continuing without.")

        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1 * self.zoom_level + self.pan_x, 1 * self.zoom_level + self.pan_x,
                -1 * self.zoom_level + self.pan_y, 1 * self.zoom_level + self.pan_y, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        self.current_visualizer.update_animation()
        self.current_visualizer.draw()

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)

    def wheelEvent(self, event):
        zoom_factor = 0.15
        delta = event.angleDelta().y()
        self.zoom_level *= (1.0 - zoom_factor) if delta > 0 else (1.0 + zoom_factor)
        self.zoom_level = max(0.1, min(self.zoom_level, 5.0))
        self.set_zoom_level(self.zoom_level)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is not None:
            pos = event.pos()
            delta_x = pos.x() - self.last_mouse_pos.x()
            delta_y = pos.y() - self.last_mouse_pos.y()
            pan_speed = 0.005 * self.zoom_level
            self.pan_x -= delta_x * pan_speed
            self.pan_y += delta_y * pan_speed
            self.last_mouse_pos = pos
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = None


# ------------------------------------------------------------------------------
# Main GUI Application (unchanged functionality)
# ------------------------------------------------------------------------------

class LLaDAGUINew(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLaDA GUI - OpenGL Viz - Prototype")
        self.resize(1200, 900)
        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.update.connect(self.update_memory_status_bar)
        self.memory_monitor.start()
        self.worker = None
        self.token_stream_data_mode = "Decoded Tokens"

        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.timeout.connect(self._delayed_cleanup_gpu_memory)
        self.gpu_cleanup_delay = 5 * 60 * 1000  # 5 minutes delay
        self.keep_gpu_loaded = False

        self.setup_ui()
        self.load_settings()

    def _create_labeled_widget(self, label_text, widget=None, tooltip=None):
        label = QLabel(label_text)
        if tooltip:
            widget.setToolTip(tooltip)
        return label, widget

    def add_widget_to_grid(self, layout, label_text, widget, row, col, tooltip=None):
        label, widget = self._create_labeled_widget(label_text, widget, tooltip)
        layout.addWidget(label, row, col)
        layout.addWidget(widget, row, col + 1)

    def setup_visualization_settings_group(self):
        viz_settings_group = QGroupBox("👁️ Visualization Settings")
        viz_settings_layout = QGridLayout()

        self.visualization_type_combo = QComboBox()
        viz_types = ["Token Stream", "Test Square", "Memory Influence Map", "Abstract Token Cloud", "Probability Distribution"]
        self.visualization_type_combo.addItems(viz_types)
        self.visualization_type_combo.currentTextChanged.connect(self.on_visualization_type_changed)
        self.add_widget_to_grid(viz_settings_layout, "Type:", self.visualization_type_combo, 0, 0)

        self.color_scheme_combo = QComboBox()
        color_schemes = ["Cool", "Warm", "GrayScale", "Rainbow", "CoolWarm", "Plasma"]
        self.color_scheme_combo.addItems(color_schemes)
        self.color_scheme_combo.setCurrentText("Cool")
        self.color_scheme_combo.currentTextChanged.connect(self.vis.set_color_scheme)
        self.add_widget_to_grid(viz_settings_layout, "Color Scheme:", self.color_scheme_combo, 1, 0)

        self.token_shape_combo = QComboBox()
        token_shapes = ["Circle", "Square", "Triangle", "Line"]
        self.token_shape_combo.addItems(token_shapes)
        self.token_shape_combo.setCurrentText("Circle")
        self.token_shape_combo.currentTextChanged.connect(self.vis.set_token_shape)
        self.add_widget_to_grid(viz_settings_layout, "Token Shape:", self.token_shape_combo, 2, 0)

        self.animation_speed_spin = self.spinbox(0.005, 0.001, 0.01, 0.1)
        self.animation_speed_spin.valueChanged.connect(self.vis.set_animation_speed)
        self.add_widget_to_grid(viz_settings_layout, "Animation Speed:", self.animation_speed_spin, 3, 0)

        self.token_size_spin = self.spinboxGrid("Token Size", 4, viz_settings_layout, self.vis.set_token_size, 0.03, 0.01, 0.1, 0.005)
        self.token_spacing_spin = self.spinboxGrid("Token Spacing", 5, viz_settings_layout, self.vis.set_token_spacing, 0.07, 0.2, 0.01, 0.01)

        self.token_data_mode_combo = QComboBox()
        data_modes = ["Decoded Tokens", "Token IDs"]
        self.token_data_mode_combo.addItems(data_modes)
        self.token_data_mode_combo.setCurrentText("Decoded Tokens")
        self.token_data_mode_combo.currentTextChanged.connect(self.on_token_stream_data_mode_changed)
        self.add_widget_to_grid(viz_settings_layout, "Token Data:", self.token_data_mode_combo, 7, 0)

        self.show_prob_bar_checkbox = QCheckBox("Show Prob. Bar")
        self.show_prob_bar_checkbox.setChecked(True)
        self.show_prob_bar_checkbox.stateChanged.connect(lambda state: self.vis.set_show_probability_bar(state == Qt.CheckState.Checked))
        self.add_widget_to_grid(viz_settings_layout, "Prob. Bar:", self.show_prob_bar_checkbox, 8, 0)

        viz_settings_group.setLayout(viz_settings_layout)
        return viz_settings_group

    def spinboxGrid(self, label: str, row: int, target, handler, val: float, min: float, max: float, step: float) -> QDoubleSpinBox:
        b = self.spinboxHandled(handler, val, min, max, step)
        self.add_widget_to_grid(target, label + ":", b, row, 0)
        return b

    def spinboxHandled(self, handler, val, min, max, step):
        b = self.spinbox(val, min, max, step)
        b.valueChanged.connect(handler)
        return b

    def spinbox(self, val, min, max, step):
        b = QDoubleSpinBox()
        b.setRange(min, max)
        b.setValue(val)
        b.setSingleStep(step)
        return b

    def setup_generation_parameters_group(self):
        gen_params_group = QGroupBox("⚙️ Generation Parameters")
        gen_params_layout = QGridLayout()

        self.gen_length_spin = QSpinBox()
        self.gen_length_spin.setRange(16, 512)
        self.gen_length_spin.setValue(DEFAULT_PARAMS['gen_length'])
        self.gen_length_spin.setSingleStep(16)
        self.add_widget_to_grid(gen_params_layout, "Length:", self.gen_length_spin, 0, 0)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(2, 512)
        self.steps_spin.setValue(DEFAULT_PARAMS['steps'])
        self.steps_spin.setSingleStep(1)
        self.add_widget_to_grid(gen_params_layout, "Steps:", self.steps_spin, 0, 2)

        self.block_length_spin = QSpinBox()
        self.block_length_spin.setRange(2, 256)
        self.block_length_spin.setValue(DEFAULT_PARAMS['block_length'])
        self.block_length_spin.setSingleStep(1)
        self.add_widget_to_grid(gen_params_layout, "Block Length:", self.block_length_spin, 1, 0)

        self.temperature_spin = self.spinbox(DEFAULT_PARAMS['temperature'], 0, 2, 0.01)
        self.add_widget_to_grid(gen_params_layout, "Temperature:", self.temperature_spin, 1, 2)

        self.cfg_scale_spin = self.spinbox(DEFAULT_PARAMS['cfg_scale'], 0, 5, 0.1)
        self.add_widget_to_grid(gen_params_layout, "CFG Scale:", self.cfg_scale_spin, 2, 0, "Classifier-Free Guidance scale")

        self.remasking_combo = QComboBox()
        remasking_methods = ["low_confidence", "random"]
        self.remasking_combo.addItems(remasking_methods)
        self.remasking_combo.setCurrentText(DEFAULT_PARAMS['remasking'])
        self.add_widget_to_grid(gen_params_layout, "Remasking:", self.remasking_combo, 2, 2, "Method for remasking tokens during generation")

        gen_params_group.setLayout(gen_params_layout)
        return gen_params_group

    def setup_hardware_memory_group(self):
        hw_memory_group = QGroupBox("⚙️ Hardware & Memory Options")
        hw_memory_layout = QGridLayout()

        device_label, _ = self._create_labeled_widget("Device:")
        self.device_group = QButtonGroup()
        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU (CUDA)")
        if torch.cuda.is_available():
            self.gpu_radio.setChecked(True)
        else:
            self.cpu_radio.setChecked(True)
            self.gpu_radio.setEnabled(False)
        self.device_group.addButton(self.cpu_radio, 0)
        self.device_group.addButton(self.gpu_radio, 1)
        hw_memory_layout.addWidget(device_label, 0, 0)
        hw_memory_layout.addWidget(self.cpu_radio, 0, 1)
        hw_memory_layout.addWidget(self.gpu_radio, 0, 2)

        self.extreme_mode_checkbox = QCheckBox("Extreme Mode")
        self.extreme_mode_checkbox.setToolTip("Enable extreme memory optimizations")
        self.fast_mode_checkbox = QCheckBox("Fast Mode")
        self.fast_mode_checkbox.setToolTip("Enable faster generation (lower quality)")
        self.enable_memory_checkbox = QCheckBox("Enable Memory Integration")
        self.enable_memory_checkbox.setToolTip("Enable memory integration for context-aware generation")
        hw_memory_layout.addWidget(self.extreme_mode_checkbox, 2, 0, 1, 3)
        hw_memory_layout.addWidget(self.fast_mode_checkbox, 3, 0, 1, 3)
        hw_memory_layout.addWidget(self.enable_memory_checkbox, 4, 0, 1, 3)

        self.keep_gpu_loaded_checkbox = QCheckBox("Keep GPU Loaded")
        self.keep_gpu_loaded_checkbox.setChecked(self.keep_gpu_loaded)
        self.keep_gpu_loaded_checkbox.setToolTip("Keep GPU loaded after generation for faster repeat requests")
        self.keep_gpu_loaded_checkbox.stateChanged.connect(self.on_keep_gpu_loaded_changed)
        hw_memory_layout.addWidget(self.keep_gpu_loaded_checkbox, 5, 0, 1, 3)

        hw_memory_group.setLayout(hw_memory_layout)
        return hw_memory_group

    def setup_realtime_stats_group(self):
        g = QGroupBox("📊 Realtime Statistics")
        l = QGridLayout()
        self.token_rate_label = QLabel("Token Rate: -")
        self.step_time_label = QLabel("Step Time: - ms/step")
        self.detailed_memory_label = QLabel("Memory Usage: -")
        l.addWidget(self.token_rate_label, 0, 0)
        l.addWidget(self.step_time_label, 1, 0)
        l.addWidget(self.detailed_memory_label, 2, 0)
        g.setLayout(l)
        return g

    def setup_memory_status_group(self):
        memory_status_group = QGroupBox("💾 Memory Status")
        memory_status_layout = QGridLayout()

        self.system_ram_progress_sidebar = QProgressBar()
        self.system_ram_progress_sidebar.setRange(0, 100)
        self.system_ram_progress_sidebar.setTextVisible(False)
        self.system_ram_label_sidebar = QLabel("- / - GB (-%)")
        self.add_widget_to_grid(memory_status_layout, "System RAM:", self.system_ram_progress_sidebar, 0, 0)
        memory_status_layout.addWidget(self.system_ram_label_sidebar, 0, 2)

        self.gpu_vram_progress_sidebar = QProgressBar()
        self.gpu_vram_progress_sidebar.setRange(0, 100)
        self.gpu_vram_progress_sidebar.setTextVisible(False)
        self.gpu_vram_label_sidebar = QLabel("- / - GB (-%)")
        self.add_widget_to_grid(memory_status_layout, "GPU VRAM:", self.gpu_vram_progress_sidebar, 1, 0)
        memory_status_layout.addWidget(self.gpu_vram_label_sidebar, 1, 2)

        memory_status_group.setLayout(memory_status_layout)
        return memory_status_group

    def setup_ui(self):
        main_widget = QSplitter(Qt.Orientation.Vertical)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt here...")
        main_widget.addWidget(self.prompt_input)

        center_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_widget.addWidget(center_splitter)

        self.vis = LladaVis(self)
        center_splitter.addWidget(self.vis)

        self.sidebar_scroll_area = QScrollArea()
        self.sidebar_scroll_area.setWidgetResizable(True)
        self.sidebar_widget = QWidget()
        self.side_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_scroll_area.setWidget(self.sidebar_widget)
        center_splitter.addWidget(self.sidebar_scroll_area)
        center_splitter.setSizes([600, 600])
        main_widget.setSizes([175, 725])

        self.side_layout.addWidget(self.setup_visualization_settings_group())
        self.side_layout.addWidget(self.setup_generation_parameters_group())
        self.side_layout.addWidget(self.setup_hardware_memory_group())
        self.side_layout.addWidget(self.setup_realtime_stats_group())
        self.side_layout.addWidget(self.setup_memory_status_group())
        self.side_layout.addStretch(1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        self.generate_button_status_bar = QPushButton("Generate")
        self.generate_button_status_bar.clicked.connect(self.on_generate_clicked)
        self.stop_button_status_bar = QPushButton("Stop")
        self.stop_button_status_bar.clicked.connect(self.on_stop_clicked)
        self.stop_button_status_bar.setEnabled(False)
        self.clear_button_status_bar = QPushButton("Clear")
        self.clear_button_status_bar.clicked.connect(self.clear_output)
        sb_layout = QHBoxLayout()
        sb_layout.addStretch(1)
        sb_layout.addWidget(self.generate_button_status_bar)
        sb_layout.addWidget(self.stop_button_status_bar)
        sb_layout.addWidget(self.clear_button_status_bar)
        status_button_widget = QWidget()
        status_button_widget.setLayout(sb_layout)
        self.status_bar.addPermanentWidget(status_button_widget)

        self.ram_indicator = QProgressBar()
        self.ram_indicator.setMaximumWidth(70)
        self.ram_indicator.setTextVisible(False)
        self.gpu_indicator = QProgressBar()
        self.gpu_indicator.setMaximumWidth(70)
        self.gpu_indicator.setTextVisible(False)
        self.status_bar.addPermanentWidget(QLabel("RAM:"))
        self.status_bar.addPermanentWidget(self.ram_indicator)
        self.status_bar.addPermanentWidget(QLabel("VRAM:"))
        self.status_bar.addPermanentWidget(self.gpu_indicator)

        self.setCentralWidget(main_widget)

    def on_visualization_type_changed(self, viz_type):
        self.vis.set_visualization_type(viz_type)
        is_token_stream = viz_type == "Token Stream"
        self.token_shape_combo.setEnabled(is_token_stream)
        self.animation_speed_spin.setEnabled(is_token_stream or viz_type == "Abstract Token Cloud")
        self.token_size_spin.setEnabled(is_token_stream)
        self.token_spacing_spin.setEnabled(is_token_stream)
        self.token_data_mode_combo.setEnabled(is_token_stream)
        self.show_prob_bar_checkbox.setEnabled(is_token_stream)

    def on_token_stream_data_mode_changed(self, mode):
        self.token_stream_data_mode = mode
        self.vis.set_token_stream_data_mode(mode)

    def on_keep_gpu_loaded_changed(self, state):
        self.keep_gpu_loaded = (state == Qt.CheckState.Checked.value)
        if self.keep_gpu_loaded:
            self.cancel_gpu_cleanup_timer()

    def get_generation_config(self):
        device = 'cuda' if self.gpu_radio.isChecked() and torch.cuda.is_available() else 'cpu'
        return {
            'gen_length': self.gen_length_spin.value(),
            'steps': self.steps_spin.value(),
            'block_length': self.block_length_spin.value(),
            'temperature': self.temperature_spin.value(),
            'cfg_scale': self.cfg_scale_spin.value(),
            'remasking': self.remasking_combo.currentText(),
            'device': device,
            'extreme_mode': self.extreme_mode_checkbox.isChecked(),
            'fast_mode': self.fast_mode_checkbox.isChecked(),
            'use_memory': self.enable_memory_checkbox.isChecked()
        }

    def start_gpu_cleanup_timer(self):
        if self.cleanup_timer.isActive():
            self.cleanup_timer.stop()
        self.cleanup_timer.start(self.gpu_cleanup_delay)
        self.status_bar.showMessage("GPU memory cleanup scheduled in 5 minutes...")

    def cancel_gpu_cleanup_timer(self):
        if self.cleanup_timer.isActive():
            self.cleanup_timer.stop()
            self.status_bar.showMessage("GPU memory cleanup cancelled.")

    @pyqtSlot()
    def _delayed_cleanup_gpu_memory(self):
        if self.keep_gpu_loaded:
            return
        self.status_bar.showMessage("Initiating GPU memory cleanup...")
        if self.worker:
            self.worker.cleanup_memory_signal.emit()
        else:
            print("Warning: No worker to signal for cleanup on timer event.")
            self.status_bar.showMessage("Warning: No worker to signal for cleanup.")

    @pyqtSlot()
    def on_generate_clicked(self):
        prompt_text = self.prompt_input.toPlainText().strip()
        if not prompt_text:
            QMessageBox.warning(self, "Input Error", "Prompt is empty")
            return

        config = self.get_generation_config()
        self.set_ui_generating(True)
        self.cancel_gpu_cleanup_timer()

        # Create a new worker for each generate click.
        w = LLaDAWorker(prompt_text, config)
        # Alternatively, use EnhancedLLaDAWorker:
        # w = EnhancedLLaDAWorker(prompt_text, config)
        self.worker = w
        w.progress.connect(self.update_progress)
        w.step_update.connect(self.update_visualization)
        w.finished.connect(self.generation_finished)
        w.error.connect(self.generation_error)
        w.realtime_stats.connect(self.update_realtime_stats_display)
        w.memory_influence_update.connect(self.vis.set_memory_influence_data)
        w.start()

    @pyqtSlot()
    def on_stop_clicked(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.set_ui_generating(False)
            if not self.keep_gpu_loaded:
                self.start_gpu_cleanup_timer()

    @pyqtSlot(int, str, dict)
    def update_progress(self, progress_percent, message, data):
        self.status_bar.showMessage(f"Generating - {message}")

    @pyqtSlot(int, list, list, list, list, list)
    def update_visualization(self, step, tokens, masks, confidences, token_ids, step_confidences):
        print(f"Step: {step}, Tokens: {tokens[:10]}..., Masks: {masks[:10]}..., Confidences: {confidences[:10]}..., Step Confidences: {step_confidences[:2]}...")
        if self.vis.vis_mode == "Token Stream":
            data_to_visualize = tokens if self.token_stream_data_mode == "Decoded Tokens" else token_ids
            self.vis.set_token_stream_data(data_to_visualize, masks, confidences, self.token_stream_data_mode)
        elif self.vis.vis_mode == "Memory Influence Map":
            self.vis.set_memory_influence_data(np.random.rand(32, 32))  # Placeholder data

    @pyqtSlot(str)
    def generation_finished(self, output_text):
        self.status_bar.showMessage("Generation Finished")
        print(f"Generated Output: {output_text}")
        self.prompt_input.setPlainText(output_text)
        self.set_ui_generating(False)
        if not self.keep_gpu_loaded:
            self.start_gpu_cleanup_timer()

    @pyqtSlot(str)
    def generation_error(self, error_message):
        self.status_bar.showMessage(f"Generation Error: {error_message}")
        QMessageBox.critical(self, "Generation Error", f"Error: {error_message}")
        self.set_ui_generating(False)
        if not self.keep_gpu_loaded:
            self.start_gpu_cleanup_timer()

    def set_ui_generating(self, is_generating):
        ui_elements = [
            self.generate_button_status_bar, self.prompt_input, self.visualization_type_combo,
            self.color_scheme_combo, self.token_shape_combo, self.animation_speed_spin,
            self.token_size_spin, self.token_spacing_spin,
            self.token_data_mode_combo, self.gen_length_spin, self.steps_spin,
            self.block_length_spin, self.temperature_spin, self.cfg_scale_spin,
            self.remasking_combo, self.cpu_radio, self.gpu_radio,
            self.extreme_mode_checkbox, self.fast_mode_checkbox,
            self.enable_memory_checkbox, self.show_prob_bar_checkbox,
            self.keep_gpu_loaded_checkbox
        ]
        for element in ui_elements:
            if element is not self.stop_button_status_bar:
                element.setEnabled(not is_generating)

        self.stop_button_status_bar.setEnabled(is_generating)
        self.token_shape_combo.setEnabled(not is_generating and self.visualization_type_combo.currentText() == "Token Stream")
        self.token_data_mode_combo.setEnabled(not is_generating and self.visualization_type_combo.currentText() == "Token Stream")
        self.show_prob_bar_checkbox.setEnabled(not is_generating and self.visualization_type_combo.currentText() == "Token Stream")

    @pyqtSlot(dict)
    def update_realtime_stats_display(self, stats):
        self.token_rate_label.setText(f"Token Rate: {stats.get('token_rate', '-')}")
        self.step_time_label.setText(f"Step Time: {stats.get('step_time', '-')} ms/step")
        self.detailed_memory_label.setText(f"Memory Usage: {stats.get('memory_usage', '-')}")

    @pyqtSlot()
    def clear_output(self):
        self.prompt_input.clear()
        self.vis.set_visualization_type("Token Stream")
        self.vis.set_color_scheme("Cool")
        self.vis.set_token_shape("Circle")
        self.vis.set_animation_speed(0.01)
        self.vis.set_token_size(0.03)
        self.vis.set_token_spacing(0.07)
        self.vis.set_zoom_level(1.0)
        self.token_data_mode_combo.setCurrentText("Decoded Tokens")
        self.show_prob_bar_checkbox.setChecked(True)
        self.cancel_gpu_cleanup_timer()

    @pyqtSlot(dict)
    def update_memory_status_bar(self, memory_stats):
        system_percent = memory_stats.get('system_percent', 0)
        gpu_percent = memory_stats.get('gpu_percent', 0)
        system_used = memory_stats.get('system_used', 0)
        system_total = memory_stats.get('system_total', 0)
        gpu_used = memory_stats.get('gpu_used', 0)
        gpu_total = memory_stats.get('gpu_total', 0)
        gpu_available = memory_stats.get('gpu_available', False)
        self.ram_indicator.setValue(int(system_percent))
        self.gpu_indicator.setValue(int(gpu_percent))
        self.system_ram_progress_sidebar.setValue(int(system_percent))
        self.gpu_vram_progress_sidebar.setValue(int(gpu_percent))
        self.system_ram_label_sidebar.setText(f"{system_used:.2f} / {system_total:.2f} GB ({system_percent:.0f}%)")
        self.gpu_vram_label_sidebar.setText(f"{gpu_used:.2f} / {gpu_total:.2f} GB ({gpu_percent:.0f}%)" if gpu_available else "N/A")

    def save_settings(self):
        settings = QSettings("LLaDA_GUI")
        s = settings.setValue
        s("visualization_type", self.visualization_type_combo.currentText())
        s("color_scheme", self.color_scheme_combo.currentText())
        s("token_shape", self.token_shape_combo.currentText())
        s("animation_speed", self.animation_speed_spin.value())
        s("token_size", self.token_size_spin.value())
        s("token_spacing", self.token_spacing_spin.value())
        s("gen_length", self.gen_length_spin.value())
        s("steps", self.steps_spin.value())
        s("block_length", self.block_length_spin.value())
        s("temperature", self.temperature_spin.value())
        s("cfg_scale", self.cfg_scale_spin.value())
        s("remasking", self.remasking_combo.currentText())
        s("device", "cuda" if self.gpu_radio.isChecked() else "cpu")
        s("extreme_mode", self.extreme_mode_checkbox.isChecked())
        s("fast_mode", self.fast_mode_checkbox.isChecked())
        s("use_memory", self.enable_memory_checkbox.isChecked())
        s("token_stream_data_mode", self.token_data_mode_combo.currentText())
        s("show_prob_bar", self.show_prob_bar_checkbox.isChecked())
        s("keep_gpu_loaded", self.keep_gpu_loaded_checkbox.isChecked())

    def load_settings(self):
        settings = QSettings("LLaDA_GUI")
        self.visualization_type_combo.setCurrentText(settings.value("visualization_type", "Token Stream"))
        self.color_scheme_combo.setCurrentText(settings.value("color_scheme", "Cool"))
        self.token_shape_combo.setCurrentText(settings.value("token_shape", "Circle"))
        self.animation_speed_spin.setValue(float(settings.value("animation_speed", 0.01)))
        self.token_size_spin.setValue(float(settings.value("token_size", 0.03)))
        self.token_spacing_spin.setValue(float(settings.value("token_spacing", 0.07)))
        self.gen_length_spin.setValue(int(settings.value("gen_length", DEFAULT_PARAMS['gen_length'])))
        self.steps_spin.setValue(int(settings.value("steps", DEFAULT_PARAMS['steps'])))
        self.block_length_spin.setValue(int(settings.value("block_length", DEFAULT_PARAMS['block_length'])))
        self.temperature_spin.setValue(float(settings.value("temperature", DEFAULT_PARAMS['temperature'])))
        self.cfg_scale_spin.setValue(float(settings.value("cfg_scale", DEFAULT_PARAMS['cfg_scale'])))
        self.remasking_combo.setCurrentText(settings.value("remasking", DEFAULT_PARAMS['remasking']))
        device = settings.value("device", "cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and torch.cuda.is_available():
            self.gpu_radio.setChecked(True)
        else:
            self.cpu_radio.setChecked(True)
        self.extreme_mode_checkbox.setChecked(bool(settings.value("extreme_mode", False)))
        self.fast_mode_checkbox.setChecked(bool(settings.value("fast_mode", False)))
        self.enable_memory_checkbox.setChecked(bool(settings.value("use_memory", False)))
        self.token_data_mode_combo.setCurrentText(settings.value("token_stream_data_mode", "Decoded Tokens"))
        self.show_prob_bar_checkbox.setChecked(bool(settings.value("show_prob_bar", True)))
        self.keep_gpu_loaded_checkbox.setChecked(bool(settings.value("keep_gpu_loaded", True)))

    def closeEvent(self, event):
        self.save_settings()
        self.memory_monitor.stop()
        self.cancel_gpu_cleanup_timer()
        if not self.keep_gpu_loaded:
            self._delayed_cleanup_gpu_memory()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = LLaDAGUINew()
    window.show()
    sys.exit(app.exec())


profile = False

if __name__ == "__main__":
    if profile:
        import cProfile
        sort = 'cumtime'
        cProfile.run('main()', None, sort)
    else:
        main()

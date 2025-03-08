#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diffusion process visualization module for the LLaDA GUI.
"""

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QPainter, QFont, QBrush, QPen
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QGridLayout,
    QSizePolicy, QGroupBox
)


class TokenVisualizer(QWidget):
    """Visualization widget for displaying tokens and their confidence."""

    def __init__(self, token_text="", confidence=0.0, is_masked=True, parent=None):
        super().__init__(parent)
        self.token_text = str(token_text)  # Convert to string to prevent type issues
        self.confidence = confidence
        self.is_masked = is_masked
        self.setMinimumSize(60, 30)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        return QSize(60, 30)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Define colors
        if self.is_masked:
            bg_color = QColor(230, 230, 230)  # Light gray for masked tokens
            text_color = QColor(120, 120, 120)
        else:
            # Color based on confidence
            confidence = max(0, min(self.confidence, 1.0))
            bg_color = QColor(
                int(255 * (1 - confidence)),
                int(200 * confidence),
                int(255 * (1 - confidence / 2))
            )
            text_color = QColor(0, 0, 0)

        # Draw background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 5, 5)

        # Draw border
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 5, 5)

        # Draw text
        painter.setPen(text_color)
        painter.setFont(QFont("Monospace", 8))

        # Handle different token text lengths - ensure we're working with a string
        display_text = str(self.token_text)
        if len(display_text) > 8:
            display_text = display_text[:7] + "…"

        painter.drawText(
            0, 0, self.width(), self.height(),
            Qt.AlignmentFlag.AlignCenter,
            "[MASK]" if self.is_masked else display_text
        )


class DiffusionVisualizer(QWidget):
    """Widget for visualizing the diffusion process."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.token_grid = []
        self.current_step = 0
        self.max_steps = 0

    def init_ui(self):
        """Initialize the UI elements."""
        layout = QVBoxLayout(self)

        # Step information
        self.step_label = QLabel("Diffusion Process: Step 0 of 0")
        self.step_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.step_label)

        # Scrollable token grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.token_container = QWidget()
        self.token_layout = QGridLayout(self.token_container)
        self.token_layout.setSpacing(4)

        self.scroll_area.setWidget(self.token_container)
        layout.addWidget(self.scroll_area)

    def setup_visualization(self, num_tokens, num_steps):
        """Initialize the visualization grid."""
        # Clear existing grid
        for i in reversed(range(self.token_layout.count())):
            widget = self.token_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Reset token grid
        self.token_grid = []
        self.current_step = 0
        self.max_steps = num_steps

        # Create visualization widgets
        for i in range(num_tokens):
            token_row = []
            for j in range(num_steps + 1):  # +1 for final state
                token_widget = TokenVisualizer(is_masked=(j == 0))
                self.token_layout.addWidget(token_widget, i, j)
                token_row.append(token_widget)
            self.token_grid.append(token_row)

        # Update labels
        self.step_label.setText(f"Diffusion Process: Step 0 of {num_steps}")

    def update_step(self, step, tokens, mask_indices, confidences=None):
        """Update the visualization for the current diffusion step."""
        if not isinstance(tokens, list):
            # If tokens is not a list, try to convert it
            try:
                if hasattr(tokens, 'tolist'):  # For numpy arrays or torch tensors
                    tokens = tokens.tolist()
                else:
                    tokens = [tokens]  # If it's a single value, make it a list
            except:
                tokens = ["?"]  # Fallback for unconvertible types

        if not isinstance(mask_indices, list):
            try:
                if hasattr(mask_indices, 'tolist'):
                    mask_indices = mask_indices.tolist()
                else:
                    mask_indices = [bool(mask_indices)]
            except:
                mask_indices = [True]

        if confidences is None:
            confidences = [0.5] * len(tokens)
        elif not isinstance(confidences, list):
            try:
                if hasattr(confidences, 'tolist'):
                    confidences = confidences.tolist()
                else:
                    confidences = [float(confidences)]
            except:
                confidences = [0.5] * len(tokens)

        # Ensure all lists are the same length
        max_len = max(len(tokens), len(mask_indices), len(confidences))
        tokens = tokens + ["?"] * (max_len - len(tokens))
        mask_indices = mask_indices + [True] * (max_len - len(mask_indices))
        confidences = confidences + [0.5] * (max_len - len(confidences))

        self.current_step = step

        # Update each token in the current column
        for i, (token, masked, conf) in enumerate(zip(tokens, mask_indices, confidences)):
            if i < len(self.token_grid):
                # Update the current and all future steps
                for j in range(step, self.max_steps + 1):
                    if j < len(self.token_grid[i]):
                        self.token_grid[i][j].token_text = str(token)
                        self.token_grid[i][j].is_masked = bool(masked)
                        self.token_grid[i][j].confidence = float(conf)
                        self.token_grid[i][j].update()

        # Update step label
        self.step_label.setText(f"Diffusion Process: Step {step} of {self.max_steps}")


class DiffusionProcessVisualizer(QGroupBox):
    """Combined widget for visualizing the whole diffusion process."""

    def __init__(self, parent=None):
        super().__init__("Diffusion Process Visualization", parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(
            "This visualization shows how tokens evolve during the diffusion process. "
            "Masked tokens are shown in gray, while predicted tokens are colored by confidence."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Diffusion visualizer
        self.visualizer = DiffusionVisualizer()
        layout.addWidget(self.visualizer)

    def setup_process(self, num_tokens, num_steps):
        """Set up the visualization process."""
        try:
            # Ensure we're working with integers
            num_tokens = int(num_tokens)
            num_steps = int(num_steps)

            # Apply reasonable limits
            num_tokens = min(max(num_tokens, 1), 256)
            num_steps = min(max(num_steps, 1), 128)

            self.visualizer.setup_visualization(num_tokens, num_steps)
        except Exception as e:
            import logging
            logging.error(f"Error in setup_process: {e}")
            # Set up with default values if there's an error
            self.visualizer.setup_visualization(16, 8)

    def update_process(self, step, tokens, mask_indices, confidences=None):
        """Update the visualization for the current step."""
        try:
            # Make sure step is an integer
            step = int(step)

            # Update visualization
            self.visualizer.update_step(step, tokens, mask_indices, confidences)
        except Exception as e:
            import logging
            logging.error(f"Error in update_process: {e}")


if __name__ == "__main__":
    # Test code
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    window = DiffusionProcessVisualizer()
    window.setup_process(10, 5)

    # Example updates
    window.update_process(1,
                          ["the", "cat", "sat", "on", "the", "mat", "and", "was", "happy", "."],
                          [False, True, True, False, False, True, True, True, True, False],
                          [0.9, 0.0, 0.0, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 0.95])

    window.resize(800, 400)
    window.show()

    sys.exit(app.exec())

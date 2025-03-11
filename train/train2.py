import random
import sys
import threading
import time
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QThread, Qt
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QGroupBox,
    QProgressBar, QSpinBox, QDoubleSpinBox, QFileDialog, QTabWidget,
    QFormLayout, QSlider
)
from torch.utils.data import Dataset, DataLoader
import pyqtgraph as pg

# Constants
ASCII_VOCAB_SIZE = 128
MASK_TOKEN_ID = ASCII_VOCAB_SIZE  # Align with embedding size
SEQ_LEN = 512
MAX_PLOT_POINTS = 1000
MOVING_AVG_WINDOW = 50
NUM_DIFFUSION_STEPS = SEQ_LEN

# Noise Schedule (Linear Beta Schedule as in DDPM)
def linear_beta_schedule(num_steps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


def add_gumbel_noise(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Add Gumbel noise to logits for sampling."""
    if temperature == 0:
        return logits
    
    # Sample Gumbel noise
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-9) + 1e-9)
    
    # Apply temperature and add noise
    return logits + gumbel_noise * temperature
# Helper Functions
def process_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tokens, 0, ASCII_VOCAB_SIZE)

def forward_process(input_ids: torch.Tensor, t: torch.Tensor, betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    b, l = input_ids.shape
    alpha_t = torch.gather(1 - betas.cpu(), 0, t.cpu()).to(input_ids.device)  # 1 - beta_t
    p_mask = (1 - alpha_t).sqrt()  # Simplified noise probability
    p_mask = p_mask[:, None].repeat(1, l)
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, torch.tensor(MASK_TOKEN_ID, device=input_ids.device), input_ids)
    return noisy_batch, masked_indices

@torch.no_grad()
def generate0(model: nn.Module, prompt: torch.Tensor, steps: int = 128, gen_length: int = 128,
             block_length: int = 128, temperature: float = 0.0, cfg_scale: float = 0.0,
             remasking: str = 'low_confidence', mask_id: int = MASK_TOKEN_ID) -> torch.Tensor:
    """Generates text iteratively."""
    device = prompt.device
    total_length = prompt.shape[1] + gen_length
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by the number of blocks"
    steps_per_block = steps // num_blocks

    for block in range(num_blocks):
        block_start = prompt.shape[1] + block * block_length
        block_end = prompt.shape[1] + (block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_cat = torch.cat([x, un_x], dim=0)
                logits = model(x_cat)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x)

            logits_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float32), dim=-1)  # float32 for stability
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == 'random':
                x0_p = torch.rand_like(x0, dtype=torch.float32)
            else:
                raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

            if block_end < x.shape[1]:
                x0_p[:, block_end:] = float('-inf')
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, float('-inf'))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x

@torch.no_grad()
def generate(model: nn.Module, prompt: torch.Tensor, gen_length: int = 128, num_steps: int = NUM_DIFFUSION_STEPS,
             temperature: float = 0.7, cfg_scale: float = 0.0) -> torch.Tensor:
    device = prompt.device
    total_length = prompt.shape[1] + gen_length
    x = torch.full((1, total_length), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt
    prompt_index = (x != MASK_TOKEN_ID)

    betas = linear_beta_schedule(num_steps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for step in range(num_steps - 1, -1, -1):
        t = torch.tensor([step], device=device)
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            break

        logits = model(x, t)
        if cfg_scale > 0:
            un_x = x.clone()
            un_x[prompt_index] = MASK_TOKEN_ID
            un_logits = model(un_x, t)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

        probs = F.softmax(logits.to(torch.float32), dim=-1)
        x0 = torch.multinomial(probs.squeeze(0), num_samples=1).squeeze(-1).unsqueeze(0)
        x0 = torch.where(mask_index, x0, x)

        # Simplified DDPM-style update: replace a fraction of tokens
        confidence = torch.max(probs, dim=-1)[0]
        confidence = torch.where(mask_index, confidence, float('-inf'))
        num_replace = max(1, mask_index.sum().item() // (step + 1))
        _, top_indices = torch.topk(confidence.view(-1), k=num_replace)
        mask_update = torch.zeros_like(x, dtype=torch.bool).view(-1)
        mask_update[top_indices] = True
        mask_update = mask_update.view_as(x)
        x = torch.where(mask_update, x0, x)

    return x

# Dataset and Model
class TextDataset(Dataset):
    def __init__(self, text: str, seq_length: int = SEQ_LEN):
        self.seq_length = seq_length
        self.tokens = torch.tensor([b for b in text.encode('ascii', errors='ignore')], dtype=torch.long)

    def __len__(self) -> int:
        return max(1, len(self.tokens) - self.seq_length)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tokens[idx: idx + self.seq_length]

class TextDiffusionModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.time_embedding = nn.Embedding(NUM_DIFFUSION_STEPS, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        input_ids = process_tokens(input_ids)
        x = self.embedding(input_ids)
        if t is not None:
            t_emb = self.time_embedding(t).unsqueeze(1).expand(-1, input_ids.shape[1], -1)
            x += t_emb
        x += self.pos_embedding[:, :input_ids.shape[1], :]
        x = self.transformer_encoder(x)
        return self.linear(x)

# Trainer
class TrainerThread(QThread):
    progress_signal = pyqtSignal(str, float, int, int, str, list)
    epoch_complete_signal = pyqtSignal(int, float)

    def __init__(self, model: nn.Module, dataloader: DataLoader, learning_rate: float = 1e-4,
                 optimizer_name: str = "ADAM", sample_interval: int = 50):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.sample_interval = sample_interval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.running = True
        self.epoch = 0
        self.model_lock = threading.Lock()
        self.speed = 100
        self.base_sleep = 0.01
        self.loss_history = []
        self.betas = linear_beta_schedule(NUM_DIFFUSION_STEPS).to(self.device)
        self.init_optimizer()

    def init_optimizer(self):
        optimizers = {
            "ADAM": torch.optim.Adam,
            "SGD": lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
            "LION": lambda params, lr: torch.optim.Adam(params, lr=lr)  # Fallback
        }
        self.optimizer = optimizers.get(self.optimizer_name, torch.optim.Adam)(self.model.parameters(), lr=self.learning_rate)

    def set_speed(self, speed: int):
        self.speed = speed
        self.running = speed != 0

    def run(self):
        while self.running:
            self.epoch += 1
            epoch_loss = 0.0
            count = 0
            for batch in self.dataloader:
                if not self.running:
                    break
                batch = batch.to(self.device)
                t = torch.randint(0, NUM_DIFFUSION_STEPS, (batch.shape[0],), device=self.device)
                noisy_batch, _ = forward_process(batch, t, self.betas)
                with self.model_lock:
                    logits = self.model(noisy_batch, t)
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), batch.view(-1), reduction='mean')
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                count += 1
                self.loss_history.append(loss.item())
                sample_text = self.generate_sample() if count % self.sample_interval == 0 else ""
                self.progress_signal.emit(
                    f"Epoch {self.epoch} Iter {count} | Loss: {loss.item():.4f}",
                    loss.item(), self.epoch, count, sample_text, self.loss_history
                )
                time.sleep(max(0, self.base_sleep * (100 - self.speed) / 100))
            avg_loss = epoch_loss / count if count > 0 else float('inf')
            self.epoch_complete_signal.emit(self.epoch, avg_loss)

    def stop_training(self):
        self.running = False

    def generate_sample(self) -> str:
        with self.model_lock:
            self.model.eval()
            prompt_tensor = torch.tensor([[ord('A')]], dtype=torch.long, device=self.device)
            out = generate(self.model, prompt_tensor, gen_length=64)
            self.model.train()
            return "".join(chr(t) for t in out[:, prompt_tensor.shape[1]:].squeeze(0).tolist())



# Main Application
class TextDiffusionTrainerApp(QMainWindow):
    DATASETS = {
        "Predefined": ["WikiText-103", "BookCorpus", "OpenWebText", "C4", "ArXiv", "PubMed", "DailyDialog", "PersonaChat"],
        "Custom": ["Custom"]
    }
    DATASET_TEXTS = {
        "WikiText-103": "Wikipedia articles " * 500,
        "BookCorpus": "Once upon a time " * 400,
        "OpenWebText": "Internet content " * 600,
        "C4": "Web crawl data " * 300,
        "ArXiv": "Scientific papers " * 700,
        "PubMed": "Biomedical abstracts " * 450,
        "DailyDialog": "Conversations " * 800,
        "PersonaChat": "Dialogues " * 550
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Diffusion Trainer")
        self.model = TextDiffusionModel(vocab_size=ASCII_VOCAB_SIZE)
        self.trainer_thread = None
        self.dataloader = None
        self.loss_data_buffer = []
        self.displayed_loss_data = []
        self.is_dark_mode = False
        self.selected_category = "Predefined"
        self.selected_dataset = "WikiText-103"
        self.custom_text = ""
        self.initUI()
        self.start_trainer()

    def initUI(self):
        main_layout = QHBoxLayout()
        self.custom_text_edit = QTextEdit(placeholderText="Enter custom text...")
        self.load_file_button = QPushButton("Load from File")

        # Left Panel (Plot and Metrics)
        left_panel = QVBoxLayout()
        self.theme_toggle = QPushButton("Toggle Dark Mode")
        self.theme_toggle.clicked.connect(self.toggle_theme)
        left_panel.addWidget(self.theme_toggle)

        self.model_specs_label = QLabel("Model Specs: Not initialized")
        left_panel.addWidget(self.model_specs_label)

        plot_group = QGroupBox("Training Loss Plot")
        plot_layout = QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Loss', color='black', size='12pt')
        self.plot_widget.setLabel('bottom', 'Iteration', color='black', size='12pt')
        self.plot_widget.setTitle("Training Loss Over Time", color='black', size='14pt')
        self.loss_curve = self.plot_widget.plot(pen=pg.mkPen('r', width=2), name='Raw Loss')
        self.smoothed_curve = self.plot_widget.plot(pen=pg.mkPen('b', width=2), name='Smoothed Loss')
        self.plot_widget.addLegend()
        plot_layout.addWidget(self.plot_widget)
        self.export_plot_button = QPushButton("Export Plot Data")
        self.export_plot_button.clicked.connect(self.export_plot_data)
        plot_layout.addWidget(self.export_plot_button)
        plot_group.setLayout(plot_layout)
        left_panel.addWidget(plot_group)

        self.model_state_label = QLabel()
        self.model_state_label.setFixedSize(128, 128)
        left_panel.addWidget(QLabel("Model State Visualization:"))
        left_panel.addWidget(self.model_state_label)

        # Right Panel (Tabs and Controls)
        right_panel = QVBoxLayout()
        self.tab_widget = QTabWidget()
        train_tab = QWidget()
        generate_tab = QWidget()
        self.tab_widget.addTab(train_tab, "Train")
        self.tab_widget.addTab(generate_tab, "Generate")
        right_panel.addWidget(self.tab_widget)

        # Train Tab
        train_layout = QHBoxLayout()
        train_tab.setLayout(train_layout)

        # Left Side of Train Tab (Logs and Samples)
        train_left = QVBoxLayout()
        self.log_widget = QTextEdit(readOnly=True, font=QFont("Courier New", 10))
        train_left.addWidget(QLabel("Training Log:"))
        train_left.addWidget(self.log_widget)

        self.sample_text_display = QTextEdit(readOnly=True, font=QFont("Courier New", 10))
        train_left.addWidget(QLabel("Sample Text:"))
        train_left.addWidget(self.sample_text_display)
        train_layout.addLayout(train_left)

        # Right Side of Train Tab (Controls)
        train_right = QVBoxLayout()
        self.progress_bar = QProgressBar()
        train_right.addWidget(self.progress_bar)

        self.throttle_slider = QSlider(Qt.Orientation.Horizontal)
        self.throttle_slider.setRange(0, 100)
        self.throttle_slider.setValue(100)
        self.throttle_slider.valueChanged.connect(self.update_training_speed)
        train_right.addWidget(QLabel("Training Speed (0=Stop, 100=Full):"))
        train_right.addWidget(self.throttle_slider)

        settings_group = QGroupBox("Training Settings")
        settings_layout = QFormLayout()
        self.category_combo = QComboBox()
        self.category_combo.addItems(list(self.DATASETS.keys()))
        self.category_combo.currentTextChanged.connect(self.update_dataset_list)
        self.dataset_list = QComboBox()
        self.update_dataset_list(self.selected_category)
        settings_layout.addRow("Category:", self.category_combo)
        settings_layout.addRow("Dataset:", self.dataset_list)

        self.custom_text_edit.hide()
        self.load_file_button.clicked.connect(self.load_custom_text)
        self.load_file_button.hide()
        settings_layout.addRow("Custom Text:", self.custom_text_edit)
        settings_layout.addRow(self.load_file_button)

        self.batch_size_spinbox = QSpinBox(value=4, minimum=1, maximum=128)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["ADAM", "SGD", "LION"])
        self.learning_rate_spinbox = QDoubleSpinBox(value=1e-4, minimum=1e-5, maximum=1e-2, decimals=7, singleStep=1e-5)
        self.sample_interval_spinbox = QSpinBox(value=50, minimum=1, maximum=1000)
        settings_layout.addRow("Batch Size:", self.batch_size_spinbox)
        settings_layout.addRow("Optimizer:", self.optimizer_combo)
        settings_layout.addRow("Learning Rate:", self.learning_rate_spinbox)
        settings_layout.addRow("Sample Interval:", self.sample_interval_spinbox)

        self.apply_settings_button = QPushButton("Apply Settings and Restart")
        self.apply_settings_button.clicked.connect(self.apply_settings)
        settings_layout.addRow(self.apply_settings_button)
        settings_group.setLayout(settings_layout)
        train_right.addWidget(settings_group)

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_trainer)
        train_right.addWidget(self.stop_button)
        train_layout.addLayout(train_right)

        # Generate Tab
        generate_layout = QVBoxLayout()
        generate_tab.setLayout(generate_layout)
        gen_input_layout = QHBoxLayout()
        self.gen_prompt_input = QLineEdit(placeholderText="Enter prompt...")
        self.gen_response_button = QPushButton("Generate")
        self.gen_response_button.clicked.connect(self.generate_response)
        gen_input_layout.addWidget(self.gen_prompt_input)
        gen_input_layout.addWidget(self.gen_response_button)
        generate_layout.addLayout(gen_input_layout)

        self.gen_response_widget = QTextEdit(readOnly=True, font=QFont("Courier New", 10))
        generate_layout.addWidget(QLabel("Generated Text:"))
        generate_layout.addWidget(self.gen_response_widget)

        gen_settings_group = QGroupBox("Generation Settings")
        gen_settings_layout = QFormLayout()
        self.gen_temperature_spinbox = QDoubleSpinBox(value=0.7, minimum=0.0, maximum=2.0, singleStep=0.1)
        self.gen_cfg_scale_spinbox = QDoubleSpinBox(value=0.0, minimum=0.0, maximum=5.0, singleStep=0.1)
        gen_settings_layout.addRow("Temperature:", self.gen_temperature_spinbox)
        gen_settings_layout.addRow("CFG Scale:", self.gen_cfg_scale_spinbox)
        gen_settings_group.setLayout(gen_settings_layout)
        generate_layout.addWidget(gen_settings_group)

        # Combine Panels
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)

        # Styling
        self.light_stylesheet = """
            QMainWindow {background-color: #f0f0f0;}
            QGroupBox {border: 2px solid gray; border-radius: 5px; margin-top: 1ex; font-weight: bold;}
            QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px;}
            QLabel {font-size: 14px; color: black;}
            QPushButton {background-color: #4CAF50; color: white; border: none; padding: 10px; border-radius: 5px;}
            QPushButton:hover {background-color: #367c39;}
        """
        self.dark_stylesheet = """
            QMainWindow {background-color: #2b2b2b;}
            QGroupBox {border: 2px solid #555; border-radius: 5px; margin-top: 1ex; font-weight: bold;}
            QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; color: #ddd;}
            QLabel {font-size: 14px; color: #ddd;}
            QPushButton {background-color: #4CAF50; color: white; border: none; padding: 10px; border-radius: 5px;}
            QPushButton:hover {background-color: #367c39;}
        """
        self.setStyleSheet(self.light_stylesheet)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def get_dataset_text(self, dataset_name: str) -> str:
        return self.custom_text if dataset_name == "Custom" else self.DATASET_TEXTS.get(dataset_name, "Default text " * 1000)

    def start_trainer(self):
        dataset_text = self.get_dataset_text(self.selected_dataset)
        dataset = TextDataset(dataset_text)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size_spinbox.value(), shuffle=True)
        self.model_specs_label.setText(f"Model Specs:\nParameters: {sum(p.numel() for p in self.model.parameters())}")
        self.stop_trainer()
        self.trainer_thread = TrainerThread(
            self.model, self.dataloader, learning_rate=self.learning_rate_spinbox.value(),
            optimizer_name=self.optimizer_combo.currentText(), sample_interval=self.sample_interval_spinbox.value()
        )
        self.trainer_thread.progress_signal.connect(self.update_log)
        self.trainer_thread.epoch_complete_signal.connect(self.update_epoch_complete)
        self.trainer_thread.start()
        self.progress_bar.setRange(0, len(self.dataloader))
        self.progress_bar.setValue(0)
        self.loss_data_buffer.clear()
        self.displayed_loss_data.clear()

    @pyqtSlot()
    def stop_trainer(self):
        if self.trainer_thread and self.trainer_thread.isRunning():
            self.trainer_thread.stop_training()
            self.trainer_thread.wait()

    @pyqtSlot(int)
    def update_training_speed(self, value):
        if self.trainer_thread:
            self.trainer_thread.set_speed(value)

    def update_dataset_list(self, category: str):
        self.selected_category = category
        self.dataset_list.clear()
        self.dataset_list.addItems(self.DATASETS[category])
        self.custom_text_edit.setVisible(category == "Custom")
        self.load_file_button.setVisible(category == "Custom")

    def load_custom_text(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.custom_text = file.read()
                self.custom_text_edit.setText(self.custom_text)
            except Exception as e:
                self.log_widget.append(f"Error loading file: {e}\n")

    def apply_settings(self):
        self.selected_category = self.category_combo.currentText()
        self.selected_dataset = self.dataset_list.currentText()
        if self.selected_category == "Custom":
            self.custom_text = self.custom_text_edit.toPlainText()
            if not self.custom_text.strip():
                self.log_widget.append("Error: Custom text cannot be empty.\n")
                return
        self.start_trainer()

    @pyqtSlot(str, float, int, int, str, list)
    def update_log(self, message: str, loss: float, epoch: int, iteration: int, sample_text: str, loss_history: list):
        self.log_widget.clear()
        self.log_widget.append(message)
        self.progress_bar.setValue(iteration)
        if sample_text:
            self.sample_text_display.setText(f"Sample Text (Epoch {epoch}, Iteration {iteration}):\n{sample_text}")

        self.loss_data_buffer.append(loss)
        if len(self.loss_data_buffer) > MAX_PLOT_POINTS:
            self.loss_data_buffer.pop(0)

        start_idx = max(0, len(self.loss_data_buffer) - MAX_PLOT_POINTS)
        if len(self.loss_data_buffer) >= MOVING_AVG_WINDOW:
            smoothed_data = [
                sum(self.loss_data_buffer[max(0, i - MOVING_AVG_WINDOW + 1):i + 1]) /
                len(self.loss_data_buffer[max(0, i - MOVING_AVG_WINDOW + 1):i + 1])
                for i in range(start_idx, len(self.loss_data_buffer))
            ]
            self.displayed_loss_data = smoothed_data
        else:
            self.displayed_loss_data = self.loss_data_buffer.copy()

        self.loss_curve.setData(list(range(len(self.loss_data_buffer))), self.loss_data_buffer)
        self.smoothed_curve.setData(list(range(len(self.displayed_loss_data))), self.displayed_loss_data)

    @pyqtSlot(int, float)
    def update_epoch_complete(self, epoch: int, avg_loss: float):
        self.log_widget.append(f"*** Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f} ***\n")
        self.progress_bar.setValue(0)
        if self.dataloader:
            self.progress_bar.setRange(0, len(self.dataloader))
        self.update_model_state_visual()

    def update_model_state_visual(self):
        if self.model:
            with torch.no_grad():
                weights = self.model.embedding.weight[:32, :32].cpu().numpy()
                weights = (weights - weights.min()) / (weights.max() - weights.min())
                pixels = (weights * 255).astype(np.uint8)
                image = QImage(pixels.data, 32, 32, 32, QImage.Format.Format_Grayscale8)
                self.model_state_label.setPixmap(QPixmap.fromImage(image).scaled(128, 128))

    @pyqtSlot()
    def generate_response(self):
        prompt_text = self.gen_prompt_input.text().strip()
        if not prompt_text:
            self.gen_response_widget.setText("Response: Please enter a prompt.")
            return

        prompt_tokens = [ord(c) for c in prompt_text if ord(c) < ASCII_VOCAB_SIZE]
        if not prompt_tokens:
            self.gen_response_widget.setText("Response: Prompt contains no valid ASCII characters.")
            return

        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.trainer_thread.device)
        with self.trainer_thread.model_lock:
            self.model.eval()
            out = generate(self.model, prompt_tensor,
                           gen_length=128,
                           temperature=self.gen_temperature_spinbox.value(),
                           cfg_scale=self.gen_cfg_scale_spinbox.value())
            self.model.train()
            response_text = "".join(chr(t) for t in out[:, prompt_tensor.shape[1]:].squeeze(0).tolist())
            self.gen_response_widget.setText(response_text)

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.setStyleSheet(self.dark_stylesheet if self.is_dark_mode else self.light_stylesheet)
        self.plot_widget.setBackground('k' if self.is_dark_mode else 'w')
        color = 'white' if self.is_dark_mode else 'black'
        self.plot_widget.setTitle("Training Loss Over Time", color=color, size='14pt')
        self.plot_widget.setLabel('left', 'Loss', color=color, size='12pt')
        self.plot_widget.setLabel('bottom', 'Iteration', color=color, size='12pt')

    def export_plot_data(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot Data", "", "CSV Files (*.csv)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("Iteration,Raw Loss,Smoothed Loss\n")
                    for i, (raw, smoothed) in enumerate(zip(self.loss_data_buffer, self.displayed_loss_data)):
                        f.write(f"{i},{raw},{smoothed}\n")
                self.log_widget.append(f"Plot data exported to: {file_path}\n")
            except Exception as e:
                self.log_widget.append(f"Error exporting plot data: {e}\n")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TextDiffusionTrainerApp()
    ex.show()
    sys.exit(app.exec())
import random
import sys
import threading
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QThread
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QGroupBox,
    QProgressBar, QSpinBox, QDoubleSpinBox, QFileDialog,
    QTabWidget, QFormLayout,
)
from torch.utils.data import Dataset, DataLoader

# --- Constants and Type Aliases ---
ASCII_VOCAB_SIZE = 128
MASK_TOKEN_ID = 126336
SPECIAL_MASK_INDEX = ASCII_VOCAB_SIZE
SEQ_LEN = 1024

# --- Helper Functions ---

def process_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Remap MASK_TOKEN_ID to the internal SPECIAL_MASK_INDEX."""
    return torch.where(tokens == MASK_TOKEN_ID,
                       torch.tensor(SPECIAL_MASK_INDEX, device=tokens.device),
                       tokens)


class TextDataset(Dataset):
    """Dataset for encoding text to ASCII tokens and slicing into sequences."""
    def __init__(self, text: str, seq_length: int = SEQ_LEN):
        self.seq_length = seq_length
        self.tokens = torch.tensor([b for b in text.encode('ascii', errors='ignore')], dtype=torch.long)

    def __len__(self) -> int:
        return max(1, len(self.tokens) - self.seq_length)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tokens[idx: idx + self.seq_length]


class TextDiffusionModel(nn.Module):
    """Transformer Encoder model for text diffusion."""
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, seq_length: int = SEQ_LEN):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for mask token
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,
                                                   dropout=0.1)  # Added dropout
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, vocab_size)

        # Initialize weights (Xavier/Glorot initialization)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = process_tokens(input_ids)
        x = self.embedding(input_ids) + self.pos_embedding[:, :input_ids.shape[1], :]
        x = self.transformer_encoder(x)
        logits = self.linear(x)
        return logits


def forward_process(input_ids: torch.Tensor, eps: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masks input tokens randomly."""
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices,
                              torch.tensor(MASK_TOKEN_ID, device=input_ids.device),
                              input_ids)
    return noisy_batch, masked_indices, p_mask


def get_dataset_text(dataset_name: str, custom_text: str = "") -> str:
    """Loads dataset text or uses custom text."""
    if dataset_name == "Custom":
        return custom_text
    example_texts = {
        "WikiText-103": "This is an example of WikiText-103 content. " * 500,
        "BookCorpus": "Once upon a time, in a land far, far away... " * 400,
        "OpenWebText": "The internet is a vast and wondrous place... " * 600,
        "C4": "Colossal Clean Crawled Corpus provides diverse text data. " * 300,
        "ArXiv": "Recent advances in quantum computing have shown... " * 700,
        "PubMed": "Studies indicate a correlation between diet and health... " * 450,
        "DailyDialog": "Hello! How are you doing today? I'm fine, thanks! " * 800,
        "PersonaChat": "My favorite hobby is playing the guitar. I love cats! " * 550,
    }
    return example_texts.get(dataset_name, "Error: Dataset not found. Default text. " * 1000)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Adds Gumbel noise to logits for sampling."""
    if temperature == 0:
        return logits
    # Use float32 for numerical stability; convert back at the end if needed.
    logits = logits.to(torch.float32)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits))) * temperature
    return logits + gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Calculates the number of tokens to transition per step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.full((mask_num.size(0), steps),
                                     base.item(), device=mask_index.device, dtype=torch.int64)
    for i in range(mask_num.size(0)):
        if remainder[i].item() > 0:
            num_transfer_tokens[i, :remainder[i].item()] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(model: nn.Module, prompt: torch.Tensor, steps: int = 128, gen_length: int = 128,
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


# --- Trainer and GUI ---
class TrainerThread(QThread):
    """Trainer thread for running the training loop."""
    progress_signal = pyqtSignal(str, float, int, int, str)  # message, loss, epoch, iteration, sample_text
    epoch_complete_signal = pyqtSignal(int, float)

    def __init__(self, model: nn.Module, dataloader: DataLoader,
                 learning_rate: float = 1e-4, sequence_randomization_prob: float = 0.01,
                 sample_interval: int = 50):  # Added sample_interval
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                                          eps=1e-08)  # AdamW params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.running = True
        self.epoch = 0
        self.model_lock = threading.Lock()
        self.lr = learning_rate
        self.sequence_randomization_prob = sequence_randomization_prob
        self.sample_interval = sample_interval  # How often to generate a sample

    def run(self):
        """Main training loop."""
        while self.running:
            self.epoch += 1
            epoch_loss = 0.0
            count = 0
            for batch in self.dataloader:
                batch = batch.to(self.device)
                if random.random() < self.sequence_randomization_prob:
                    random_length = random.randint(1, batch.shape[1])
                    batch = batch[:, :random_length]

                noisy_batch, masked_indices, p_mask = forward_process(batch)
                with self.model_lock:
                    logits = self.model(noisy_batch)
                loss = F.cross_entropy(
                    logits[masked_indices],
                    batch[masked_indices],
                    reduction='none'
                ) / p_mask[masked_indices]
                loss = loss.sum() / (batch.shape[0] * batch.shape[1])
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                self.optimizer.step()

                epoch_loss += loss.item()
                count += 1

                sample_text = ""
                if count % self.sample_interval == 0:  # Generate sample text
                    sample_text = self.generate_sample()

                self.progress_signal.emit(f"Epoch {self.epoch} Iter {count} | Loss: {loss.item():.4f}",
                                          loss.item(), self.epoch, count, sample_text)
                time.sleep(0.01)
                if not self.running:
                    break
            avg_loss = epoch_loss / count if count > 0 else float('inf')
            self.epoch_complete_signal.emit(self.epoch, avg_loss)
            time.sleep(0.1)

    def stop_training(self):
        """Stops the training loop."""
        self.running = False

    def generate_sample(self) -> str:
        """Generates a text sample for display during training."""
        with self.model_lock:
            self.model.eval()
            prompt_tensor = torch.tensor([[ord('A')]], dtype=torch.long, device=self.device)  # Start with 'A'
            out = generate(self.model, prompt_tensor, steps=64, gen_length=64, block_length=16,
                           temperature=0.7, cfg_scale=0.0, remasking='low_confidence', mask_id=MASK_TOKEN_ID)
            self.model.train()
            generated_tokens = out[:, prompt_tensor.shape[1]:].squeeze(0).tolist()
            return "".join(chr(t) for t in generated_tokens)


class TextDiffusionTrainerApp(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Diffusion Trainer - PyQt6 (Enhanced)")

        self.datasets: Dict[str, Dict[str, str]] = {
            "Predefined": {
                "WikiText-103": "High-quality Wikipedia articles.",
                "BookCorpus": "A large corpus of free books.",
                "OpenWebText": "Open web crawl dataset.",
                "C4": "Colossal Clean Crawled Corpus.",
                "ArXiv": "Collection of scientific papers.",
                "PubMed": "Biomedical abstracts.",
                "DailyDialog": "Daily conversations dataset.",
                "PersonaChat": "Persona-based dialogues.",
            },
            "Custom": {  # Custom data loading
                "Custom": "Load custom text from file or input."
            }
        }
        self.selected_category: str = "Predefined"
        self.selected_dataset: str = "WikiText-103"
        self.custom_text: str = ""  # Store custom text
        self.model: Optional[TextDiffusionModel] = None
        self.trainer_thread: Optional[TrainerThread] = None
        self.dataloader: Optional[DataLoader] = None

        # Styling
        self.setStyleSheet("""    
            QMainWindow {background-color: black;}
            QGroupBox {border: 2px solid gray; border-radius: 5px; margin-top: 1ex; font-weight: bold;}
            QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px;}
            QLabel {font-size: 14px;}
            QPushButton {background-color: orange; color: black; border: none; padding: 10px; border-radius: 5px;}
            QPushButton:hover {background-color: yellow;}
            QLineEdit {border: 1px solid gray; border-radius: 4px; padding: 5px;}
            QTextEdit {border: 1px solid gray; border-radius: 4px; padding: 5px;}
            QComboBox {border: 1px solid gray; border-radius: 4px; padding: 5px;}
            QProgressBar {border: 1px solid gray; border-radius: 5px; text-align: center;}
            QProgressBar::chunk {background-color: #05B8CC; width: 10px; margin: 0.5px;}
        """)

        self.initUI()
        self.start_trainer()

    def initUI(self):
        """Initializes the user interface."""
        main_layout = QVBoxLayout()

        # --- Tabs ---
        self.tab_widget = QTabWidget()
        train_tab = QWidget()
        generate_tab = QWidget()
        self.tab_widget.addTab(train_tab, "Train")
        self.tab_widget.addTab(generate_tab, "Generate")
        main_layout.addWidget(self.tab_widget)

        # --- Train Tab Layout ---
        train_layout = QVBoxLayout()
        train_tab.setLayout(train_layout)

        # --- Log Widget ---
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFont(QFont("Courier New", 10))  # Monospaced font
        train_layout.addWidget(QLabel("Training Log:"))
        train_layout.addWidget(self.log_widget)

        # --- Sample Text Display ---
        self.sample_text_label = QLabel("Sample Text (updated periodically):")
        self.sample_text_display = QTextEdit()
        self.sample_text_display.setReadOnly(True)
        self.sample_text_display.setFont(QFont("Courier New", 10))
        train_layout.addWidget(self.sample_text_label)
        train_layout.addWidget(self.sample_text_display)

        # Custom Data Input (Conditional)
        self.custom_text_edit = QTextEdit()
        self.custom_text_edit.setPlaceholderText("Enter custom text here...")
        self.custom_text_edit.hide()  # Initially hidden
        self.load_file_button = QPushButton("Load from File")
        self.load_file_button.clicked.connect(self.load_custom_text)
        self.load_file_button.hide()  # Initially hidden

        settings_layout = QFormLayout()  # Use QFormLayout for better alignment
        settings_layout.addRow("Custom Text:", self.custom_text_edit)
        settings_layout.addRow(self.load_file_button)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        train_layout.addWidget(self.progress_bar)

        # --- Settings Group ---
        settings_group = QGroupBox("Training Settings")

        # Dataset Selection
        self.category_combo = QComboBox()
        self.category_combo.addItems(list(self.datasets.keys()))
        self.category_combo.currentTextChanged.connect(self.update_dataset_list)
        self.dataset_list = QComboBox()
        self.update_dataset_list(self.selected_category)
        settings_layout.addRow("Category:", self.category_combo)  # Use addRow
        settings_layout.addRow("Dataset:", self.dataset_list)

        # Training Parameters
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 128)  # Increased max batch size
        self.batch_size_spinbox.setValue(4)
        self.learning_rate_spinbox = QDoubleSpinBox()
        self.learning_rate_spinbox.setRange(1e-7, 1e-2)
        self.learning_rate_spinbox.setDecimals(7)
        self.learning_rate_spinbox.setValue(1e-4)
        self.learning_rate_spinbox.setSingleStep(1e-5)
        settings_layout.addRow("Batch Size:", self.batch_size_spinbox)
        settings_layout.addRow("Learning Rate:", self.learning_rate_spinbox)
        self.sample_interval_spinbox = QSpinBox()
        self.sample_interval_spinbox.setRange(1, 1000)
        self.sample_interval_spinbox.setValue(50)
        settings_layout.addRow("Sample Interval:", self.sample_interval_spinbox)

        self.apply_settings_button = QPushButton("Apply Settings and Restart")
        self.apply_settings_button.clicked.connect(self.apply_settings)
        settings_layout.addRow(self.apply_settings_button)

        settings_group.setLayout(settings_layout)
        train_layout.addWidget(settings_group)

        # --- Stop Button ---
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_trainer)
        train_layout.addWidget(self.stop_button)

        # --- Generate Tab Layout ---
        generate_layout = QVBoxLayout()
        generate_tab.setLayout(generate_layout)

        # --- Input Bar (for Generation)---
        gen_input_layout = QHBoxLayout()
        self.gen_prompt_input = QLineEdit()
        self.gen_prompt_input.setPlaceholderText("Enter prompt for generation...")
        self.gen_response_button = QPushButton("Generate")
        self.gen_response_button.clicked.connect(self.generate_response)
        gen_input_layout.addWidget(self.gen_prompt_input)
        gen_input_layout.addWidget(self.gen_response_button)
        generate_layout.addLayout(gen_input_layout)

        # --- Response Widget (for Generation) ---
        self.gen_response_widget = QTextEdit()
        self.gen_response_widget.setReadOnly(True)
        self.gen_response_widget.setFont(QFont("Courier New", 10))
        generate_layout.addWidget(QLabel("Generated Text:"))
        generate_layout.addWidget(self.gen_response_widget)

        # --- Generation Settings ---
        gen_settings_group = QGroupBox("Generation Settings")
        gen_settings_layout = QFormLayout()

        self.gen_temperature_spinbox = QDoubleSpinBox()
        self.gen_temperature_spinbox.setRange(0.0, 2.0)
        self.gen_temperature_spinbox.setValue(0.7)  # Default to 0.7
        self.gen_temperature_spinbox.setSingleStep(0.1)
        self.gen_cfg_scale_spinbox = QDoubleSpinBox()
        self.gen_cfg_scale_spinbox.setRange(0.0, 5.0)
        self.gen_cfg_scale_spinbox.setValue(0.0)
        self.gen_cfg_scale_spinbox.setSingleStep(0.1)
        gen_settings_layout.addRow("Temperature:", self.gen_temperature_spinbox)
        gen_settings_layout.addRow("CFG Scale:", self.gen_cfg_scale_spinbox)

        gen_settings_group.setLayout(gen_settings_layout)
        generate_layout.addWidget(gen_settings_group)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def start_trainer(self):
        """Starts the training process."""
        dataset_text = get_dataset_text(self.selected_dataset, self.custom_text)
        dataset = TextDataset(dataset_text)
        batch_size = self.batch_size_spinbox.value()
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)  # Don't drop last
        if self.model is None:  # Only create a new model if one doesn't exist
            self.model = TextDiffusionModel(vocab_size=ASCII_VOCAB_SIZE)
        lr = self.learning_rate_spinbox.value()
        sample_interval = self.sample_interval_spinbox.value()

        self.stop_trainer()

        self.trainer_thread = TrainerThread(self.model, self.dataloader, learning_rate=lr,
                                            sample_interval=sample_interval)
        self.trainer_thread.progress_signal.connect(self.update_log)
        self.trainer_thread.epoch_complete_signal.connect(self.update_epoch_complete)
        self.trainer_thread.start()
        self.log_widget.append(
            f"Training started with dataset: {self.selected_dataset}, Batch Size: {batch_size}, LR: {lr}\n"
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, len(self.dataloader))

    @pyqtSlot()
    def stop_trainer(self):
        """Stops the trainer and waits for the thread to finish."""
        if self.trainer_thread and self.trainer_thread.isRunning():
            self.trainer_thread.stop_training()
            self.trainer_thread.wait()
            self.log_widget.append("Training stopped.\n")

    def update_dataset_list(self, category: str):
        """Updates the dataset list based on the selected category."""
        self.dataset_list.clear()
        self.dataset_list.addItems(list(self.datasets[category].keys()))
        self.selected_category = category  # keep track of the selection
        self.toggle_custom_input()  # show/hide custom input

    def toggle_custom_input(self):
        """Shows/hides the custom text input based on dataset selection."""
        if self.selected_category == "Custom":
            self.custom_text_edit.show()
            self.load_file_button.show()
        else:
            self.custom_text_edit.hide()
            self.load_file_button.hide()

    def load_custom_text(self):
        """Loads custom text from a file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.custom_text = file.read()
                self.custom_text_edit.setText(self.custom_text)
                self.log_widget.append(f"Loaded custom text from: {file_path}\n")
            except Exception as e:
                self.log_widget.append(f"Error loading file: {e}\n")

    def apply_settings(self):
        """Applies the selected settings and restarts the trainer."""
        selected_cat = self.category_combo.currentText()
        selected_ds = self.dataset_list.currentText()
        if selected_cat and selected_ds:
            self.selected_category = selected_cat
            self.selected_dataset = selected_ds
            # Handle custom text
            if self.selected_category == "Custom":
                self.custom_text = self.custom_text_edit.toPlainText()
                if not self.custom_text.strip():
                    self.log_widget.append("Error: Custom text cannot be empty.\n")
                    return
            self.log_widget.append(
                f"Settings applied: Category: {self.selected_category}, Dataset: {self.selected_dataset}\n"
            )
            self.start_trainer()
        else:
            self.log_widget.append("Error: Please select both category and dataset.\n")

    @pyqtSlot(str, float, int, int, str)
    def update_log(self, message: str, loss: float, epoch: int, iteration: int, sample_text: str):
        """Updates the log, progress bar, and sample text display."""
        self.log_widget.append(message)
        self.progress_bar.setValue(iteration)
        if sample_text:
            self.sample_text_display.setText(f"Sample Text (Epoch {epoch}, Iteration {iteration}):\n{sample_text}")

    @pyqtSlot(int, float)
    def update_epoch_complete(self, epoch: int, avg_loss: float):
        """Updates the log with epoch completion."""
        self.log_widget.append(f"*** Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f} ***\n")
        self.progress_bar.setValue(0)
        if self.dataloader:
            self.progress_bar.setRange(0, len(self.dataloader))

    @pyqtSlot()
    def generate_response(self):
        """Generates a response based on the input prompt."""
        prompt_text = self.gen_prompt_input.text().strip()
        if not prompt_text:
            self.gen_response_widget.setText("Response: Please enter a prompt.")
            return

        prompt_tokens = [ord(c) for c in prompt_text if ord(c) < 128]
        if not prompt_tokens:
            self.gen_response_widget.setText("Response: Prompt contains no valid ASCII characters.")
            return

        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.trainer_thread.device)

        with self.trainer_thread.model_lock:
            if self.model:
                self.model.eval()
                out = generate(self.model, prompt_tensor,
                               steps=128, gen_length=128, block_length=32,
                               temperature=self.gen_temperature_spinbox.value(),
                               cfg_scale=self.gen_cfg_scale_spinbox.value(),
                               remasking='low_confidence',
                               mask_id=MASK_TOKEN_ID)
                self.model.train()  # Switch back to training mode
                generated_tokens = out[:, prompt_tensor.shape[1]:].squeeze(0).tolist()
                response_text = "".join(chr(t) for t in generated_tokens)
                self.gen_response_widget.setText(response_text)
            else:
                self.gen_response_widget.setText("Response: Model not initialized yet. Training may be starting.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TextDiffusionTrainerApp()
    ex.show()
    sys.exit(app.exec())

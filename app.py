
import sys
import random
import cv2
import torch
import torch.nn as nn
import timm
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from torchvision import transforms

# Constants
NUM_CLASSES = 3
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['paper', 'rock', 'scissors']  # Adjust based on your training order
MODEL_PATH = 'rps_model.pth'  # Path to the imported weights (.pth file)

# Load the trained model from .pth file
class RPSModel(nn.Module):
    def __init__(self):
        super(RPSModel, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=NUM_CLASSES)

    def forward(self, x):
        return self.model(x)

model = RPSModel()
try:
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it exists.")
    sys.exit(1)
model = model.to(DEVICE)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to predict gesture
def predict_gesture(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(frame_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    return CLASS_NAMES[pred]

# Determine winner
def determine_winner(user_choice, bot_choice):
    if user_choice == bot_choice:
        return "Tie!"
    elif (user_choice == 'rock' and bot_choice == 'scissors') or \
         (user_choice == 'scissors' and bot_choice == 'paper') or \
         (user_choice == 'paper' and bot_choice == 'rock'):
        return "You win!"
    else:
        return "Bot wins!"

# Main App Window
class RPSApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Rock-Paper-Scissors Bot")
        self.setGeometry(100, 100, 640, 480)

        # Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Video label
        self.video_label = QLabel()
        layout.addWidget(self.video_label)

        # Prediction label (used in both modes)
        self.prediction_label = QLabel("Predicted gesture: Waiting...")
        layout.addWidget(self.prediction_label)

        # Game-specific labels (hidden in test mode)
        self.user_label = QLabel("Your choice: Waiting...")
        self.bot_label = QLabel("Bot choice: Waiting...")
        self.result_label = QLabel("Result: Waiting...")
        layout.addWidget(self.user_label)
        layout.addWidget(self.bot_label)
        layout.addWidget(self.result_label)

        # Mode selection buttons
        mode_layout = QHBoxLayout()
        self.game_button = QPushButton("Play Game")
        self.game_button.clicked.connect(lambda: self.set_mode("game"))
        self.test_button = QPushButton("Test Model")
        self.test_button.clicked.connect(lambda: self.set_mode("test"))
        mode_layout.addWidget(self.game_button)
        mode_layout.addWidget(self.test_button)
        layout.addLayout(mode_layout)

        # Start/Stop button
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_camera)
        layout.addWidget(self.start_button)

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_running = False
        self.current_mode = None  # 'game' or 'test'

        # Initial UI state
        self.set_mode("game")  # Start in game mode by default

    def set_mode(self, mode):
        self.current_mode = mode
        if mode == "game":
            self.user_label.setVisible(True)
            self.bot_label.setVisible(True)
            self.result_label.setVisible(True)
            self.prediction_label.setVisible(False)
            self.game_button.setEnabled(False)
            self.test_button.setEnabled(True)
        elif mode == "test":
            self.user_label.setVisible(False)
            self.bot_label.setVisible(False)
            self.result_label.setVisible(False)
            self.prediction_label.setVisible(True)
            self.game_button.setEnabled(True)
            self.test_button.setEnabled(False)

    def toggle_camera(self):
        if not self.is_running:
            if self.current_mode is None:
                return  # No mode selected
            self.is_running = True
            self.timer.start(30)  # ~33 FPS
            self.start_button.setText("Stop")
        else:
            self.is_running = False
            self.timer.stop()
            self.start_button.setText("Start")
            # Reset labels
            self.prediction_label.setText("Predicted gesture: Waiting...")
            self.user_label.setText("Your choice: Waiting...")
            self.bot_label.setText("Bot choice: Waiting...")
            self.result_label.setText("Result: Waiting...")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Predict gesture
            user_choice = predict_gesture(frame)

            if self.current_mode == "game":
                # Bot chooses randomly
                bot_choice = random.choice(CLASS_NAMES)

                # Determine winner
                result = determine_winner(user_choice, bot_choice)

                # Update labels
                self.user_label.setText(f"Your choice: {user_choice}")
                self.bot_label.setText(f"Bot choice: {bot_choice}")
                self.result_label.setText(f"Result: {result}")
            elif self.current_mode == "test":
                # Just show prediction
                self.prediction_label.setText(f"Predicted gesture: {user_choice}")

            # Display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RPSApp()
    window.show()
    sys.exit(app.exec())

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QCheckBox, QGroupBox, QFormLayout, QSpinBox, QMessageBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QCloseEvent, QShortcut
import cv2
import sys
import logging
import json
import os
import time
from src.hand_tracker import HandTracker
from src.gesture_controller import GestureController
from src.voice_controller import VoiceController

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
    return {}

def save_config(config):
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save config: {e}")

class HandTrackingThread(QThread):
    frame_updated = Signal(QImage)
    hand_landmarks_signal = Signal(list)

    def __init__(self, hand_tracker, max_fps=20):
        super().__init__()
        self.hand_tracker = hand_tracker
        self.running = False
        self.max_fps = max_fps

    def run(self):
        self.running = True
        last_time = time.time()
        while self.running:
            try:
                hand_landmarks_list, frame = self.hand_tracker.get_hand_landmarks(return_frame=True)
                if frame is not None:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.frame_updated.emit(qt_image)
                if hand_landmarks_list:
                    self.hand_landmarks_signal.emit(hand_landmarks_list)
                # Limit frame rate
                elapsed = time.time() - last_time
                sleep_time = max(0, 1.0 / self.max_fps - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_time = time.time()
            except Exception as e:
                logging.error(f"HandTrackingThread error: {e}")

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Tracking Gesture Control System")
        self.config = load_config()
        try:
            self.hand_tracker = HandTracker()
        except RuntimeError as e:
            self.show_camera_error(str(e))
            self.hand_tracker = None
        self.gesture_controller = GestureController()
        self.voice_controller = VoiceController()
        self.hand_thread = None
        self.init_ui()
        self.setup_shortcuts()
        self.restore_preferences()

    def init_ui(self):
        # Controls
        self.start_btn = QPushButton("Start Hand Tracking")
        self.stop_btn = QPushButton("Stop Hand Tracking")
        self.voice_toggle = QCheckBox("Enable Voice Control")
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(480, 360)
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(300)
        self.sensitivity_slider.setValue(int(self.gesture_controller.sensitivity * 100))
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        self.sensitivity_label = QLabel(f"Sensitivity: {self.gesture_controller.sensitivity:.2f}")

        # Settings panel for gesture thresholds
        self.settings_group = QGroupBox("Gesture Thresholds")
        settings_layout = QFormLayout()
        self.pinch_spin = QDoubleSpinBox()
        self.pinch_spin.setDecimals(3)
        self.pinch_spin.setRange(0.01, 0.2)
        self.pinch_spin.setValue(self.gesture_controller.pinch_threshold)
        self.pinch_spin.valueChanged.connect(self.update_pinch_threshold)
        self.right_pinch_spin = QDoubleSpinBox()
        self.right_pinch_spin.setDecimals(3)
        self.right_pinch_spin.setRange(0.01, 0.2)
        self.right_pinch_spin.setValue(self.gesture_controller.right_pinch_threshold)
        self.right_pinch_spin.valueChanged.connect(self.update_right_pinch_threshold)
        settings_layout.addRow("Pinch (Left Click):", self.pinch_spin)
        settings_layout.addRow("Pinch (Right Click):", self.right_pinch_spin)
        self.settings_group.setLayout(settings_layout)

        # Layouts
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.voice_toggle)
        controls_layout.addWidget(self.sensitivity_label)
        controls_layout.addWidget(self.sensitivity_slider)

        main_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.preview_label)
        main_layout.addWidget(self.settings_group)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Connect signals
        self.start_btn.clicked.connect(self.start_hand_tracking)
        self.stop_btn.clicked.connect(self.stop_hand_tracking)
        self.voice_toggle.stateChanged.connect(self.toggle_voice_control)

    def setup_shortcuts(self):
        # Ctrl+H: Toggle hand tracking
        self.shortcut_hand = QShortcut(QKeySequence("Ctrl+H"), self)
        self.shortcut_hand.activated.connect(self.toggle_hand_tracking_shortcut)
        # Ctrl+V: Toggle voice control
        self.shortcut_voice = QShortcut(QKeySequence("Ctrl+V"), self)
        self.shortcut_voice.activated.connect(self.toggle_voice_control_shortcut)

    def toggle_hand_tracking_shortcut(self):
        if self.hand_thread and self.hand_thread.isRunning():
            self.stop_hand_tracking()
        else:
            self.start_hand_tracking()

    def toggle_voice_control_shortcut(self):
        if self.voice_toggle.isChecked():
            self.voice_toggle.setChecked(False)
        else:
            self.voice_toggle.setChecked(True)

    def show_camera_error(self, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Camera Access Error")
        msg.setText("Camera could not be opened.\n\n" + message)
        msg.setInformativeText(
            "Please grant camera access to your terminal in System Settings > Privacy & Security > Camera.\n"
            "After granting access, restart your terminal and try again."
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def start_hand_tracking(self):
        if not self.hand_tracker:
            self.show_camera_error("Camera is not available.")
            return
        if self.hand_thread and self.hand_thread.isRunning():
            return
        self.hand_thread = HandTrackingThread(self.hand_tracker)
        self.hand_thread.frame_updated.connect(self.update_preview)
        self.hand_thread.hand_landmarks_signal.connect(self.handle_hand_landmarks)
        self.hand_thread.start()

    def stop_hand_tracking(self):
        if self.hand_thread:
            self.hand_thread.stop()
            self.hand_thread = None

    def update_preview(self, qt_image):
        self.preview_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def handle_hand_landmarks(self, hand_landmarks_list):
        if hand_landmarks_list:
            # Only use the first hand for cursor control
            self.gesture_controller.recognize_gesture(hand_landmarks_list[0])

    def restore_preferences(self):
        # Restore sensitivity and thresholds from config
        try:
            sens = self.config.get('sensitivity', self.gesture_controller.sensitivity)
            self.gesture_controller.sensitivity = sens
            self.sensitivity_slider.setValue(int(sens * 100))
            self.sensitivity_label.setText(f"Sensitivity: {sens:.2f}")
            pinch = self.config.get('pinch_threshold', self.gesture_controller.pinch_threshold)
            self.gesture_controller.pinch_threshold = pinch
            self.pinch_spin.setValue(pinch)
            right_pinch = self.config.get('right_pinch_threshold', self.gesture_controller.right_pinch_threshold)
            self.gesture_controller.right_pinch_threshold = right_pinch
            self.right_pinch_spin.setValue(right_pinch)
        except Exception as e:
            logging.error(f"Failed to restore preferences: {e}")

    def save_preferences(self):
        self.config['sensitivity'] = self.gesture_controller.sensitivity
        self.config['pinch_threshold'] = self.gesture_controller.pinch_threshold
        self.config['right_pinch_threshold'] = self.gesture_controller.right_pinch_threshold
        save_config(self.config)

    def update_sensitivity(self, value):
        self.gesture_controller.sensitivity = value / 100.0
        self.sensitivity_label.setText(f"Sensitivity: {self.gesture_controller.sensitivity:.2f}")
        self.save_preferences()

    def update_pinch_threshold(self, value):
        self.gesture_controller.pinch_threshold = value
        self.save_preferences()

    def update_right_pinch_threshold(self, value):
        self.gesture_controller.right_pinch_threshold = value
        self.save_preferences()

    def toggle_voice_control(self, state):
        if state == Qt.Checked:
            self.voice_controller.listen_and_execute()
        else:
            self.voice_controller.stop()

    def closeEvent(self, event: QCloseEvent):
        if self.hand_thread:
            self.hand_thread.stop()
        self.hand_tracker.release()
        self.voice_controller.stop()
        self.save_preferences()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 
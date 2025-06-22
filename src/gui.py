from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QCheckBox, QGroupBox, QFormLayout, QSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
import cv2
import sys
from src.hand_tracker import HandTracker
from src.gesture_controller import GestureController
from src.voice_controller import VoiceController

class HandTrackingThread(QThread):
    frame_updated = Signal(QImage)
    hand_landmarks_signal = Signal(list)

    def __init__(self, hand_tracker):
        super().__init__()
        self.hand_tracker = hand_tracker
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            hand_landmarks_list, frame = self.hand_tracker.get_hand_landmarks(return_frame=True)
            if frame is not None:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_updated.emit(qt_image)
            if hand_landmarks_list:
                self.hand_landmarks_signal.emit(hand_landmarks_list)

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Tracking Gesture Control System")
        self.hand_tracker = HandTracker()
        self.gesture_controller = GestureController()
        self.voice_controller = VoiceController()
        self.hand_thread = None
        self.init_ui()

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

    def start_hand_tracking(self):
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

    def update_sensitivity(self, value):
        self.gesture_controller.sensitivity = value / 100.0
        self.sensitivity_label.setText(f"Sensitivity: {self.gesture_controller.sensitivity:.2f}")

    def update_pinch_threshold(self, value):
        self.gesture_controller.pinch_threshold = value

    def update_right_pinch_threshold(self, value):
        self.gesture_controller.right_pinch_threshold = value

    def toggle_voice_control(self, state):
        if state == Qt.Checked:
            self.voice_controller.listen_and_execute()
        else:
            self.voice_controller.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 
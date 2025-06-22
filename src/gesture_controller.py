import pyautogui
import numpy as np

class GestureController:
    def __init__(self, smoothing=0.2):
        # Get screen size
        self.screen_width, self.screen_height = pyautogui.size()
        self.smoothing = smoothing
        self.prev_x = None
        self.prev_y = None

    def move_cursor_with_hand(self, hand_landmarks):
        """
        Moves the cursor based on the index finger tip position (landmark 8).
        hand_landmarks: list of (x, y, z) tuples for one hand (normalized coordinates)
        """
        if not hand_landmarks or len(hand_landmarks) < 9:
            return
        # Index finger tip is landmark 8
        x_norm, y_norm, _ = hand_landmarks[8]
        # Convert normalized coordinates to screen coordinates
        x = int(x_norm * self.screen_width)
        y = int(y_norm * self.screen_height)
        # Invert y-axis for screen coordinates if needed
        y = min(max(y, 0), self.screen_height - 1)
        x = min(max(x, 0), self.screen_width - 1)
        # Smoothing
        if self.prev_x is None or self.prev_y is None:
            smooth_x, smooth_y = x, y
        else:
            smooth_x = int(self.prev_x + self.smoothing * (x - self.prev_x))
            smooth_y = int(self.prev_y + self.smoothing * (y - self.prev_y))
        pyautogui.moveTo(smooth_x, smooth_y)
        self.prev_x, self.prev_y = smooth_x, smooth_y

    def reset(self):
        self.prev_x = None
        self.prev_y = None

    def recognize_gesture(self, hand_landmarks):
        # Placeholder for gesture recognition logic
        pass 
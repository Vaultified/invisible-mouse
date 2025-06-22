import pyautogui
import numpy as np
from collections import deque

class GestureController:
    def __init__(
        self,
        smoothing=0.2,
        pinch_threshold=0.04,
        right_pinch_threshold=0.04,
        scroll_threshold=0.03,
        scroll_sensitivity=100,
        sensitivity=1.0,
        dead_zone=0.02,
        edge_boost_factor=2.0,
        smoothing_window=5
    ):
        # Get screen size
        self.screen_width, self.screen_height = pyautogui.size()
        self.smoothing = smoothing
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone
        self.edge_boost_factor = edge_boost_factor
        self.smoothing_window = smoothing_window
        self.prev_x = None
        self.prev_y = None
        self.pinch_threshold = pinch_threshold  # Left click (index pinch)
        self.right_pinch_threshold = right_pinch_threshold  # Right click (middle pinch)
        self.scroll_threshold = scroll_threshold
        self.scroll_sensitivity = scroll_sensitivity
        self.is_pinching = False  # Left click state
        self.is_right_pinching = False  # Right click state
        self.is_dragging = False
        self.prev_scroll_y = None
        self.is_scrolling = False
        self.x_history = deque(maxlen=self.smoothing_window)
        self.y_history = deque(maxlen=self.smoothing_window)

    def move_cursor_with_hand(self, hand_landmarks):
        if not hand_landmarks or len(hand_landmarks) < 9:
            return
        x_norm, y_norm, _ = hand_landmarks[8]
        # Dead zone: ignore small movements
        if self.prev_x is not None and self.prev_y is not None:
            prev_x_norm = self.prev_x / self.screen_width
            prev_y_norm = self.prev_y / self.screen_height
            dx = abs(x_norm - prev_x_norm)
            dy = abs(y_norm - prev_y_norm)
            if dx < self.dead_zone and dy < self.dead_zone:
                return
        # Sensitivity
        x_norm = np.clip(x_norm * self.sensitivity, 0, 1)
        y_norm = np.clip(y_norm * self.sensitivity, 0, 1)
        # Edge boosting
        edge_margin = 0.05
        if x_norm < edge_margin:
            x_norm -= (edge_margin - x_norm) * (self.edge_boost_factor - 1)
            x_norm = max(0, x_norm)
        elif x_norm > 1 - edge_margin:
            x_norm += (x_norm - (1 - edge_margin)) * (self.edge_boost_factor - 1)
            x_norm = min(1, x_norm)
        if y_norm < edge_margin:
            y_norm -= (edge_margin - y_norm) * (self.edge_boost_factor - 1)
            y_norm = max(0, y_norm)
        elif y_norm > 1 - edge_margin:
            y_norm += (y_norm - (1 - edge_margin)) * (self.edge_boost_factor - 1)
            y_norm = min(1, y_norm)
        # Convert normalized coordinates to screen coordinates
        x = int(x_norm * self.screen_width)
        y = int(y_norm * self.screen_height)
        # Moving average smoothing
        self.x_history.append(x)
        self.y_history.append(y)
        smooth_x = int(np.mean(self.x_history))
        smooth_y = int(np.mean(self.y_history))
        pyautogui.moveTo(smooth_x, smooth_y)
        self.prev_x, self.prev_y = smooth_x, smooth_y

    def detect_pinch(self, hand_landmarks, finger_tip_idx=8, threshold=None):
        """
        Detects a pinch gesture (thumb tip and index finger tip close together).
        Returns True if pinch is detected, else False.
        """
        if not hand_landmarks or len(hand_landmarks) <= max(4, finger_tip_idx):
            return False
        thumb_tip = np.array(hand_landmarks[4][:2])
        finger_tip = np.array(hand_landmarks[finger_tip_idx][:2])
        dist = np.linalg.norm(thumb_tip - finger_tip)
        if threshold is None:
            threshold = self.pinch_threshold
        if dist < threshold:
            print(f"Pinch detected! Thumb-{finger_tip_idx} Distance: {dist:.4f}")
            return True
        else:
            print(f"No pinch. Thumb-{finger_tip_idx} Distance: {dist:.4f}")
            return False

    def detect_scroll(self, hand_landmarks):
        # Use index (8) and middle (12) finger tips for two-finger scroll
        if not hand_landmarks or len(hand_landmarks) < 13:
            self.is_scrolling = False
            self.prev_scroll_y = None
            return
        index_tip = hand_landmarks[8][1]
        middle_tip = hand_landmarks[12][1]
        avg_y = (index_tip + middle_tip) / 2
        if self.prev_scroll_y is not None:
            dy = avg_y - self.prev_scroll_y
            if abs(dy) > self.scroll_threshold:
                scroll_amount = int(-dy * self.scroll_sensitivity)
                pyautogui.scroll(scroll_amount)
                print(f"Scroll: {scroll_amount}")
                self.is_scrolling = True
            else:
                self.is_scrolling = False
        self.prev_scroll_y = avg_y

    def process_hand(self, hand_landmarks):
        """
        Moves cursor and handles click based on hand landmarks.
        """
        self.move_cursor_with_hand(hand_landmarks)
        # Left click (index pinch)
        pinching = self.detect_pinch(hand_landmarks, finger_tip_idx=8, threshold=self.pinch_threshold)
        # Right click (middle pinch)
        right_pinching = self.detect_pinch(hand_landmarks, finger_tip_idx=12, threshold=self.right_pinch_threshold)
        # Drag (maintain pinch)
        if pinching and not self.is_pinching:
            pyautogui.mouseDown()
            print("Left drag start!")
            self.is_pinching = True
            self.is_dragging = True
        elif not pinching and self.is_pinching:
            pyautogui.mouseUp()
            print("Left drag end!")
            self.is_pinching = False
            self.is_dragging = False
        # Right click (on pinch, not hold)
        if right_pinching and not self.is_right_pinching:
            pyautogui.rightClick()
            print("Right click!")
            self.is_right_pinching = True
        elif not right_pinching and self.is_right_pinching:
            print("Right pinch released.")
            self.is_right_pinching = False
        # Scroll (two-finger vertical movement)
        self.detect_scroll(hand_landmarks)

    def reset(self):
        self.prev_x = None
        self.prev_y = None
        self.is_pinching = False
        self.is_right_pinching = False
        self.is_dragging = False
        self.prev_scroll_y = None
        self.is_scrolling = False

    def recognize_gesture(self, hand_landmarks):
        # For compatibility; delegates to process_hand
        self.process_hand(hand_landmarks) 
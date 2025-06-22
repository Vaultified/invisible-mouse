import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        self.max_num_hands = max_num_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open camera.")

    def get_hand_landmarks(self, return_frame=False):
        success, frame = self.cap.read()
        if not success:
            print("Error: Failed to capture frame from camera.")
            return ([] if not return_frame else ([], None))
        # Flip the frame for natural interaction
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        hand_landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                normalized_landmarks = [
                    (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
                ]
                hand_landmarks_list.append(normalized_landmarks)
                # Draw landmarks for preview
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        if return_frame:
            return hand_landmarks_list, frame
        return hand_landmarks_list

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows() 
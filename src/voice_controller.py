import speech_recognition as sr
import threading
import pyautogui

class VoiceController:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.listening = False
        self.thread = None
        self.last_text = ""
        try:
            self.microphone = sr.Microphone()
        except OSError as e:
            print(f"Microphone error: {e}")
            self.microphone = None

    def _callback(self, recognizer, audio):
        try:
            text = recognizer.recognize_google(audio)
            print(f"Voice input: {text}")
            self.last_text = text
            self.handle_command(text)
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")

    def listen_and_execute(self):
        if not self.microphone:
            print("No microphone available.")
            return
        if self.listening:
            print("Already listening.")
            return
        self.listening = True
        print("VoiceController: Listening for voice commands...")
        self.thread = threading.Thread(target=self._background_listen)
        self.thread.daemon = True
        self.thread.start()

    def _background_listen(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        stop_listening = self.recognizer.listen_in_background(
            self.microphone, self._callback
        )
        try:
            while self.listening:
                pass  # Keep thread alive
        except KeyboardInterrupt:
            stop_listening(wait_for_stop=False)
            print("VoiceController: Stopped listening.")

    def handle_command(self, text):
        cmd = text.lower().strip()
        if cmd == "enter":
            pyautogui.press("enter")
            print("[Voice] Enter key pressed.")
        elif cmd == "click":
            pyautogui.click()
            print("[Voice] Mouse click.")
        elif cmd == "scroll up":
            pyautogui.scroll(300)
            print("[Voice] Scrolled up.")
        elif cmd == "scroll down":
            pyautogui.scroll(-300)
            print("[Voice] Scrolled down.")
        else:
            # Dictation: type the text
            pyautogui.typewrite(text + " ")
            print(f"[Voice] Dictated: {text}")

    def stop(self):
        self.listening = False
        print("VoiceController: Listening stopped.") 
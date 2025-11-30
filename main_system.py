import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import face_recognition
import pyttsx3
import os
import threading
import datetime
import time
from PIL import Image, ImageTk
import numpy as np
import sys

# Print status to console for debugging
print("--- System Initializing ---")
print("Loading libraries... (This might take a few seconds)")

class FaceAttendanceApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Attendance System v2.0")
        self.window.geometry("1000x650")
        
        # --- Config & Init ---
        self.dataset_path = "dataset"
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        # Initialize TTS Engine
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
        except Exception as e:
            print(f"Warning: Voice engine failed to init: {e}")
            self.engine = None
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_attendance_time = {}
        self.cooldown_seconds = 10  # Seconds between greetings
        self.is_running = False
        self.mode = "IDLE"  # IDLE, RECOGNIZE, REGISTER
        self.cap = None

        # --- UI Setup ---
        self.setup_ui()
        
        # --- Load Data in background to prevent UI freeze ---
        # We perform initial load in main thread to ensure readiness, 
        # but print status so user knows it's working.
        self.load_face_data()

    def setup_ui(self):
        # Left Panel (Controls)
        left_panel = tk.Frame(self.window, width=280, bg="#f0f0f0")
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        # Title
        tk.Label(left_panel, text="Control Panel", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=20)

        # Registration Section
        tk.Label(left_panel, text="--- New User Registration ---", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=5)
        
        tk.Label(left_panel, text="User ID (e.g., 1001):", bg="#f0f0f0").pack(anchor="w", padx=15)
        self.entry_id = tk.Entry(left_panel)
        self.entry_id.pack(fill=tk.X, padx=15)
        
        tk.Label(left_panel, text="Name (English):", bg="#f0f0f0").pack(anchor="w", padx=15)
        self.entry_name = tk.Entry(left_panel)
        self.entry_name.pack(fill=tk.X, padx=15)

        self.btn_register = tk.Button(left_panel, text="📸 Register Face", command=self.start_register_mode, bg="#4CAF50", fg="white", font=("Arial", 10))
        self.btn_register.pack(pady=10, fill=tk.X, padx=15)

        # Separator
        tk.Frame(left_panel, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=5, pady=15)

        # Attendance Section
        tk.Label(left_panel, text="--- Attendance Tracking ---", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=5)
        self.btn_start = tk.Button(left_panel, text="▶ Start Tracking", command=self.start_recognition_mode, bg="#2196F3", fg="white", font=("Arial", 11, "bold"))
        self.btn_start.pack(pady=10, fill=tk.X, padx=15)
        
        self.btn_stop = tk.Button(left_panel, text="⏹ Stop Camera", command=self.stop_camera, bg="#f44336", fg="white", font=("Arial", 10))
        self.btn_stop.pack(pady=5, fill=tk.X, padx=15)

        # Logs Section
        tk.Label(left_panel, text="System Logs:", bg="#f0f0f0").pack(anchor="w", padx=15, pady=(20, 0))
        self.log_list = tk.Listbox(left_panel, height=12, font=("Courier", 9))
        self.log_list.pack(fill=tk.X, padx=15, pady=5)
        
        # Scrollbar for logs
        scrollbar = tk.Scrollbar(self.log_list)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_list.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_list.yview)

        # Right Panel (Video)
        self.right_panel = tk.Frame(self.window, bg="black")
        self.right_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.video_label = tk.Label(self.right_panel, text="Camera Stopped", fg="white", bg="black", font=("Arial", 14))
        self.video_label.pack(expand=True)

    def load_face_data(self):
        """Loads images from dataset folder"""
        self.log("Loading face data...")
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        images = os.listdir(self.dataset_path)
        count = 0
        for image_name in images:
            if image_name.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(self.dataset_path, image_name)
                try:
                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img)
                    if len(encodings) > 0:
                        self.known_face_encodings.append(encodings[0])
                        # Filename format: ID_Name.jpg -> Remove extension
                        name = os.path.splitext(image_name)[0]
                        self.known_face_names.append(name)
                        count += 1
                except Exception as e:
                    print(f"Error loading {image_name}: {e}")
        
        self.log(f"Data loaded: {count} users found.")

    def speak(self, text):
        """Runs TTS in a separate thread"""
        if self.engine is None:
            return
        def run():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except:
                pass
        threading.Thread(target=run, daemon=True).start()

    def get_greeting_text(self, name_info):
        """Generates English greeting based on time"""
        # name_info might be "1001_Steve"
        display_name = name_info.split('_')[-1] if '_' in name_info else name_info
        
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            return f"Good morning, {display_name}."
        elif 12 <= hour < 18:
            return f"Good afternoon, {display_name}."
        else:
            return f"Good evening, {display_name}. Thank you for your hard work."

    def start_camera(self):
        if self.cap is None:
            # Try index 0, then 1 if failed
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                 self.cap = cv2.VideoCapture(1)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open webcam.")
                self.cap = None
                return

            self.is_running = True
            self.update_frame()

    def stop_camera(self):
        self.is_running = False
        self.mode = "IDLE"
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='', text="Camera Stopped", bg="black")
        self.log("Camera stopped.")

    def start_register_mode(self):
        user_id = self.entry_id.get().strip()
        name = self.entry_name.get().strip()
        
        if not user_id or not name:
            messagebox.showwarning("Warning", "Please enter User ID and Name first!")
            return
            
        self.mode = "REGISTER"
        self.log(f"Registration mode: {name}")
        self.start_camera()
        
        # Bind key 'c' or 'C' to capture
        self.window.bind('<c>', self.capture_face)
        self.window.bind('<C>', self.capture_face)
        
        messagebox.showinfo("Instructions", "Look at the camera.\nPress 'C' on your keyboard to capture.")

    def capture_face(self, event=None):
        if self.mode == "REGISTER" and hasattr(self, 'current_frame'):
            user_id = self.entry_id.get().strip()
            name = self.entry_name.get().strip()
            
            # Sanitize filename
            safe_name = "".join([c for c in name if c.isalpha() or c.isdigit()])
            filename = f"{user_id}_{safe_name}.jpg"
            filepath = os.path.join(self.dataset_path, filename)
            
            # Save Image (Convert RGB back to BGR for OpenCV)
            frame_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, frame_bgr)
            
            self.log(f"User saved: {filename}")
            self.speak("Registration successful.")
            messagebox.showinfo("Success", f"User {name} registered successfully!\nReloading data...")
            
            # Reset UI
            self.entry_id.delete(0, tk.END)
            self.entry_name.delete(0, tk.END)
            self.load_face_data()
            self.stop_camera()
            self.window.unbind('<c>')
            self.window.unbind('<C>')

    def start_recognition_mode(self):
        if not self.known_face_encodings:
            messagebox.showwarning("Warning", "No user data found!\nPlease register a user first.")
            return
        self.mode = "RECOGNIZE"
        self.log("Tracking started...")
        self.start_camera()

    def log(self, text):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {text}"
        self.log_list.insert(0, msg)
        print(msg) # Also print to console

    def process_recognition(self, frame_rgb):
        # Resize frame for faster processing (1/4 size)
        small_frame = cv2.resize(frame_rgb, (0, 0), fx=0.25, fy=0.25)
        
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Check for matches
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.45)
            name = "Unknown"
            
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            
            face_names.append(name)
            
            # Voice Greeting Logic
            if name != "Unknown":
                now = time.time()
                # Check Cooldown
                if name not in self.last_attendance_time or (now - self.last_attendance_time[name] > self.cooldown_seconds):
                    self.last_attendance_time[name] = now
                    greeting = self.get_greeting_text(name)
                    self.log(f"Identified: {name}")
                    self.speak(greeting)

        return face_locations, face_names

    def update_frame(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            # Flip horizontally for mirror effect (optional)
            frame = cv2.flip(frame, 1)
            
            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb # Store for capture

            # Draw Logic based on Mode
            if self.mode == "RECOGNIZE":
                locations, names = self.process_recognition(frame_rgb)
                
                for (top, right, bottom, left), name in zip(locations, names):
                    # Scale back up (x4)
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    color = (0, 255, 0) if name != "Unknown" else (255, 0, 0) # Green or Red
                    cv2.rectangle(frame_rgb, (left, top), (right, bottom), color, 2)
                    
                    # Draw background for text
                    cv2.rectangle(frame_rgb, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    display_name = name.split('_')[-1] if '_' in name else name
                    cv2.putText(frame_rgb, display_name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

            elif self.mode == "REGISTER":
                # Draw Helper Text
                cv2.putText(frame_rgb, "Press 'C' to Capture", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                # Draw a guide box
                h, w, _ = frame_rgb.shape
                cv2.rectangle(frame_rgb, (w//2 - 100, h//2 - 120), (w//2 + 100, h//2 + 120), (255, 255, 255), 1)

            # Update GUI Image
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Schedule next frame update
        self.window.after(15, self.update_frame)

if __name__ == "__main__":
    print("Launching GUI...")
    try:
        root = tk.Tk()
        # Attempt to set icon if exists (optional)
        # root.iconbitmap('icon.ico') 
        app = FaceAttendanceApp(root)
        root.mainloop()
    except Exception as e:
        print("CRITICAL ERROR:")
        print(e)
        input("Press Enter to exit...")
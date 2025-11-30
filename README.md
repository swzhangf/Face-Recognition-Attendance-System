# Face Recognition Attendance System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A robust, single-file Desktop Application for **Face Recognition Attendance**. Built with Python, it features a GUI for user registration, real-time face tracking, and time-aware voice greetings (Text-to-Speech).

## 📖 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Theoretical Background & Algorithms](#-theoretical-background--algorithms)
    - [1. Face Detection (HOG)](#1-face-detection-hog)
    - [2. Face Encoding (ResNet & Triplet Loss)](#2-face-encoding-resnet--triplet-loss)
    - [3. Similarity Measurement (Euclidean Distance)](#3-similarity-measurement-euclidean-distance)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 🚀 Features

*   **User Registration**: Simple GUI to input Employee ID and Name. captures face data instantly via webcam.
*   **Real-time Recognition**: Identifies registered users in real-time video streams.
*   **Smart Voice Greeting (TTS)**: 
    *   *05:00 - 12:00*: "Good morning, [Name]."
    *   *12:00 - 18:00*: "Good afternoon, [Name]."
    *   *18:00 - Onwards*: "Good evening, [Name]. Thank you for your hard work."
*   **Anti-Spam Mechanism**: Includes a "Cooldown" timer (default 10s) to prevent repetitive voice announcements for the same user.
*   **Single File Deployment**: The entire logic is encapsulated in `main_system.py` for easy portability.

---

## 🛠 Installation

### Prerequisites
Ensure you have Python 3.8 or higher installed. You also need a working webcam.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies
This project relies on `dlib`, `face_recognition`, `opencv-python`, and `pyttsx3`.

```bash
pip install opencv-python face_recognition pyttsx3 numpy Pillow
```

> **Note**: Installing `face_recognition` (and its dependency `dlib`) might require CMake and Visual Studio C++ compilers on Windows. If installation fails, try installing a pre-compiled dlib wheel or consult the [face_recognition installation guide](https://github.com/ageitgey/face_recognition).

---

## 💻 Usage

1.  **Run the Application**:
    ```bash
    python main_system.py
    ```

2.  **Register a User**:
    *   Enter **User ID** and **Name** in the left panel.
    *   Click **"📸 Register Face"**.
    *   Press **'C'** on your keyboard to capture the photo.

3.  **Start Attendance**:
    *   Click **"▶ Start Tracking"**.
    *   The system will detect faces and greet users audibly.
    *   Logs are displayed in the bottom-left panel.

---

## 🧠 Theoretical Background & Algorithms

This system utilizes a pipeline of computer vision algorithms to achieve high-accuracy recognition.

### 1. Face Detection: HOG
To locate the face in the image, we use the **Histogram of Oriented Gradients (HOG)** method.

*   **Concept**: It analyzes the distribution of intensity gradients or edge directions. The image is divided into small connected regions (cells), and a histogram of gradient directions is compiled for pixels within each cell.
*   **Gradient Calculation**:
    For a pixel $I(x,y)$, the gradients in $x$ and $y$ directions are:
    $$G_x = I(x+1, y) - I(x-1, y)$$
    $$G_y = I(x, y+1) - I(x, y-1)$$
    
    The magnitude $g$ and angle $\theta$ are:
    $$g = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)$$

### 2. Face Encoding: ResNet & Triplet Loss
Once the face is detected and aligned (using 68 facial landmarks), it is passed through a **Deep Convolutional Neural Network (CNN)** based on the ResNet-34 architecture.

*   **Output**: The network maps the face image to a **128-dimensional vector** (embedding).
*   **Training Objective**: The network is trained using **Triplet Loss**. The goal is to ensure that an anchor image ($A$) is closer to a positive image ($P$, same person) than to a negative image ($N$, different person) by a margin $\alpha$.

    **Formula**:
    $$\mathcal{L}(A, P, N) = \max \left( || f(A) - f(P) ||^2 - || f(A) - f(N) ||^2 + \alpha, \quad 0 \right)$$

    Where:
    *   $f(x)$ is the embedding.
    *   $\alpha$ is the margin enforced between positive and negative pairs.

### 3. Similarity Measurement: Euclidean Distance
To recognize a person, the system calculates the **Euclidean Distance** between the 128-d encoding of the live face and the known faces in the database.

*   **Formula**:
    Given two vectors $\vec{p}$ (live face) and $\vec{q}$ (stored face) in 128-dimensional space:
    $$d(\vec{p}, \vec{q}) = \sqrt{\sum_{i=1}^{128} (p_i - q_i)^2}$$

*   **Decision**: 
    If $d(\vec{p}, \vec{q}) < \text{Threshold}$ (default 0.45), the faces are considered a match.

---

## 🏗 System Architecture

The application follows a modular event-driven architecture using `Tkinter`:

1.  **GUI Thread (Main)**: Handles UI rendering and user inputs.
2.  **Video Loop**: Captures frames from OpenCV, converts BGR (OpenCV) to RGB (Pillow), and updates the canvas.
3.  **Voice Thread**: `pyttsx3` operations are offloaded to a daemon thread to prevent the GUI from freezing during speech synthesis.

```python
# Pseudo-code for Threading Logic
def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    thread = threading.Thread(target=run)
    thread.start()
```

---

## 📂 Project Structure

```text
.
├── dataset/             # Stores registered user images (Generated automatically)
│   ├── 1001_John.jpg
│   └── 1002_Jane.jpg
├── main_system.py       # Main Entry Point (GUI + Logic)
└── README.md            # Documentation
```

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
```

---

# Rock, Paper, Scissors ✊✋✌️

A real-time Rock Paper Scissors game that uses your webcam to detect hand gestures and play against a CPU opponent — built with Python, OpenCV, and MediaPipe's hand-tracking model.

## How It Works

1. Your webcam feed is captured and mirrored for a natural experience.
2. MediaPipe's **HandLandmarker** model detects 21 key points on your hand in real time.
3. The number of extended fingers is calculated to determine your move — Rock (0 fingers), Scissors (2 fingers), or Paper (5 fingers).
4. A short **deque-based smoothing** window takes the most frequent detected gesture over the last few frames, reducing flicker and misreads.
5. Once a stable gesture is detected, the CPU randomly picks its move and the winner is calculated instantly.

## Features

- Real-time hand tracking with a visual skeleton overlay
- Orientation-independent gesture detection (works whether your palm or the back of your hand faces the camera)
- Live score tracking for both player and CPU
- Custom font support for on-screen text (falls back to a default font automatically if unavailable)
- Automatic one-time download of the hand-tracking model — no manual setup needed

## Tech Stack

- **Python**
- **OpenCV** — webcam capture and rendering
- **MediaPipe (Tasks API)** — hand landmark detection
- **Pillow (PIL)** — custom font rendering
- **NumPy**

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Kaivalya-n2/Rock-Paper-Scissors.git
   cd Rock-Paper-Scissors
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the game:
   ```bash
   python main.py
   ```

   On first run, the script automatically downloads the hand-tracking model file (`hand_landmarker.task`, ~8 MB). This requires an internet connection just for that first launch.

> **Note:** MediaPipe's hand-tracking models work best with Python 3.9–3.12. If you're on a newer Python version and run into compatibility issues, consider using a supported version via a virtual environment.

## How to Play

- Hold your hand up to the webcam and show:
  - ✊ **Fist** → Rock
  - ✌️ **Two fingers** → Scissors
  - ✋ **Open palm** → Paper
- The CPU will play its move automatically once your gesture is detected.
- Scores update live at the top of the window.
- Press **Esc** to quit.

## Optional: Custom Font

To use the custom "Alumni Sans Medium Italic" font for on-screen text, place `AlumniSans-MediumItalic.ttf` in the project root. If it's not found, the game automatically falls back to OpenCV's default font.

## Author

Built by Kaivalya as a computer vision project exploring real-time gesture recognition.
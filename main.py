
import cv2
import mediapipe as mp
import random
import os
import urllib.request
import time
from collections import deque
import statistics as st
from PIL import ImageFont, ImageDraw, Image
import numpy as np

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


def calculate_winner(cpu_choice, player_choice):
    # Determines the winner of each round when passed the computer's and player's moves
    if player_choice == "Invalid":
        return "Invalid!"

    if player_choice == cpu_choice:
        return "Tie!"

    elif player_choice == "Rock" and cpu_choice == "Scissors":
        return "You win!"

    elif player_choice == "Rock" and cpu_choice == "Paper":
        return "CPU wins!"

    elif player_choice == "Scissors" and cpu_choice == "Rock":
        return "CPU wins!"

    elif player_choice == "Scissors" and cpu_choice == "Paper":
        return "You win!"

    elif player_choice == "Paper" and cpu_choice == "Rock":
        return "You win!"

    elif player_choice == "Paper" and cpu_choice == "Scissors":
        return "CPU wins!"


def compute_fingers(hand_landmarks, count):
    # hand_landmarks here is a list of [id, xPos, yPos, handedness_label]
    # Index Finger
    if hand_landmarks[8][2] < hand_landmarks[6][2]:
        count += 1

    # Middle Finger
    if hand_landmarks[12][2] < hand_landmarks[10][2]:
        count += 1

    # Ring Finger
    if hand_landmarks[16][2] < hand_landmarks[14][2]:
        count += 1

    # Pinky Finger
    if hand_landmarks[20][2] < hand_landmarks[18][2]:
        count += 1

    # Thumb - orientation-independent check (works whether palm or back faces camera)
    # Compares distance from thumb tip to pinky base vs thumb middle-joint to pinky base.
    # If the tip is farther away, the thumb is extended.
    pinky_base = (hand_landmarks[17][1], hand_landmarks[17][2])
    thumb_tip = (hand_landmarks[4][1], hand_landmarks[4][2])
    thumb_ip = (hand_landmarks[3][1], hand_landmarks[3][2])

    dist_tip = ((thumb_tip[0] - pinky_base[0]) ** 2 + (thumb_tip[1] - pinky_base[1]) ** 2) ** 0.5
    dist_ip = ((thumb_ip[0] - pinky_base[0]) ** 2 + (thumb_ip[1] - pinky_base[1]) ** 2) ** 0.5

    if dist_tip > dist_ip:
        count += 1

    return count


def put_text_pil(img, text, position, font, color):
    """Draw text on image using PIL for custom font support"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# Standard 21-point hand landmark connections (used for drawing the skeleton)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_hand_skeleton(image, landmark_points):
    """landmark_points: list of (x_px, y_px) tuples, 21 entries"""
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(image, landmark_points[start_idx], landmark_points[end_idx], (255, 255, 255), 2)
    for point in landmark_points:
        cv2.circle(image, point, 4, (0, 200, 0), -1)


# ---------------------------------------------------------------------------
# Download the hand landmark model file automatically if it's not present
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (one-time, ~8 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
hand_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# Load custom font - adjust path to your font file location
try:
    possible_paths = [
        "AlumniSans-MediumItalic.ttf",
        "static/AlumniSans-MediumItalic.ttf",
        "Alumni_Sans/static/AlumniSans-MediumItalic.ttf",
        os.path.join(os.path.dirname(__file__), "AlumniSans-MediumItalic.ttf"),
        os.path.join(os.path.dirname(__file__), "static", "AlumniSans-MediumItalic.ttf"),
    ]

    font_path = None
    for path in possible_paths:
        if os.path.exists(path):
            font_path = path
            break

    if font_path is None:
        raise FileNotFoundError("Font file not found")

    font_small = ImageFont.truetype(font_path, 30)
    font_medium = ImageFont.truetype(font_path, 40)
    font_large = ImageFont.truetype(font_path, 50)
    print("Custom font (Alumni Sans Medium Italic) loaded successfully!")
except Exception as e:
    print("Custom font not found. Using default OpenCV font.")
    print(f"Error: {e}")
    print("Place 'AlumniSans-MediumItalic.ttf' in the same folder as this script.")
    font_small = None
    font_medium = None
    font_large = None

# Using OpenCV to capture from the webcam
webcam = cv2.VideoCapture(0)

# Define colors (RGB format for PIL)
dark_pink_rgb = (255, 150, 180)
light_blue_rgb = (0, 191, 255)

# Define colors (BGR format for OpenCV)
dark_pink = (180, 150, 255)
light_blue = (255, 191, 0)
yellow = (0, 255, 255)
green = (0, 255, 0)

cpu_choices = ["Rock", "Paper", "Scissors"]
cpu_choice = "Nothing"
cpu_score, player_score = 0, 0
winner_colour = green
winner_colour_rgb = (0, 255, 0)
player_choice = "Nothing"
hand_valid = False
display_values = ["Rock", "Invalid", "Scissors", "Invalid", "Invalid", "Paper"]
winner = "None"
de = deque(['Nothing'] * 5, maxlen=5)

start_time = time.time()

while webcam.isOpened():
    success, image = webcam.read()
    if not success:
        print("Camera isn't working")
        continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    imgH, imgW, imgC = image.shape

    # Run hand detection using the new Tasks API
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    timestamp_ms = int((time.time() - start_time) * 1000)
    detection_result = hand_detector.detect_for_video(mp_image, timestamp_ms)

    hand_landmarks = []
    isCounting = False
    count = 0

    if detection_result.hand_landmarks:
        isCounting = True

        if player_choice in ["Rock", "Paper", "Scissors"] and not hand_valid:
            hand_valid = True
            cpu_choice = random.choice(cpu_choices)
            winner = calculate_winner(cpu_choice, player_choice)

            if winner == "You win!":
                player_score += 1
                winner_colour = dark_pink
                winner_colour_rgb = dark_pink_rgb
                print(f"You win! Score: You {player_score} - CPU {cpu_score}")
            elif winner == "CPU wins!":
                cpu_score += 1
                winner_colour = light_blue
                winner_colour_rgb = light_blue_rgb
                print(f"CPU wins! Score: You {player_score} - CPU {cpu_score}")
            elif winner == "Tie!":
                winner_colour = yellow
                winner_colour_rgb = (255, 255, 0)
                print(f"Tie! Score: You {player_score} - CPU {cpu_score}")
            elif winner == "Invalid!":
                winner_colour = yellow
                winner_colour_rgb = (255, 255, 0)

        for hand_idx, hand in enumerate(detection_result.hand_landmarks):
            label = detection_result.handedness[hand_idx][0].category_name  # "Left" or "Right"

            landmark_points = []
            for id, landmark in enumerate(hand):
                xPos, yPos = int(landmark.x * imgW), int(landmark.y * imgH)
                hand_landmarks.append([id, xPos, yPos, label])
                landmark_points.append((xPos, yPos))

            draw_hand_skeleton(image, landmark_points)
            count = compute_fingers(hand_landmarks, count)
    else:
        hand_valid = False

    if isCounting and count <= 5:
        player_choice = display_values[count]
    elif isCounting and count > 5:
        player_choice = "Invalid"
    else:
        player_choice = "Nothing"

    de.appendleft(player_choice)

    try:
        player_choice = st.mode(de)
    except st.StatisticsError:
        print("Stats Error")
        continue

    if font_small is not None:
        image = put_text_pil(image, "You", (50, 30), font_medium, dark_pink_rgb)
        image = put_text_pil(image, "CPU", (imgW - 120, 30), font_medium, light_blue_rgb)
        image = put_text_pil(image, str(player_score), (80, 80), font_large, dark_pink_rgb)
        image = put_text_pil(image, str(cpu_score), (imgW - 100, 80), font_large, light_blue_rgb)

        if player_choice != "Nothing":
            image = put_text_pil(image, player_choice, (30, 230), font_medium, dark_pink_rgb)

        if cpu_choice != "Nothing":
            image = put_text_pil(image, cpu_choice, (imgW - 200, 230), font_medium, light_blue_rgb)

        if winner != "None":
            text_size = font_large.getbbox(winner)
            text_width = text_size[2] - text_size[0]
            text_x = (imgW - text_width) // 2
            image = put_text_pil(image, winner, (text_x, imgH - 120), font_large, winner_colour_rgb)
    else:
        cv2.putText(image, "You", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, dark_pink, 2)
        cv2.putText(image, "CPU", (imgW - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, light_blue, 2)
        cv2.putText(image, str(player_score), (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, dark_pink, 2)
        cv2.putText(image, str(cpu_score), (imgW - 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, light_blue, 2)

        if player_choice != "Nothing":
            cv2.putText(image, player_choice, (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, dark_pink, 2)

        if cpu_choice != "Nothing":
            cv2.putText(image, cpu_choice, (imgW - 200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, light_blue, 2)

        if winner != "None":
            text_size = cv2.getTextSize(winner, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
            text_x = (imgW - text_size[0]) // 2
            cv2.putText(image, winner, (text_x, imgH - 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, winner_colour, 2)

    cv2.imshow('Rock, Paper, Scissors', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()
hand_detector.close()
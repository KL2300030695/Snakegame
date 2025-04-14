import cv2
import mediapipe as mp
import numpy as np
import random
import math
import pygame
import time

# Initialize Pygame for sound
pygame.init()
try:
    eat_sound = pygame.mixer.Sound('eat.wav')
    gameover_sound = pygame.mixer.Sound('over.wav')
except pygame.error as e:
    print("Sound loading error:", e)
    eat_sound = None
    gameover_sound = None

# MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Game variables
snake_points = []
snake_length = 0
allowed_length = 150
previous_head = None
score = 0

apple_pos = [random.randint(100, 500), random.randint(100, 400)]
apple_size = 20

font = cv2.FONT_HERSHEY_SIMPLEX
game_started = False
paused = False

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def reset_game():
    global snake_points, snake_length, allowed_length, previous_head
    global score, apple_pos, game_started, paused
    snake_points.clear()
    snake_length = 0
    allowed_length = 150
    previous_head = None
    score = 0
    apple_pos = [random.randint(100, 500), random.randint(100, 400)]
    game_started = False
    paused = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    results = hands.process(img_rgb)

    key = cv2.waitKey(1)
    if key == ord('p') or key == ord('P'):
        paused = not paused

    # Start screen
    if not game_started:
        cv2.putText(img, "Show index finger to START", (w // 6, h // 2), font, 1, (255, 255, 255), 2)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                if index_tip.y < middle_tip.y:
                    game_started = True
        cv2.imshow("Snake Game with Hand Tracking", img)
        if key == 27:
            break
        continue

    if paused:
        cv2.putText(img, "PAUSED (Press 'P' to resume)", (w // 5, h // 2), font, 1, (0, 255, 255), 2)
        cv2.imshow("Snake Game with Hand Tracking", img)
        if key == 27:
            break
        continue

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark[8]
            cx, cy = int(lm.x * w), int(lm.y * h)

            if previous_head is None:
                previous_head = (cx, cy)

            snake_points.append((cx, cy))
            snake_length += distance(previous_head, (cx, cy))
            previous_head = (cx, cy)

            while snake_length > allowed_length and len(snake_points) >= 2:
                dist = distance(snake_points[0], snake_points[1])
                snake_length -= dist
                snake_points.pop(0)

            # Draw Snake
            for i in range(1, len(snake_points)):
                cv2.line(img, snake_points[i - 1], snake_points[i], (0, 255, 0), 15)

            # Draw Apple
            cv2.circle(img, tuple(apple_pos), apple_size, (0, 0, 255), -1)

            # Apple collision
            if distance((cx, cy), apple_pos) < apple_size:
                apple_pos = [random.randint(50, w - 50), random.randint(50, h - 50)]
                allowed_length += 30
                score += 1
                if eat_sound:
                    eat_sound.play()

            # Self collision
            if len(snake_points) > 20:
                for i in range(len(snake_points) - 20):
                    if distance(snake_points[i], (cx, cy)) < 15:
                        if gameover_sound:
                            gameover_sound.play()
                        cv2.putText(img, "GAME OVER!", (w // 2 - 150, h // 2), font, 2, (0, 0, 255), 4)
                        cv2.imshow("Snake Game with Hand Tracking", img)
                        cv2.waitKey(2000)
                        reset_game()
                        break

            # Score display
            cv2.putText(img, f'Score: {score}', (10, 40), font, 1, (255, 0, 0), 2)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        previous_head = None

    cv2.imshow("Snake Game with Hand Tracking", img)
    if key == 27:
        break

    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()

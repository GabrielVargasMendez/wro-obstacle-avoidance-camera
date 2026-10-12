# wro-obstacle-avoidance-camera
"""
Leg/Obstacle Detection via Background Subtraction
--------------------------------------------------
Platform:     MATRIX M-Vision Camera (OpenMV-based, STM32H7)
Firmware:     OpenMV v5.0.0 / MicroPython v1.28
Competition:  World Robot Olympiad (WRO) - Future Innovators

Description:
    This script detects moving obstacles (human legs) in front of the robot
    using a background-subtraction technique. It distinguishes between:
      - A single leg/narrow obstacle (position: left or right) -> avoid
      - A wide, solid obstacle (a wall) -> stop
      - No obstacle -> proceed normally

    Detection results are sent over UART to the MATRIX Mini R4 controller
    using the `matrix_mini` library, which then decides how to move the
    robot's motors and servos.

Author: Gabriel Vargas
"""

import gc
import time
import sensor
from matrix_mini import send_data

# ---------------------------------------------------------------------------
# 1. CAMERA SENSOR CONFIGURATION
# ---------------------------------------------------------------------------
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QCIF)       # 176x144 resolution
sensor.skip_frames(time=2000)           # allow sensor to stabilize

sensor.set_vflip(True)
sensor.set_hmirror(True)

# Lock gain/white balance so the stored background stays stable over time
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)

clock = time.clock()
IMG_W = 176
IMG_H = 144
FRAME_CENTER = IMG_W // 2

# ---------------------------------------------------------------------------
# 2. DETECTION PARAMETERS
# ---------------------------------------------------------------------------
DIFF_THRESHOLD = 20                     # sensitivity to background change
MIN_AREA = 250                          # minimum blob area (pixels) to count
ROI_DETECTION = (0, 30, IMG_W, IMG_H - 30)   # ignore top of frame (sky/ceiling)
ROI_AREA = ROI_DETECTION[2] * ROI_DETECTION[3]

# A "wall" is wide and flat (low height/width ratio) and spans a large
# portion of the frame width. A "leg" is narrow and tall (high ratio).
WALL_MIN_WIDTH = int(IMG_W * 0.5)
WALL_MAX_ASPECT = 0.5
LEG_MIN_ASPECT = 0.4

# ---------------------------------------------------------------------------
# 3. AUTOMATIC BACKGROUND REFRESH
# ---------------------------------------------------------------------------
# Lighting conditions drift over time, which can cause false positives.
# The background reference is refreshed periodically when nothing is
# detected, and immediately if a sudden large-scale change suggests a
# lighting shift rather than a real object.
FREE_FRAMES_TO_REFRESH = 60
NOISE_RATIO_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# 4. UART DATA PROTOCOL (sent to MATRIX Mini R4)
# ---------------------------------------------------------------------------
# send_data([code, x_position])
#   code 1 -> leg detected on the LEFT   -> robot should steer left to avoid
#   code 2 -> leg detected on the RIGHT  -> robot should steer right to avoid
#   code 3 -> nothing detected           -> proceed straight
#   code 5 -> wide solid obstacle (wall) -> stop

bg_img = None
free_counter = 0

while True:
    clock.tick()
    gc.collect()

    img = sensor.snapshot()

    # Capture the initial background reference frame
    if bg_img is None:
        bg_img = img.copy()
        print("-> Background reference captured successfully.")
        continue

    diff_img = img.copy()
    diff_img.difference(bg_img)

    blobs = diff_img.find_blobs(
        [(DIFF_THRESHOLD, 255)],
        roi=ROI_DETECTION,
        pixels_threshold=100,
        area_threshold=MIN_AREA,
        merge=True,
    )

    # --- Detect sudden global change (likely a lighting shift, not an object) ---
    total_changed_pixels = sum(b.pixels for b in blobs)
    change_ratio = total_changed_pixels / float(ROI_AREA)

    if change_ratio > NOISE_RATIO_THRESHOLD:
        print(f"Lighting shift detected (ratio={change_ratio:.2f}) -> refreshing background")
        bg_img = img.copy()
        send_data([3, 0])
        print("FPS:", clock.fps())
        continue

    wall_blobs = []
    leg_blobs = []

    for b in blobs:
        x, y, w, h = b.rect
        aspect_ratio = h / float(w)

        if aspect_ratio <= WALL_MAX_ASPECT and w >= WALL_MIN_WIDTH:
            wall_blobs.append((x, y, w, h))
        elif aspect_ratio >= LEG_MIN_ASPECT:
            leg_blobs.append((x, y, w, h))

    if wall_blobs:
        # Wide solid object detected -> treat as a wall, stop
        free_counter = 0
        for (x, y, w, h) in wall_blobs:
            img.draw_rectangle((x, y, w, h), color=(255, 255, 255))
        img.draw_string((5, 5), "WALL", color=(255, 255, 255))
        send_data([5, 0])
        print("State: WALL detected (wide object) -> STOP")

    elif leg_blobs:
        # One or more narrow/tall blobs (could be both legs of the same
        # person) -> treat as a single obstacle, average their X position
        free_counter = 0
        x_centers = [x + (w // 2) for (x, y, w, h) in leg_blobs]
        avg_x = sum(x_centers) // len(x_centers)

        for (x, y, w, h) in leg_blobs:
            img.draw_rectangle((x, y, w, h), color=(255, 255, 255))
        img.draw_cross((avg_x, IMG_H // 2), color=(255, 255, 255))

        position = 1 if avg_x < FRAME_CENTER else 2
        send_data([position, avg_x])
        print(f"Leg(s) DETECTED at avg X: {avg_x} | Position: {position}")

    else:
        free_counter += 1
        send_data([3, 0])
        print("State: Clear (no detection)")

        # Periodically refresh the background when the scene stays clear
        if free_counter >= FREE_FRAMES_TO_REFRESH:
            bg_img = img.copy()
            free_counter = 0
            print("-> Background refreshed (periodic update).")

    img.draw_rectangle(ROI_DETECTION, color=(128, 128, 128))
    print("FPS:", clock.fps())

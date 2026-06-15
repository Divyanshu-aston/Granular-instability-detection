import os
import random
import cv2
import imageio
import tifffile as tiff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

# ==================================================
# USER PARAMETERS
# ==================================================
TIF_PATH = r"C:\Users\250442661\Downloads\Yolo\cropped_boxes\cf7e2d2f-screenshot_037_frame_740_kh_1.tif"

PIXEL_TO_MM = 0.212
MAX_FRAMES = 2200

THRESHOLD = 150
MIN_COMPONENT_AREA = 2

ANGULAR_VELOCITY = 1.60
ROTATION_FACTOR = 0.05001268

SG_WINDOW = 101
SG_POLYORDER = 3

NUM_VIS_FRAMES = 10

# ==================================================
# OUTPUT NAMES
# ==================================================
FILE_LABEL = os.path.basename(TIF_PATH)
INPUT_STEM = os.path.splitext(FILE_LABEL)[0]

CSV_FILENAME = f"amplitude_vs_rotation_normalized_{INPUT_STEM}.csv"
VIS_TIF_FILENAME = f"highest_points_check_{INPUT_STEM}.tif"

# ==================================================
# LOAD TIFF
# ==================================================
stack = tiff.imread(TIF_PATH)

# Handle single-frame TIFF
if stack.ndim == 2:
    stack = stack[np.newaxis, :, :]

# Handle RGB/RGBA TIFF image
elif stack.ndim == 3 and stack.shape[-1] in [3, 4]:
    gray = cv2.cvtColor(stack, cv2.COLOR_RGB2GRAY)
    stack = gray[np.newaxis, :, :]

# Handle RGB/RGBA TIFF stack
elif stack.ndim == 4 and stack.shape[-1] in [3, 4]:
    gray_frames = []
    for frame in stack:
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    stack = np.array(gray_frames)

n_frames, height, width = stack.shape

print(f"Loaded TIFF: {TIF_PATH}")
print(f"Frames: {n_frames}, Height: {height}, Width: {width}")

# ==================================================
# STORAGE
# ==================================================
frame_numbers = []
rotations = []
amplitudes_mm = []
highest_points = []

# ==================================================
# PROCESS EACH FRAME
# ==================================================
for frame_idx in range(min(MAX_FRAMES, n_frames)):

    frame = stack[frame_idx]

    # Convert to 8-bit if needed
    if frame.dtype != np.uint8:
        frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_8bit = frame_norm.astype(np.uint8)
    else:
        frame_8bit = frame

    # Threshold bright pixels
    # _, binary = cv2.threshold(
    #     frame_8bit,
    #     THRESHOLD,
    #     255,
    #     cv2.THRESH_BINARY
    # )

    # binary = binary.astype(np.uint8)

    # # Clean small noise
    # kernel = np.ones((3, 3), np.uint8)
    # # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # # Connected components
    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    #     binary,
    #     connectivity=8
    # )

    # if num_labels <= 1:
    #     continue

    # # Ignore very small components
    # component_areas = stats[1:, cv2.CC_STAT_AREA]
    # valid_component_indices = np.where(component_areas >= MIN_COMPONENT_AREA)[0]

    # if len(valid_component_indices) == 0:
    #     continue

    # largest_component_index = valid_component_indices[
    #     np.argmax(component_areas[valid_component_indices])
    # ]

    # largest_label = largest_component_index + 1
    # main_mask = labels == largest_label

    # ys, xs = np.where(main_mask)

    # if len(xs) == 0:
    #     continue

    # # Highest point = smallest y-row in image coordinates
    # top_row = ys.min()

    # # Use center x of all pixels on the highest row
    # top_xs = xs[ys == top_row]
    # x_top = int(np.mean(top_xs))
    # y_cv = int(top_row)

    # # Convert to bottom-origin coordinate
    # y_bottom = height - y_cv

    # amplitude_mm = y_bottom * PIXEL_TO_MM

    # Threshold bright pixels --- -2nd method
    # _, binary = cv2.threshold(frame_8bit, THRESHOLD, 255, cv2.THRESH_BINARY)
    # binary = binary.astype(np.uint8)

    # # Remove only very tiny noise, but keep thin peaks
    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    #     binary,
    #     connectivity=8
    # )

    # clean_mask = np.zeros_like(binary, dtype=np.uint8)

    # for label in range(1, num_labels):
    #     area = stats[label, cv2.CC_STAT_AREA]

    #     if area >= MIN_COMPONENT_AREA:
    #         clean_mask[labels == label] = 255

    # ys, xs = np.where(clean_mask > 0)

    # if len(xs) == 0:
    #     continue

    # # Find true highest bright row
    # top_row = ys.min()

    # # Use average x of all bright pixels at that highest row
    # top_xs = xs[ys == top_row]
    # x_top = int(np.mean(top_xs))
    # y_cv = int(top_row)

    # # Convert to bottom-origin coordinate
    # y_bottom = height - y_cv
    # amplitude_mm = y_bottom * PIXEL_TO_MM

    # Threshold
    _, binary = cv2.threshold(frame_8bit, THRESHOLD, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        continue

    # Keep the largest component as the main wave, even if it touches the border
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_mask = labels == largest_label

    ys, xs = np.where(main_mask)

    if len(xs) == 0:
        continue

    # Highest point of the main wave only
    top_row = ys.min()
    top_xs = xs[ys == top_row]

    x_top = int(np.mean(top_xs))
    y_cv = int(top_row)

    y_bottom = height - y_cv
    amplitude_mm = y_bottom * PIXEL_TO_MM

    frame_no = frame_idx + 1
    rotation_value = frame_no * ANGULAR_VELOCITY * ROTATION_FACTOR

    frame_numbers.append(frame_no)
    rotations.append(rotation_value)
    amplitudes_mm.append(amplitude_mm)
    highest_points.append((frame_idx, x_top, y_cv, y_bottom))

print(f"Processed valid frames: {len(amplitudes_mm)}")

if len(amplitudes_mm) == 0:
    raise RuntimeError("No valid bright components found. Try lowering THRESHOLD.")

# ==================================================
# SAVE RANDOM VISUAL CHECK TIFF
# ==================================================
num_vis = min(NUM_VIS_FRAMES, len(highest_points))
selected_points = random.sample(highest_points, num_vis)

output_frames = []

for frame_idx, x_top, y_cv, y_bottom in selected_points:

    frame = stack[frame_idx]

    if frame.dtype != np.uint8:
        frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_8bit = frame_norm.astype(np.uint8)
    else:
        frame_8bit = frame.copy()

    frame_bgr = cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2BGR)

    cv2.circle(frame_bgr, (x_top, y_cv), 4, (0, 0, 255), -1)

    # cv2.putText(
    #     frame_bgr,
    #     f"Frame {frame_idx + 1}",
    #     (10, 30),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7,
    #     (0, 255, 0),
    #     2
    # )

    # cv2.putText(
    #     frame_bgr,
    #     f"x={x_top}, y_bottom={y_bottom}",
    #     (10, 60),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7,
    #     (0, 255, 0),
    #     2
    # )

    output_frames.append(frame_bgr)

imageio.mimwrite(VIS_TIF_FILENAME, output_frames)
print(f"Saved visual check TIFF: {VIS_TIF_FILENAME}")

# ==================================================
# CONVERT TO ARRAYS
# ==================================================
frames = np.array(frame_numbers)
rotations = np.array(rotations)
amp = np.array(amplitudes_mm)

# ==================================================
# SMOOTHING
# ==================================================
if len(amp) > SG_POLYORDER + 3:

    sg_window = min(SG_WINDOW, len(amp))

    if sg_window % 2 == 0:
        sg_window -= 1

    if sg_window <= SG_POLYORDER:
        sg_window = SG_POLYORDER + 3

    if sg_window % 2 == 0:
        sg_window += 1

    if sg_window > len(amp):
        sg_window = len(amp) if len(amp) % 2 == 1 else len(amp) - 1

    amp_smooth = savgol_filter(
        amp,
        window_length=sg_window,
        polyorder=SG_POLYORDER
    )

else:
    amp_smooth = amp.copy()

# ==================================================
# NORMALIZE
# ==================================================
amp_min = amp.min()
amp_max = amp.max()

if amp_max == amp_min:
    amp_norm = np.zeros_like(amp)
    amp_smooth_norm = np.zeros_like(amp_smooth)
else:
    amp_norm = (amp - amp_min) / (amp_max - amp_min)
    amp_smooth_norm = (amp_smooth - amp_min) / (amp_max - amp_min)

# Shift first value to zero
amp_norm = amp_norm - amp_norm[0]
amp_smooth_norm = amp_smooth_norm - amp_smooth_norm[0]

amp_norm = np.clip(amp_norm, 0, None)
amp_smooth_norm = np.clip(amp_smooth_norm, 0, None)

# ==================================================
# ISOLATE MAIN PEAK
# ==================================================
y = amp_smooth_norm

peak_idx = np.argmax(y)

left_candidates = np.where(y[:peak_idx] <= 0)[0]
left_idx = left_candidates[-1] + 1 if len(left_candidates) > 0 else 0

right_candidates = np.where(y[peak_idx:] <= 0)[0]
right_idx = peak_idx + right_candidates[0] if len(right_candidates) > 0 else len(y)

frames_peak = frames[left_idx:right_idx]
rotations_peak = rotations[left_idx:right_idx]
amp_peak = amp[left_idx:right_idx]
amp_smooth_peak = amp_smooth[left_idx:right_idx]
amp_norm_peak = amp_norm[left_idx:right_idx]
amp_smooth_norm_peak = amp_smooth_norm[left_idx:right_idx]

# ==================================================
# SAVE CSV
# ==================================================
output_df = pd.DataFrame({
    "Frame": frames_peak,
    "Rotations": rotations_peak,
    "Amplitude_mm_raw": amp_peak,
    "Amplitude_mm_smoothed": amp_smooth_peak,
    "Amplitude_normalized_raw": amp_norm_peak,
    "Amplitude_normalized_smoothed": amp_smooth_norm_peak
})

output_df.to_csv(CSV_FILENAME, index=False)
print(f"CSV saved: {CSV_FILENAME}")

# ==================================================
# PLOT RAW AMPLITUDE
# ==================================================
plt.figure(figsize=(10, 5))
plt.plot(rotations_peak, amp_peak, linewidth=1.5, label="Raw amplitude")
plt.xlabel("Number of Rotations")
plt.ylabel("Amplitude (mm)")
plt.title(f"Raw Amplitude vs Rotation\n{FILE_LABEL}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==================================================
# PLOT NORMALIZED AMPLITUDE
# ==================================================
plt.figure(figsize=(10, 5))
plt.plot(rotations_peak, amp_norm_peak, alpha=0.35, linewidth=1, label="Raw normalized")
plt.plot(rotations_peak, amp_smooth_norm_peak, linewidth=2.5, label="Smoothed normalized")
plt.xlabel("Number of Rotations")
plt.ylabel("Normalized Amplitude")
plt.title(f"Normalized Amplitude vs Rotation\n{FILE_LABEL}")
plt.ylim(bottom=0)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
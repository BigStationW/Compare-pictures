# === Adjustable Text Parameters ===
TITLE_FONT_SCALE_BASE = 0.3        # Adjusts title font size (default 1.0)
FILENAME_FONT_SCALE_BASE = 1.0     # Adjusts filename caption font size (default 1.0)

# === Font Customization Options ===
FONT_FACE = 'SIMPLEX'              # Choose from: SIMPLEX, PLAIN, DUPLEX, COMPLEX, TRIPLEX, COMPLEX_SMALL, SCRIPT_SIMPLEX, SCRIPT_COMPLEX
FONT_BOLD = True                  # Set to True for bold text, False for normal
# ==================================

import cv2
import numpy as np
from moviepy.editor import VideoFileClip, VideoClip
import os
import glob
import math
import re

# Title text for the video
title_text = '"This is a title"'

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def find_media_files():
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    media_files = []

    script_dir = os.path.dirname(os.path.abspath(__file__))

    for ext in video_extensions + image_extensions:
        media_files.extend(glob.glob(os.path.join(script_dir, f'*{ext}')))

    output_file = os.path.join(script_dir, "combined_video.mp4")
    output_image = os.path.join(script_dir, "combined_image.jpg")
    if output_file in media_files:
        media_files.remove(output_file)
    if output_image in media_files:
        media_files.remove(output_image)

    media_files.sort(key=natural_sort_key)
    return media_files

font_map = {
    'SIMPLEX': cv2.FONT_HERSHEY_SIMPLEX,
    'PLAIN': cv2.FONT_HERSHEY_PLAIN,
    'DUPLEX': cv2.FONT_HERSHEY_DUPLEX,
    'COMPLEX': cv2.FONT_HERSHEY_COMPLEX,
    'TRIPLEX': cv2.FONT_HERSHEY_TRIPLEX,
    'COMPLEX_SMALL': cv2.FONT_HERSHEY_COMPLEX_SMALL,
    'SCRIPT_SIMPLEX': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    'SCRIPT_COMPLEX': cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
}
font = font_map.get(FONT_FACE.upper(), cv2.FONT_HERSHEY_SIMPLEX)

media_files = find_media_files()
if len(media_files) < 1:
    print("Error: No media files found.")
    exit(1)

print(f"Found {len(media_files)} media files to combine")

clips = []
display_names = []
clip_types = []
max_width = 0
max_height = 0
image_duration = 5
has_videos = False
max_video_duration = 0

for media_file in media_files:
    ext = os.path.splitext(media_file)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        has_videos = True
        temp_clip = VideoFileClip(media_file)
        max_video_duration = max(max_video_duration, temp_clip.duration)
        temp_clip.close()

if has_videos:
    image_duration = max_video_duration
    print(f"Videos detected. Using max video duration: {max_video_duration:.2f}s")
else:
    print("Only images detected. Will output a single combined image.")

for media_file in media_files:
    ext = os.path.splitext(media_file)[1].lower()
    basename = os.path.splitext(os.path.basename(media_file))[0]
    display_name = re.sub(r'^\s*\d+\.\s*', '', basename).replace('@', ':')

    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        print(f"Loading video: {media_file}")
        clip = VideoFileClip(media_file)
        clip_types.append('video')
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        print(f"Loading image: {media_file}")
        img = cv2.imread(media_file)
        if img is None:
            print(f"Warning: Unable to load image {media_file}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        max_width = max(max_width, width)
        max_height = max(max_height, height)

        clip = VideoClip(lambda t, img=img: img, duration=image_duration)
        clip.size = (width, height)
        clip_types.append('image')
    else:
        continue

    clips.append(clip)
    display_names.append(display_name)

    if hasattr(clip, 'size'):
        width, height = clip.size
        max_width = max(max_width, width)
        max_height = max(max_height, height)

target_width = max_width
target_height = max_height

num_clips = len(clips)
if num_clips <= 3:
    grid_cols = num_clips
    grid_rows = 1
else:
    grid_cols = math.ceil(math.sqrt(num_clips))
    grid_rows = math.ceil(num_clips / grid_cols)

def resize_frame(frame, target_width, target_height):
    h, w = frame.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return result

def add_text_to_frame(frame, text, position, font_scale=1, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return frame

def process_frame(t):
    combined_width = target_width * grid_cols

    title_font_scale = TITLE_FONT_SCALE_BASE * (combined_width / 700)
    title_thickness = 2 if FONT_BOLD else 1
    title_padding = 40

    (text_width, text_height), _ = cv2.getTextSize(title_text, font, title_font_scale, title_thickness)

    margin = 40
    if text_width > combined_width - margin:
        avg_char_width = text_width / len(title_text)
        chars_per_line = int((combined_width - margin) / avg_char_width)
        words = title_text.split()
        lines = []
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + " " + word
            (test_width, _), _ = cv2.getTextSize(test_line, font, title_font_scale, title_thickness)
            if test_width <= combined_width - margin:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

        line_spacing = int(text_height * 1.4)
        title_height = line_spacing * len(lines) + title_padding * 2
        title_bar = np.zeros((title_height, combined_width, 3), dtype=np.uint8)

        for i, line in enumerate(lines):
            (line_width, _), _ = cv2.getTextSize(line, font, title_font_scale, title_thickness)
            line_x = (combined_width - line_width) // 2
            line_y = title_padding + i * line_spacing + text_height
            add_text_to_frame(title_bar, line, (line_x, line_y), font_scale=title_font_scale, thickness=title_thickness)
    else:
        title_height = text_height + title_padding * 2
        title_bar = np.zeros((title_height, combined_width, 3), dtype=np.uint8)
        line_x = (combined_width - text_width) // 2
        line_y = title_padding + text_height
        add_text_to_frame(title_bar, title_text, (line_x, line_y), font_scale=title_font_scale, thickness=title_thickness)

    rows = []
    for row_idx in range(grid_rows):
        cols = []
        row_heights = []

        for col_idx in range(grid_cols):
            clip_idx = row_idx * grid_cols + col_idx
            if clip_idx < num_clips:
                clip = clips[clip_idx]
                clip_type = clip_types[clip_idx]
                display_name = display_names[clip_idx]

                if clip_type == 'image':
                    frame = clip.get_frame(0)
                else:
                    if t < clip.duration:
                        frame = clip.get_frame(min(t, clip.duration - 0.001))
                    else:
                        frame = clip.get_frame(clip.duration - 0.001)

                frame = resize_frame(frame, target_width, target_height)

                caption_font_scale = FILENAME_FONT_SCALE_BASE * (target_width / 800)
                caption_thickness = 2 if FONT_BOLD else 1
                caption_padding = 30

                wrapped_lines = []
                words = display_name.split()
                if words:
                    current_line = words[0]
                    for word in words[1:]:
                        test_line = current_line + ' ' + word
                        (line_width, _), _ = cv2.getTextSize(test_line, font, caption_font_scale, caption_thickness)
                        if line_width <= target_width - 20:
                            current_line = test_line
                        else:
                            wrapped_lines.append(current_line)
                            current_line = word
                    wrapped_lines.append(current_line)

                line_height = int(35 * caption_font_scale)
                text_block_height = len(wrapped_lines) * line_height + caption_padding * 2
                frame_with_text = np.zeros((target_height + text_block_height, target_width, 3), dtype=np.uint8)
                frame_with_text[:target_height, :] = frame

                for i, line in enumerate(wrapped_lines):
                    (line_width, _), _ = cv2.getTextSize(line, font, caption_font_scale, caption_thickness)
                    text_x = int((target_width - line_width) / 2)
                    text_y = target_height + caption_padding + i * line_height + int(line_height * 0.8)
                    add_text_to_frame(frame_with_text, line, (text_x, text_y), font_scale=caption_font_scale, thickness=caption_thickness)

                cols.append(frame_with_text)
                row_heights.append(frame_with_text.shape[0])
            else:
                placeholder = np.zeros((target_height + 80, target_width, 3), dtype=np.uint8)
                cols.append(placeholder)
                row_heights.append(placeholder.shape[0])

        max_row_height = max(row_heights)
        padded_cols = []
        for frame in cols:
            pad_height = max_row_height - frame.shape[0]
            if pad_height > 0:
                padding = np.zeros((pad_height, frame.shape[1], 3), dtype=np.uint8)
                frame = np.vstack([frame, padding])
            padded_cols.append(frame)

        row = np.hstack(padded_cols)
        rows.append(row)

    grid = np.vstack(rows)
    combined = np.vstack([title_bar, grid])
    return combined.astype(np.uint8)

if has_videos:
    output_file = "combined_video.mp4"
    max_duration = max(clip.duration for clip in clips)
    final_clip = VideoClip(make_frame=process_frame, duration=max_duration)
    final_clip.write_videofile(output_file, fps=24)
    print(f"Video saved as: {output_file}")
    final_clip.close()
else:
    output_file = "combined_image.jpg"
    combined_frame = process_frame(0)
    combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file, combined_frame_bgr)
    print(f"Image saved as: {output_file}")

for clip in clips:
    clip.close()

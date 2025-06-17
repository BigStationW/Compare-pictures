import cv2
import numpy as np
from moviepy.editor import VideoFileClip, VideoClip
import os
import glob
import math
import re

# Title text for the video
title_text = "I2V 720x720x81f, 4steps, LCM sampler, FlowMatchSigmas Scheduler, cfg 1"

# Natural sort helper
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Function to find both video and image files
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

# Get media files
media_files = find_media_files()
if len(media_files) < 1:
    print("Error: No media files found.")
    exit(1)

print(f"Found {len(media_files)} media files to combine")

clips = []
display_names = []
clip_types = []  # Track whether each clip is an image or video
max_width = 0
max_height = 0
image_duration = 5  # default duration for static images when no videos present
has_videos = False
max_video_duration = 0

# First pass: determine if we have videos and find max video duration
for media_file in media_files:
    ext = os.path.splitext(media_file)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        has_videos = True
        temp_clip = VideoFileClip(media_file)
        max_video_duration = max(max_video_duration, temp_clip.duration)
        temp_clip.close()

# Set image duration based on whether we have videos
if has_videos:
    image_duration = max_video_duration
    print(f"Videos detected. Using max video duration: {max_video_duration:.2f}s")
else:
    print("Only images detected. Will output a single combined image.")

for media_file in media_files:
    ext = os.path.splitext(media_file)[1].lower()
    basename = os.path.splitext(os.path.basename(media_file))[0]
    display_name = re.sub(r'^\s*\d+\.\s*', '', basename)  # Remove leading numeric prefix

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
    # frame is already in RGB format from MoviePy
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    return frame

def process_frame(t):
    combined_width = target_width * grid_cols
    title_height = 70
    title_bar = np.zeros((title_height, combined_width, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    margin = 40
    (text_width, text_height), _ = cv2.getTextSize(title_text, font, font_scale, thickness)

    if text_width > combined_width - margin:
        avg_char_width = text_width / len(title_text)
        chars_per_line = int((combined_width - margin) / avg_char_width)
        words = title_text.split()
        lines = []
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + " " + word
            (test_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if test_width <= combined_width - margin:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        line_spacing = int(text_height * 1.3)
        new_title_height = line_spacing * len(lines) + 30
        title_bar = np.zeros((new_title_height, combined_width, 3), dtype=np.uint8)
        title_height = new_title_height
        for i, line in enumerate(lines):
            (line_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            line_y = 30 + i * line_spacing
            line_position = (int((combined_width - line_width)/2), line_y)
            add_text_to_frame(title_bar, line, line_position, font_scale=font_scale, thickness=thickness)
    else:
        title_position = (int((combined_width - text_width)/2), int(title_height/2) + int(text_height/2))
        add_text_to_frame(title_bar, title_text, title_position, font_scale=font_scale, thickness=thickness)

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
                
                # Fixed logic for handling images vs videos
                if clip_type == 'image':
                    # For images, always show the same frame regardless of time
                    frame = clip.get_frame(0)
                else:
                    # For videos, handle normally but show last frame if time exceeds duration
                    if t < clip.duration:
                        frame = clip.get_frame(min(t, clip.duration - 0.001))
                    else:
                        # Show the last frame of the video instead of black
                        frame = clip.get_frame(clip.duration - 0.001)
                
                frame = resize_frame(frame, target_width, target_height)

                wrapped_lines = []
                words = display_name.split()
                if words:  # Check if there are any words
                    current_line = words[0]
                    for word in words[1:]:
                        test_line = current_line + ' ' + word
                        (line_width, _), _ = cv2.getTextSize(test_line, font, 1.0, thickness)
                        if line_width <= target_width - 20:
                            current_line = test_line
                        else:
                            wrapped_lines.append(current_line)
                            current_line = word
                    wrapped_lines.append(current_line)

                line_height = 30
                text_block_height = len(wrapped_lines) * line_height + 10 if wrapped_lines else 50
                frame_with_text = np.zeros((target_height + text_block_height, target_width, 3), dtype=np.uint8)
                frame_with_text[:target_height, :] = frame

                for i, line in enumerate(wrapped_lines):
                    (line_width, _), _ = cv2.getTextSize(line, font, 1.0, thickness)
                    text_x = int((target_width - line_width) / 2)
                    text_y = target_height + 30 + i * line_height
                    add_text_to_frame(frame_with_text, line, (text_x, text_y), font_scale=1.0)

                cols.append(frame_with_text)
                row_heights.append(frame_with_text.shape[0])
            else:
                placeholder = np.zeros((target_height + 50, target_width, 3), dtype=np.uint8)
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

# Determine output based on content type
if has_videos:
    # Create video output
    output_file = "combined_video.mp4"
    max_duration = max(clip.duration for clip in clips)
    final_clip = VideoClip(make_frame=process_frame, duration=max_duration)
    final_clip.write_videofile(output_file, fps=24)
    print(f"Video saved as: {output_file}")
    final_clip.close()
else:
    # Create image output
    output_file = "combined_image.jpg"
    combined_frame = process_frame(0)  # Get frame at t=0 since all are images
    # Convert from RGB to BGR for OpenCV
    combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file, combined_frame_bgr)
    print(f"Image saved as: {output_file}")

# Cleanup
for clip in clips:
    clip.close()

# === Adjustable Text Parameters ===
TITLE_FONT_SCALE_BASE = 0.3        # Adjusts title font size (default 1.0)
FILENAME_FONT_SCALE_BASE = 1.5     # Adjusts filename caption font size (default 1.0)

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
title_text = 'This is a title.'

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
        clip = VideoClip(lambda t, img=img: img, duration=image_duration)
        height, width = img.shape[:2]
        clip.size = (width, height)
        clip_types.append('image')
    else:
        continue

    clips.append(clip)
    display_names.append(display_name)

# Calculate actual dimensions for each clip
clip_dimensions = []
for i, clip in enumerate(clips):
    if clip_types[i] == 'image':
        frame = clip.get_frame(0)
    else:
        frame = clip.get_frame(0)
    h, w = frame.shape[:2]
    clip_dimensions.append((w, h))

# Calculate grid dimensions
num_clips = len(clips)
if num_clips <= 3:
    grid_cols = num_clips
    grid_rows = 1
else:
    grid_cols = math.ceil(math.sqrt(num_clips))
    grid_rows = math.ceil(num_clips / grid_cols)

# Calculate row heights and column widths for each grid position
def calculate_grid_dimensions():
    row_heights = []
    col_widths_by_row = []  # Track column widths for each row separately
    
    for row_idx in range(grid_rows):
        max_height_in_row = 0
        col_widths_in_row = []
        
        for col_idx in range(grid_cols):
            clip_idx = row_idx * grid_cols + col_idx
            if clip_idx < len(clips):
                w, h = clip_dimensions[clip_idx]
                max_height_in_row = max(max_height_in_row, h)
                col_widths_in_row.append(w)
            else:
                col_widths_in_row.append(0)
        
        row_heights.append(max_height_in_row)
        col_widths_by_row.append(col_widths_in_row)
    
    return row_heights, col_widths_by_row

row_heights, col_widths_by_row = calculate_grid_dimensions()

def wrap_text(text, max_width, font_scale, thickness):
    """Wrap text to fit within specified width"""
    words = text.split()
    if not words:
        return []
    
    lines = []
    current_line = words[0]
    
    for word in words[1:]:
        test_line = current_line + ' ' + word
        (line_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if line_width <= max_width - 20:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

def calculate_max_caption_height_for_row(row_idx, grid_cols, display_names, row_height):
    """Calculate the maximum caption height needed for any item in this row"""
    caption_font_scale = FILENAME_FONT_SCALE_BASE * (row_height / 800)
    caption_thickness = 2 if FONT_BOLD else 1
    caption_padding = 30
    line_height = int(35 * caption_font_scale)
    
    max_lines = 0
    
    for col_idx in range(grid_cols):
        clip_idx = row_idx * grid_cols + col_idx
        if clip_idx < len(display_names):
            display_name = display_names[clip_idx]
            # Use original image width for text wrapping
            if clip_idx < len(clip_dimensions):
                img_width = clip_dimensions[clip_idx][0]
                wrapped_lines = wrap_text(display_name, img_width, caption_font_scale, caption_thickness)
                max_lines = max(max_lines, len(wrapped_lines))
    
    return max_lines * line_height + caption_padding * 2

def add_text_to_frame(frame, text, position, font_scale=1, color=(255, 255, 255), thickness=2):
    """Add text to frame"""
    cv2.putText(frame, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return frame

def resize_and_center_frame(frame, target_height):
    """Resize frame to target height while maintaining aspect ratio"""
    h, w = frame.shape[:2]
    
    # Calculate new width based on target height
    scale = target_height / h
    new_width = int(w * scale)
    new_height = target_height
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    return resized_frame

def create_frame_with_caption(frame, display_name, target_height, caption_height):
    """Create a frame with properly aligned caption using uniform height"""
    # Resize frame to target height
    resized_frame = resize_and_center_frame(frame, target_height)
    frame_width = resized_frame.shape[1]
    
    caption_font_scale = FILENAME_FONT_SCALE_BASE * (target_height / 800)
    caption_thickness = 2 if FONT_BOLD else 1
    caption_padding = 30
    line_height = int(35 * caption_font_scale)
    
    # Create the final frame with caption area
    frame_with_text = np.zeros((target_height + caption_height, frame_width, 3), dtype=np.uint8)
    frame_with_text[:target_height, :] = resized_frame
    
    # Add caption text
    wrapped_lines = wrap_text(display_name, frame_width, caption_font_scale, caption_thickness)
    
    # Center the text block vertically within the caption area
    total_text_height = len(wrapped_lines) * line_height
    text_start_y = target_height + (caption_height - total_text_height) // 2
    
    for i, line in enumerate(wrapped_lines):
        (line_width, _), _ = cv2.getTextSize(line, font, caption_font_scale, caption_thickness)
        text_x = (frame_width - line_width) // 2
        text_y = text_start_y + i * line_height + int(line_height * 0.8)
        add_text_to_frame(frame_with_text, line, (text_x, text_y), 
                         font_scale=caption_font_scale, thickness=caption_thickness)
    
    return frame_with_text

def process_frame(t):
    # Calculate target width for proportional scaling
    def calculate_proportional_widths(row_frames_info, target_width, uniform_height):
        """Calculate proportional widths so all frames in a row sum to target_width"""
        if not row_frames_info:
            return []
        
        # Calculate what each frame's width would be at uniform height
        natural_widths = []
        for clip_idx in row_frames_info:
            original_w, original_h = clip_dimensions[clip_idx]
            # Width when scaled to uniform height
            scaled_width = int(original_w * (uniform_height / original_h))
            natural_widths.append(scaled_width)
        
        # Scale all widths proportionally to fit target width
        total_natural_width = sum(natural_widths)
        if total_natural_width == 0:
            return [target_width // len(natural_widths)] * len(natural_widths)
        
        scale_factor = target_width / total_natural_width
        proportional_widths = [int(w * scale_factor) for w in natural_widths]
        
        # Adjust for rounding errors - make sure total equals target_width
        width_diff = target_width - sum(proportional_widths)
        if width_diff != 0:
            proportional_widths[-1] += width_diff
        
        return proportional_widths

    # First pass: determine the maximum natural row width
    max_natural_width = 0
    row_info = []
    
    for row_idx in range(grid_rows):
        uniform_height = row_heights[row_idx]
        row_clip_indices = []
        row_natural_width = 0
        
        for col_idx in range(grid_cols):
            clip_idx = row_idx * grid_cols + col_idx
            if clip_idx < num_clips:
                row_clip_indices.append(clip_idx)
                original_w, original_h = clip_dimensions[clip_idx]
                scaled_width = int(original_w * (uniform_height / original_h))
                row_natural_width += scaled_width
        
        row_info.append((row_clip_indices, row_natural_width))
        max_natural_width = max(max_natural_width, row_natural_width)
    
    # Use max natural width as target width for all rows
    target_width = max_natural_width
    
    # Second pass: create frames with proportional scaling
    rows = []
    
    for row_idx in range(grid_rows):
        uniform_height = row_heights[row_idx]
        row_caption_height = calculate_max_caption_height_for_row(row_idx, grid_cols, display_names, uniform_height)
        row_clip_indices, _ = row_info[row_idx]
        
        if not row_clip_indices:
            continue
        
        # Calculate proportional widths for this row
        proportional_widths = calculate_proportional_widths(row_clip_indices, target_width, uniform_height)
        
        row_frames = []
        
        for i, clip_idx in enumerate(row_clip_indices):
            target_frame_width = proportional_widths[i]
            clip = clips[clip_idx]
            clip_type = clip_types[clip_idx]
            display_name = display_names[clip_idx]

            # Get frame data
            if clip_type == 'image':
                frame = clip.get_frame(0)
            else:
                if t < clip.duration:
                    frame = clip.get_frame(min(t, clip.duration - 0.001))
                else:
                    frame = clip.get_frame(clip.duration - 0.001)

            # Resize frame to exact target width and uniform height
            resized_frame = cv2.resize(frame, (target_frame_width, uniform_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Create frame with caption
            frame_with_text = np.zeros((uniform_height + row_caption_height, target_frame_width, 3), dtype=np.uint8)
            frame_with_text[:uniform_height, :] = resized_frame
            
            # Add caption text
            caption_font_scale = FILENAME_FONT_SCALE_BASE * (uniform_height / 800)
            caption_thickness = 2 if FONT_BOLD else 1
            line_height = int(35 * caption_font_scale)
            
            wrapped_lines = wrap_text(display_name, target_frame_width, caption_font_scale, caption_thickness)
            
            # Center the text block vertically within the caption area
            total_text_height = len(wrapped_lines) * line_height
            text_start_y = uniform_height + (row_caption_height - total_text_height) // 2
            
            for j, line in enumerate(wrapped_lines):
                (line_width, _), _ = cv2.getTextSize(line, font, caption_font_scale, caption_thickness)
                text_x = (target_frame_width - line_width) // 2
                text_y = text_start_y + j * line_height + int(line_height * 0.8)
                add_text_to_frame(frame_with_text, line, (text_x, text_y), 
                                 font_scale=caption_font_scale, thickness=caption_thickness)
            
            row_frames.append(frame_with_text)
        
        # Combine frames in this row
        if row_frames:
            row = np.hstack(row_frames)
            rows.append(row)

    # Use the target width for title
    combined_width = target_width

    # Create title bar (rest of the title code remains the same)
    title_font_scale = TITLE_FONT_SCALE_BASE * (combined_width / 700)
    title_thickness = 2 if FONT_BOLD else 1
    title_padding = 40

    (text_width, text_height), _ = cv2.getTextSize(title_text, font, title_font_scale, title_thickness)

    margin = 40
    if text_width > combined_width - margin:
        # Wrap title text
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
            add_text_to_frame(title_bar, line, (line_x, line_y), 
                             font_scale=title_font_scale, thickness=title_thickness)
    else:
        title_height = text_height + title_padding * 2
        title_bar = np.zeros((title_height, combined_width, 3), dtype=np.uint8)
        line_x = (combined_width - text_width) // 2
        line_y = title_padding + text_height
        add_text_to_frame(title_bar, title_text, (line_x, line_y), 
                         font_scale=title_font_scale, thickness=title_thickness)

    # Combine all rows
    if rows:
        grid = np.vstack(rows)
        combined = np.vstack([title_bar, grid])
    else:
        combined = title_bar
    
    return combined.astype(np.uint8)

# Generate output
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

# Clean up
for clip in clips:
    clip.close()

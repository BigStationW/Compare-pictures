import cv2
import numpy as np
import os
import glob
import math
import re

# Title text for the composite image - easily modifiable
title_text = "Same settings"

# Function to automatically find image files in the script's directory
def find_image_files():
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif']
    image_files = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir == '':
        script_dir = '.'

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(script_dir, f'*{ext}')))

    output_file = os.path.join(script_dir, "combined_image.jpg")
    if output_file in image_files:
        image_files.remove(output_file)

    return image_files

# Get image files from the script directory
image_files = find_image_files()

if len(image_files) < 1:
    print("Error: No image files found.")
    exit(1)

output_file = "combined_image.jpg"
print(f"Found {len(image_files)} images to combine")

# Load all images
images = []
display_names = []
max_width = 0
max_height = 0

for image_file in image_files:
    print(f"Loading: {image_file}")
    img = cv2.imread(image_file)
    if img is None:
        print(f"Warning: Could not load {image_file}, skipping.")
        continue

    images.append(img)

    height, width = img.shape[:2]
    max_width = max(max_width, width)
    max_height = max(max_height, height)

    base_name = os.path.splitext(os.path.basename(image_file))[0].replace('@', ':')
    display_name = re.sub(r'^\d+\.\s*', '', base_name)
    display_names.append(display_name)

target_width = max_width
target_height = max_height
num_images = len(images)

# Grid layout
if num_images <= 3:
    grid_cols = num_images
    grid_rows = 1
else:
    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)

# Resize image function
def resize_image(img, target_width, target_height):
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return result

# Add text to image with wrapping
def add_text_to_image(img, text, position, font_scale=1, color=(255, 255, 255), thickness=2, max_width=None):
    font = cv2.FONT_HERSHEY_SIMPLEX

    if max_width is not None:
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            (w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if w <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        y = position[1]
        for line in lines:
            (line_width, line_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            x = position[0] + (max_width - line_width) // 2
            cv2.putText(img, line, (x, y), font, font_scale, color, thickness)
            y += int(line_height * 1.3)
    else:
        cv2.putText(img, text, position, font, font_scale, color, thickness)

    return img

# Create combined width
combined_width = target_width * grid_cols

# Title bar handling
include_title = bool(title_text.strip())
title_bar = None
title_height = 0

if include_title:
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
            line_position = (int((combined_width - line_width) / 2), line_y)
            add_text_to_image(title_bar, line, line_position, font_scale=font_scale, thickness=thickness)
    else:
        title_position = (int((combined_width - text_width) / 2), int(title_height / 2) + int(text_height / 2))
        add_text_to_image(title_bar, title_text, title_position, font_scale=font_scale, thickness=thickness)

# Build grid
label_area_height = 80  # extra space for wrapped text
rows = []

for row_idx in range(grid_rows):
    cols = []
    for col_idx in range(grid_cols):
        img_idx = row_idx * grid_cols + col_idx
        if img_idx < num_images:
            img = images[img_idx]
            display_name = display_names[img_idx]
            img = resize_image(img, target_width, target_height)

            img_with_text = np.zeros((target_height + label_area_height, target_width, 3), dtype=np.uint8)
            img_with_text[:target_height, :] = img

            # Prepare for vertical centering
            font_scale = 1.0
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            max_label_width = target_width - 20

            # Word-wrap the text first
            words = display_name.split()
            lines = []
            current_line = ""

            for word in words:
                test_line = f"{current_line} {word}".strip()
                (w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                if w <= max_label_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

            # Estimate total text block height
            (_, line_height), _ = cv2.getTextSize("Test", font, font_scale, thickness)
            text_block_height = int(line_height * len(lines) * 1.3)

            # Calculate starting y-position for vertical centering
            start_y = target_height + (label_area_height - text_block_height) // 2 + line_height

            # Draw each line
            y = start_y
            for line in lines:
                (line_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
                x = (target_width - line_width) // 2
                cv2.putText(img_with_text, line, (x, y), font, font_scale, (255, 255, 255), thickness)
                y += int(line_height * 1.3)

            cols.append(img_with_text)
        else:
            empty_frame = np.zeros((target_height + label_area_height, target_width, 3), dtype=np.uint8)
            cols.append(empty_frame)
    row = np.hstack(cols)
    rows.append(row)

grid = np.vstack(rows)

# Final combine
if include_title:
    combined = np.vstack([title_bar, grid])
else:
    combined = grid

cv2.imwrite(output_file, combined)
print(f"Combined image saved as {output_file}")

import cv2
import numpy as np

# Resize the image
def resize_image(image, scale_percent=75):
    """Resize the given image by a scale percentage."""
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Preprocess images to extract road-like structures
def preprocess_image(image):
    """Enhance the image to detect road structures."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Smooth image
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection
    dilated = cv2.dilate(edges, None, iterations=2)  # Enhance roads
    return dilated

# Detect lines using Hough Line Transform
def find_lines(image):
    """Find line segments in the preprocessed image."""
    return cv2.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

# Load images
map_img = cv2.imread('camurlim.jpg', cv2.IMREAD_GRAYSCALE)  # Map image
template = cv2.imread('camurlim.jpg', cv2.IMREAD_GRAYSCALE)  # Template image

if map_img is None or template is None:
    print("Error: One or both images could not be loaded.")
    exit()

# Resize images
map_img = resize_image(map_img)
template = resize_image(template)

# Preprocess the images to emphasize roads
processed_map = preprocess_image(map_img)
processed_template = preprocess_image(template)

# Detect lines in the preprocessed images
lines_map = find_lines(processed_map)
lines_template = find_lines(processed_template)

# Function to draw lines on images for visualization
def draw_lines(image, lines, color=(0, 255, 0)):
    """Draw detected lines on the image."""
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), color, 2)

# Create copies for visualization
map_with_lines = map_img.copy()
template_with_lines = template.copy()

# Draw the detected lines on the images
draw_lines(map_with_lines, lines_map)
draw_lines(template_with_lines, lines_template)

# Match lines (simple geometric matching by angle and distance)
def match_lines(lines1, lines2, angle_threshold=10, distance_threshold=50):
    """Match lines between two sets based on angle and distance."""
    matches = []
    for line1 in lines1:
        x1, y1, x2, y2 = line1[0]
        angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        for line2 in lines2:
            x3, y3, x4, y4 = line2[0]
            angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi
            # Check angle similarity
            if abs(angle1 - angle2) < angle_threshold:
                # Check distance similarity
                dist = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
                if dist < distance_threshold:
                    matches.append((line1, line2))
    return matches

# Match lines from map and template
matched_lines = match_lines(lines_map, lines_template)

# Draw matched lines
matched_image = np.hstack((map_with_lines, template_with_lines))
offset = map_with_lines.shape[1]  # Offset for the template in the matched image

for line1, line2 in matched_lines:
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    # Draw lines with connecting matches
    cv2.line(matched_image, (x1, y1), (x3 + offset, y3), (0, 0, 255), 2)
    cv2.line(matched_image, (x2, y2), (x4 + offset, y4), (0, 255, 255), 2)

# Save the images with results
cv2.imwrite("processed_map.jpg", map_with_lines)
cv2.imwrite("processed_template.jpg", template_with_lines)
cv2.imwrite("matched_lines.jpg", matched_image)

# Optionally display results
# cv2.imshow("Map with Roads", map_with_lines)
# cv2.imshow("Template with Roads", template_with_lines)
# cv2.imshow("Matched Roads", matched_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

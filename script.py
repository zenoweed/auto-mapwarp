import cv2
import numpy as np
import logging
from time import time

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)

def preprocess_road_image(image):
    """Enhance image to detect road contours."""
    logging.debug("Preprocessing image for road detection...")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    logging.debug("Applied Gaussian blur.")
    edges = cv2.Canny(blurred, 50, 150)
    logging.debug("Detected edges using Canny.")
    return edges

def find_contours(image):
    """Find contours in the preprocessed image."""
    logging.debug("Finding contours in the image...")
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    logging.debug(f"Found {len(contours)} contours.")
    return contours

def filter_and_approximate_contours(contours, epsilon_factor=0.01, min_area=100):
    """Filter and simplify contours."""
    logging.debug(f"Filtering and approximating contours with epsilon_factor={epsilon_factor}, min_area={min_area}...")
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            filtered_contours.append(approx)
    logging.debug(f"Filtered to {len(filtered_contours)} contours.")
    return filtered_contours

def draw_contours(image, contours, color=(0, 255, 0)):
    """Draw contours on the image."""
    logging.debug(f"Drawing {len(contours)} contours on the image.")
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, color, 2)
    return image_with_contours

def match_contours(contours1, contours2, similarity_threshold=0.5):
    """Match contours based on shape similarity."""
    logging.debug("Matching contours between images...")
    matches = []
    for contour1 in contours1:
        for contour2 in contours2:
            # Calculate similarity using cv2.matchShapes
            similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)
            if similarity < similarity_threshold:
                matches.append((contour1, contour2))
    logging.debug(f"Matched {len(matches)} contour pairs.")
    return matches

def process_land_usage_images(map_path, template_path):
    start_time = time()
    logging.info("Loading images...")
    map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if map_img is None or template_img is None:
        logging.error("Failed to load one or both images.")
        return

    # Preprocess images for road detection
    logging.info("Preprocessing images...")
    map_edges = preprocess_road_image(map_img)
    template_edges = preprocess_road_image(template_img)

    # Find contours
    map_contours = find_contours(map_edges)
    template_contours = find_contours(template_edges)

    # Filter and simplify contours
    map_approx_contours = filter_and_approximate_contours(map_contours)
    template_approx_contours = filter_and_approximate_contours(template_contours)

    # Draw and save processed images with contours
    map_with_contours = draw_contours(map_img, map_approx_contours)
    template_with_contours = draw_contours(template_img, template_approx_contours)
    cv2.imwrite("map_with_road_contours.jpg", map_with_contours)
    cv2.imwrite("template_with_road_contours.jpg", template_with_contours)

    # Match contours
    logging.info("Matching contours...")
    matches = match_contours(map_approx_contours, template_approx_contours)

    # Visualize matched contours
    matched_img = np.hstack((map_img, template_img))
    offset = map_img.shape[1]
    for contour1, contour2 in matches:
        for point in contour1:
            cv2.circle(matched_img, tuple(point[0]), 2, (0, 255, 0), -1)
        for point in contour2:
            cv2.circle(matched_img, (point[0][0] + offset, point[0][1]), 2, (255, 0, 0), -1)

    cv2.imwrite("matched_road_contours.jpg", matched_img)
    logging.info(f"Processing complete. Matches found: {len(matches)}")
    logging.info(f"Total processing time: {time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    process_land_usage_images("camurlim.jpg", "camurlim.jpg")

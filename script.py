import cv2

# Load the images
map_img = cv2.imread('camurlim.jpg')  # Load the map image
template = cv2.imread('camurlim.jpg')  # Load the template (smaller part of the map)

if map_img is None:
    print("Error: map_img could not be loaded.")
    exit()

if template is None:
    print("Error: template image could not be loaded.")
    exit()

# Define a function to resize images
def resize_image(image, scale_percent=50):
    """Resize the given image by a scale percentage."""
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Resize the map and template images
resized_map = resize_image(map_img)
resized_template = resize_image(template)

# Perform template matching
res = cv2.matchTemplate(resized_map, resized_template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Print matched coordinates and confidence
print(f"Matched Coordinates: {max_loc}")  # Gives X, Y on the map
print(f"Match confidence (max_val): {max_val}")  # Confidence score

import cv2

# Load images

def resize_image(image, scale_percent=75):
    """Resize the given image by a scale percentage."""
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

map_img = resize_image(cv2.imread('camurlim.jpg', cv2.IMREAD_GRAYSCALE))  # Map image
template = resize_image(cv2.imread('camurlim.jpg', cv2.IMREAD_GRAYSCALE))  # Template image

if map_img is None or template is None:
    print("Error: One or both images could not be loaded.")
    exit()

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)  # Increase the number of features detected

# Detect keypoints and descriptors in both the map and template
keypoints1, descriptors1 = orb.detectAndCompute(map_img, None)
keypoints2, descriptors2 = orb.detectAndCompute(template, None)

# Match descriptors using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance, best matches first
matches = sorted(matches, key=lambda x: x.distance)

# Ensure that there are at least 3 good matches
if len(matches) < 3:
    print("Not enough matches found!")
    exit()

# Extract top 3 control points (corresponding points in the map and template)
control_points_map = []
control_points_template = []

for match in matches[:3]:  # Get the best 3 matches
    control_points_map.append(keypoints1[match.queryIdx].pt)  # Points in map
    control_points_template.append(keypoints2[match.trainIdx].pt)  # Points in template

# Print control points for debugging
print("Control Points in Map (X, Y):", control_points_map)
print("Control Points in Template (X, Y):", control_points_template)

# Draw matches on map and template images
map_with_keypoints = cv2.drawKeypoints(map_img, keypoints1, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
template_with_keypoints = cv2.drawKeypoints(template, keypoints2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Draw matches between control points
matched_image = cv2.drawMatches(
    map_with_keypoints, keypoints1, 
    template_with_keypoints, keypoints2,
    matches[:3], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Mark the control points with yellow circles on both images
for pt in control_points_map:
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(map_with_keypoints, (x, y), radius=10, color=(0, 255, 255), thickness=3)  # Yellow circle

for pt in control_points_template:
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(template_with_keypoints, (x, y), radius=10, color=(0, 255, 255), thickness=3)  # Yellow circle

# Save the images with keypoints and control points marked
cv2.imwrite("marked_map.jpg", map_with_keypoints)
cv2.imwrite("marked_template.jpg", template_with_keypoints)
cv2.imwrite("matched_control_points.jpg", matched_image)

# Show marked matches
# # cv2.imshow("Matched Control Points", matched_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
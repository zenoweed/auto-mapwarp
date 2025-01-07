import cv2
import numpy as np
import osmnx as ox
import requests

# 1. Load Regional Map and OpenStreetMap Base
regional_map = cv2.imread('camurlim.jpg', cv2.IMREAD_GRAYSCALE)
osm_map = cv2.imread('scr.jpg', cv2.IMREAD_GRAYSCALE)

scale_percent = 50  # Adjust to reduce size (50% in this case)
regional_map = cv2.resize(regional_map, (0, 0), fx=scale_percent/100, fy=scale_percent/100)
# osm_map = cv2.resize(osm_map, (0, 0), fx=scale_percent/100, fy=scale_percent/100)
# 2. Detect and Match Features
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(regional_map, None)
kp2, des2 = sift.detectAndCompute(osm_map, None)

# Use FLANN for Matching
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Filter Matches Using Lowe's Ratio Test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Extract Coordinates of Matches
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

# 3. Compute Transformation Matrix and Warp
H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
warped_map = cv2.warpPerspective(regional_map, H, (osm_map.shape[1], osm_map.shape[0]))

# 4. Generate Control Points
control_points = []
for s, d in zip(src_pts, dst_pts):
    control_points.append({'pixel_x': s[0], 'pixel_y': s[1], 'lon': d[0], 'lat': d[1]})

visualized_map = regional_map.copy()
if len(visualized_map.shape) == 2:  # If grayscale, convert to color
    visualized_map = cv2.cvtColor(visualized_map, cv2.COLOR_GRAY2BGR)

font = cv2.FONT_HERSHEY_SIMPLEX

for idx, cp in enumerate(control_points):
    x = int(cp['pixel_x'])
    y = int(cp['pixel_y'])
    lon = cp['lon']
    lat = cp['lat']
    
    # Draw a circle at the control point
    cv2.circle(visualized_map, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    
    # Annotate with control point information
    label = f"#{idx} ({lon:.4f}, {lat:.4f})"
    cv2.putText(visualized_map, label, (x + 10, y - 10), font, 0.4, (0, 255, 0), thickness=1)
cv2.imwrite('control_points_visualized_regional.jpg', visualized_map)  # Save the output

visualized_map = osm_map.copy()
if len(visualized_map.shape) == 2:  # If grayscale, convert to color
    visualized_map = cv2.cvtColor(visualized_map, cv2.COLOR_GRAY2BGR)

font = cv2.FONT_HERSHEY_SIMPLEX

for idx, cp in enumerate(control_points):
    x = int(cp['pixel_x'])
    y = int(cp['pixel_y'])
    lon = cp['lon']
    lat = cp['lat']
    
    # Draw a circle at the control point
    cv2.circle(visualized_map, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    
    # Annotate with control point information
    label = f"#{idx} ({lon:.4f}, {lat:.4f})"
    cv2.putText(visualized_map, label, (x + 10, y - 10), font, 0.4, (0, 255, 0), thickness=1)
cv2.imwrite('control_points_visualized_osm.jpg', visualized_map)  # Save the output

# 5. Upload to Mapwarper
# api_key = 'your_api_key'
# image_id = 'your_image_id'
# for cp in control_points:
#     payload = {
#         'control_point': {
#             'x': cp['pixel_x'],
#             'y': cp['pixel_y'],
#             'lon': cp['lon'],
#             'lat': cp['lat']
#         }
#     }
    # print("Payload being sent to API:", payload) 
    # url = f"https://mapwarper.net/maps/{image_id}/gcps.json?key={api_key}"
    # requests.post(url, json=payload)

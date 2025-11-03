import os
import cv2
import math
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from rasterio.windows import Window
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
from shapely.geometry import Point

# === CONFIGURATION ===
input_file = 'orthomosiacs/model_farm_2.tif'
model_path = 'best.pt'
chip_width, chip_height = 300, 300
conf_threshold = 0.1
MST_THRESHOLD = 27
north_vector = np.array([0.5, 0.866])
detection_csv = 'detected_plant_points.csv'
points_wkt_name = 'model_farm_1_points.csv'
lines_wkt = 'model_farm_5_lines_wkt.csv'

# === Utility Functions ===
def rotate_point(pt, center, angle_rad):
    x, y = pt[0] - center[0], pt[1] - center[1]
    xr = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    yr = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return (xr + center[0], yr + center[1])

def save_points_to_csv(points, filename):
    df = pd.DataFrame(points, columns=['x', 'y'])
    df.to_csv(filename, index=False)

def load_points_from_csv(filename):
    df = pd.read_csv(filename)
    return df[['x', 'y']].to_numpy()

def detect_plants():
    model = YOLO(model_path)
    plant_coords = []
    with rasterio.open(input_file) as src:
        img_width, img_height = src.width, src.height
        num_cols = img_width // chip_width
        num_rows = img_height // chip_height
        for row in range(num_rows):
            for col in range(num_cols):
                x_off = col * chip_width
                y_off = row * chip_height
                window = Window(x_off, y_off, chip_width, chip_height)
                chip = src.read([1, 2, 3], window=window).transpose(1, 2, 0)
                chip = np.clip((chip - chip.min()) / (chip.ptp() + 1e-5) * 255, 0, 255).astype(np.uint8)
                temp_path = 'temp_chip.png'
                cv2.imwrite(temp_path, chip[:, :, ::-1])
                results = model(temp_path)[0]
                for box in results.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if label.lower() == 'sugarcane' and conf > conf_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        global_x, global_y = x_off + cx, y_off + cy
                        plant_coords.append((global_x, global_y))
    return np.array(plant_coords)

def compute_mean_direction(plant_coords):
    G = nx.Graph()
    for i, pt in enumerate(plant_coords):
        G.add_node(i, pos=(pt[0], pt[1]))
    dists = distance_matrix(plant_coords, plant_coords)
    for i in range(len(plant_coords)):
        for j in range(i + 1, len(plant_coords)):
            G.add_edge(i, j, weight=dists[i, j])
    mst = nx.minimum_spanning_tree(G)
    filtered_edges = [(u, v) for u, v, d in mst.edges(data=True) if d['weight'] < MST_THRESHOLD]
    subgraph = mst.edge_subgraph(filtered_edges).copy()

    directions = []
    for component in nx.connected_components(subgraph):
        nodes = list(component)
        if len(nodes) < 2:
            continue
        pts = np.array([plant_coords[n] for n in nodes])
        X = pts[:, 0].reshape(-1, 1)
        y = pts[:, 1]
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        dx, dy = 1.0, slope
        norm = np.sqrt(dx**2 + dy**2)
        directions.append([dx / norm, dy / norm])

    directions = np.array(directions)
    mean_vector = np.mean(directions, axis=0)
    return mean_vector / np.linalg.norm(mean_vector)

def rotate_image_and_points(image, points, angle_deg):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    rotated_coords = np.array([rotate_point(p, center, np.radians(angle_deg)) for p in points])
    return rotated_image, rotated_coords



if os.path.exists(detection_csv):
    print(f"Loading points from {detection_csv}")
    plant_coords = load_points_from_csv(detection_csv)
else:
    print("Running model inference to detect plants...")
    plant_coords = detect_plants()
    save_points_to_csv(plant_coords, detection_csv)



# === Save points as WKT ===
def save_points_as_wkt(points, output_csv=points_wkt_name):
    wkt_rows = []
    for i, (x, y) in enumerate(points):
        pt = Point(x, y)
        wkt_rows.append({'id': i + 1, 'wkt': pt.wkt})
    
    df_wkt = pd.DataFrame(wkt_rows)
    df_wkt.to_csv(output_csv, index=False)
    print(f"Saved {len(wkt_rows)} points to '{output_csv}' in WKT format.")

# Call this function at the end
save_points_as_wkt(plant_coords)

mean_vector = compute_mean_direction(plant_coords)
target_vector = np.array([0, -1])  # Up is North in image space
dot = np.dot(mean_vector, target_vector)
det = mean_vector[0] * target_vector[1] - mean_vector[1] * target_vector[0]
angle_to_north = np.degrees(np.arctan2(det, dot))
print(f"Rotation angle to align with North: {angle_to_north:.2f}°")
with rasterio.open(input_file) as src:
    full_img = src.read([1, 2, 3]).transpose(1, 2, 0)
    full_img = np.clip((full_img - full_img.min()) / (full_img.ptp() + 1e-5) * 255, 0, 255).astype(np.uint8)
# Rotate image and points (no plotting here)
rotated_img, rotated_coords = rotate_image_and_points(full_img, plant_coords, angle_to_north)

from sklearn.cluster import DBSCAN
# Step 1: Cluster rotated points based on X (column grouping)
# Step 1: Set bin size (approximate column width in pixels after rotation)
bin_size = 25  # adjust based on spacing

# Step 2: Create bins and find top-most point in each bin
x_vals = rotated_coords[:, 0]
min_x, max_x = np.min(x_vals), np.max(x_vals)
bins = np.arange(min_x, max_x + bin_size, bin_size)

first_points = []

for i in range(len(bins) - 1):
    # Find points within this bin
    in_bin = (x_vals >= bins[i]) & (x_vals < bins[i + 1])
    bin_points = rotated_coords[in_bin]

    if len(bin_points) == 0:
        continue

    # Find top-most (min Y) point in the bin
    top_point = bin_points[np.argmin(bin_points[:, 1])]
    first_points.append(top_point)

first_points = np.array(first_points)

# Parameters
step_size = 1  # how densely to scan columns (1 pixel = full resolution)
min_column_spacing = 20  # minimum X distance between columns (pixels)
tolerance = 3  # how close a point must be in X to be counted as intersecting the line

# Sort by Y (top to bottom), then X
sorted_points = rotated_coords[np.argsort(rotated_coords[:, 1])]
used_x = []

top_row_points = []

for pt in sorted_points:
    x, y = pt

    # Check if this X is far enough from already found columns
    if all(abs(x - ux) > min_column_spacing for ux in used_x):
        # Accept this as a new column
        top_row_points.append(pt)
        used_x.append(x)

top_row_points = np.array(top_row_points)

# (Visualization of all plants and top-row suppressed; we'll only plot gaps later)

import numpy as np
from shapely.geometry import Point
import pandas as pd

# Assume you already have:
# - lon_lat_points: list of (lon, lat)
# - rotation_angle: angle you used to rotate image (in degrees)
# import pandas as pd
# from shapely.geometry import Point
# import rasterio
# import rasterio.transform

# # Load raster to access georeferencing
# with rasterio.open(input_file) as src:
#     transform = src.transform
#     crs = src.crs

#     # Convert pixel (x, y) to geographic (lon, lat)
#     lon_lat_points = [transform * (x, y) for x, y in top_row_points]

# # Step 1: Compute center of all points in geographic space
# xs, ys = zip(*lon_lat_points)
# center_lon = np.mean(xs)
# center_lat = np.mean(ys)
# center = (center_lon, center_lat)

# # Step 2: Define anti-rotation function
# def rotate_geo_point(pt, center, angle_deg):
#     angle_rad = np.radians(-angle_deg)  # negative for anti-rotation
#     x, y = pt[0] - center[0], pt[1] - center[1]
#     xr = x * np.cos(angle_rad) - y * np.sin(angle_rad)
#     yr = x * np.sin(angle_rad) + y * np.cos(angle_rad)
#     return (xr + center[0], yr + center[1])

# # Step 3: Apply anti-rotation
# anti_rotated_lonlat = [rotate_geo_point(pt, center, angle_to_north) for pt in lon_lat_points]

# # Step 4: Save to CSV with WKT
# wkt_points = [f"POINT({lon} {lat})" for lon, lat in anti_rotated_lonlat]
# df = pd.DataFrame({
#     'id': range(len(wkt_points)),
#     'longitude': [pt[0] for pt in anti_rotated_lonlat],
#     'latitude': [pt[1] for pt in anti_rotated_lonlat],
#     'wkt': wkt_points
# })
# df.to_csv('top_row_plants_anti_rotated.csv', index=False)

gap_lines = []
gap_pixel_segments = []
# print("Saved anti-rotated georeferenced points to top_row_plants_anti_rotated.csv")
import numpy as np
import cv2
import matplotlib.pyplot as plt

# === Parameters ===
step_size = 1  # how densely to scan columns (1 pixel = full resolution)
min_column_spacing = 20  # minimum X distance between columns (pixels)
tolerance = 3  # how close a point must be in X to be counted as intersecting the line

# Sort by Y (top to bottom), then X
sorted_points = rotated_coords[np.argsort(rotated_coords[:, 1])]
used_x = []
top_row_points = []

for pt in sorted_points:
    x, y = pt
    if all(abs(x - ux) > min_column_spacing for ux in used_x):
        top_row_points.append(pt)
        used_x.append(x)

top_row_points = np.array(top_row_points)

# === Report top-row (topmost) count ===
print(f"Identified {len(top_row_points)} topmost points.")

# === Bottom Points Extraction (single, cleaned pass) ===
bottom_points = []
column_spacing_thresh = 20  # minimum spacing between sugarcane columns (in pixels)

min_y = int(np.min(rotated_coords[:, 1]))
max_y = int(np.max(rotated_coords[:, 1]))
mid_y = min_y + (max_y - min_y) // 2

sorted_points = sorted(rotated_coords, key=lambda x: -x[1])  # Bottom-up

for pt in sorted_points:
    x, y = pt
    if y >= mid_y and all(abs(x - bx) > column_spacing_thresh for bx, by in bottom_points):
        bottom_points.append((x, y))

print(f"Identified {len(bottom_points)} bottommost points.")

# === Step 3: One-to-One Matching (Bottom to Closest Unused Top) ===
columns = []
used_top_indices = set()
used_top_indices = set()
top_array = np.array(top_row_points)

for bx, by in bottom_points:
    min_dist = float('inf')
    chosen_index = -1
    for idx, (tx, ty) in enumerate(top_array):
        if idx in used_top_indices:
            continue
        dist = abs(tx - bx)
        if dist < min_dist:
            min_dist = dist
            chosen_index = idx
    if chosen_index != -1:
        tx, ty = top_array[chosen_index]
        columns.append(((tx, ty), (bx, by)))
        used_top_indices.add(chosen_index)

# === Step 4: Draw Lines with Angle Constraint (85°–95° Only) ===
img_copy = rotated_img.copy()
angle_threshold = 3  # Allowed deviation from 90° (vertical)

for (x1, y1), (x2, y2) in columns:
    dx = x2 - x1
    dy = y2 - y1
    angle = abs(math.degrees(math.atan2(dy, dx)))
    angle = 180 - angle if angle > 90 else angle  # Normalize to [0, 90]

    if 90 - angle_threshold <= angle <= 90 + angle_threshold:
        cv2.line(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# === Step 5: Show Result ===
# === Step 6: Improved Recursive Column Point Tracing ===
step_y = 10  # pixels to move down each step
x_tolerance = 15  # lateral deviation tolerance
y_tolerance = 15  # vertical distance tolerance
max_skip_steps = 70  # max number of consecutive misses before breaking

rotated_coords_arr = np.array(rotated_coords)
unassigned = set(range(len(rotated_coords)))
column_points_map = []

for (x_start, y_start) in top_row_points:
    col_points = []
    x, y = x_start, y_start
    skips = 0
    last_direction = 0

    while y < rotated_img.shape[0] and skips < max_skip_steps:
        # Search window: find unassigned points nearby
        candidates = []
        for idx in list(unassigned):
            px, py = rotated_coords_arr[idx]
            if (
                abs(px - x) < x_tolerance and
                0 < py - y < y_tolerance  # point should be below and within Y range
            ):
                candidates.append((idx, px, py))

        if candidates:
            # Choose nearest in Y (closest downward point)
            best_idx, px, py = sorted(candidates, key=lambda tup: tup[2])[0]
            col_points.append((px, py))
            x, y = px, py
            unassigned.remove(best_idx)
            skips = 0
        else:
            # No point found in this step; skip down and try again
            y += step_y
            skips += 1

    if len(col_points) > 1:
        column_points_map.append(col_points)

# (Column-wise visualizations suppressed; only gap segments will be plotted)

# plt.show()
# === Step 8: Fit Best-Fit Line for Each Column and Plot ===
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import rasterio

# === Load raster for transform and pixel size ===
with rasterio.open(input_file) as src:
    transform = src.transform
    pixel_size_x, pixel_size_y = src.res
    image_height, image_width = src.height, src.width

# === Column statistics ===
column_stats = []
img_lines = rotated_img.copy()
gap_threshold_px = 3 / pixel_size_x  # 3 meters in pixels

from shapely.geometry import LineString, box
import pandas as pd
import rasterio
import numpy as np
import math

# === Anti-rotation function ===
def rotate_geo_point(pt, center, angle_deg):
    angle_rad = np.radians(angle_deg)
    x, y = pt[0] - center[0], pt[1] - center[1]
    xr = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    yr = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return (xr + center[0], yr + center[1])

# === Padding function for gap lines ===
def add_gap_padding(p1, p2, padding_px):
    vec = np.array(p2) - np.array(p1)
    dist = np.linalg.norm(vec)
    if dist == 0:
        return p1, p2
    unit_vec = vec / dist
    new_p1 = np.array(p1) + padding_px * unit_vec
    new_p2 = np.array(p2) - padding_px * unit_vec
    return new_p1.tolist(), new_p2.tolist()

# === Load raster and compute required info ===
with rasterio.open(input_file) as src:
    transform = src.transform
    pixel_size_x, pixel_size_y = src.res
    width, height = src.width, src.height
    center_pixel = (width / 2, height / 2)
    center_geo = transform * center_pixel
    bounds = src.bounds
    raster_rect = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

# === Setup ===
anti_rotation_angle = angle_to_north
padding_meters = 1
padding_px = padding_meters / pixel_size_x

top_lines = []
bottom_lines = []
for i, col_pts in enumerate(column_points_map):
    col_pts = np.array(col_pts)
    if len(col_pts) < 2:
        continue

    col_pts_sorted = col_pts[np.argsort(col_pts[:, 1])]
    top_pt = col_pts_sorted[0]
    bottom_pt = col_pts_sorted[-1]

    # === Top and Bottom Lines ===
    top_geo = rotate_geo_point(transform * (top_pt[0], top_pt[1]), center_geo, anti_rotation_angle)
    top_img_edge_geo = rotate_geo_point(transform * (top_pt[0], 0), center_geo, anti_rotation_angle)
    bottom_geo = rotate_geo_point(transform * (bottom_pt[0], bottom_pt[1]), center_geo, anti_rotation_angle)
    bottom_img_edge_geo = rotate_geo_point(transform * (bottom_pt[0], height - 1), center_geo, anti_rotation_angle)

    top_line_geom = LineString([top_geo, top_img_edge_geo]).intersection(raster_rect)
    bottom_line_geom = LineString([bottom_geo, bottom_img_edge_geo]).intersection(raster_rect)

    if not top_line_geom.is_empty:
        top_lines.append({'type': 'top', 'column_id': i + 1, 'wkt': top_line_geom.wkt})
    if not bottom_line_geom.is_empty:
        bottom_lines.append({'type': 'bottom', 'column_id': i + 1, 'wkt': bottom_line_geom.wkt})

    # === Gap Lines with Padding ===
    for j in range(len(col_pts) - 1):
        pt1, pt2 = col_pts[j], col_pts[j + 1]
        dist = np.linalg.norm(pt2 - pt1)
        if dist > (3 / pixel_size_x):
            pt1_pad, pt2_pad = add_gap_padding(pt1, pt2, padding_px)
            geo1 = rotate_geo_point(transform * (pt1_pad[0], pt1_pad[1]), center_geo, anti_rotation_angle)
            geo2 = rotate_geo_point(transform * (pt2_pad[0], pt2_pad[1]), center_geo, anti_rotation_angle)

            gap_line_geom = LineString([geo1, geo2]).intersection(raster_rect)
            if not gap_line_geom.is_empty:
                gap_lines.append({
                    'type': 'gap',
                    'column_id': i + 1,
                    'from_index': j,
                    'to_index': j + 1,
                    'wkt': gap_line_geom.wkt
                })
                # keep pixel coordinates (after padding) to visualize gaps on the rotated image
                try:
                    gap_pixel_segments.append((pt1_pad, pt2_pad))
                except Exception:
                    pass

# === Save to CSV ===
df_lines = pd.DataFrame(top_lines + bottom_lines + gap_lines)

# Ensure parent directory exists before saving
parent_dir = os.path.dirname(lines_wkt)
if parent_dir:
    os.makedirs(parent_dir, exist_ok=True)

df_lines.to_csv(lines_wkt, index=False)
print(f"Saved top, bottom, and clipped gap lines to '{lines_wkt}'")
# --- Plot only the detected gap segments over the rotated image ---
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
# Draw column lines (green) and gaps (red), and plant points (black)
try:
    # 1) Draw columns as green lines (extend slightly below bottom point)
    extension_px_min = 20  # minimum extension in pixels below bottom point
    # create a boolean mask of valid (non-black) pixels to avoid drawing into rotated-image background
    mask = np.any(rotated_img > 0, axis=2)
    img_h, img_w = mask.shape

    def clip_segment_to_mask(p1, p2, mask):
        """Return (p1_clip, p2_clip) clipped to the contiguous True region along the segment, or None if no overlap.

        Samples the segment and keeps the first-to-last indices where mask is True.
        """
        x1, y1 = p1
        x2, y2 = p2
        length = max(int(np.hypot(x2 - x1, y2 - y1)) * 2, 2)
        xs = np.linspace(x1, x2, length)
        ys = np.linspace(y1, y2, length)
        xi = np.clip(np.round(xs).astype(int), 0, img_w - 1)
        yi = np.clip(np.round(ys).astype(int), 0, img_h - 1)
        vals = mask[yi, xi]
        true_idxs = np.where(vals)[0]
        if true_idxs.size == 0:
            return None
        first, last = true_idxs[0], true_idxs[-1]
        return (xs[first], ys[first]), (xs[last], ys[last])
    for col_pts in column_points_map:
        try:
            col_arr = np.array(col_pts)
            if col_arr.shape[0] < 2:
                continue
            top = col_arr[0]
            bottom = col_arr[-1]
            vec = bottom - top
            dist = np.linalg.norm(vec)
            if dist == 0:
                ext_bottom = bottom
            else:
                unit = vec / dist
                ext = max(extension_px_min, int(0.1 * dist))
                ext_bottom = bottom + unit * ext
            # clip to non-black rotated image area
            clipped = clip_segment_to_mask(top, ext_bottom, mask)
            if clipped is None:
                continue
            (cx1, cy1), (cx2, cy2) = clipped
            plt.plot([cx1, cx2], [cy1, cy2], c='cyan', linewidth=2)
        except Exception:
            continue

    # 2) Draw gap segments in red
    for p1, p2 in gap_pixel_segments:
        # ensure gap segments are inside image mask
        clipped = clip_segment_to_mask(p1, p2, mask)
        if clipped is None:
            continue
        (gx1, gy1), (gx2, gy2) = clipped
        plt.plot([gx1, gx2], [gy1, gy2], c='red', linewidth=3)

    # 3) Plot all plant detections as black dots on top
    try:
        pts = np.array(rotated_coords)
        plt.scatter(pts[:, 0], pts[:, 1], c='black', s=8, zorder=10)
    except Exception:
        pass

    plt.title(f"Columns (green), Gaps (red), Plants (black) — gaps={len(gap_pixel_segments)}")
except Exception as e:
    plt.title(f"Visualization error: {e}")

plt.gca().invert_yaxis()
plt.axis('equal')
plt.tight_layout()
plt.show()
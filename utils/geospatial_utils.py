"""
Geospatial utilities for processing UAV imagery
Handles coordinate transformations and geospatial operations
"""

import numpy as np
from pathlib import Path

try:
    import rasterio
    from rasterio.windows import Window
    import geopandas as gpd
    from shapely.geometry import box
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    print("Warning: Geospatial libraries not available. Install with: pip install rasterio geopandas")


def load_geotiff(file_path):
    """
    Load a GeoTIFF file.
    
    Args:
        file_path: Path to GeoTIFF file
        
    Returns:
        tuple: (image array, transform, crs)
    """
    if not GEOSPATIAL_AVAILABLE:
        raise ImportError("rasterio is required for GeoTIFF operations")
    
    with rasterio.open(file_path) as src:
        image = src.read()
        transform = src.transform
        crs = src.crs
    
    return image, transform, crs


def extract_tiles(geotiff_path, tile_size=640, overlap=0.1, output_dir='tiles'):
    """
    Extract tiles from a large GeoTIFF for training.
    
    Args:
        geotiff_path: Path to input GeoTIFF
        tile_size: Size of tiles in pixels
        overlap: Overlap ratio between tiles (0-1)
        output_dir: Directory to save tiles
        
    Returns:
        list: Paths to created tiles
    """
    if not GEOSPATIAL_AVAILABLE:
        raise ImportError("rasterio is required for tile extraction")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tile_paths = []
    stride = int(tile_size * (1 - overlap))
    
    with rasterio.open(geotiff_path) as src:
        width = src.width
        height = src.height
        
        tile_idx = 0
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                # Read tile
                window = Window(x, y, tile_size, tile_size)
                tile = src.read(window=window)
                
                # Get transform for this tile
                tile_transform = rasterio.windows.transform(window, src.transform)
                
                # Save tile
                tile_path = output_path / f"tile_{tile_idx:04d}.tif"
                
                with rasterio.open(
                    tile_path,
                    'w',
                    driver='GTiff',
                    height=tile_size,
                    width=tile_size,
                    count=src.count,
                    dtype=tile.dtype,
                    crs=src.crs,
                    transform=tile_transform
                ) as dst:
                    dst.write(tile)
                
                tile_paths.append(tile_path)
                tile_idx += 1
    
    print(f"Extracted {len(tile_paths)} tiles to {output_dir}")
    return tile_paths


def pixel_to_geo_coords(pixel_x, pixel_y, transform):
    """
    Convert pixel coordinates to geographic coordinates.
    
    Args:
        pixel_x: X pixel coordinate
        pixel_y: Y pixel coordinate
        transform: Affine transform from rasterio
        
    Returns:
        tuple: (geo_x, geo_y)
    """
    if GEOSPATIAL_AVAILABLE:
        # Use rasterio's built-in method for better reliability
        import rasterio.transform
        geo_x, geo_y = rasterio.transform.xy(transform, pixel_y, pixel_x)
        return geo_x, geo_y
    else:
        # Fallback to manual calculation
        geo_x = transform[2] + pixel_x * transform[0] + pixel_y * transform[1]
        geo_y = transform[5] + pixel_x * transform[3] + pixel_y * transform[4]
        return geo_x, geo_y


def geo_to_pixel_coords(geo_x, geo_y, transform):
    """
    Convert geographic coordinates to pixel coordinates.
    
    Args:
        geo_x: X geographic coordinate
        geo_y: Y geographic coordinate
        transform: Affine transform from rasterio
        
    Returns:
        tuple: (pixel_x, pixel_y)
    """
    if GEOSPATIAL_AVAILABLE:
        # Use rasterio's built-in inverse transformation for better reliability
        import rasterio.transform
        pixel_y, pixel_x = rasterio.transform.rowcol(transform, geo_x, geo_y)
        return int(pixel_x), int(pixel_y)
    else:
        # Fallback to manual calculation
        det = transform[0] * transform[4] - transform[1] * transform[3]
        pixel_x = (transform[4] * (geo_x - transform[2]) - transform[1] * (geo_y - transform[5])) / det
        pixel_y = (-transform[3] * (geo_x - transform[2]) + transform[0] * (geo_y - transform[5])) / det
        return int(pixel_x), int(pixel_y)


def create_bbox_from_coords(coords, transform, img_width, img_height):
    """
    Create YOLO format bbox from geographic coordinates.
    
    Args:
        coords: List of (geo_x, geo_y) coordinates
        transform: Affine transform
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        tuple: (x_center, y_center, width, height) in YOLO format
    """
    # Convert all coordinates to pixels
    pixel_coords = [geo_to_pixel_coords(x, y, transform) for x, y in coords]
    
    # Get bounding box
    x_coords = [p[0] for p in pixel_coords]
    y_coords = [p[1] for p in pixel_coords]
    
    x_min = max(0, min(x_coords))
    x_max = min(img_width, max(x_coords))
    y_min = max(0, min(y_coords))
    y_max = min(img_height, max(y_coords))
    
    # Convert to YOLO format
    x_center = (x_min + x_max) / (2 * img_width)
    y_center = (y_min + y_max) / (2 * img_height)
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return x_center, y_center, width, height


def load_shapefile_annotations(shapefile_path, class_field='class'):
    """
    Load annotations from a shapefile.
    
    Args:
        shapefile_path: Path to shapefile
        class_field: Name of the field containing class labels
        
    Returns:
        GeoDataFrame with annotations
    """
    if not GEOSPATIAL_AVAILABLE:
        raise ImportError("geopandas is required for shapefile operations")
    
    gdf = gpd.read_file(shapefile_path)
    return gdf


def calculate_vegetation_indices(image, red_band=0, nir_band=3):
    """
    Calculate vegetation indices from multispectral imagery.
    
    Args:
        image: Multispectral image array (bands, height, width)
        red_band: Index of red band
        nir_band: Index of near-infrared band
        
    Returns:
        dict: Dictionary containing various vegetation indices
    """
    red = image[red_band].astype(float)
    nir = image[nir_band].astype(float)
    
    # NDVI (Normalized Difference Vegetation Index)
    ndvi = (nir - red) / (nir + red + 1e-8)
    
    # EVI (Enhanced Vegetation Index) - requires blue band
    # Simplified version without blue band
    evi = 2.5 * (nir - red) / (nir + 2.4 * red + 1.0 + 1e-8)
    
    return {
        'ndvi': ndvi,
        'evi': evi
    }


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Geospatial utilities')
    parser.add_argument('--action', type=str, required=True,
                        choices=['extract-tiles', 'info'],
                        help='Action to perform')
    parser.add_argument('--input', type=str, help='Input GeoTIFF file')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--tile-size', type=int, default=640,
                        help='Tile size in pixels')
    parser.add_argument('--overlap', type=float, default=0.1,
                        help='Overlap ratio between tiles')
    
    args = parser.parse_args()
    
    if not GEOSPATIAL_AVAILABLE:
        print("Error: Geospatial libraries not available")
        print("Install with: pip install rasterio geopandas")
        exit(1)
    
    if args.action == 'extract-tiles':
        if not args.input or not args.output:
            parser.error('extract-tiles requires --input and --output')
        extract_tiles(args.input, args.tile_size, args.overlap, args.output)
    
    elif args.action == 'info':
        if not args.input:
            parser.error('info requires --input')
        with rasterio.open(args.input) as src:
            print(f"Width: {src.width}")
            print(f"Height: {src.height}")
            print(f"Bands: {src.count}")
            print(f"CRS: {src.crs}")
            print(f"Bounds: {src.bounds}")
            print(f"Transform: {src.transform}")

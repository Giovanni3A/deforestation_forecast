import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import box


def compute_frames(
    total_bounds: np.ndarray,
    box_side: float,
    ix: float,
    fx: float,
    iy: float,
    fy: float,
):
    min_x, min_y, max_x, max_y = total_bounds
    km_x = (max_x - min_x) / box_side
    km_y = (max_y - min_y) / box_side
    matrix_size_x = int(np.ceil(km_x))
    matrix_size_y = int(np.ceil(km_y))

    boxes = []
    rows = []
    for x in range(ix, fx):
        for y in range(iy, fy):
            frame = box(
                min_x + box_side * (x),
                min_y + box_side * (y),
                min_x + box_side * (x + 1),
                min_y + box_side * (y + 1),
            )
            boxes.append(frame)
            rows.append((x, y, y + x * matrix_size_y))

    return gpd.GeoDataFrame(data=rows, geometry=boxes, columns=["x", "y", "frame_id"])

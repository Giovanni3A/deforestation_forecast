import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Polygon


def compute_frames(
    total_bounds: np.ndarray,
    box_side: float,
    ix: float,
    fx: float,
    iy: float,
    fy: float,
):
    min_x, min_y, max_x, max_y = total_bounds
    nx = int(np.ceil((max_x - min_x) / box_side))
    ny = int(np.ceil((max_y - min_y) / box_side))
    gx, gy = np.linspace(min_x,max_x,nx), np.linspace(min_y,max_y,ny)

    boxes = []
    rows = []
    for i in range(ix, fx):
        for j in range(iy, fy):
            frame = Polygon([
                [gx[i],gy[j]],
                [gx[i],gy[j+1]],
                [gx[i+1],gy[j+1]],
                [gx[i+1],gy[j]]
            ])
            
            boxes.append(frame)
            rows.append((i, j, j + i * (ny-1)))

    return gpd.GeoDataFrame(data=rows, geometry=boxes, columns=["x", "y", "frame_id"])

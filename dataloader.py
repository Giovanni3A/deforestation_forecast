import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import config
from utils import compute_frames

# set device to GPU
dev = "cuda:0"

class CustomDataset(Dataset):
    def __init__(
        self, 
        X, 
        patches, 
        frames_idx, 
        county_data=None, 
        county_defor=None,
        precip_data=None,
        tpi_data=None,
        landcover_data=None,
        scores_data=None,
        night_data=None,
        sentinel_data=None,
        channels=None,
    ):
        super(CustomDataset, self).__init__()

        self.patches = patches
        self.frames_idx = frames_idx
        self.X = X
        self.county_data = county_data
        self.county_defor = county_defor
        self.precip_data = precip_data
        self.tpi_data = tpi_data
        self.landcover_data = landcover_data
        self.scores_data = scores_data
        self.night_data = night_data
        self.sentinel_data = sentinel_data
        self.channels = channels

        self.autor_window = 4
        self.ix = frames_idx["x"].min()
        self.iy = frames_idx["y"].min()

    def __len__(self):
        return len(self.patches) * (self.X.shape[0]-self.autor_window)

    def __getitem__(self, index):

        # get index info
        idx_patch = index // (self.X.shape[0] - self.autor_window)
        idx_time   = index % (self.X.shape[0] - self.autor_window)
        idx_frames = self.frames_idx.loc[self.patches[idx_patch]]

        # get input
        input_matrix = self.X[
            idx_time:idx_time+self.autor_window, 
            idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
            idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1
        ]

        if self.county_data is not None:
            input_matrix = np.concatenate([
                input_matrix,
                self.county_data[
                    :,
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1
                ]
            ])
        
        if self.county_defor is not None:
            input_matrix = np.concatenate([
                input_matrix,
                self.county_defor[
                    idx_time:idx_time+self.autor_window,
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1
                ]
            ])
        
        if self.precip_data is not None:
            input_matrix = np.concatenate([
                input_matrix,
                self.precip_data[
                    idx_time:idx_time+self.autor_window,
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1
                ]
            ])
        
        if self.tpi_data is not None:
            input_matrix = np.concatenate([
                input_matrix,
                self.tpi_data[
                    :,
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1
                ]
            ])
        
        if self.landcover_data is not None:
            input_matrix = np.concatenate([
                input_matrix,
                self.landcover_data[
                    :,
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1
                ]
            ])
        
        if self.scores_data is not None:
            input_matrix = np.concatenate([
                input_matrix,
                self.scores_data[
                    [idx_time+self.autor_window],
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1
                ]
            ])
        
        if self.night_data is not None:
            input_matrix = np.concatenate([
                input_matrix,
                self.night_data[
                    :,
                    idx_time+self.autor_window-1,
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1
                ]
            ])

        if self.sentinel_data is not None:
            input_matrix = np.concatenate([
                input_matrix,
                self.sentinel_data[
                    [idx_time+self.autor_window-1],
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1
                ]
            ])

        if self.channels is not None:
            input_matrix = input_matrix[self.channels, :, :]
        
        data = torch.tensor(input_matrix).float().to(dev)

        # get output
        labels = np.zeros(
            (
                2, 
                idx_frames["x"].max()-idx_frames["x"].min() + 1, 
                idx_frames["y"].max()-idx_frames["y"].min() + 1
            )
        )
        target_idx = np.where(
            self.X[
                idx_time+self.autor_window, 
                idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1
            ] > 1e-7
        )
        labels[0, :, :] = 1
        labels[0, :, :][target_idx] = 0
        labels[1, :, :][target_idx] = 1
        labels = torch.tensor(labels).float().to(dev)
        return data, labels

def load_data():

    # part 1 - read data

    # load legal amazon limits
    am_bounds = gpd.read_file(config.AMAZON_FRONTIER_DATA)
    # load frames idx detail
    frames_idx = pd.read_csv(config.TR_FRAMES_IDX, index_col=0)
    # load frames deforestation area history
    deforestation = pd.read_csv(config.TR_DEFORESTATION, index_col=0)
    deforestation["quarter_date"] = pd.to_datetime(deforestation["quarter_date"])
    # counties
    frames_county = pd.read_csv(config.TR_COUNTIES, index_col=0)
    counties_defor = pd.read_csv(config.TR_COUNTIES_DEFOR, index_col=0)
    # precipitations
    precip = pd.read_csv(config.TR_RAIN_AVG)
    precip["quarter_date"] = pd.to_datetime(precip["dt"])
    # terrain position index
    tpi = pd.read_csv(config.TR_TPI, skiprows=1)\
        .rename(columns={"Unnamed: 0": "frame_id"})
    # land cover
    landcover = pd.read_csv(config.TR_LANDCOVER)
    landcover = pd.pivot_table(
        landcover, 
        index=["frame_id"], 
        columns=["landcover"], 
        values="geometry", 
        aggfunc="sum"
    ).fillna(0).astype(int)
    # convert to percentage
    sum_by_frame = landcover.sum(axis=1)
    for col in landcover.columns:
        landcover[col] = landcover[col] / sum_by_frame
    # past scores
    past_scores = pd.read_csv(config.TR_PAST_SCORES)
    past_scores["variable"] = pd.to_datetime(past_scores["variable"])
    # night lights
    night_light = pd.read_csv(config.TR_NIGHT_LIGHT)
    night_light["dt"] = pd.to_datetime(night_light["dt"])
    # sentinel 1
    sentinel = pd.read_csv(config.TR_SENTINEL1)
    sentinel["dt"] = pd.to_datetime(sentinel["dt"])

    # part 2 - create grid and mount matrix with features

    # create limits history grid
    time_grid = np.zeros((len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in enumerate(config.TIME_STEPS):
        defor_area = (
            deforestation[
                deforestation["quarter_date"] == dt
            ].set_index("frame_id")["area"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        time_grid[t, :, :] = defor_area.values.reshape(time_grid[0, :, :].shape)
    county_data = np.zeros((2, frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    county_data[0] = (
        frames_county.set_index("frame_id")["populacao"] +\
        pd.Series(0, index=frames_idx.index)
    ).fillna(0).\
        values.reshape(county_data.shape[1:])

    county_data[1] = (
        frames_county.set_index("frame_id")["densidade"] +\
        pd.Series(0, index=frames_idx.index)
    ).fillna(0).\
        values.reshape(county_data.shape[1:])
    frames_counties_defor = pd.merge(
        counties_defor,
        frames_county[["frame_id", "county_id"]],
        on="county_id",
        how="right"
    )
    frames_counties_defor["quarter_date"] = pd.to_datetime(frames_counties_defor["quarter_date"])
    # create limits history grid
    counties_time_grid = np.zeros((len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in tqdm(enumerate(config.TIME_STEPS), desc="Creating counties defor time grid"):
        defor_area = (
            frames_counties_defor[
                frames_counties_defor["quarter_date"] == dt
            ].set_index("frame_id")["area"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        counties_time_grid[t, :, :] = defor_area.values.reshape(counties_time_grid[0, :, :].shape)
    # create limits history grid
    precip_time_grid = np.zeros((len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in tqdm(enumerate(config.TIME_STEPS), desc="Creating precipitations time grid"):
        precip_sum = (
            precip[
                precip["quarter_date"] == dt
            ].set_index("frame_id")["precipitation"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        precip_time_grid[t, :, :] = precip_sum.values.reshape(counties_time_grid[0, :, :].shape)
    cols = ["mean", "min", "max", "std"]
    tpi_array = np.zeros((len(cols), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for icol, col in enumerate(cols):
        v = (
            tpi.set_index("frame_id")[col] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        tpi_array[icol, :, :] = v.values.reshape(tpi_array[0, :, :].shape)
    landcover_categories = [[20], [40, 50], [180]]
    landcover_array = np.zeros((len(landcover_categories), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for i, cols in enumerate(landcover_categories):
        v = (
            landcover[cols].sum(axis=1) +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        landcover_array[i, :, :] = v.values.reshape(landcover_array[0, :, :].shape)
    # create history grid for scores
    scores_time_grid = np.zeros((len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in tqdm(enumerate(config.TIME_STEPS), desc="Creating scores time grid"):
        t_scores = (
            past_scores[
                past_scores["variable"] == dt
            ].set_index("frame_id")["value"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        scores_time_grid[t, :, :] = t_scores.values.reshape(scores_time_grid[0, :, :].shape)
    # create history grid for scores
    night_time_grid = np.zeros((2, len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in tqdm(enumerate(config.TIME_STEPS), desc="Creating night lights time grid"):
        avg_light = (
            night_light[
                night_light["dt"] == dt
            ].set_index("frame_id")["avg_light"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        night_time_grid[0, t, :, :] = avg_light.values.reshape(night_time_grid[0, 0, :, :].shape)
        
        max_light = (
            night_light[
                night_light["dt"] == dt
            ].set_index("frame_id")["max_light"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        night_time_grid[1, t, :, :] = max_light.values.reshape(night_time_grid[0, 0, :, :].shape)
    # create limits history grid
    sentinel_time_grid = np.zeros((len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in tqdm(enumerate(config.TIME_STEPS), desc="Creating sentinel time grid"):
        dt_sentinel = (
            sentinel[
                sentinel["dt"] == dt
            ].set_index("frame_id")["value"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        sentinel_time_grid[t, :, :] = dt_sentinel.values.reshape(sentinel_time_grid[0, :, :].shape)
    
    # part 3 - compute frame patches

    out_condition = "both"  # deforestation | borders | both
    bundle_step = 32
    patches = []
    for ix in tqdm(list(range(frames_idx["x"].min(), frames_idx["x"].max()+1, bundle_step)), desc="Computing patches"):
        fx = ix + config.INPUT_BOXES_SIZE
        for iy in range(frames_idx["y"].min(), frames_idx["y"].max()+1, bundle_step):
            fy = iy + config.INPUT_BOXES_SIZE

            iframes = frames_idx[
                (frames_idx["x"] >= ix) & 
                (frames_idx["x"] < fx) &
                (frames_idx["y"] >= iy) &
                (frames_idx["y"] < fy)
            ]
            
            if out_condition == "borders":
                if iframes["in_borders"].mean() >= 0.5:  # condition: bundle has to be at least half inside borders
                    patches.append(iframes.index)
                    
            elif out_condition == "deforestation":
                out_of_borders_frames = len(set(iframes.index) - set(deforestation["frame_id"].values))
                if out_of_borders_frames < len(iframes):  # condition: bundle has to contain some deforestation
                    patches.append(iframes.index) 

            elif out_condition == "both":
                out_of_borders_frames = len(set(iframes.index) - set(deforestation["frame_id"].values))
                if (out_of_borders_frames < len(iframes)) and (iframes["in_borders"].mean() >= 0.5):
                    patches.append(iframes.index) 
    # remove patches that represent reduced regions
    patches = [b for b in patches if (len(b)==len(patches[0]))]

    # part 4 - train test split and normalization
    train_time_idx = range(0,12)
    val_time_idx = range(12,20)
    test_time_idx = range(20,28)
    train_data = time_grid[train_time_idx, :, :]
    val_data = time_grid[val_time_idx, :, :]
    test_data = time_grid[test_time_idx, :, :]
    
    norm_pop = (county_data[0, :, :] - np.median(county_data[0, :, :])) / 1e5
    norm_den = (county_data[1, :, :] - np.median(county_data[1, :, :])) / 30
    county_data[0, :, :] = norm_pop
    county_data[1, :, :] = norm_den
    counties_time_grid = (counties_time_grid-counties_time_grid[train_time_idx, :, :].mean()) / counties_time_grid[train_time_idx, :, :].std()
    precip_time_grid = (precip_time_grid-precip_time_grid[train_time_idx, :, :].mean()) / precip_time_grid[train_time_idx, :, :].std()
    for i in range(tpi_array.shape[0]):
        tpi_array[i, :, :] = (tpi_array[i, :, :] - tpi_array[i, :, :].mean()) / tpi_array[i, :, :].std()
    for i in [0, 1]:
        s = (
            (
                night_time_grid[i, :, :, :] - 
                night_time_grid[i, train_time_idx, :, :].mean()
            ) / night_time_grid[i, train_time_idx, :, :].std()
        )
        s[np.where(s > 3)] = 3
        night_time_grid[i, :, :, :] = s.copy()

    return (
        train_data, val_data, test_data,
        patches, frames_idx, 
        county_data,
        counties_time_grid,
        precip_time_grid,
        tpi_array,
        landcover_array,
        scores_time_grid,
        night_time_grid,
        sentinel_time_grid
    )
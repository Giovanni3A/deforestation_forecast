import os
import pandas as pd

# data folder
DATA_PATH = r"C:\Users\giovanni\DL2CV\data"

# data layers
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
TRUSTED_DATA_PATH = os.path.join(DATA_PATH, "trusted")

# raw data
AMAZON_FRONTIER_DATA = os.path.join(RAW_DATA_PATH, "brazilian_legal_amazon")
INITIAL_DEFORESTATION = os.path.join(RAW_DATA_PATH, "accumulated_deforestation_2007")
DETER_DATA = os.path.join(RAW_DATA_PATH, "deter-amz-09-03-2024-22_27_53")
PRODES_DATA = os.path.join(RAW_DATA_PATH, "yearly_deforestation")
COUNTIES_DATA = os.path.join(RAW_DATA_PATH, "municipios")
RAIN_DATA = os.path.join(RAW_DATA_PATH, "precipitations")
TPI_DATA = os.path.join(RAW_DATA_PATH, "TPI.tif")
LANDCOVER_DATA = os.path.join(RAW_DATA_PATH, "landcover.tif")
NIGHT_LIGHT_DATA = os.path.join(RAW_DATA_PATH, "night_lights")
SENTINEL1_DATA = os.path.join(RAW_DATA_PATH, "sentinel1")

# trusted data
TR_DEFORESTATION = os.path.join(TRUSTED_DATA_PATH, "deforestation.csv")
TR_PAST_SCORES = os.path.join(TRUSTED_DATA_PATH, "past_scores.csv")
TR_FRAMES = os.path.join(TRUSTED_DATA_PATH, "grid")
TR_FRAMES_IDX = os.path.join(TRUSTED_DATA_PATH, "frames_idx.csv")
TR_COUNTIES = os.path.join(TRUSTED_DATA_PATH, "counties.csv")
TR_COUNTIES_DEFOR = os.path.join(TRUSTED_DATA_PATH, "counties_defor.csv")
TR_RAIN_AVG = os.path.join(TRUSTED_DATA_PATH, "avg_precipitation.csv")
TR_TPI = os.path.join(TRUSTED_DATA_PATH, "tpi.csv")
TR_LANDCOVER = os.path.join(TRUSTED_DATA_PATH, "landcover.csv")
TR_NIGHT_LIGHT = os.path.join(TRUSTED_DATA_PATH, "night_light.csv")
TR_SENTINEL1 = os.path.join(TRUSTED_DATA_PATH, "sentinel1.csv")

TR_FOREST = os.path.join(TRUSTED_DATA_PATH, "forest.csv")
TR_DISTANCES = os.path.join(TRUSTED_DATA_PATH, "distances.csv")
TR_CLOUDS = os.path.join(TRUSTED_DATA_PATH, "clouds.csv")
TR_IBAMA = os.path.join(TRUSTED_DATA_PATH, "IBAMA.csv")
TR_PAST_DEFOR = os.path.join(TRUSTED_DATA_PATH, "past_defor.csv")

# temporal limits
DT_INIT = "2019-01-21"
DT_FIM = "2023-12-31"
# TIME_STEPS = list(pd.date_range("2017-01-01", "2023-08-01", freq="MS"))
# TIME_STEPS = TIME_STEPS + [dt.replace(day=16) for dt in TIME_STEPS]
# TIME_STEPS.sort()

# TIME_STEPS = []
# for ms in pd.date_range(DT_INIT, DT_FIM, freq="MS"):
#     TIME_STEPS.append(ms.date())
#     TIME_STEPS.append(ms.replace(day=16).date())
# TIME_STEPS = pd.DatetimeIndex(TIME_STEPS)
# TIME_STEPS = pd.date_range(
#     DT_INIT,
#     DT_FIM,
#     freq="QS",
# )
TIME_STEPS = pd.date_range(DT_INIT, DT_FIM, freq="7D")

# spatial and input size parameters
BOX_SIDE = 0.01
INPUT_BOXES_SIZE = 32

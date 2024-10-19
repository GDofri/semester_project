

import utils
import streak_cut
import numpy as np
import astropy.io.fits as fits
import os
from tqdm import tqdm
# def create_dataset_strips(height: int):
#     path = os.path.join(utils.get_project_root(), "datasets", "strips_181024")
#     data_csv = utils.read_streaks_csv()
#     # load all the images into memory
#     labels = np.zeros(len(data_csv), dtype=np.float32)
#     # # Images are (4200, 2144)
#     # max_length = np.sqrt(4200 ** 2 + 2144 ** 2).round().astype(int)
#     auxiliary_data = np.zeros((len(data_csv), 3), dtype=np.float32)
#     # images = np.zeros((len(data_csv), height, max_length), dtype=np.float32)
#     # Sort the data by file_name
#     data_csv = data_csv.sort_values(by="file_name")
#     # Go through the data, only open the fits files once
#
#     rows = data_csv.iterrows()
#     next_row = next(rows)
#
#
#
#     for(idx, streak) in list(data_csv.iterrows())[:10]:
#         streakName = streak["file_name"]
#         extension = streak["extension"]
#         with fits.open(utils.get_fits_path(streakName), memmap=False) as hdul:
#             img = hdul[extension].data
#             streak_x_start = streak["x_start[px]"]
#             streak_y_start = streak["y_start[px]"]
#             streak_x_end = streak["x_end[px]"]
#             streak_y_end = streak["y_end[px]"]
#             img = streak_cut.cut_around_line(img, (streak_x_start, streak_y_start), (streak_x_end, streak_y_end), height)
#             # images[idx] = img


def create_dataset_strips(height: int):
    path = os.path.join(utils.get_project_root(), "datasets", "strips_181024")
    data_csv = utils.read_streaks_csv()
    # load all the images into memory
    # Sort the data by file_name
    data_csv = data_csv.sort_values(by="file_name")
    # Go through the data, only open the fits files once
    file_names = data_csv["file_name"].unique()

    pbar = tqdm(total=len(file_names), desc="Creating strips")
    for file_name in file_names:
        # Load the fits file
        with fits.open(utils.get_fits_path(file_name)) as hdul:
            for idx, row in data_csv[data_csv["file_name"] == file_name].iterrows():
                streak_x_start = row["x_start[px]"]
                streak_y_start = row["y_start[px]"]
                streak_x_end = row["x_end[px]"]
                streak_y_end = row["y_end[px]"]
                img = hdul[row["extension"]].data
                strip = streak_cut.cut_around_line(img, (streak_x_start, streak_y_start), (streak_x_end, streak_y_end), height)
                np.save(os.path.join(path, f"{row['file_name']}_strip_{row['extension']}_{row['ID']}.npy"), strip)
            pbar.update(1)

def create_auxiliary_data_csv():

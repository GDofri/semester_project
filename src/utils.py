import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import download_file
from matplotlib.colors import LogNorm
from matplotlib import colormaps

import pandas as pd
import cv2
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import os

# Demo from https://learn.astropy.org/tutorials/FITS-images.html

def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__ + '/..'))

def get_cell_order():
    print("Warning: This cell order should be updated with a new one, refer to the OmegaCam manual.")
    cell_order = np.array([
        32, 31, 30, 29, 16, 15, 14, 13,
        28, 27, 26, 25, 12, 11, 10, 9,
        24, 23, 22, 21, 8, 7, 6, 5,
        20, 19, 18, 17, 4, 3, 2, 1
    ])
    return cell_order


def astropy_demo():
    image_file = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits', cache=True)
    hdu_list = fits.open(image_file)
    print(hdu_list.info())
    image_data = hdu_list[0].data
    # or just do this from the get-go
    # image_data = fits.getdata(image_file)
    print(type(image_data))
    print(image_data.shape)
    hdu_list.close()
    plt.imshow(image_data, cmap='gray')
    plt.colorbar()
    plt.show()


def read_our_fits():
    # with fits.open('data/2022-01-01/r_SDSS/OMEGA.2022-01-02T00:59:16.697.astred.cal.fits') as hdu_list:
    with fits.open('../data/2022-01-01/r_SDSS/OMEGA.2022-01-02T00:35:59.042.astred.cal.fits') as hdu_list:
        # hdu_list = fits.open('./2022-01-01/r_SDSS/OMEGA.2022-01-02T00:59:16.697.astred.cal.fits')
        # print(hdu_list.info())
        print("Loading image data")
        images = ([hdu_list[i].data for i in range(1, len(hdu_list))])

    # or just do this from the get-go
    # image_data = fits.getdata(image_file)
    # print(type(image_data))
    # print(image_data.shape)
    # print(image_data.dtype)
    # # print(type(hdu_list[1].header))
    # for head in   hdu_list[1].header:
    #     print(head + ":", hdu_list[1].header[head])
    # hdu_list.close()
    fig, axes = plt.subplots(4, 8, figsize=(8, 8))
    print("Enumerating images")
    for i, ax in enumerate(axes.flatten()):  #'F')):
        ax.text(0.15, 0.9, str(i + 1), color='red', fontsize=12, ha='center', va='center', transform=ax.transAxes)
        ax.imshow(images[i], cmap='viridis', norm=LogNorm())
        ax.set_xticks([])
        ax.set_yticks([])
    # plt.colorbar()
    print("Adjusting plt settings")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    print("Showing")
    plt.show()


def read_streaks_csv():
    streaks = pd.read_csv(os.path.join(get_project_root(), "good_data.csv"))

    return streaks


def file_exists(file_path: str) -> bool:
    try:
        with open(file_path):
            pass
    except IOError:
        return False
    return True


def get_fits_path(streakName: str, external_drive: bool = False, drive: str = "MedTina") -> str:
    # Parse date from streakName correcting for
    # the fact that the directory is one day behind
    date = streakName[6:16]
    date = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)
    date = date.strftime("%Y-%m-%d") + "/"
    root = os.path.join(get_project_root(), "data", "VST_BUFFER", "2022-01")
    if external_drive:
        # root = "/media/dofri/" + drive + "/VST_BUFFER/2022-01/"
        root = os.path.join("/media/dofri/", drive, "VST_BUFFER/2022-01/")
    path = os.path.join(root, date, "L2_REDUCTION", "r_SDSS", streakName[:-5] + ".astred.cal.fits")
    # check if the file exists
    if not file_exists(path):
        path = os.path.join(root, date, "L2_REDUCTION", "r_SDSS", streakName)
    if not file_exists(path):
        raise FileNotFoundError("File not found: " + path)
    return path

def get_streaks_path(streakName: str):
    date = streakName[6:16]
    date = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)
    date = date.strftime("%Y-%m-%d") + "/"
    root = os.path.join(get_project_root(), "data", "VST_BUFFER", "2022-01")
    path = os.path.join(root, date, "L2_REDUCTION", "r_SDSS", streakName[:-5] + ".astred.cal.fits")
    # check if the file exists
    if not file_exists(path):
        path = os.path.join(root, date, "L2_REDUCTION", "r_SDSS", streakName)
    if not file_exists(path):
        raise FileNotFoundError("File not found: " + path)
    path += ".streaks"
    if not file_exists(path):
        raise FileNotFoundError("File not found: " + path)
    return path

def get_lc_path(streak_name: str, extension: int, id: int):
    date = streak_name[6:16]
    date = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)
    date = date.strftime("%Y-%m-%d") + "/"
    root = os.path.join(get_project_root(), "data", "VST_BUFFER", "2022-01")
    name = streak_name[:-5]
    lc_file_name = f"{name}.{extension}.{id}.lc"

    path = os.path.join(root, date, "L2_REDUCTION", "r_SDSS", lc_file_name)
    return path
def show_streak_with_endpoints(streak_name: str, extension: int, id: int, external_drive: bool = False):
    streaks_csv = read_streaks_csv()
    print(streaks_csv[(streaks_csv["file_name"] == streak_name)])

    streak = streaks_csv[(streaks_csv["file_name"] == streak_name) &
                         (streaks_csv["extension"] == extension) &
                         (streaks_csv["ID"] == id)].iloc[0]

    streak_x_start = streak["x_start[px]"].astype(np.int32)
    streak_y_start = streak["y_start[px]"].astype(np.int32)
    streak_x_end = streak["x_end[px]"].astype(np.int32)
    streak_y_end = streak["y_end[px]"].astype(np.int32)
    print(f"Start: ({streak_x_start}, {streak_y_start}), End: ({streak_x_end}, {streak_y_end})")

    with fits.open(get_fits_path(streak_name, external_drive)) as hdu_list:
        img = hdu_list[extension].data[streak_y_start:streak_y_end, streak_x_start:streak_x_end]
        plt.imshow(img, cmap='gray', norm=LogNorm())
        # Plot the two endpoints with points
        plt.plot([0, streak_x_end - streak_x_start], [0, streak_y_end - streak_y_start], 'r.')

        plt.show()


def highlight_streak(streakName: str, id: int):
    """
    :param streakName: Name of the streak as it appears in the csv file
    :param id: 1 indexed id of the streak for given streakName
    :return:
    """

    streaks_csv: pd.DataFrame = read_streaks_csv()
    streaks_csv = streaks_csv[(streaks_csv["file_name"] == streakName) & (streaks_csv["ID"] == id)]
    streak_ext_id = streaks_csv["extension"].values[0]
    streak_x_start = streaks_csv["x_start[px]"].astype(int).values[0]
    streak_y_start = streaks_csv["y_start[px]"].astype(int).values[0]
    streak_x_end = streaks_csv["x_end[px]"].astype(int).values[0]
    streak_y_end = streaks_csv["y_end[px]"].astype(int).values[0]

    img = fitsio.read(get_fits_path(streakName), ext=streak_ext_id)
    # print histogram of image data
    data_sample = np.random.choice(img.flatten(), size=10000, replace=False)
    plt.hist(data_sample, bins=256, log=True)
    plt.show()

    normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # draw streak as a line on image
    highlighted_img = normalized_img.copy()
    cv2.line(highlighted_img, (streak_x_start, streak_y_start), (streak_x_end, streak_y_end), (255, 0, 0), 2)
    # display  images in a subplot
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[1].imshow(normalized_img, cmap='gray')
    axes[1].set_title("Highlighted streak")
    axes[1].plot([streak_x_start, streak_x_end], [streak_y_start, streak_y_end], color='red', linewidth=1)
    axes[2].imshow(img, cmap='gray', norm=LogNorm())
    axes[2].set_title("Log scaled image")
    plt.show()


def highlight_fits(streak_name: str, save: bool = False, show: bool = True):
    scale = 10
    streaks_csv: pd.DataFrame = read_streaks_csv()
    streaks_csv = streaks_csv[(streaks_csv["file_name"] == streak_name)]
    fits = fitsio.FITS(get_fits_path(streak_name))

    fig, axes = plt.subplots(4, 8, figsize=(8, 8))
    flat_axes = axes.flatten()

    norads = streaks_csv["norad_number"].unique()
    colors = colormaps['tab10'].resampled(len(norads))

    norad_color_map = {norad: colors(i) for i, norad in enumerate(norads)}

    # Set image data to each subplot and add the extension number
    for idx, ax in enumerate(flat_axes):
        extension_id = get_cell_order()[idx]
        img = fitsio.read(get_fits_path(streak_name), ext=extension_id).astype(np.float32)
        # if img is None:
        #     raise ValueError(f"Failed to load image for extension {extension_id}")
        # if not isinstance(img, np.ndarray):
        #     raise TypeError("The image is not a valid NumPy array")
        # if len(img.shape) < 2:
        #     raise ValueError("The image does not have enough dimensions to be resized")
        # if img.shape[0] < 2 or img.shape[1] < 2:
        #     raise ValueError("The image is too small to be resized")
        #print(img.dtype)
        img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_LINEAR)

        # mirror the image along the y axis
        img = np.flip(img)
        ax.imshow(img, cmap='gray', norm=LogNorm())
        ax.text(0.15, 0.9, str(extension_id), color='red', fontsize=12, ha='center', va='center',
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    if save:
        plt.savefig("data/figures/" + streak_name + ".png", transparent=False)
    # Highlight streaks on each subplot
    for subplot_idx in range(32):
        extension_id = get_cell_order()[subplot_idx]
        streaks_in_cell = streaks_csv[streaks_csv["extension"] == extension_id]
        if streaks_in_cell.empty:
            continue
        streaks_in_cell = streaks_in_cell.astype({
            'x_start[px]': int,
            'y_start[px]': int,
            'x_end[px]': int,
            'y_end[px]': int
        })
        img_height, img_width = fits[extension_id].get_info()['dims']
        # Loop over the filtered DataFrame
        for index, row in streaks_in_cell.iterrows():
            streak_x_start = (img_width - row['x_start[px]']) // scale
            streak_y_start = (img_height - row['y_start[px]']) // scale
            streak_x_end = (img_width - row['x_end[px]']) // scale
            streak_y_end = (img_height - row['y_end[px]']) // scale
            norad = row['norad_number']
            flat_axes[subplot_idx].plot([streak_x_start, streak_x_end], [streak_y_start, streak_y_end],
                                        color=norad_color_map[norad], linewidth=2)

    # Add legend
    handles = [plt.Line2D([0], [0], color=norad_color_map[norad], label=norad) for norad in norads]
    plt.legend(handles=handles, title="Norad number", loc='upper right')

    if save:
        plt.savefig("data/figures/" + streak_name + ".highlighted.png", transparent=False)
    if show:
        plt.show()

    #normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def create_correleted_streaks_csv():
    streaks_csv: pd.DataFrame = read_streaks_csv()
    correlated_streaks = streaks_csv[streaks_csv["JD_start[UTC]"].notna()]

    # check if there are any nan in correlated_streaks. If so print them
    print(correlated_streaks[correlated_streaks.isna().any(axis=1)])
    print(len(correlated_streaks))

    # save the correlated streaks to a new csv file
    correlated_streaks.to_csv("data/r_SDSS.correlated_streaks.csv", index=False)


def read_fits_header(streak_name: str):
    fits = fitsio.FITS(get_fits_path(streak_name))
    for i in range(1, len(fits)):
        print(fits[i].read_header())
    fits.close()

def get_strip_file_name(row: pd.Series):
    return f"{row['file_name']}_strip_{row['extension']}_{row['ID']}.npy"

def get_strip_file_name_from_items(file_name: str, extension: str, id: str):
    return f"{file_name}_strip_{extension}_{id}.npy"

def get_strip_file_path_from_items(file_name: str, extension: str, id: str):
    filename = f"{file_name}_strip_{extension}_{id}.npy"
    return os.path.join(get_project_root(), 'datasets', 'strips_181024', filename)

def get_strip_file_path(row: pd.Series):
    filename = f"{row['file_name']}_strip_{row['extension']}_{row['ID']}.npy"
    return os.path.join(get_project_root(), 'datasets', 'strips_181024', filename)



def main():
    pass
    # show_32_highlighted("OMEGA.2022-01-02T00:35:59.042.fits")
    # read_fits_header("OMEGA.2022-01-02T00:35:59.042.fits")
    # show_32_highlighted("OMEGA.2022-01-03T02:20:08.139.fits")

    # highlight_fits("OMEGA.2022-01-02T00:59:16.697.fits", save=True, show=False)

    # streaks_csv = read_streaks_csv()
    # fits_names = streaks_csv["file_name"].unique()
    # for name in tqdm(fits_names):
    #     tqdm.write(" " + name)
    #     try:
    #         highlight_fits(name, save=True, show=False)
    #     except FileNotFoundError as e:
    #         break


if __name__ == '__main__':
    main()

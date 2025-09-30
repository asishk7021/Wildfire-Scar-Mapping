import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    examples = [(2017, 40), (2017, 49), (2017, 53), (2017, 86),
                (2018, 7), (2018, 10), (2018, 14),
                (2019, 11), (2019, 15), (2019, 24), (2019, 35), (2019, 41), (2019, 44), (2019, 50), (2019, 74),
                (2020, 35), (2020, 42), (2020, 47), (2020, 63), 
                (2021, 4), (2021, 8), (2021, 14), (2021, 17), (2021, 45), (2021, 48), (2021, 50), (2021, 59), (2021, 62), (2021, 65), (2021, 69), (2021, 71), (2021, 75), (2021, 78), (2021, 84), (2021, 89), (2021, 93), (2021, 100), (2021, 108), (2021, 110), (2021, 130), (2021, 134) ]
    for year, event in tqdm(examples):
        year = str(year)
        event = str(event)
        floga_path = Path(f'/mnt/FLOGA/data/dataset/FLOGA_dataset_{year}_sen2_60_mod_500.h5')
        hdf = h5py.File(floga_path, 'r')
        # for event in tqdm(hdf[year].keys()):
        make_plot(hdf, year, event)

def make_plot(hdf, year, event, bands='nrg'):
    SAVE_DIR = Path(f"/mnt/FLOGA/data/data_visualization_examples/{year}/{bands}/")
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    # Define custom colormap for the labels
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', [(0, 0, 0, 10), (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0), (0.8647058823529412, 0.30980392156862746, 0.45882352941176474, 1.0)], 3)
    cmap_sea = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap clouds', [(0, 0, 0, 1.0), (1.0, 1.0, 1.0, 1.0)], 2)
    cmap_clc = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap clc', 
        [
            (0, 0, 0, 1.0),  # NODATA
            (0.902, 0.0, 0.302, 1.0),  # Continuous urban fabric
            (1.0, 0.0, 0.0, 1.0),  # Discontinuous urban fabric
            (0.8, 0.302, 0.949, 1.0),  # Industrial or commercial units
            (0.8, 0.0, 0.0, 1.0),  # Road and rail networks and associated land
            (0.902, 0.8, 0.8, 1.0),  # Port areas
            (0.901, 0.8, 0.902, 1.0),  # Airports
            (0.651, 0.0, 0.8, 1.0),  # Mineral extraction sites
            (0.651, 0.302, 1.0, 1.0),  # Dump sites
            (1.0, 0.302, 1.0, 1.0),  # Construction sites
            (1.0, 0.651, 1.0, 1.0),  # Green urban areas
            (1.0, 0.902, 1.0, 1.0),  # Sport and leisure facilities
            (1.0, 1.0, 0.659, 1.0),  # Non-irrigated arable land
            (1.0, 1.0, 0.0, 1.0),  # Permanently irrigated land
            (0.902, 0.902, 0.0, 1.0),  # Rice fields
            (0.902, 0.502, 0.0, 1.0),  # Vineyards
            (0.949, 0.651, 0.302, 1.0),  # Fruit trees and berry plantations
            (0.902, 0.651, 0.0, 1.0),  # Olive groves
            (0.902, 0.902, 0.302, 1.0),  # Pastures
            (1.0, 0.902, 0.651, 1.0),  # Annual crops associated with permanent crops
            (1.0, 0.902, 0.302, 1.0),  # Complex cultivation patterns
            (0.902, 0.8, 0.302, 1.0),  # Land principally occupied by agriculture with significant areas of natural vegetation
            (0.949, 0.8, 0.651, 1.0),  # Agro-forestry areas
            (0.502, 1.0, 0.0, 1.0),  # Broad-leaved forest
            (0.0, 0.651, 0.0, 1.0),  # Coniferous forest
            (0.302, 1.0, 0.0, 1.0),  # Mixed forest
            (0.8, 0.949, 0.302, 1.0),  # Natural grasslands
            (0.651, 1.0, 0.502, 1.0),  # Moors and heathland
            (0.651, 0.902, 0.302, 1.0),  # Sclerophyllous vegetation
            (0.651, 0.949, 0.0, 1.0),  # Transitional woodland-shrub
            (0.902, 0.902, 0.902, 1.0),  # Beaches dunes sands
            (0.8, 0.8, 0.8, 1.0),  # Bare rocks
            (0.8, 1.0, 0.8, 1.0),  # Sparsely vegetated areas
            (0.0, 0.0, 0.0, 1.0),  # Burnt areas
            (0.651, 0.902, 0.8, 1.0),  # Glaciers and perpetual snow
            (0.651, 0.651, 1.0, 1.0),  # Inland marshes
            (0.302, 0.302, 1.0, 1.0),  # Peat bogs
            (0.8, 0.8, 1.0, 1.0),  # Salt marshes
            (0.902, 0.902, 1.0, 1.0),  # Salines
            (0.651, 0.651, 0.902, 1.0),  # Intertidal flats
            (0.0, 0.8, 0.949, 1.0),  # Water courses
            (0.0, 1.0, 0.651, 1.0),  # Coastal lagoons
            (0.651, 1.0, 0.902, 1.0),  # Estuaries
            (0.902, 0.949, 1.0, 1.0),  # Sea and ocean
        ],
        44
    )

    if bands == 'nrg':
        # Get band indices for R, G, B
        sen2_plot_bands = [3, 2, 1]
        mod_plot_bands = [0, 3, 2]
    else:
        # Get band indices for NIR, R, G
        sen2_plot_bands = [10, 3, 2]
        mod_plot_bands = [1, 0, 3]

    fig, ax = plt.subplots(2, 4, figsize=(20, 11))

    # MODIS pre-fire image
    img = hdf[year][event]['mod_500_pre'][:][mod_plot_bands, ...]
    img = scale_image(img)
    img = np.moveaxis(img, 0, -1)
    ax[0, 0].imshow(img)
    ax[0, 0].set_title('MODIS pre-fire')

    # MODIS post-fire image
    img = hdf[year][event]['mod_500_post'][:][mod_plot_bands, ...]
    img = scale_image(img)
    img = np.moveaxis(img, 0, -1)
    ax[0, 1].imshow(img)
    ax[0, 1].set_title('MODIS post-fire')

    # Sentinel-2 pre-fire image
    img = hdf[year][event]['sen2_60_pre'][:][sen2_plot_bands, ...]
    img = scale_image(img)
    img = np.moveaxis(img, 0, -1)
    ax[0, 2].imshow(img * 7)
    ax[0, 2].set_title('Sentinel-2 pre-fire')

    # Sentinel-2 post-fire image
    img = hdf[year][event]['sen2_60_post'][:][sen2_plot_bands, ...]
    img = scale_image(img)
    img = np.moveaxis(img, 0, -1)
    ax[0, 3].imshow(img * 7)
    ax[0, 3].set_title('Sentinel-2 post-fire')

    # CLC mask
    img = hdf[year][event]['clc_100_mask'][:]
    img[(img == 48) | (img == 128)] = 0  # NODATA
    img = np.moveaxis(img, 0, -1)
    ax[1, 0].imshow(img, vmin=0, vmax=43, cmap=cmap_clc)
    ax[1, 0].set_title('CLC mask')

    # Sea mask
    img = hdf[year][event]['sea_mask'][:]
    ax[1, 1].imshow(img.squeeze(), vmin=0, vmax=1, cmap=cmap_sea)
    ax[1, 1].set_title('Sea mask')

    # Label
    img = hdf[year][event]['label'][:]
    ax[1, 2].imshow(img.squeeze(), vmin=0, vmax=2, cmap=cmap)
    ax[1, 2].set_title('Label')

    # Remove axes and ticks
    for i in range(2):
        for j in range(4):
            # Remove all axis labels
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['bottom'].set_visible(False)
            ax[i, j].spines['left'].set_visible(False)

    plt.subplots_adjust(wspace=0.05, hspace=0.01)
    plt.savefig(SAVE_DIR / f"{event}.png", dpi=500)

def scale_image(img):
    img = img.astype(np.float32)
    return img / img.max()

if __name__ == '__main__':
    main()
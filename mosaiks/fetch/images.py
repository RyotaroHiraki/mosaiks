import logging
import math
import warnings
from typing import List

import geopandas as gpd
import numpy as np
import pyproj
import stackstac
import torch
from pystac.item import Item
from torch.utils.data import DataLoader, Dataset

from mosaiks.fetch.stacs import fetch_stac_item_from_id

# temp warning suppress for stackstac 0.4.4's  UserWarning @ stackstac/prepare.py:364
# "pd.to_datetime() - UserWarning: The argument 'infer_datetime_format'"
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = [
    "create_data_loader",
    "fetch_image_crop_from_stac_id",
    "fetch_image_crop",
]


def fetch_image_crop(
    lon: float,
    lat: float,
    stac_items: list[Item],
    image_width: int,
    bands: List[str],
    resolution: int,
    dtype: str = "float64",
    image_composite_method: str = "least_cloudy",
    normalise: bool = True,
) -> np.array:
    """
    Fetches a crop of satellite imagery referenced by the given STAC item(s),
    centered around the given point and with the given image_width.
    """

    size = (
        len(bands),
        math.ceil(image_width / resolution + 1),
        math.ceil(image_width / resolution + 1),
    )

    if stac_items is None:
        return np.ones(size) * np.nan

    assert isinstance(stac_items, list)

    # ③ None を除外
    stac_items_not_none = [item for item in stac_items if item is not None]
    if len(stac_items_not_none) == 0:
        return np.ones(size) * np.nan

    def get_item_crs(item):
        crs = item.properties.get("proj:epsg")
        if crs is None:
            code = item.properties.get("proj:code")
            if isinstance(code, str) and code.upper().startswith("EPSG:"):
                crs = int(code.split(":")[1])
        return crs or 3857

    def get_bounds(item):
        crs = get_item_crs(item)
        transformer = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
        buffer_distance = (image_width // 2) * resolution
        x_utm, y_utm = transformer.transform(lon, lat)
        x_min, x_max = x_utm - buffer_distance, x_utm + buffer_distance
        y_min, y_max = y_utm - buffer_distance, y_utm + buffer_distance
        return crs, [x_min, y_min, x_max, y_max]

    if image_composite_method == "all":
        first = stac_items_not_none[0]
        crs, bounds = get_bounds(first)

        try:
            xarray = stackstac.stack(
                stac_items_not_none,
                assets=bands,
                resolution=resolution,
                rescale=False,
                dtype="float64",
                epsg=crs,
                bounds=bounds,
                fill_value=np.nan,
            )
            image = xarray.median(dim="time").values
        except Exception as e:
            logging.warning(f"Failed composite for point {lon}, {lat}: {e}")
            return np.ones(size) * np.nan

    elif image_composite_method == "least_cloudy":
        for i, item in enumerate(stac_items_not_none):
            crs, bounds = get_bounds(item)

            try:
                xarray = stackstac.stack(
                    item,
                    assets=bands,
                    resolution=resolution,
                    rescale=False,
                    dtype="float64",
                    epsg=crs,
                    bounds=bounds,
                    fill_value=np.nan,
                )
                image = xarray.values

                if len(image.shape) > 3:
                    image = image.squeeze(0)

                if not np.all(np.isnan(image)):
                    break

            except Exception as e:
                logging.warning(
                    f"Skipping item {i} for point {lon}, {lat}: {e}"
                )
                continue
        else:
            logging.warning(f"All images in the stack failed for point {lon}, {lat}")
            return np.ones(size) * np.nan

    else:
        raise ValueError(
            f"image_composite_method must be 'least_cloudy' or 'all', "
            f"not {image_composite_method}"
        )

    if normalise:
        image = _minmax_normalize_image(image)

    return image


def _minmax_normalize_image(image: np.array) -> np.array:
    img_min, img_max = image.min(), image.max()
    return (image - img_min) / (img_max - img_min)


def create_data_loader(
    points_gdf_with_stac: gpd.GeoDataFrame,
    image_bands: List[str],
    image_resolution: int,
    image_dtype: str,
    image_width: int,
    image_composite_method: str = "least_cloudy",
) -> DataLoader:
    """
    Creates a PyTorch DataLoader which returns cropped images based on the given
    coordinate points and associated STAC image references.

    Parameters
    ----------
    points_gdf_with_stac : A GeoDataFrame with the points we want to fetch imagery for
        alongside the STAC item references to pictures for each point. The stac_item
        entry for each row should be a list of stac items that cover that point.
    image_bands : The bands to use for the image crops
    image_resolution : The resolution to use for the image crops
    image_dtype : The data type to use for the image crops
    image_width : Desired width of the image to be fetched (in meters).
    image_composite_method : The type of composite to make if multiple images are given.
        Defaults to "least_cloudy".

    Returns
    -------
    data_loader : A PyTorch DataLoader which returns cropped images as tensors
    """

    dataset = CustomDataset(
        latitudes=points_gdf_with_stac["Lat"].to_numpy(),
        longitudes=points_gdf_with_stac["Lon"].to_numpy(),
        stac_items=points_gdf_with_stac["stac_item"].tolist(),
        image_width=image_width,
        bands=image_bands,
        resolution=image_resolution,
        dtype=image_dtype,
        image_composite_method=image_composite_method,
    )
    data_loader = DataLoader(dataset, batch_size=None)

    return data_loader


class CustomDataset(Dataset):
    def __init__(
        self,
        latitudes: np.array,
        longitudes: np.array,
        stac_items: List[Item],
        image_width: int,
        bands: List[str],
        resolution: int,
        dtype: str = "float64",
        image_composite_method: str = "least_cloudy",
    ) -> None:
        """
        Parameters
        ----------
        latitudes : Array of latitudes to sample from
        longitudes : Array of longitudes to sample from
        stac_items : List of STAC items to sample from
        image_width : Desired width of the image to be fetched (in meters).
        bands : List of bands to sample
        resolution : Resolution of the image to sample
        dtype : Data type of the image to sample. Defaults to "int16".
            NOTE - np.uint8 results in loss of signal in the features
            and np.uint16 is not supported by PyTorch.
        image_composite_method : The type of composite to make if multiple images are given.
        """

        self.latitudes = latitudes
        self.longitudes = longitudes
        self.stac_items = stac_items
        self.image_width = image_width
        self.bands = bands
        self.resolution = resolution
        self.dtype = dtype
        self.image_composite_method = image_composite_method

    def __len__(self):
        """Returns the number of points in the dataset"""

        return len(self.latitudes)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Parameters
        ----------
        idx :Index of the point to get imagery for

        Returns
        -------
        out_image : Image tensor of shape (C, H, W)
        """

        lat = self.latitudes[idx]
        lon = self.longitudes[idx]
        stac_items = self.stac_items[idx]

        try:
            image = fetch_image_crop(
                lon=lon,
                lat=lat,
                stac_items=stac_items,
                image_width=self.image_width,
                bands=self.bands,
                resolution=self.resolution,
                dtype=self.dtype,
                image_composite_method=self.image_composite_method,
                normalise=True,
            )
            torch_image = torch.from_numpy(image).float()

            # if all 0s, replace with NaNs
            if torch.all(torch_image == 0):
                torch_image = np.ones_like(torch_image) * np.nan
            return torch_image

        except Exception as e:
            logging.warn(f"Skipping {idx}: {e}")
            return None


# for debugging


def fetch_image_crop_from_stac_id(
    stac_id: str,
    lon: float,
    lat: float,
    image_width: int,
    bands: List[str],
    resolution: int,
    dtype: str = "float16",
    image_composite_method: str = "least_cloudy",
    normalise: bool = True,
    stac_api_name: str = "planetary-compute",
) -> np.array:
    """
    Note: This function is necessary since STAC Items cannot be directly saved to file
    alongside the features in `.parquet` format, and so only their ID can be saved.
    This function uses the ID to first fetch the correct STAC Items, runs these
    through fetch_image_crop(), and optionally displays the image.

    Takes a stac_id (or list of stac_ids), lat, lon, and satellite
    parameters and returns and displays a cropped image. This image is fetched using the
    same process as when fetching for featurization.

    If multiple STAC items are given, the median composite of the images is returned.

    Parameters
    ----------
    stac_id : The STAC ID of the image to fetch
    lon : Longitude of the centerpoint to fetch imagery for
    lat : Latitude of the centerpoint to fetch imagery for
    image_width : Desired width of the image to be fetched (in meters).
    bands : The satellite image bands to fetch
    resolution : The resolution of the image to fetch
    dtype : The data type of the image to fetch. Defaults to "int16".
    image_composite_method : The type of composite to make if multiple images are given.
    normalise : Whether to normalise the image. Defaults to True.
    stac_api_name : The name of the STAC API to use. Defaults to "planetary-compute".

    Returns
    -------
    image_crop : A numpy array of the cropped image
    """

    if not isinstance(stac_id, list):
        stac_id = [stac_id]

    stac_items = fetch_stac_item_from_id(stac_id, stac_api_name)
    if len(stac_items) == 0:
        logging.warn("No STAC items found.")
        return None

    else:
        image_crop = fetch_image_crop(
            lon=lon,
            lat=lat,
            stac_items=stac_items,
            image_width=image_width,
            bands=bands,
            resolution=resolution,
            dtype=dtype,
            image_composite_method=image_composite_method,
            normalise=normalise,
        )

        return image_crop

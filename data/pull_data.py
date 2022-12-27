""" 
This script is to pull sentinel or landsat imagery and store in local disk
This script mostly contains modified code from benchmark blogpost of tick tick bloom
"""


import os
import odc
import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
import planetary_computer as pc
from pystac_client.client import Client
import geopy.distance as distance
import rioxarray

import cv2
import odc.stac

from tqdm import tqdm
from pathlib import Path
from datetime import timedelta


# read metadata
metadata = pd.read_csv('metadata.csv')
metadata.date = pd.to_datetime(metadata.date)

# read trainlabels
trainlabels = pd.read_csv('train_labels.csv')


# make connection with planetary stack
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
)

# get our bounding box to search latitude and longitude coordinates


def get_bounding_box(latitude, longitude, meter_buffer=2_000):
    """
    Given a latitude, longitude, and buffer in meters, returns a bounding
    box around the point with the buffer on the left, right, top, and bottom.

    Returns a list of [minx, miny, maxx, maxy]
    """
    distance_search = distance.distance(meters=meter_buffer)

    # calculate the lat/long bounds based on ground distance
    # bearings are cardinal directions to move (south, west, north, and east)
    min_lat = distance_search.destination(
        (latitude, longitude), bearing=180)[0]
    min_long = distance_search.destination(
        (latitude, longitude), bearing=270)[1]
    max_lat = distance_search.destination((latitude, longitude), bearing=0)[0]
    max_long = distance_search.destination(
        (latitude, longitude), bearing=90)[1]

    return [min_long, min_lat, max_long, max_lat]

# get our date range to search, and format correctly for query
def get_date_range(date, time_buffer_days=15):
    """Get a date range to search for in the planetary computer based
    on a sample's date. The time range will include the sample date
    and time_buffer_days days prior

    Returns a string"""
    datetime_format = "%Y-%m-%dT"
    range_start = pd.to_datetime(date) - timedelta(days=time_buffer_days)
    date_range = f"{range_start.strftime(datetime_format)}/{pd.to_datetime(date).strftime(datetime_format)}"

    return date_range


def crop_sentinel_image(item, bounding_box):
    """
    Given a STAC item from Sentinel-2 and a bounding box tuple in the format
    (minx, miny, maxx, maxy), return a cropped portion of the item's visual
    imagery in the bounding box.

    Returns the image as a numpy array with dimensions (color band, height, width)
    """
    (minx, miny, maxx, maxy) = bounding_box

    image = rioxarray.open_rasterio(pc.sign(item.assets["visual"].href)).rio.clip_box(
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        crs="EPSG:4326",
    )

    return image.to_numpy()


def crop_landsat_image(item, bounding_box):
    """
    Given a STAC item from Landsat and a bounding box tuple in the format
    (minx, miny, maxx, maxy), return a cropped portion of the item's visual
    imagery in the bounding box.

    Returns the image as a numpy array with dimensions (color band, height, width)
    """
    (minx, miny, maxx, maxy) = bounding_box

    image = odc.stac.stac_load(
        [pc.sign(item)], bands=["red", "green", "blue"], bbox=[minx, miny, maxx, maxy]
    ).isel(time=0)
    image_array = image[["red", "green", "blue"]].to_array().to_numpy()

    # normalize to 0 - 255 values
    image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)

    return image_array


def get_sentinel_best_img(item_details):
    # 1 - filter to sentinel
    item_details[item_details.platform.str.contains("Sentinel")]

    # 2 - take closest by date
    sentinel_best_item = (
        item_details[item_details.platform.str.contains("Sentinel")]
        .sort_values(by="datetime", ascending=False)
        .iloc[0]
    )
    item = sentinel_best_item.item_obj

    # get a smaller geographic bounding box
    minx, miny, maxx, maxy = get_bounding_box(
        row.latitude, row.longitude, meter_buffer=2_000
    )

    # get the zoomed in image array
    bbox = (minx, miny, maxx, maxy)
    # needs to be transposed.
    zoomed_img_array = crop_sentinel_image(item, bbox)

    return zoomed_img_array


def get_landsat_best_img(item_details, xarray=True):
    # 1 - filter to landsat
    landsat_item = (
        item_details[item_details.platform.str.contains("landsat")]
        .sample(n=1, random_state=3)
        .iloc[0]
    )
    landsat_item
    item = landsat_item.item_obj

    # we'll use the same cropped area
    landsat_zoomed_arr = crop_landsat_image(
        item, bbox)  # (needs to be transposed)
    return landsat_zoomed_arr


IMAGE_DIR = 'imgs'
errored_ids = []

# sample 1000 rows
metadata_subset = metadata.sample(1_000, random_state=42)

#  for every sample get a bbox and daterange and search for images in the catalog.
for row in tqdm(metadata_subset.itertuples(), total=len(metadata_subset), colour='green'):

    uid = row.uid
    img_arr_path = IMAGE_DIR + '/' + f'{uid}.npy'

    if os.path.exists(img_arr_path):
        # print(f"Image already exists for {uid}")
        pass
    else:
        date_range = get_date_range(row.date)
        bbox = get_bounding_box(row.latitude, row.longitude)

        # search the planetary computer sentinel-l2a and landsat level-2 collections
        search = catalog.search(
            collections=["sentinel-2-l2a", "landsat-c2-l2"], bbox=bbox, datetime=date_range
        )

        # see how many items were returned
        items = [item for item in search.get_all_items()]

        # if len(items) == 0:
        #     print(f"No items found for {uid}")
        #     errored_ids.append(uid)
        #     continue

        # get details of all of the items returned into a dataframe
        item_details = pd.DataFrame(
            [
                {
                    "datetime": item.datetime.strftime("%Y-%m-%d"),
                    "platform": item.properties["platform"],
                    "min_long": item.bbox[0],
                    "max_long": item.bbox[2],
                    "min_lat": item.bbox[1],
                    "max_lat": item.bbox[3],
                    "bbox": item.bbox,
                    "item_obj": item,
                }
                for item in items
            ]
        )

        # check which rows actually contain the sample location
        item_details["contains_sample_point"] = (
            (item_details.min_lat < row.latitude)
            & (item_details.max_lat > row.latitude)
            & (item_details.min_long < row.longitude)
            & (item_details.max_long > row.longitude)
        )

        # print(
        #     f"Filtering from {len(item_details)} returned to {item_details.contains_sample_point.sum()} items that contain the sample location"
        # )

        item_details = item_details[item_details["contains_sample_point"]]
        item_details[["datetime", "platform", "contains_sample_point", "bbox"]].sort_values(
            by="datetime"
        )

        # get the best sentinel and landsat images
        try:
            img_arr = get_landsat_best_img(item_details=item_details)

            img_arr = get_sentinel_best_img(item_details=item_details)
        except:
            # print(f"No items found for {uid}")
            errored_ids.append(uid)

        # save images
        with open(img_arr_path, "wb") as f:
            np.save(f, img_arr)

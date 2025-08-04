import panel as pn
import pandas as pd
import numpy as np

import tkinter as tk
from tkinter import filedialog

from bokeh.models import OpenHead, WMTSTileSource

try:
    from mapbox_token import mapbox_access_token
except:
    mapbox_access_token = None

from bokeh.plotting import figure
from bokeh.models import (
    Slider,
    CheckboxGroup,
    RadioButtonGroup,
    CustomJS,
    ColumnDataSource,
    LinearAxis,
    WheelZoomTool,
    ResetTool,
    PanTool,
    Div,
    Button,
    LinearColorMapper,
    ColorBar,
    HoverTool,
    Arrow,
    VeeHead,
    NormalHead,
)
from bokeh.palettes import brewer, viridis
from bokeh.colors import RGB

pn.extension()
# Suppress copy/slice warning
pd.options.mode.copy_on_write = True


VELOCITY_SCALE = 1000

arrow_head_type = NormalHead  # --▶
# arrow_head_type = VeeHead     #-->
arrow_head_size = 4

if (
    mapbox_access_token == "INSERT_TOKEN_HERE"
    or mapbox_access_token is None
    or mapbox_access_token == ""
):
    has_mapbox_token = False
else:
    has_mapbox_token = True

# Define some basic coordinate transformation functions
KM2M = 1.0e3
RADIUS_EARTH = 6371000


def sph2cart(lon, lat, radius):
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z


def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def wrap2360(lon):
    lon[np.where(lon < 0.0)] += 360.0
    return lon


def calculate_fault_bottom_edge(lon1, lat1, lon2, lat2, depth_km, dip_degrees):
    """
    Calculate the approximate longitude and latitude coordinates of the bottom edge
    of a fault plane.

    Parameters:
    -----------
    lon1, lat1 : float
        Longitude and latitude of the western-most fault endpoint (degrees)
    lon2, lat2 : float
        Longitude and latitude of the eastern-most fault endpoint (degrees)
    depth_km : float
        Depth of the fault plane in kilometers
    dip_degrees : float
        Dip angle of the fault in degrees (0-90, where 90 is vertical)

    Returns:
    --------
    tuple: (lon1_bottom, lat1_bottom, lon2_bottom, lat2_bottom)
        Bottom edge coordinates in longitude/latitude
    """

    # Convert angles to radians
    dip_rad = np.radians(dip_degrees)
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)

    # Earth radius in kilometers
    R = 6371.0

    # For a vertical fault (90 degrees), bottom coordinates are the same as top
    if np.abs(dip_degrees - 90.0) < 1e-6:
        return lon1, lat1, lon2, lat2

    # Calculate the strike direction (along the fault trace)
    # This is the bearing from point 1 to point 2
    delta_lon = lon2_rad - lon1_rad

    # Calculate bearing using spherical trigonometry
    y = np.sin(delta_lon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
        lat2_rad
    ) * np.cos(delta_lon)
    strike_bearing = np.arctan2(y, x)

    # The dip direction is perpendicular to strike (90 degrees clockwise from strike)
    dip_direction = strike_bearing + np.pi / 2

    # Calculate horizontal distance the fault extends due to dip
    # horizontal_distance = depth / tan(dip)
    horizontal_distance_km = depth_km / np.tan(dip_rad)

    # Convert horizontal distance to angular distance on Earth's surface
    angular_distance = horizontal_distance_km / R

    # Calculate the bottom edge coordinates for both endpoints
    # Point 1 bottom coordinates
    lat1_bottom_rad = np.arcsin(
        np.sin(lat1_rad) * np.cos(angular_distance)
        + np.cos(lat1_rad) * np.sin(angular_distance) * np.cos(dip_direction)
    )

    lon1_bottom_rad = lon1_rad + np.arctan2(
        np.sin(dip_direction) * np.sin(angular_distance) * np.cos(lat1_rad),
        np.cos(angular_distance) - np.sin(lat1_rad) * np.sin(lat1_bottom_rad),
    )

    # Point 2 bottom coordinates
    lat2_bottom_rad = np.arcsin(
        np.sin(lat2_rad) * np.cos(angular_distance)
        + np.cos(lat2_rad) * np.sin(angular_distance) * np.cos(dip_direction)
    )

    lon2_bottom_rad = lon2_rad + np.arctan2(
        np.sin(dip_direction) * np.sin(angular_distance) * np.cos(lat2_rad),
        np.cos(angular_distance) - np.sin(lat2_rad) * np.sin(lat2_bottom_rad),
    )

    # Convert back to degrees
    lon1_bottom = np.degrees(lon1_bottom_rad)
    lat1_bottom = np.degrees(lat1_bottom_rad)
    lon2_bottom = np.degrees(lon2_bottom_rad)
    lat2_bottom = np.degrees(lat2_bottom_rad)

    return lon1_bottom, lat1_bottom, lon2_bottom, lat2_bottom


def wgs84_to_web_mercator(lon, lat):
    # Converts decimal (longitude, latitude) to Web Mercator (x, y)
    EARTH_RADIUS = 6378137.0  # Earth's radius (m)
    x = EARTH_RADIUS * np.deg2rad(lon)
    y = EARTH_RADIUS * np.log(np.tan((np.pi / 4.0 + np.deg2rad(lat) / 2.0)))
    return x, y


##################################
# Declare empty ColumnDataStores #
##################################

# Source for stations. Dict of length n_sta
stasource_1 = ColumnDataSource(
    data={
        "lon_1": [],
        "lat_1": [],
        "obs_east_vel_1": [],
        "obs_north_vel_1": [],
        "obs_east_vel_lon_1": [],
        "obs_north_vel_lat_1": [],
        "mod_east_vel_1": [],
        "mod_north_vel_1": [],
        "mod_east_vel_lon_1": [],
        "mod_north_vel_lat_1": [],
        "res_east_vel_1": [],
        "res_north_vel_1": [],
        "res_east_vel_lon_1": [],
        "res_north_vel_lat_1": [],
        "rot_east_vel_1": [],
        "rot_north_vel_1": [],
        "rot_east_vel_lon_1": [],
        "rot_north_vel_lat_1": [],
        "seg_east_vel_1": [],
        "seg_north_vel_1": [],
        "seg_east_vel_lon_1": [],
        "seg_north_vel_lat_1": [],
        "tde_east_vel_1": [],
        "tde_north_vel_1": [],
        "tde_east_vel_lon_1": [],
        "tde_north_vel_lat_1": [],
        "str_east_vel_1": [],
        "str_north_vel_1": [],
        "str_east_vel_lon_1": [],
        "str_north_vel_lat_1": [],
        "mog_east_vel_1": [],
        "mog_north_vel_1": [],
        "mog_east_vel_lon_1": [],
        "mog_north_vel_lat_1": [],
        "res_mag_1": [],
        "sized_res_mag_1": [],
        "name_1": [],
    }
)

# Source for block bounding segments. Dict of length n_segments
segsource_1 = ColumnDataSource(
    data={
        "xseg": [],
        "yseg": [],
        "ssrate": [],
        "dsrate": [],
        "active_comp": [],
        "name": [],
    },
)

# Source for fault surface projections. Dict of length n_segments
fault_proj_source_1 = ColumnDataSource(
    data={
        "xpoly": [],
        "ypoly": [],
        "dip": [],
        "name": [],
    },
)

# Source for triangular dislocation elements. Dict of length n_tde
tdesource_1 = ColumnDataSource(
    data={
        "xseg": [],
        "yseg": [],
        "ssrate": [],
        "dsrate": [],
        "active_comp": [],
    },
)

# Source for TDE outlines. Dict of length n_tde
tde_perim_source_1 = ColumnDataSource(
    data={
        "xseg": [],
        "yseg": [],
        "proj_col": [],
    },
)

# Make copies for folder 2
stasource_2 = ColumnDataSource(
    data={
        "lon_2": [],
        "lat_2": [],
        "obs_east_vel_2": [],
        "obs_north_vel_2": [],
        "obs_east_vel_lon_2": [],
        "obs_north_vel_lat_2": [],
        "mod_east_vel_2": [],
        "mod_north_vel_2": [],
        "mod_east_vel_lon_2": [],
        "mod_north_vel_lat_2": [],
        "res_east_vel_2": [],
        "res_north_vel_2": [],
        "res_east_vel_lon_2": [],
        "res_north_vel_lat_2": [],
        "rot_east_vel_2": [],
        "rot_north_vel_2": [],
        "rot_east_vel_lon_2": [],
        "rot_north_vel_lat_2": [],
        "seg_east_vel_2": [],
        "seg_north_vel_2": [],
        "seg_east_vel_lon_2": [],
        "seg_north_vel_lat_2": [],
        "tde_east_vel_2": [],
        "tde_north_vel_2": [],
        "tde_east_vel_lon_2": [],
        "tde_north_vel_lat_2": [],
        "str_east_vel_2": [],
        "str_north_vel_2": [],
        "str_east_vel_lon_2": [],
        "str_north_vel_lat_2": [],
        "mog_east_vel_2": [],
        "mog_north_vel_2": [],
        "mog_east_vel_lon_2": [],
        "mog_north_vel_lat_2": [],
        "res_mag_2": [],
        "sized_res_mag_2": [],
        "name_2": [],
    }
)
segsource_2 = ColumnDataSource(segsource_1.data.copy())
fault_proj_source_2 = ColumnDataSource(fault_proj_source_1.data.copy())
tdesource_2 = ColumnDataSource(tdesource_1.data.copy())
tde_perim_source_2 = ColumnDataSource(tde_perim_source_1.data.copy())

# Source for common stations (used in residual comparison). Dict of length n_common_sta
commonsta = ColumnDataSource(
    data={
        "lon_c": [],
        "lat_c": [],
        "res_mag_diff": [],
        "abs_res_mag_diff": [],
        "sized_res_mag_diff": [],
    }
)
# Source for unique stations (used in residual comparison). Dict of length n_unique_sta
uniquesta = ColumnDataSource(
    data={
        "lon_u": [],
        "lat_u": [],
    }
)

################################
# START: Load data from button #
################################

folder_load_button_1 = Button(label="load", button_type="success")
folder_label_1 = Div(text="---")

folder_load_button_2 = Button(label="load", button_type="success")
folder_label_2 = Div(text="---")


def load_data(folder_number):
    # Read data from a local folder
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_name = filedialog.askdirectory(title="load")

    # Set display of folder name
    if folder_number == 1:
        folder_label = folder_label_1
        stasource = stasource_1
        segsource = segsource_1
        fault_proj_source = fault_proj_source_1
        tdesource = tdesource_1
        tde_perim_source = tde_perim_source_1
    else:
        folder_label = folder_label_2
        stasource = stasource_2
        segsource = segsource_2
        fault_proj_source = fault_proj_source_2
        tdesource = tdesource_2
        tde_perim_source = tde_perim_source_2

    folder_label.text = folder_name.split("/")[-1]

    # Read model output as dataframes
    station = pd.read_csv(folder_name + "/model_station.csv")
    resmag = np.sqrt(
        np.power(station.model_east_vel_residual, 2)
        + np.power(station.model_north_vel_residual, 2)
    )
    lon_station = station.lon.values
    lat_station = station.lat.values
    x_station, y_station = wgs84_to_web_mercator(lon_station, lat_station)
    segment = pd.read_csv(folder_name + "/model_segment.csv")

    lon1_seg = segment.lon1.values
    lat1_seg = segment.lat1.values
    lon2_seg = segment.lon2.values
    lat2_seg = segment.lat2.values
    x1_seg, y1_seg = wgs84_to_web_mercator(lon1_seg, lat1_seg)
    x2_seg, y2_seg = wgs84_to_web_mercator(lon2_seg, lat2_seg)

    meshes = pd.read_csv(folder_name + "/model_meshes.csv")
    lon1_mesh = meshes["lon1"]
    lat1_mesh = meshes["lat1"]
    dep1_mesh = meshes["dep1"]
    lon2_mesh = meshes["lon2"]
    lat2_mesh = meshes["lat2"]
    dep2_mesh = meshes["dep2"]
    lon3_mesh = meshes["lon3"]
    lat3_mesh = meshes["lat3"]
    dep3_mesh = meshes["dep3"]
    mesh_idx = meshes["mesh_idx"]

    # Calculate element geometry
    tri_leg1 = np.transpose(
        [
            np.deg2rad(lon2_mesh - lon1_mesh),
            np.deg2rad(lat2_mesh - lat1_mesh),
            (1 + KM2M * dep2_mesh / RADIUS_EARTH)
            - (1 + KM2M * dep1_mesh / RADIUS_EARTH),
        ]
    )
    tri_leg2 = np.transpose(
        [
            np.deg2rad(lon3_mesh - lon1_mesh),
            np.deg2rad(lat3_mesh - lat1_mesh),
            (1 + KM2M * dep3_mesh / RADIUS_EARTH)
            - (1 + KM2M * dep1_mesh / RADIUS_EARTH),
        ]
    )
    norm_vec = np.cross(tri_leg1, tri_leg2)
    azimuth, elevation, r = cart2sph(norm_vec[:, 0], norm_vec[:, 1], norm_vec[:, 2])
    strike = wrap2360(-np.rad2deg(azimuth))
    dip = 90 - np.rad2deg(elevation)

    # Project steeply dipping meshes so they're visible
    mesh_list = np.unique(mesh_idx)
    proj_mesh_flag = np.zeros_like(mesh_list)
    for i in mesh_list:
        this_mesh_els = mesh_idx == i
        this_mesh_dip = np.mean(dip[this_mesh_els])
        if this_mesh_dip > 75:
            proj_mesh_flag[i] = 1
            dip_dir = np.mean(np.deg2rad(strike[this_mesh_els] + 90))
            lon1_mesh[this_mesh_els] += np.sin(dip_dir) * np.rad2deg(
                np.abs(KM2M * dep1_mesh[this_mesh_els] / RADIUS_EARTH)
            )
            lat1_mesh[this_mesh_els] += np.cos(dip_dir) * np.rad2deg(
                np.abs(KM2M * dep1_mesh[this_mesh_els] / RADIUS_EARTH)
            )
            lon2_mesh[this_mesh_els] += np.sin(dip_dir) * np.rad2deg(
                np.abs(KM2M * dep2_mesh[this_mesh_els] / RADIUS_EARTH)
            )
            lat2_mesh[this_mesh_els] += np.cos(dip_dir) * np.rad2deg(
                np.abs(KM2M * dep2_mesh[this_mesh_els] / RADIUS_EARTH)
            )
            lon3_mesh[this_mesh_els] += np.sin(dip_dir) * np.rad2deg(
                np.abs(KM2M * dep3_mesh[this_mesh_els] / RADIUS_EARTH)
            )
            lat3_mesh[this_mesh_els] += np.cos(dip_dir) * np.rad2deg(
                np.abs(KM2M * dep3_mesh[this_mesh_els] / RADIUS_EARTH)
            )
    proj_mesh_idx = np.where(proj_mesh_flag)[0]
    x1_mesh, y1_mesh = wgs84_to_web_mercator(lon1_mesh, lat1_mesh)
    x2_mesh, y2_mesh = wgs84_to_web_mercator(lon2_mesh, lat2_mesh)
    x3_mesh, y3_mesh = wgs84_to_web_mercator(lon3_mesh, lat3_mesh)

    # Determine mesh perimeter
    edge1_lon = np.array((lon1_mesh, lon2_mesh))
    edge1_lat = np.array((lat1_mesh, lat2_mesh))
    edge1_array = np.vstack(
        (np.sort(edge1_lon, axis=0), np.sort(edge1_lat, axis=0), mesh_idx)
    )
    edge2_lon = np.array((lon2_mesh, lon3_mesh))
    edge2_lat = np.array((lat2_mesh, lat3_mesh))
    edge2_array = np.vstack(
        (np.sort(edge2_lon, axis=0), np.sort(edge2_lat, axis=0), mesh_idx)
    )
    edge3_lon = np.array((lon3_mesh, lon1_mesh))
    edge3_lat = np.array((lat3_mesh, lat1_mesh))
    edge3_array = np.vstack(
        (np.sort(edge3_lon, axis=0), np.sort(edge3_lat, axis=0), mesh_idx)
    )
    all_edge_array = np.concatenate((edge1_array, edge2_array, edge3_array), axis=1)

    edge1_array_unsorted = np.vstack((edge1_lon, edge1_lat, mesh_idx))
    edge2_array_unsorted = np.vstack((edge2_lon, edge2_lat, mesh_idx))
    edge3_array_unsorted = np.vstack((edge3_lon, edge3_lat, mesh_idx))
    all_edge_array_unsorted = np.concatenate(
        (edge1_array_unsorted, edge2_array_unsorted, edge3_array_unsorted), axis=1
    )

    # Find unique edges and their counts
    unique_edges, unique_edge_index, edge_count = np.unique(
        all_edge_array, return_index=True, return_counts=True, axis=1
    )
    unique_edges_unsorted = all_edge_array_unsorted[:, unique_edge_index]
    perim_edges = unique_edges_unsorted[:, edge_count == 1]
    proj_mesh_edge_flag = np.isin(perim_edges[-1, :], proj_mesh_idx).astype(int)

    x1_perim_seg, y1_perim_seg = wgs84_to_web_mercator(
        perim_edges[0, :], perim_edges[2, :]
    )
    x2_perim_seg, y2_perim_seg = wgs84_to_web_mercator(
        perim_edges[1, :], perim_edges[3, :]
    )

    suffix = f"_{folder_number}"
    stasource.data = {
        f"lon{suffix}": x_station,
        f"lat{suffix}": y_station,
        f"obs_east_vel{suffix}": station.east_vel,
        f"obs_north_vel{suffix}": station.north_vel,
        f"obs_east_vel_lon{suffix}": x_station + VELOCITY_SCALE * station.east_vel,
        f"obs_north_vel_lat{suffix}": y_station + VELOCITY_SCALE * station.north_vel,
        f"mod_east_vel{suffix}": station.model_east_vel,
        f"mod_north_vel{suffix}": station.model_north_vel,
        f"mod_east_vel_lon{suffix}": x_station
        + VELOCITY_SCALE * station.model_east_vel,
        f"mod_north_vel_lat{suffix}": y_station
        + VELOCITY_SCALE * station.model_north_vel,
        f"res_east_vel{suffix}": station.model_east_vel_residual,
        f"res_north_vel{suffix}": station.model_north_vel_residual,
        f"res_east_vel_lon{suffix}": x_station
        + VELOCITY_SCALE * station.model_east_vel_residual,
        f"res_north_vel_lat{suffix}": y_station
        + VELOCITY_SCALE * station.model_north_vel_residual,
        f"rot_east_vel{suffix}": station.model_east_vel_rotation,
        f"rot_north_vel{suffix}": station.model_north_vel_rotation,
        f"rot_east_vel_lon{suffix}": x_station
        + VELOCITY_SCALE * station.model_east_vel_rotation,
        f"rot_north_vel_lat{suffix}": y_station
        + VELOCITY_SCALE * station.model_north_vel_rotation,
        f"seg_east_vel{suffix}": station.model_east_elastic_segment,
        f"seg_north_vel{suffix}": station.model_north_elastic_segment,
        f"seg_east_vel_lon{suffix}": x_station
        + VELOCITY_SCALE * station.model_east_elastic_segment,
        f"seg_north_vel_lat{suffix}": y_station
        + VELOCITY_SCALE * station.model_north_elastic_segment,
        f"tde_east_vel{suffix}": station.model_east_vel_tde,
        f"tde_north_vel{suffix}": station.model_north_vel_tde,
        f"tde_east_vel_lon{suffix}": x_station
        + VELOCITY_SCALE * station.model_east_vel_tde,
        f"tde_north_vel_lat{suffix}": y_station
        + VELOCITY_SCALE * station.model_north_vel_tde,
        f"str_east_vel{suffix}": station.model_east_vel_block_strain_rate,
        f"str_north_vel{suffix}": station.model_north_vel_block_strain_rate,
        f"str_east_vel_lon{suffix}": x_station
        + VELOCITY_SCALE * station.model_east_vel_block_strain_rate,
        f"str_north_vel_lat{suffix}": y_station
        + VELOCITY_SCALE * station.model_north_vel_block_strain_rate,
        f"mog_east_vel{suffix}": station.model_east_vel_mogi,
        f"mog_north_vel{suffix}": station.model_north_vel_mogi,
        f"mog_east_vel_lon{suffix}": x_station
        + VELOCITY_SCALE * station.model_east_vel_mogi,
        f"mog_north_vel_lat{suffix}": y_station
        + VELOCITY_SCALE * station.model_north_vel_mogi,
        f"res_mag{suffix}": resmag,
        f"sized_res_mag{suffix}": VELOCITY_SCALE / 2500 * resmag,
        f"name{suffix}": station.name,
    }

    segsource.data = {
        "xseg": [np.array([x1_seg[i], x2_seg[i]]) for i in range(len(segment))],
        "yseg": [np.array([y1_seg[i], y2_seg[i]]) for i in range(len(segment))],
        "ssrate": list(segment["model_strike_slip_rate"]),
        "dsrate": list(
            segment["model_dip_slip_rate"] - segment["model_tensile_slip_rate"]
        ),
        "active_comp": list(segment["model_strike_slip_rate"]),
        "name_1": list(segment["name"]),
        "tsrate": list(segment["model_tensile_slip_rate"]),
        "lonstart": list(segment["lon1"]),
        "latstart": list(segment["lat1"]),
        "lonend": list(segment["lon2"]),
        "latend": list(segment["lat2"]),
    }

    # Calculate fault surface projections for non-vertical segments
    fault_proj_polygons_x = []
    fault_proj_polygons_y = []
    fault_proj_dips = []
    fault_proj_names = []
    
    for i in range(len(segment)):
        dip_deg = segment["dip"].iloc[i]
        locking_depth = segment["locking_depth"].iloc[i]
        
        # Only create projection polygons for non-vertical faults
        if abs(dip_deg - 90.0) > 1e-6:
            # Calculate bottom edge coordinates
            lon1_bot, lat1_bot, lon2_bot, lat2_bot = calculate_fault_bottom_edge(
                segment["lon1"].iloc[i],
                segment["lat1"].iloc[i], 
                segment["lon2"].iloc[i],
                segment["lat2"].iloc[i],
                locking_depth,
                dip_deg
            )
            
            # Convert to web mercator
            x1_bot, y1_bot = wgs84_to_web_mercator(lon1_bot, lat1_bot)
            x2_bot, y2_bot = wgs84_to_web_mercator(lon2_bot, lat2_bot)
            
            # Create polygon coordinates (top edge -> bottom edge -> close)
            poly_x = np.array([x1_seg[i], x2_seg[i], x2_bot, x1_bot, x1_seg[i]])
            poly_y = np.array([y1_seg[i], y2_seg[i], y2_bot, y1_bot, y1_seg[i]])
            
            fault_proj_polygons_x.append(poly_x)
            fault_proj_polygons_y.append(poly_y)
            fault_proj_dips.append(dip_deg)
            fault_proj_names.append(segment["name"].iloc[i])
    
    fault_proj_source.data = {
        "xpoly": fault_proj_polygons_x,
        "ypoly": fault_proj_polygons_y,
        "dip": fault_proj_dips,
        "name": fault_proj_names,
    }

    tdesource.data = {
        "xseg": [
            np.array([x1_mesh[j], x2_mesh[j], x3_mesh[j]]) for j in range(len(meshes))
        ],
        "yseg": [
            np.array([y1_mesh[j], y2_mesh[j], y3_mesh[j]]) for j in range(len(meshes))
        ],
        "ssrate": list(meshes["strike_slip_rate"]),
        "dsrate": list(meshes["dip_slip_rate"]),
        "active_comp": list(meshes["strike_slip_rate"]),
    }

    tde_perim_source.data = {
        "xseg": [
            np.array([x1_perim_seg[j], x2_perim_seg[j]])
            for j in range(np.shape(perim_edges)[1])
        ],
        "yseg": [
            np.array([y1_perim_seg[j], y2_perim_seg[j]])
            for j in range(np.shape(perim_edges)[1])
        ],
        "proj_col": list(proj_mesh_edge_flag),
    }

    # Residual magnitude comparison
    # This needs to be inside load_data to appropriately update the CDS
    # Will only really run if both stasource_1 and stasource_2 aren't empty

    # Do DataFrame comparisons for residual improvement
    if (len(stasource_1.data["lon_1"]) > 0) & (len(stasource_2.data["lon_2"]) > 0):
        # Generate temporary dataframes from ColumnDataSources
        station_1 = pd.DataFrame(stasource_1.data)
        station_2 = pd.DataFrame(stasource_2.data)
        # Intersect station dataframes based on lon, lat and retain residual velocity components
        common = pd.merge(
            station_1,
            station_2,
            how="inner",
            left_on=["lon_1", "lat_1"],
            right_on=["lon_2", "lat_2"],
        )
        # Stations unique to either
        unique = pd.concat(
            (
                station_1[["lon_1", "lat_1"]],
                station_2[["lon_2", "lat_2"]].rename(
                    columns={"lon_2": "lon_1", "lat_2": "lat_1"}
                ),
            )
        ).drop_duplicates(keep=False, ignore_index=True)

        # Calculate residual magnitude difference (magnitudes already calculated in load_data)
        common["res_mag_diff"] = common["res_mag_2"] - common["res_mag_1"]

        # ColumnDataSource to hold data for stations common to both folders
        commonsta.data = {
            "lon_c": common.lon_1,
            "lat_c": common.lat_1,
            "res_mag_diff": common.res_mag_diff,
            "abs_res_mag_diff": np.abs(common.res_mag_diff),
            "sized_res_mag_diff": VELOCITY_SCALE / 2500 * np.abs(common.res_mag_diff),
        }
        # ColumnDataSource to hold data for stations unique to either
        uniquesta.data = {
            "lon_u": unique.lon_1,
            "lat_u": unique.lat_1,
        }


# Update the button callbacks
folder_load_button_1.on_click(lambda: load_data(1))
folder_load_button_2.on_click(lambda: load_data(2))


##############################
# END: Load data from button #
##############################

# TODO: See if res compare checkbox can become activated upon completion


################
# Figure setup #
################
def get_coastlines():
    coastlines = np.load("GSHHS_c_L1_0_360.npz")
    lon = coastlines["lon"]
    lat = coastlines["lat"]
    x, y = wgs84_to_web_mercator(lon, lat)
    return {"x": x, "y": y}


fig = figure(
    x_axis_type="mercator",  # Set x-axis to Mercator projection
    y_axis_type="mercator",  # Set y-axis to Mercator projection
    width=800,
    height=400,
    match_aspect=True,
    tools=[WheelZoomTool(), ResetTool(), PanTool()],
    output_backend="webgl",
)
fig.toolbar.active_scroll = fig.select_one(WheelZoomTool)
if has_mapbox_token:
    style_id = "maxballison/cm2i6wejr00b101pbbck1f3to"
    # Construct tile URL
    tile_url = f"https://api.mapbox.com/styles/v1/{style_id}/tiles/{{z}}/{{x}}/{{y}}?access_token={mapbox_access_token}"

    # Create a tile source with the Mapbox tiles
    tile_source = WMTSTileSource(url=tile_url)

    fig.add_tile(tile_source)

fig.xgrid.visible = False
fig.ygrid.visible = False
fig.add_layout(LinearAxis(), "above")  # Add axis on the top
fig.add_layout(LinearAxis(), "right")  # Add axis on the right

# Grid layout
grid_layout = pn.GridSpec(sizing_mode="stretch_both", max_height=700)

###############################
# Color mappers and colorbars #
###############################

# Set up dummy figure to just hold colorbars
colorbar_fig = figure(
    width=fig.width,
    height=0,
    toolbar_location=None,
    min_border=0,
    outline_line_color=None,
    x_axis_type="mercator",  # Set x-axis to Mercator projection
    y_axis_type="mercator",  # Set y-axis to Mercator projection
    match_aspect=True,
    output_backend="webgl",
)
colorbar_fig.xgrid.visible = False
colorbar_fig.ygrid.visible = False

# Slip rate color mapper
slip_color_mapper = LinearColorMapper(palette=brewer["RdBu"][11], low=-100, high=100)
slip_colorbar = ColorBar(
    color_mapper=slip_color_mapper,
    height=15,
    width=200,
    title="Slip rate (mm/yr)",
    orientation="horizontal",
    location=(0, 0),
)
colorbar_fig.add_layout(slip_colorbar)

# Residual magnitude color mapper
resmag_color_mapper = LinearColorMapper(palette=viridis(10), low=0, high=5)
resmag_colorbar = ColorBar(
    color_mapper=resmag_color_mapper,
    height=15,
    width=200,
    title="Resid. mag. (mm/yr)",
    orientation="horizontal",
    location=(250, 0),
)
colorbar_fig.add_layout(resmag_colorbar)

# Residual comparison color mapper
resmag_diff_color_mapper = LinearColorMapper(palette=brewer["RdBu"][11], low=-5, high=5)

resmag_diff_colorbar = ColorBar(
    color_mapper=resmag_diff_color_mapper,
    height=15,
    width=200,
    title="Resid. diff. (mm/yr)",
    orientation="horizontal",
    location=(500, 0),
)
colorbar_fig.add_layout(resmag_diff_colorbar)

# Simple colorbar to guide mesh edge color
mesh_edge_color_mapper = LinearColorMapper(palette=["black", "red"], low=0, high=1)

##############
# UI objects #
##############
# Folder 1 controls
folder_name_1 = "0000000292"
# folder_label_1 = Div(text="folder_1: 0000000292")
loc_checkbox_1 = CheckboxGroup(labels=["locs"], active=[])
name_checkbox_1 = CheckboxGroup(labels=["name"], active=[])
obs_vel_checkbox_1 = CheckboxGroup(labels=["obs"], active=[])
mod_vel_checkbox_1 = CheckboxGroup(labels=["mod"], active=[])
res_vel_checkbox_1 = CheckboxGroup(labels=["res"], active=[])
rot_vel_checkbox_1 = CheckboxGroup(labels=["rot"], active=[])
seg_vel_checkbox_1 = CheckboxGroup(labels=["seg"], active=[])
tde_vel_checkbox_1 = CheckboxGroup(labels=["tri"], active=[])
str_vel_checkbox_1 = CheckboxGroup(labels=["str"], active=[])
mog_vel_checkbox_1 = CheckboxGroup(labels=["mog"], active=[])
res_mag_checkbox_1 = CheckboxGroup(labels=["res mag"], active=[])

ss_text_checkbox_1 = CheckboxGroup(labels=["ss"], active=[])
ds_text_checkbox_1 = CheckboxGroup(labels=["ds"], active=[])
ss_color_checkbox_1 = CheckboxGroup(labels=["ss"], active=[])
ds_color_checkbox_1 = CheckboxGroup(labels=["ds"], active=[])

seg_text_checkbox_1 = CheckboxGroup(labels=["slip"], active=[])
seg_text_radio_1 = RadioButtonGroup(labels=["ss", "ds"], active=0)
seg_color_checkbox_1 = CheckboxGroup(labels=["slip"], active=[])
seg_color_radio_1 = RadioButtonGroup(labels=["ss", "ds"], active=0)
tde_checkbox_1 = CheckboxGroup(labels=["tde"], active=[])
tde_radio_1 = RadioButtonGroup(labels=["ss", "ds"], active=0)
fault_proj_checkbox_1 = CheckboxGroup(labels=["fault proj"], active=[])


# Folder 2 controls
folder_name_2 = "0000000293"
# folder_label_2 = Div(text="folder_2: 0000000293")
loc_checkbox_2 = CheckboxGroup(labels=["locs"], active=[])
name_checkbox_2 = CheckboxGroup(labels=["name"], active=[])
obs_vel_checkbox_2 = CheckboxGroup(labels=["obs"], active=[])
mod_vel_checkbox_2 = CheckboxGroup(labels=["mod"], active=[])
res_vel_checkbox_2 = CheckboxGroup(labels=["res"], active=[])
rot_vel_checkbox_2 = CheckboxGroup(labels=["rot"], active=[])
seg_vel_checkbox_2 = CheckboxGroup(labels=["seg"], active=[])
tde_vel_checkbox_2 = CheckboxGroup(labels=["tri"], active=[])
str_vel_checkbox_2 = CheckboxGroup(labels=["str"], active=[])
mog_vel_checkbox_2 = CheckboxGroup(labels=["mog"], active=[])
res_mag_checkbox_2 = CheckboxGroup(labels=["res mag"], active=[])

seg_text_checkbox_2 = CheckboxGroup(labels=["slip"], active=[])
seg_text_radio_2 = RadioButtonGroup(labels=["ss", "ds"], active=0)
seg_color_checkbox_2 = CheckboxGroup(labels=["slip"], active=[])
seg_color_radio_2 = RadioButtonGroup(labels=["ss", "ds"], active=0)
tde_checkbox_2 = CheckboxGroup(labels=["tde"], active=[])
tde_radio_2 = RadioButtonGroup(labels=["ss", "ds"], active=0)
fault_proj_checkbox_2 = CheckboxGroup(labels=["fault proj"], active=[])


# Other controls
velocity_scaler = Slider(
    start=0.0, end=50, value=1, step=1.0, title="vel scale", width=200
)

residual_compare_checkbox = CheckboxGroup(labels=["res compare"], active=[])


###############
# Map objects #
###############

# Velocity colors
obs_color_1 = RGB(r=0, g=0, b=256)
mod_color_1 = RGB(r=256, g=0, b=0)
res_color_1 = RGB(r=256, g=0, b=256)
rot_color_1 = RGB(r=0, g=256, b=0)
seg_color_1 = RGB(r=0, g=256, b=256)
tde_color_1 = RGB(r=256, g=166, b=0)
str_color_1 = RGB(r=0, g=128, b=128)
mog_color_1 = RGB(r=128, g=128, b=128)

# Folder 1: TDE slip rates
# Plotting these first so that coastlines, segments, and stations lie above
tde_obj_1 = fig.patches(
    xs="xseg",
    ys="yseg",
    source=tdesource_1,
    fill_color={"field": "active_comp", "transform": slip_color_mapper},
    line_width=0,
    visible=False,
)

tde_perim_obj_1 = fig.multi_line(
    xs="xseg",
    ys="yseg",
    line_color={"field": "proj_col", "transform": mesh_edge_color_mapper},
    source=tde_perim_source_1,
    line_width=1,
    visible=False,
)

# Folder 2: TDE slip rates
# Plotting these first so that coastlines, segments, and stations lie above
tde_obj_2 = fig.patches(
    xs="xseg",
    ys="yseg",
    source=tdesource_2,
    fill_color={"field": "active_comp", "transform": slip_color_mapper},
    line_width=0,
    visible=False,
)

tde_perim_obj_2 = fig.multi_line(
    xs="xseg",
    ys="yseg",
    line_color={"field": "proj_col", "transform": mesh_edge_color_mapper},
    source=tde_perim_source_2,
    line_width=1,
    visible=False,
)

# Folder 1: Fault surface projections
fault_proj_obj_1 = fig.patches(
    xs="xpoly",
    ys="ypoly", 
    source=fault_proj_source_1,
    fill_alpha=0.3,
    fill_color="lightblue",
    line_color="blue",
    line_width=1,
    visible=False,
)

# Folder 2: Fault surface projections  
fault_proj_obj_2 = fig.patches(
    xs="xpoly",
    ys="ypoly",
    source=fault_proj_source_2, 
    fill_alpha=0.3,
    fill_color="lightcoral",
    line_color="red",
    line_width=1,
    line_dash="dashed",
    visible=False,
)

# Residual magnitude differences
res_mag_diff_obj = fig.scatter(
    "lon_c",
    "lat_c",
    source=commonsta,
    size="sized_res_mag_diff",
    color={"field": "res_mag_diff", "transform": resmag_diff_color_mapper},
    visible=False,
)

# Unique stations (only present in one folder)
res_mag_diff_obj_unique = fig.scatter(
    "lon_u",
    "lat_u",
    source=uniquesta,
    size=15,
    marker="x",
    line_color="black",
    visible=False,
)

# Folder 1: Static segments. Always shown
seg_obj_1 = fig.multi_line(
    xs="xseg",
    ys="yseg",
    line_color="blue",
    source=segsource_1,
    line_width=1,
    visible=True,
)

# Folder 2: Static segments. Always shown
seg_obj_2 = fig.multi_line(
    xs="xseg",
    ys="yseg",
    line_color="blue",
    source=segsource_2,
    line_width=1,
    line_dash="dashed",
    visible=True,
)

# Folder 1: Colored line rates
seg_color_obj_1 = fig.multi_line(
    xs="xseg",
    ys="yseg",
    line_color={"field": "active_comp", "transform": slip_color_mapper},
    source=segsource_1,
    line_width=4,
    visible=False,
)

# Folder 2: Colored line rates
seg_color_obj_2 = fig.multi_line(
    xs="xseg",
    ys="yseg",
    line_color={"field": "active_comp", "transform": slip_color_mapper},
    source=segsource_2,
    line_width=4,
    visible=False,
)

# Coastlines
COASTLINES = get_coastlines()
fig.line(
    COASTLINES["x"],
    COASTLINES["y"],
    color="black",
    line_width=0.5 if not has_mapbox_token else 0.0,
)


# Create glyphs all potential plotting elements and hide them as default
loc_obj_1 = fig.scatter(
    "lon_1", "lat_1", source=stasource_1, size=2.7, color="black", visible=False
)


seg_hov_tool_1 = HoverTool(
    tooltips=[
        ("Name", "@name_1"),
        ("Start", "(@lonstart, @latstart)"),
        ("End", "(@lonend, @latend)"),
        ("Strike-Slip Rate", "@ssrate"),
        ("Dip-Slip Rate", "@dsrate"),
        ("Tensile-Slip Rate", "@tsrate"),
    ],
    renderers=[seg_obj_1],
)
fig.add_tools(seg_hov_tool_1)

loc_hov_tool_1 = HoverTool(
    tooltips=[
        ("Name", "@name_1"),
    ],
    renderers=[loc_obj_1],
)
fig.add_tools(loc_hov_tool_1)


# Folder 1: residual magnitudes
res_mag_obj_1 = fig.scatter(
    "lon_1",
    "lat_1",
    source=stasource_1,
    size="sized_res_mag_1",
    color={"field": "res_mag_1", "transform": resmag_color_mapper},
    visible=False,
)


# Folder 1: observed velocities
obs_vel_obj_1 = Arrow(
    end=arrow_head_type(
        fill_color=obs_color_1,
        fill_alpha=1.0,
        line_color=obs_color_1,
        size=arrow_head_size,
    ),
    x_start="lon_1",
    y_start="lat_1",
    x_end="obs_east_vel_lon_1",
    y_end="obs_north_vel_lat_1",
    line_color=obs_color_1,
    line_width=1,
    source=stasource_1,
    visible=False,
)
fig.add_layout(obs_vel_obj_1)

# Folder 1: modeled velocities
mod_vel_obj_1 = Arrow(
    end=arrow_head_type(
        fill_color=mod_color_1,
        fill_alpha=0.5,
        line_color=mod_color_1,
        size=arrow_head_size,
    ),
    x_start="lon_1",
    y_start="lat_1",
    x_end="mod_east_vel_lon_1",
    y_end="mod_north_vel_lat_1",
    line_color=mod_color_1,
    line_width=1,
    source=stasource_1,
    visible=False,
)
fig.add_layout(mod_vel_obj_1)

# Folder 1: residual velocities
res_vel_obj_1 = Arrow(
    end=arrow_head_type(
        fill_color=res_color_1,
        fill_alpha=0.5,
        line_color=res_color_1,
        size=arrow_head_size,
    ),
    x_start="lon_1",
    y_start="lat_1",
    x_end="res_east_vel_lon_1",
    y_end="res_north_vel_lat_1",
    line_color=res_color_1,
    line_width=1,
    source=stasource_1,
    visible=False,
)
fig.add_layout(res_vel_obj_1)

# Folder 1: Rotation Velocities
rot_vel_obj_1 = Arrow(
    end=arrow_head_type(
        fill_color=rot_color_1,
        fill_alpha=0.5,
        line_color=rot_color_1,
        size=arrow_head_size,
    ),
    x_start="lon_1",
    y_start="lat_1",
    x_end="rot_east_vel_lon_1",
    y_end="rot_north_vel_lat_1",
    line_color=rot_color_1,
    line_width=1,
    source=stasource_1,
    visible=False,
)
fig.add_layout(rot_vel_obj_1)

# Folder 1: Elastic Velocities
seg_vel_obj_1 = Arrow(
    end=arrow_head_type(
        fill_color=seg_color_1,
        fill_alpha=0.5,
        line_color=seg_color_1,
        size=arrow_head_size,
    ),
    x_start="lon_1",
    y_start="lat_1",
    x_end="seg_east_vel_lon_1",
    y_end="seg_north_vel_lat_1",
    line_color=seg_color_1,
    line_width=1,
    source=stasource_1,
    visible=False,
)
fig.add_layout(seg_vel_obj_1)

# Folder 1: TDE Velocities
tde_vel_obj_1 = Arrow(
    end=arrow_head_type(
        fill_color=tde_color_1,
        fill_alpha=0.5,
        line_color=tde_color_1,
        size=arrow_head_size,
    ),
    x_start="lon_1",
    y_start="lat_1",
    x_end="tde_east_vel_lon_1",
    y_end="tde_north_vel_lat_1",
    line_color=tde_color_1,
    line_width=1,
    source=stasource_1,
    visible=False,
)
fig.add_layout(tde_vel_obj_1)

# Folder 1: Strain Velocities
str_vel_obj_1 = Arrow(
    end=arrow_head_type(
        fill_color=str_color_1,
        fill_alpha=0.5,
        line_color=str_color_1,
        size=arrow_head_size,
    ),
    x_start="lon_1",
    y_start="lat_1",
    x_end="str_east_vel_lon_1",
    y_end="str_north_vel_lat_1",
    line_color=str_color_1,
    line_width=1,
    source=stasource_1,
    visible=False,
)
fig.add_layout(str_vel_obj_1)

# Folder 1: Mogi Velocities
mog_vel_obj_1 = Arrow(
    end=arrow_head_type(
        fill_color=mog_color_1,
        fill_alpha=0.5,
        line_color=mog_color_1,
        size=arrow_head_size,
    ),
    x_start="lon_1",
    y_start="lat_1",
    x_end="mog_east_vel_lon_1",
    y_end="mog_north_vel_lat_1",
    line_color=mog_color_1,
    line_width=1,
    source=stasource_1,
    visible=False,
)
fig.add_layout(mog_vel_obj_1)

############
# Folder 2 #
############

# Velocity colors for folder 2
obs_color_2 = RGB(r=0, g=0, b=205)
mod_color_2 = RGB(r=205, g=0, b=0)
res_color_2 = RGB(r=205, g=0, b=205)
rot_color_2 = RGB(r=0, g=205, b=0)
seg_color_2 = RGB(r=0, g=205, b=205)
tde_color_2 = RGB(r=205, g=133, b=0)
str_color_2 = RGB(r=0, g=102, b=102)
mog_color_2 = RGB(r=102, g=102, b=102)

loc_obj_2 = fig.scatter(
    "lon_2", "lat_2", source=stasource_2, size=1, color="black", visible=False
)
hover_tool_2 = HoverTool(
    tooltips=[
        ("Name", "@name_2"),
    ],
    renderers=[loc_obj_2],
)
fig.add_tools(hover_tool_2)

# Folder 2: residual magnitudes
res_mag_obj_2 = fig.scatter(
    "lon_2",
    "lat_2",
    source=stasource_2,
    size="sized_res_mag_2",
    color={"field": "res_mag_2", "transform": resmag_color_mapper},
    visible=False,
)

# Folder 2: observed velocities
obs_vel_obj_2 = Arrow(
    end=arrow_head_type(
        fill_color=obs_color_2,
        fill_alpha=2.0,
        line_color=obs_color_2,
        size=arrow_head_size,
    ),
    x_start="lon_2",
    y_start="lat_2",
    x_end="obs_east_vel_lon_2",
    y_end="obs_north_vel_lat_2",
    line_color=obs_color_2,
    line_width=2,
    source=stasource_2,
    visible=False,
)
fig.add_layout(obs_vel_obj_2)

# Folder 2: modeled velocities
mod_vel_obj_2 = Arrow(
    end=arrow_head_type(
        fill_color=mod_color_2,
        fill_alpha=0.5,
        line_color=mod_color_2,
        size=arrow_head_size,
    ),
    x_start="lon_2",
    y_start="lat_2",
    x_end="mod_east_vel_lon_2",
    y_end="mod_north_vel_lat_2",
    line_color=mod_color_2,
    line_width=2,
    source=stasource_2,
    visible=False,
)
fig.add_layout(mod_vel_obj_2)

# Folder 2: residual velocities
res_vel_obj_2 = Arrow(
    end=arrow_head_type(
        fill_color=res_color_2,
        fill_alpha=0.5,
        line_color=res_color_2,
        size=arrow_head_size,
    ),
    x_start="lon_2",
    y_start="lat_2",
    x_end="res_east_vel_lon_2",
    y_end="res_north_vel_lat_2",
    line_color=res_color_2,
    line_width=2,
    source=stasource_2,
    visible=False,
)
fig.add_layout(res_vel_obj_2)

# Folder 2: Rotation Velocities
rot_vel_obj_2 = Arrow(
    end=arrow_head_type(
        fill_color=rot_color_2,
        fill_alpha=0.5,
        line_color=rot_color_2,
        size=arrow_head_size,
    ),
    x_start="lon_2",
    y_start="lat_2",
    x_end="rot_east_vel_lon_2",
    y_end="rot_north_vel_lat_2",
    line_color=rot_color_2,
    line_width=2,
    source=stasource_2,
    visible=False,
)
fig.add_layout(rot_vel_obj_2)

# Folder 2: Elastic Velocities
seg_vel_obj_2 = Arrow(
    end=arrow_head_type(
        fill_color=seg_color_2,
        fill_alpha=0.5,
        line_color=seg_color_2,
        size=arrow_head_size,
    ),
    x_start="lon_2",
    y_start="lat_2",
    x_end="seg_east_vel_lon_2",
    y_end="seg_north_vel_lat_2",
    line_color=seg_color_2,
    line_width=2,
    source=stasource_2,
    visible=False,
)
fig.add_layout(seg_vel_obj_2)

# Folder 2: TDE Velocities
tde_vel_obj_2 = Arrow(
    end=arrow_head_type(
        fill_color=tde_color_2,
        fill_alpha=0.5,
        line_color=tde_color_2,
        size=arrow_head_size,
    ),
    x_start="lon_2",
    y_start="lat_2",
    x_end="tde_east_vel_lon_2",
    y_end="tde_north_vel_lat_2",
    line_color=tde_color_2,
    line_width=2,
    source=stasource_2,
    visible=False,
)
fig.add_layout(tde_vel_obj_2)

# Folder 2: Strain Velocities
str_vel_obj_2 = Arrow(
    end=arrow_head_type(
        fill_color=str_color_2,
        fill_alpha=0.5,
        line_color=str_color_2,
        size=arrow_head_size,
    ),
    x_start="lon_2",
    y_start="lat_2",
    x_end="str_east_vel_lon_2",
    y_end="str_north_vel_lat_2",
    line_color=str_color_2,
    line_width=2,
    source=stasource_2,
    visible=False,
)
fig.add_layout(str_vel_obj_2)

# Folder 2: Mogi Velocities
mog_vel_obj_2 = Arrow(
    end=arrow_head_type(
        fill_color=mog_color_2,
        fill_alpha=0.5,
        line_color=mog_color_2,
        size=arrow_head_size,
    ),
    x_start="lon_2",
    y_start="lat_2",
    x_end="mog_east_vel_lon_2",
    y_end="mog_north_vel_lat_2",
    line_color=mog_color_2,
    line_width=2,
    source=stasource_2,
    visible=False,
)
fig.add_layout(mog_vel_obj_2)

#######################
# Vector scale object #
#######################

# velocity_scale_obj = colorbar_fig.segment(
#     0.07, 0, 0.07 + VELOCITY_SCALE * 10, 0, line_width=3, color="black"
# )
#


velocity_scale_obj = fig.scatter(
    fig.x_range.start,
    fig.y_range.start,
    size=25,
)

#############
# Callbacks #
#############

# Define JavaScript callback to toggle visibility
checkbox_callback_js = """
    plot_object.visible = cb_obj.active.includes(0);
"""

# JavaScript callback for velocity magnitude scaling
velocity_scaler_callback = CustomJS(
    args=dict(
        source1=stasource_1,
        source2=stasource_2,
        source3=commonsta,
        velocity_scaler=velocity_scaler,
        VELOCITY_SCALE=VELOCITY_SCALE,
    ),
    code="""
    const velocity_scale_slider = velocity_scaler.value
    const lon_1 = source1.data.lon_1
    const lat_1 = source1.data.lat_1
    const obs_east_vel_1 =  source1.data.obs_east_vel_1
    const obs_north_vel_1 = source1.data.obs_north_vel_1
    const mod_east_vel_1 =  source1.data.mod_east_vel_1
    const mod_north_vel_1 = source1.data.mod_north_vel_1
    const res_east_vel_1 =  source1.data.res_east_vel_1
    const res_north_vel_1 = source1.data.res_north_vel_1
    const rot_east_vel_1 =  source1.data.rot_east_vel_1
    const rot_north_vel_1 = source1.data.rot_north_vel_1
    const seg_east_vel_1 =  source1.data.seg_east_vel_1
    const seg_north_vel_1 = source1.data.seg_north_vel_1
    const tde_east_vel_1 =  source1.data.tde_east_vel_1
    const tde_north_vel_1 = source1.data.tde_north_vel_1
    const str_east_vel_1 =  source1.data.str_east_vel_1
    const str_north_vel_1 = source1.data.str_north_vel_1
    const mog_east_vel_1 =  source1.data.mog_east_vel_1
    const mog_north_vel_1 = source1.data.mog_north_vel_1
    const res_mag_1 =       source1.data.res_mag_1
    const name_1 = source1.data.name_1
    
    const lon_2 = source2.data.lon_2
    const lat_2 = source2.data.lat_2
    const obs_east_vel_2 =  source2.data.obs_east_vel_2
    const obs_north_vel_2 = source2.data.obs_north_vel_2
    const mod_east_vel_2 =  source2.data.mod_east_vel_2
    const mod_north_vel_2 = source2.data.mod_north_vel_2
    const res_east_vel_2 =  source2.data.res_east_vel_2
    const res_north_vel_2 = source2.data.res_north_vel_2
    const rot_east_vel_2 =  source2.data.rot_east_vel_2
    const rot_north_vel_2 = source2.data.rot_north_vel_2
    const seg_east_vel_2 =  source2.data.seg_east_vel_2
    const seg_north_vel_2 = source2.data.seg_north_vel_2
    const tde_east_vel_2 =  source2.data.tde_east_vel_2
    const tde_north_vel_2 = source2.data.tde_north_vel_2
    const str_east_vel_2 =  source2.data.str_east_vel_2
    const str_north_vel_2 = source2.data.str_north_vel_2
    const mog_east_vel_2 =  source2.data.mog_east_vel_2
    const mog_north_vel_2 = source2.data.mog_north_vel_2
    const res_mag_2 =       source2.data.res_mag_2
    const name_2 = source2.data.name_2

    const lon_c = source3.data.lon_c
    const lat_c = source3.data.lat_c
    const res_mag_diff = source3.data.res_mag_diff
    const abs_res_mag_diff = source3.data.abs_res_mag_diff
    
    // Update velocities with current magnitude scaling
    let obs_east_vel_lon_1 = [];
    let obs_north_vel_lat_1 = [];
    let mod_east_vel_lon_1 = [];
    let mod_north_vel_lat_1 = [];
    let res_east_vel_lon_1 = [];
    let res_north_vel_lat_1 = [];
    let rot_east_vel_lon_1 = [];
    let rot_north_vel_lat_1 = [];
    let seg_east_vel_lon_1 = [];
    let seg_north_vel_lat_1 = [];
    let tde_east_vel_lon_1 = [];
    let tde_north_vel_lat_1 = [];
    let str_east_vel_lon_1 = [];
    let str_north_vel_lat_1 = [];
    let mog_east_vel_lon_1 = [];
    let mog_north_vel_lat_1 = [];
    let sized_res_mag_1 = [];
    for (let i = 0; i < lon_1.length; i++) {
        obs_east_vel_lon_1.push(lon_1[i] + VELOCITY_SCALE * velocity_scale_slider *  obs_east_vel_1[i]);
        obs_north_vel_lat_1.push(lat_1[i] + VELOCITY_SCALE * velocity_scale_slider * obs_north_vel_1[i]);
        mod_east_vel_lon_1.push(lon_1[i] + VELOCITY_SCALE * velocity_scale_slider *  mod_east_vel_1[i]);
        mod_north_vel_lat_1.push(lat_1[i] + VELOCITY_SCALE * velocity_scale_slider * mod_north_vel_1[i]);
        res_east_vel_lon_1.push(lon_1[i] + VELOCITY_SCALE * velocity_scale_slider *  res_east_vel_1[i]);
        res_north_vel_lat_1.push(lat_1[i] + VELOCITY_SCALE * velocity_scale_slider * res_north_vel_1[i]);
        rot_east_vel_lon_1.push(lon_1[i] + VELOCITY_SCALE * velocity_scale_slider *  rot_east_vel_1[i]);
        rot_north_vel_lat_1.push(lat_1[i] + VELOCITY_SCALE * velocity_scale_slider * rot_north_vel_1[i]);
        seg_east_vel_lon_1.push(lon_1[i] + VELOCITY_SCALE * velocity_scale_slider *  seg_east_vel_1[i]);
        seg_north_vel_lat_1.push(lat_1[i] + VELOCITY_SCALE * velocity_scale_slider * seg_north_vel_1[i]);
        tde_east_vel_lon_1.push(lon_1[i] + VELOCITY_SCALE * velocity_scale_slider *  tde_east_vel_1[i]);
        tde_north_vel_lat_1.push(lat_1[i] + VELOCITY_SCALE * velocity_scale_slider * tde_north_vel_1[i]);
        str_east_vel_lon_1.push(lon_1[i] + VELOCITY_SCALE * velocity_scale_slider *  str_east_vel_1[i]);
        str_north_vel_lat_1.push(lat_1[i] + VELOCITY_SCALE * velocity_scale_slider * str_north_vel_1[i]);
        mog_east_vel_lon_1.push(lon_1[i] + VELOCITY_SCALE * velocity_scale_slider *  mog_east_vel_1[i]);
        mog_north_vel_lat_1.push(lat_1[i] + VELOCITY_SCALE * velocity_scale_slider * mog_north_vel_1[i]);
        sized_res_mag_1.push(VELOCITY_SCALE/2500 * velocity_scale_slider * res_mag_1[i]);
    }

    // Update velocities with current magnitude scaling
    let obs_east_vel_lon_2 = [];
    let obs_north_vel_lat_2 = [];
    let mod_east_vel_lon_2 = [];
    let mod_north_vel_lat_2 = [];
    let res_east_vel_lon_2 = [];
    let res_north_vel_lat_2 = [];
    let rot_east_vel_lon_2 = [];
    let rot_north_vel_lat_2 = [];
    let seg_east_vel_lon_2 = [];
    let seg_north_vel_lat_2 = [];
    let tde_east_vel_lon_2 = [];
    let tde_north_vel_lat_2 = [];
    let str_east_vel_lon_2 = [];
    let str_north_vel_lat_2 = [];
    let mog_east_vel_lon_2 = [];
    let mog_north_vel_lat_2 = [];
    let sized_res_mag_2 = [];
    for (let j = 0; j < lon_2.length; j++) {
        obs_east_vel_lon_2.push(lon_2[j] + VELOCITY_SCALE * velocity_scale_slider *  obs_east_vel_2[j]);
        obs_north_vel_lat_2.push(lat_2[j] + VELOCITY_SCALE * velocity_scale_slider * obs_north_vel_2[j]);
        mod_east_vel_lon_2.push(lon_2[j] + VELOCITY_SCALE * velocity_scale_slider *  mod_east_vel_2[j]);
        mod_north_vel_lat_2.push(lat_2[j] + VELOCITY_SCALE * velocity_scale_slider * mod_north_vel_2[j]);
        res_east_vel_lon_2.push(lon_2[j] + VELOCITY_SCALE * velocity_scale_slider *  res_east_vel_2[j]);
        res_north_vel_lat_2.push(lat_2[j] + VELOCITY_SCALE * velocity_scale_slider * res_north_vel_2[j]);
        rot_east_vel_lon_2.push(lon_2[j] + VELOCITY_SCALE * velocity_scale_slider *  rot_east_vel_2[j]);
        rot_north_vel_lat_2.push(lat_2[j] + VELOCITY_SCALE * velocity_scale_slider * rot_north_vel_2[j]);
        seg_east_vel_lon_2.push(lon_2[j] + VELOCITY_SCALE * velocity_scale_slider *  seg_east_vel_2[j]);
        seg_north_vel_lat_2.push(lat_2[j] + VELOCITY_SCALE * velocity_scale_slider * seg_north_vel_2[j]);
        tde_east_vel_lon_2.push(lon_2[j] + VELOCITY_SCALE * velocity_scale_slider *  tde_east_vel_2[j]);
        tde_north_vel_lat_2.push(lat_2[j] + VELOCITY_SCALE * velocity_scale_slider * tde_north_vel_2[j]);
        str_east_vel_lon_2.push(lon_2[j] + VELOCITY_SCALE * velocity_scale_slider *  str_east_vel_2[j]);
        str_north_vel_lat_2.push(lat_2[j] + VELOCITY_SCALE * velocity_scale_slider * str_north_vel_2[j]);
        mog_east_vel_lon_2.push(lon_2[j] + VELOCITY_SCALE * velocity_scale_slider *  mog_east_vel_2[j]);
        mog_north_vel_lat_2.push(lat_2[j] + VELOCITY_SCALE * velocity_scale_slider * mog_north_vel_2[j]);
        sized_res_mag_2.push(VELOCITY_SCALE/2500 * velocity_scale_slider * res_mag_2[j]);
    }

    let sized_res_mag_diff = [];
    for (let k = 0; k < lon_c.length; k++) {
        sized_res_mag_diff.push(VELOCITY_SCALE/2500 * velocity_scale_slider * abs_res_mag_diff[k]);
    }

    // Package everthing back into dictionary
    // Try source.change.emit();???
    source1.data = { lon_1, lat_1, obs_east_vel_1, obs_north_vel_1, obs_east_vel_lon_1, obs_north_vel_lat_1, mod_east_vel_1, mod_north_vel_1, mod_east_vel_lon_1, mod_north_vel_lat_1, res_east_vel_1, res_north_vel_1, res_east_vel_lon_1, res_north_vel_lat_1, rot_east_vel_1, rot_north_vel_1, rot_east_vel_lon_1, rot_north_vel_lat_1, seg_east_vel_1, seg_north_vel_1, seg_east_vel_lon_1, seg_north_vel_lat_1, tde_east_vel_1, tde_north_vel_1, tde_east_vel_lon_1, tde_north_vel_lat_1, str_east_vel_1, str_north_vel_1, str_east_vel_lon_1, str_north_vel_lat_1, mog_east_vel_1, mog_north_vel_1, mog_east_vel_lon_1, mog_north_vel_lat_1, res_mag_1, sized_res_mag_1, name_1}
    source2.data = { lon_2, lat_2, obs_east_vel_2, obs_north_vel_2, obs_east_vel_lon_2, obs_north_vel_lat_2, mod_east_vel_2, mod_north_vel_2, mod_east_vel_lon_2, mod_north_vel_lat_2, res_east_vel_2, res_north_vel_2, res_east_vel_lon_2, res_north_vel_lat_2, rot_east_vel_2, rot_north_vel_2, rot_east_vel_lon_2, rot_north_vel_lat_2, seg_east_vel_2, seg_north_vel_2, seg_east_vel_lon_2, seg_north_vel_lat_2, tde_east_vel_2, tde_north_vel_2, tde_east_vel_lon_2, tde_north_vel_lat_2, str_east_vel_2, str_north_vel_2, str_east_vel_lon_2, str_north_vel_lat_2, mog_east_vel_2, mog_north_vel_2, mog_east_vel_lon_2, mog_north_vel_lat_2, res_mag_2, sized_res_mag_2, name_2}
    source3.data = { lon_c, lat_c, res_mag_diff, abs_res_mag_diff, sized_res_mag_diff}
""",
)

slip_component_callback_js = """
    const radio_value = cb_obj.active;
    const xseg = source.data.xseg
    const yseg = source.data.yseg
    const ssrate = source.data.ssrate
    const dsrate = source.data.dsrate  
    let active_comp = []                    
    for (let i = 0; i < dsrate.length; i++) {
        if(radio_value ==0) {
            active_comp[i] = ssrate[i];
        } else {
            active_comp[i] = dsrate[i];
        }
    }
   //source.change.emit();
   source.data = { xseg, yseg, ssrate, dsrate, active_comp}
"""


###################################
# Attach the callbacks to handles #
###################################
loc_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": loc_obj_1}, code=checkbox_callback_js)
)
obs_vel_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": obs_vel_obj_1}, code=checkbox_callback_js)
)
mod_vel_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": mod_vel_obj_1}, code=checkbox_callback_js)
)
res_vel_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": res_vel_obj_1}, code=checkbox_callback_js)
)
rot_vel_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": rot_vel_obj_1}, code=checkbox_callback_js)
)
seg_vel_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": seg_vel_obj_1}, code=checkbox_callback_js)
)
tde_vel_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": tde_vel_obj_1}, code=checkbox_callback_js)
)
str_vel_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": str_vel_obj_1}, code=checkbox_callback_js)
)
mog_vel_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": mog_vel_obj_1}, code=checkbox_callback_js)
)
res_mag_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": res_mag_obj_1}, code=checkbox_callback_js)
)
seg_color_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": seg_color_obj_1}, code=checkbox_callback_js)
)
seg_color_radio_1.js_on_change(
    "active", CustomJS(args=dict(source=segsource_1), code=slip_component_callback_js)
)
tde_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": tde_obj_1}, code=checkbox_callback_js)
)
tde_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": tde_perim_obj_1}, code=checkbox_callback_js)
)
tde_radio_1.js_on_change(
    "active", CustomJS(args=dict(source=tdesource_1), code=slip_component_callback_js)
)
fault_proj_checkbox_1.js_on_change(
    "active", CustomJS(args={"plot_object": fault_proj_obj_1}, code=checkbox_callback_js)
)

# Folder 2
loc_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": loc_obj_2}, code=checkbox_callback_js)
)
obs_vel_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": obs_vel_obj_2}, code=checkbox_callback_js)
)
mod_vel_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": mod_vel_obj_2}, code=checkbox_callback_js)
)
res_vel_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": res_vel_obj_2}, code=checkbox_callback_js)
)
rot_vel_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": rot_vel_obj_2}, code=checkbox_callback_js)
)
seg_vel_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": seg_vel_obj_2}, code=checkbox_callback_js)
)
tde_vel_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": tde_vel_obj_2}, code=checkbox_callback_js)
)
str_vel_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": str_vel_obj_2}, code=checkbox_callback_js)
)
mog_vel_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": mog_vel_obj_2}, code=checkbox_callback_js)
)
res_mag_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": res_mag_obj_2}, code=checkbox_callback_js)
)
seg_color_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": seg_color_obj_2}, code=checkbox_callback_js)
)
seg_color_radio_2.js_on_change(
    "active", CustomJS(args=dict(source=segsource_2), code=slip_component_callback_js)
)
tde_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": tde_obj_2}, code=checkbox_callback_js)
)
tde_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": tde_perim_obj_2}, code=checkbox_callback_js)
)
tde_radio_2.js_on_change(
    "active", CustomJS(args=dict(source=tdesource_2), code=slip_component_callback_js)
)
fault_proj_checkbox_2.js_on_change(
    "active", CustomJS(args={"plot_object": fault_proj_obj_2}, code=checkbox_callback_js)
)

# Shared between folder 1 and 2

# Velocity slider
velocity_scaler.js_on_change("value", velocity_scaler_callback)

# Residual comparison. Two nearly identical callbacks to control the two plot objects (common and unique stations)
residual_compare_checkbox.js_on_change(
    "active",
    CustomJS(args={"plot_object": res_mag_diff_obj}, code=checkbox_callback_js),
)

residual_compare_checkbox.js_on_change(
    "active",
    CustomJS(args={"plot_object": res_mag_diff_obj_unique}, code=checkbox_callback_js),
)

##############################
# Place objects on panel grid #
###############################
# Placing controls for folder 1
grid_layout[0:1, 0] = pn.Column(
    # pn.pane.Bokeh(button),
    pn.pane.Bokeh(folder_load_button_1),
    # pn.pane.Bokeh(load_folder_text_1),
    pn.pane.Bokeh(folder_label_1),
    pn.pane.Bokeh(loc_checkbox_1),
    pn.pane.Bokeh(obs_vel_checkbox_1),
    pn.pane.Bokeh(mod_vel_checkbox_1),
    pn.pane.Bokeh(res_vel_checkbox_1),
    pn.pane.Bokeh(rot_vel_checkbox_1),
    pn.pane.Bokeh(seg_vel_checkbox_1),
    pn.pane.Bokeh(tde_vel_checkbox_1),
    pn.pane.Bokeh(str_vel_checkbox_1),
    pn.pane.Bokeh(mog_vel_checkbox_1),
    pn.pane.Bokeh(res_mag_checkbox_1),
)

grid_layout[6, 0] = pn.Column(
    pn.pane.Bokeh(seg_color_checkbox_1),
    pn.pane.Bokeh(seg_color_radio_1),
    pn.pane.Bokeh(tde_checkbox_1),
    pn.pane.Bokeh(tde_radio_1),
    pn.pane.Bokeh(fault_proj_checkbox_1),
)

# Placing controls for folder 2
grid_layout[0:1, 1] = pn.Column(
    pn.pane.Bokeh(folder_load_button_2),
    pn.pane.Bokeh(folder_label_2),
    pn.pane.Bokeh(loc_checkbox_2),
    pn.pane.Bokeh(obs_vel_checkbox_2),
    pn.pane.Bokeh(mod_vel_checkbox_2),
    pn.pane.Bokeh(res_vel_checkbox_2),
    pn.pane.Bokeh(rot_vel_checkbox_2),
    pn.pane.Bokeh(seg_vel_checkbox_2),
    pn.pane.Bokeh(tde_vel_checkbox_2),
    pn.pane.Bokeh(str_vel_checkbox_2),
    pn.pane.Bokeh(mog_vel_checkbox_2),
    pn.pane.Bokeh(res_mag_checkbox_2),
)

grid_layout[6, 1] = pn.Column(
    pn.pane.Bokeh(seg_color_checkbox_2),
    pn.pane.Bokeh(seg_color_radio_2),
    pn.pane.Bokeh(tde_checkbox_2),
    pn.pane.Bokeh(tde_radio_2),
    pn.pane.Bokeh(fault_proj_checkbox_2),
)

grid_layout[5, 0:1] = pn.Column(
    pn.pane.Bokeh(residual_compare_checkbox), pn.pane.Bokeh(velocity_scaler)
)

grid_layout[8, 2:10] = colorbar_fig


# Place map
grid_layout[0:8, 2:10] = fig

# grid_layout[0:8, 2:10] = pn.Column(fig, colorbar_fig)

api_message = pn.pane.Markdown(
    """
    **Note:** Add your mapbox api key in 'mapbox_token.py' for better map styling
    """
)

if not has_mapbox_token:
    grid_layout[8, :] = api_message
# Show the app
grid_layout.show()

import scipy
import panel as pn
import pandas as pd
import numpy as np
import copy
import tkinter as tk
from tkinter import filedialog

from bokeh.plotting import figure
from bokeh.models import (
    Slider,
    CheckboxGroup,
    RadioButtonGroup,
    CustomJS,
    ColumnDataSource,
    LinearAxis,
    BoxZoomTool,
    ResetTool,
    PanTool,
    Div,
    Button,
    MultiLine,
    Patches,
    ColorBar,
    LinearColorMapper,
)
from bokeh.palettes import brewer, viridis
from bokeh.colors import RGB

pn.extension()

VELOCITY_SCALE = 0.01

##################################
# Declare empty ColumnDataStores #
##################################
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
    },
)

tdesource_1 = ColumnDataSource(
    data={
        "xseg": [],
        "yseg": [],
        "ssrate": [],
        "dsrate": [],
        "active_comp": [],
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
    }
)
segsource_2 = ColumnDataSource(segsource_1.data.copy())
tdesource_2 = ColumnDataSource(tdesource_1.data.copy())

# ColumnDataSource to hold data for stations common to both folders
commonsta = ColumnDataSource(
    data={
        "lon_c": [],
        "lat_c": [],
        "res_mag_diff": [],
        "res_mag_diff_sized": [],
    }
)

# Declare empty station dataframes
station1 = pd.DataFrame()
station2 = pd.DataFrame()

################################
# START: Load data from button #
################################

folder_load_button_1 = Button(label="load", button_type="success")
folder_label_1 = Div(text="---")

folder_load_button_2 = Button(label="load", button_type="success")
folder_label_2 = Div(text="---")


# Define the load_data callback function
def load_data1():
    # Read data from a local folder
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_name = filedialog.askdirectory(title="load")
    # Set display of folder name
    folder_label_1.text = folder_name.split("/")[-1]

    # Read model out put as dataframes
    station1 = pd.read_csv(folder_name + "/model_station.csv")
    resmag = np.sqrt(
        np.power(station1.model_east_vel_residual, 2)
        + np.power(station1.model_north_vel_residual, 2)
    )
    segment = pd.read_csv(folder_name + "/model_segment.csv")
    meshes = pd.read_csv(folder_name + "/model_meshes.csv")

    stasource_1.data = {
        "lon_1": station1.lon,
        "lat_1": station1.lat,
        "obs_east_vel_1": station1.east_vel,
        "obs_north_vel_1": station1.north_vel,
        "obs_east_vel_lon_1": station1.lon + VELOCITY_SCALE * station1.east_vel,
        "obs_north_vel_lat_1": station1.lat + VELOCITY_SCALE * station1.north_vel,
        "mod_east_vel_1": station1.model_east_vel,
        "mod_north_vel_1": station1.model_north_vel,
        "mod_east_vel_lon_1": station1.lon + VELOCITY_SCALE * station1.model_east_vel,
        "mod_north_vel_lat_1": station1.lat + VELOCITY_SCALE * station1.model_north_vel,
        "res_east_vel_1": station1.model_east_vel_residual,
        "res_north_vel_1": station1.model_north_vel_residual,
        "res_east_vel_lon_1": station1.lon
        + VELOCITY_SCALE * station1.model_east_vel_residual,
        "res_north_vel_lat_1": station1.lat
        + VELOCITY_SCALE * station1.model_north_vel_residual,
        "rot_east_vel_1": station1.model_east_vel_rotation,
        "rot_north_vel_1": station1.model_north_vel_rotation,
        "rot_east_vel_lon_1": station1.lon
        + VELOCITY_SCALE * station1.model_east_vel_rotation,
        "rot_north_vel_lat_1": station1.lat
        + VELOCITY_SCALE * station1.model_north_vel_rotation,
        "seg_east_vel_1": station1.model_east_elastic_segment,
        "seg_north_vel_1": station1.model_north_elastic_segment,
        "seg_east_vel_lon_1": station1.lon
        + VELOCITY_SCALE * station1.model_east_elastic_segment,
        "seg_north_vel_lat_1": station1.lat
        + VELOCITY_SCALE * station1.model_north_elastic_segment,
        "tde_east_vel_1": station1.model_east_vel_tde,
        "tde_north_vel_1": station1.model_north_vel_tde,
        "tde_east_vel_lon_1": station1.lon
        + VELOCITY_SCALE * station1.model_east_vel_tde,
        "tde_north_vel_lat_1": station1.lat
        + VELOCITY_SCALE * station1.model_north_vel_tde,
        "str_east_vel_1": station1.model_east_vel_block_strain_rate,
        "str_north_vel_1": station1.model_north_vel_block_strain_rate,
        "str_east_vel_lon_1": station1.lon
        + VELOCITY_SCALE * station1.model_east_vel_block_strain_rate,
        "str_north_vel_lat_1": station1.lat
        + VELOCITY_SCALE * station1.model_north_vel_block_strain_rate,
        "mog_east_vel_1": station1.model_east_vel_mogi,
        "mog_north_vel_1": station1.model_north_vel_mogi,
        "mog_east_vel_lon_1": station1.lon
        + VELOCITY_SCALE * station1.model_east_vel_mogi,
        "mog_north_vel_lat_1": station1.lat
        + VELOCITY_SCALE * station1.model_north_vel_mogi,
        "res_mag_1": resmag,
        "sized_res_mag_1": 10 * VELOCITY_SCALE * resmag,
    }

    # Source for block bounding segments. Dict of length n_segments
    segsource_1.data = {
        "xseg": [
            np.array((segment.loc[i, "lon1"], segment.loc[i, "lon2"]))
            for i in range(len(segment))
        ],
        "yseg": [
            np.array((segment.loc[i, "lat1"], segment.loc[i, "lat2"]))
            for i in range(len(segment))
        ],
        "ssrate": list(segment["model_strike_slip_rate"]),
        "dsrate": list(
            segment["model_dip_slip_rate"] - segment["model_tensile_slip_rate"]
        ),
        "active_comp": list(segment["model_strike_slip_rate"]),
    }

    tdesource_1.data = {
        "xseg": [
            np.array((meshes.lon1[j], meshes.lon2[j], meshes.lon3[j]))
            for j in range(len(meshes.lon1))
        ],
        "yseg": [
            np.array((meshes.lat1[j], meshes.lat2[j], meshes.lat3[j]))
            for j in range(len(meshes.lon1))
        ],
        "ssrate": list(meshes["strike_slip_rate"]),
        "dsrate": list(meshes["dip_slip_rate"]),
        "active_comp": list(meshes["strike_slip_rate"]),
    }


# Define the load_data callback function
def load_data2():
    # Read data from a local folder
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_name = filedialog.askdirectory(title="load")

    # Set display of folder name
    folder_label_2.text = folder_name.split("/")[-1]

    # Read model out put as dataframes
    station2 = pd.read_csv(folder_name + "/model_station.csv")
    resmag = np.sqrt(
        np.power(station2.model_east_vel_residual, 2)
        + np.power(station2.model_north_vel_residual, 2)
    )
    segment = pd.read_csv(folder_name + "/model_segment.csv")
    meshes = pd.read_csv(folder_name + "/model_meshes.csv")

    stasource_2.data = {
        "lon_2": station2.lon,
        "lat_2": station2.lat,
        "obs_east_vel_2": station2.east_vel,
        "obs_north_vel_2": station2.north_vel,
        "obs_east_vel_lon_2": station2.lon + VELOCITY_SCALE * station2.east_vel,
        "obs_north_vel_lat_2": station2.lat + VELOCITY_SCALE * station2.north_vel,
        "mod_east_vel_2": station2.model_east_vel,
        "mod_north_vel_2": station2.model_north_vel,
        "mod_east_vel_lon_2": station2.lon + VELOCITY_SCALE * station2.model_east_vel,
        "mod_north_vel_lat_2": station2.lat + VELOCITY_SCALE * station2.model_north_vel,
        "res_east_vel_2": station2.model_east_vel_residual,
        "res_north_vel_2": station2.model_north_vel_residual,
        "res_east_vel_lon_2": station2.lon
        + VELOCITY_SCALE * station2.model_east_vel_residual,
        "res_north_vel_lat_2": station2.lat
        + VELOCITY_SCALE * station2.model_north_vel_residual,
        "rot_east_vel_2": station2.model_east_vel_rotation,
        "rot_north_vel_2": station2.model_north_vel_rotation,
        "rot_east_vel_lon_2": station2.lon
        + VELOCITY_SCALE * station2.model_east_vel_rotation,
        "rot_north_vel_lat_2": station2.lat
        + VELOCITY_SCALE * station2.model_north_vel_rotation,
        "seg_east_vel_2": station2.model_east_elastic_segment,
        "seg_north_vel_2": station2.model_north_elastic_segment,
        "seg_east_vel_lon_2": station2.lon
        + VELOCITY_SCALE * station2.model_east_elastic_segment,
        "seg_north_vel_lat_2": station2.lat
        + VELOCITY_SCALE * station2.model_north_elastic_segment,
        "tde_east_vel_2": station2.model_east_vel_tde,
        "tde_north_vel_2": station2.model_north_vel_tde,
        "tde_east_vel_lon_2": station2.lon
        + VELOCITY_SCALE * station2.model_east_vel_tde,
        "tde_north_vel_lat_2": station2.lat
        + VELOCITY_SCALE * station2.model_north_vel_tde,
        "str_east_vel_2": station2.model_east_vel_block_strain_rate,
        "str_north_vel_2": station2.model_north_vel_block_strain_rate,
        "str_east_vel_lon_2": station2.lon
        + VELOCITY_SCALE * station2.model_east_vel_block_strain_rate,
        "str_north_vel_lat_2": station2.lat
        + VELOCITY_SCALE * station2.model_north_vel_block_strain_rate,
        "mog_east_vel_2": station2.model_east_vel_mogi,
        "mog_north_vel_2": station2.model_north_vel_mogi,
        "mog_east_vel_lon_2": station2.lon
        + VELOCITY_SCALE * station2.model_east_vel_mogi,
        "mog_north_vel_lat_2": station2.lat
        + VELOCITY_SCALE * station2.model_north_vel_mogi,
        "res_mag_2": resmag,
        "sized_res_mag_2": 10 * VELOCITY_SCALE * resmag,
    }

    # Source for block bounding segments. Dict of length n_segments
    segsource_2.data = {
        "xseg": [
            np.array((segment.loc[i, "lon1"], segment.loc[i, "lon2"]))
            for i in range(len(segment))
        ],
        "yseg": [
            np.array((segment.loc[i, "lat1"], segment.loc[i, "lat2"]))
            for i in range(len(segment))
        ],
        "ssrate": list(segment["model_strike_slip_rate"]),
        "dsrate": list(
            segment["model_dip_slip_rate"] - segment["model_tensile_slip_rate"]
        ),
        "active_comp": list(segment["model_strike_slip_rate"]),
    }

    tdesource_2.data = {
        "xseg": [
            np.array((meshes.lon1[j], meshes.lon2[j], meshes.lon3[j]))
            for j in range(len(meshes.lon1))
        ],
        "yseg": [
            np.array((meshes.lat1[j], meshes.lat2[j], meshes.lat3[j]))
            for j in range(len(meshes.lon1))
        ],
        "ssrate": list(meshes["strike_slip_rate"]),
        "dsrate": list(meshes["dip_slip_rate"]),
        "active_comp": list(meshes["strike_slip_rate"]),
    }


# Link the callback function to the button
folder_load_button_1.on_click(load_data1)
folder_load_button_2.on_click(load_data2)

# Do DataFrame comparisons
if (not station1.empty) & (not station2.empty):
    # Intersect station dataframes based on lon, lat and retain residual velocity components
    common = pd.merge(
        station1, station2, how="inner", on=["lon", "lat"], suffixes=("_1", "_2")
    )
    not_in_2 = station1[~station1.isin(station2)]
    not_in_1 = station2[~station2.isin(station1)]

    # Calculate residual magnitudes
    common["res_mag_1"] = np.sqrt(
        common["model_east_vel_residual_1"] ** 2
        + common["model_north_vel_residual_1"] ** 2
    )
    common["res_mag_2"] = np.sqrt(
        common["model_east_vel_residual_2"] ** 2
        + common["model_north_vel_residual_2"] ** 2
    )
    common["res_mag_diff"] = common["res_mag_2"] - common["res_mag_1"]

    # ColumnDataSource to hold data for stations common to both folders
    commonsta = ColumnDataSource(
        data={
            "lon_c": common.lon,
            "lat_c": common.lat,
            "res_mag_diff": common.res_mag_diff,
            "res_mag_diff_sized": 5 * np.abs(common.res_mag_diff),
        }
    )


##############################
# END: Load data from button #
##############################


################
# Figure setup #
################
def get_coastlines():
    # COASTLINES = scipy.io.loadmat("coastlines.mat")
    # COASTLINES["lon"] = COASTLINES["lon"].flatten()
    # COASTLINES["lat"] = COASTLINES["lat"].flatten()
    COASTLINES = np.load("GSHHS_c_L1_0_360.npz")
    return COASTLINES


fig = figure(
    x_range=(0, 360),
    y_range=(-90, 90),
    width=800,
    height=400,
    match_aspect=True,
    tools=[BoxZoomTool(match_aspect=True), ResetTool(), PanTool()],
    output_backend="webgl",
)
fig.xgrid.visible = False
fig.ygrid.visible = False
fig.add_layout(LinearAxis(), "above")  # Add axis on the top
fig.add_layout(LinearAxis(), "right")  # Add axis on the right

# Grid layout
grid_layout = pn.GridSpec(sizing_mode="stretch_both", max_height=600)

# Slip rate color mapper
slip_color_mapper = LinearColorMapper(palette=brewer["RdBu"][11], low=-100, high=100)

# Residual magnitude color mapper
resmag_color_mapper = LinearColorMapper(palette=viridis(10), low=0, high=5)

# Residual comparison color mapper
rescompare_color_mapper = LinearColorMapper(palette=brewer["RdBu"][11], low=-5, high=5)


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


# Other controls
velocity_scaler = Slider(
    start=0.0, end=50, value=1, step=1.0, title="vel scale", width=200
)


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
    COASTLINES["lon"],
    COASTLINES["lat"],
    color="black",
    line_width=0.5,
)

# Create glyphs all potential plotting elements and hide them as default
loc_obj_1 = fig.scatter(
    "lon_1", "lat_1", source=stasource_1, size=1, color="black", visible=False
)

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
obs_vel_obj_1 = fig.segment(
    "lon_1",
    "lat_1",
    "obs_east_vel_lon_1",
    "obs_north_vel_lat_1",
    source=stasource_1,
    line_width=1,
    color=obs_color_1,
    alpha=0.5,
    visible=False,
)

# Folder 1: modeled velocities
mod_vel_obj_1 = fig.segment(
    "lon_1",
    "lat_1",
    "mod_east_vel_lon_1",
    "mod_north_vel_lat_1",
    source=stasource_1,
    line_width=1,
    color=mod_color_1,
    alpha=0.5,
    visible=False,
)

# Folder 1: residual velocities
res_vel_obj_1 = fig.segment(
    "lon_1",
    "lat_1",
    "res_east_vel_lon_1",
    "res_north_vel_lat_1",
    source=stasource_1,
    line_width=1,
    color=res_color_1,
    visible=False,
)

# Folder 1: rotation velocities
rot_vel_obj_1 = fig.segment(
    "lon_1",
    "lat_1",
    "rot_east_vel_lon_1",
    "rot_north_vel_lat_1",
    source=stasource_1,
    line_width=1,
    color=rot_color_1,
    visible=False,
)

# Folder 1: elastic velocities
seg_vel_obj_1 = fig.segment(
    "lon_1",
    "lat_1",
    "seg_east_vel_lon_1",
    "seg_north_vel_lat_1",
    source=stasource_1,
    line_width=1,
    color=seg_color_1,
    visible=False,
)

# Folder 1: tde velocities
tde_vel_obj_1 = fig.segment(
    "lon_1",
    "lat_1",
    "tde_east_vel_lon_1",
    "tde_north_vel_lat_1",
    source=stasource_1,
    line_width=1,
    color=tde_color_1,
    alpha=0.5,
    visible=False,
)

# Folder 1: strain velocities
str_vel_obj_1 = fig.segment(
    "lon_1",
    "lat_1",
    "str_east_vel_lon_1",
    "str_north_vel_lat_1",
    source=stasource_1,
    line_width=1,
    color=str_color_1,
    alpha=0.5,
    visible=False,
)

# Folder 1: mogi velocities
mog_vel_obj_1 = fig.segment(
    "lon_1",
    "lat_1",
    "mog_east_vel_lon_1",
    "mog_north_vel_lat_1",
    source=stasource_1,
    line_width=1,
    color=mog_color_1,
    alpha=0.5,
    visible=False,
)

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
obs_vel_obj_2 = fig.segment(
    "lon_2",
    "lat_2",
    "obs_east_vel_lon_2",
    "obs_north_vel_lat_2",
    source=stasource_2,
    line_width=1,
    color=obs_color_2,
    alpha=0.5,
    visible=False,
)

# Folder 2: modeled velocities
mod_vel_obj_2 = fig.segment(
    "lon_2",
    "lat_2",
    "mod_east_vel_lon_2",
    "mod_north_vel_lat_2",
    source=stasource_2,
    line_width=1,
    color=mod_color_2,
    alpha=0.5,
    visible=False,
)

# Folder 2: residual velocities
res_vel_obj_2 = fig.segment(
    "lon_2",
    "lat_2",
    "res_east_vel_lon_2",
    "res_north_vel_lat_2",
    source=stasource_2,
    line_width=1,
    color=res_color_2,
    visible=False,
)

# Folder 2: rotation velocities
rot_vel_obj_2 = fig.segment(
    "lon_2",
    "lat_2",
    "rot_east_vel_lon_2",
    "rot_north_vel_lat_2",
    source=stasource_2,
    line_width=1,
    color=rot_color_2,
    visible=False,
)

# Folder 2: elastic velocities
seg_vel_obj_2 = fig.segment(
    "lon_2",
    "lat_2",
    "seg_east_vel_lon_2",
    "seg_north_vel_lat_2",
    source=stasource_2,
    line_width=1,
    color=seg_color_2,
    visible=False,
)

# Folder 2: tde velocities
tde_vel_obj_2 = fig.segment(
    "lon_2",
    "lat_2",
    "tde_east_vel_lon_2",
    "tde_north_vel_lat_2",
    source=stasource_2,
    line_width=1,
    color=tde_color_2,
    alpha=0.5,
    visible=False,
)

# Folder 2: strain velocities
str_vel_obj_2 = fig.segment(
    "lon_2",
    "lat_2",
    "str_east_vel_lon_2",
    "str_north_vel_lat_2",
    source=stasource_2,
    line_width=1,
    color=str_color_2,
    alpha=0.5,
    visible=False,
)

# Folder 2: mogi velocities
mog_vel_obj_2 = fig.segment(
    "lon_2",
    "lat_2",
    "mog_east_vel_lon_2",
    "mog_north_vel_lat_2",
    source=stasource_2,
    line_width=1,
    color=mog_color_2,
    alpha=0.5,
    visible=False,
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
        sized_res_mag_1.push(10 * VELOCITY_SCALE * velocity_scale_slider * res_mag_1[i]);
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
        sized_res_mag_2.push(10 * VELOCITY_SCALE * velocity_scale_slider * res_mag_2[j]);
    }

    // Package everthing back into dictionary
    // Try source.change.emit();???
    source1.data = { lon_1, lat_1, obs_east_vel_1, obs_north_vel_1, obs_east_vel_lon_1, obs_north_vel_lat_1, mod_east_vel_1, mod_north_vel_1, mod_east_vel_lon_1, mod_north_vel_lat_1, res_east_vel_1, res_north_vel_1, res_east_vel_lon_1, res_north_vel_lat_1, rot_east_vel_1, rot_north_vel_1, rot_east_vel_lon_1, rot_north_vel_lat_1, seg_east_vel_1, seg_north_vel_1, seg_east_vel_lon_1, seg_north_vel_lat_1, tde_east_vel_1, tde_north_vel_1, tde_east_vel_lon_1, tde_north_vel_lat_1, str_east_vel_1, str_north_vel_1, str_east_vel_lon_1, str_north_vel_lat_1, mog_east_vel_1, mog_north_vel_1, mog_east_vel_lon_1, mog_north_vel_lat_1, res_mag_1, sized_res_mag_1}
    source2.data = { lon_2, lat_2, obs_east_vel_2, obs_north_vel_2, obs_east_vel_lon_2, obs_north_vel_lat_2, mod_east_vel_2, mod_north_vel_2, mod_east_vel_lon_2, mod_north_vel_lat_2, res_east_vel_2, res_north_vel_2, res_east_vel_lon_2, res_north_vel_lat_2, rot_east_vel_2, rot_north_vel_2, rot_east_vel_lon_2, rot_north_vel_lat_2, seg_east_vel_2, seg_north_vel_2, seg_east_vel_lon_2, seg_north_vel_lat_2, tde_east_vel_2, tde_north_vel_2, tde_east_vel_lon_2, tde_north_vel_lat_2, str_east_vel_2, str_north_vel_2, str_east_vel_lon_2, str_north_vel_lat_2, mog_east_vel_2, mog_north_vel_2, mog_east_vel_lon_2, mog_north_vel_lat_2, res_mag_2, sized_res_mag_2}
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
tde_radio_1.js_on_change(
    "active", CustomJS(args=dict(source=tdesource_1), code=slip_component_callback_js)
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
tde_radio_2.js_on_change(
    "active", CustomJS(args=dict(source=tdesource_2), code=slip_component_callback_js)
)

# Shared between folder 1 and 2

# Velocity slider
velocity_scaler.js_on_change("value", velocity_scaler_callback)

# Residual comparison


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
)

grid_layout[5, 0:1] = pn.Column(
    pn.pane.Bokeh(velocity_scaler),
)

# Place map
grid_layout[0:8, 2:10] = fig

# Show the app
grid_layout.show()

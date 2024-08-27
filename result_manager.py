import scipy
import panel as pn
import pandas as pd
import numpy as np
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
from bokeh.palettes import brewer
from bokeh.colors import RGB

pn.extension()

VELOCITY_SCALE = 0.01

##################################
# Declare empty ColumnDataStores #
##################################
source = ColumnDataSource(
    data={
        "lon": [],
        "lat": [],
        "obs_east_vel": [],
        "obs_north_vel": [],
        "obs_east_vel_lon": [],
        "obs_north_vel_lat": [],
        "mod_east_vel": [],
        "mod_north_vel": [],
        "mod_east_vel_lon": [],
        "mod_north_vel_lat": [],
    }
)

# Source for block bounding segments. Dict of length n_segments
segsource = ColumnDataSource(
    data={
        "xseg": [],
        "yseg": [],
        "ssrate": [],
        "dsrate": [],
        "active_comp": [],
    },
)

tdesource = ColumnDataSource(
    data={
        "xseg": [],
        "yseg": [],
        "ssrate": [],
        "dsrate": [],
        "active_comp": [],
    },
)


################################
# START: Load data from button #
################################

folder_load_button_1 = Button(label="load", button_type="success")
folder_label_1 = Div(text="---")


# Define the load_data callback function
def load_data():
    # Read data from a local folder
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_name = filedialog.askdirectory(title="load")

    # Set display of folder name
    folder_label_1.text = folder_name.split("/")[-1]

    # Read model out put as dataframes
    station = pd.read_csv(folder_name + "/model_station.csv")
    segment = pd.read_csv(folder_name + "/model_segment.csv")
    meshes = pd.read_csv(folder_name + "/model_meshes.csv")

    source.data = {
        "lon": station.lon,
        "lat": station.lat,
        "obs_east_vel": station.east_vel,
        "obs_north_vel": station.north_vel,
        "obs_east_vel_lon": station.lon + VELOCITY_SCALE * station.east_vel,
        "obs_north_vel_lat": station.lat + VELOCITY_SCALE * station.north_vel,
        "mod_east_vel": station.model_east_vel,
        "mod_north_vel": station.model_north_vel,
        "mod_east_vel_lon": station.lon + VELOCITY_SCALE * station.model_east_vel,
        "mod_north_vel_lat": station.lat + VELOCITY_SCALE * station.model_north_vel,
        "res_east_vel": station.model_east_vel_residual,
        "res_north_vel": station.model_north_vel_residual,
        "res_east_vel_lon": station.lon
        + VELOCITY_SCALE * station.model_east_vel_residual,
        "res_north_vel_lat": station.lat
        + VELOCITY_SCALE * station.model_north_vel_residual,
        "rot_east_vel": station.model_east_vel_rotation,
        "rot_north_vel": station.model_north_vel_rotation,
        "rot_east_vel_lon": station.lon
        + VELOCITY_SCALE * station.model_east_vel_rotation,
        "rot_north_vel_lat": station.lat
        + VELOCITY_SCALE * station.model_north_vel_rotation,
        "seg_east_vel": station.model_east_elastic_segment,
        "seg_north_vel": station.model_north_elastic_segment,
        "seg_east_vel_lon": station.lon
        + VELOCITY_SCALE * station.model_east_elastic_segment,
        "seg_north_vel_lat": station.lat
        + VELOCITY_SCALE * station.model_north_elastic_segment,
        "tde_east_vel": station.model_east_vel_tde,
        "tde_north_vel": station.model_north_vel_tde,
        "tde_east_vel_lon": station.lon + VELOCITY_SCALE * station.model_east_vel_tde,
        "tde_north_vel_lat": station.lat + VELOCITY_SCALE * station.model_north_vel_tde,
        "str_east_vel": station.model_east_vel_block_strain_rate,
        "str_north_vel": station.model_north_vel_block_strain_rate,
        "str_east_vel_lon": station.lon
        + VELOCITY_SCALE * station.model_east_vel_block_strain_rate,
        "str_north_vel_lat": station.lat
        + VELOCITY_SCALE * station.model_north_vel_block_strain_rate,
        "mog_east_vel": station.model_east_vel_mogi,
        "mog_north_vel": station.model_north_vel_mogi,
        "mog_east_vel_lon": station.lon + VELOCITY_SCALE * station.model_east_vel_mogi,
        "mog_north_vel_lat": station.lat
        + VELOCITY_SCALE * station.model_north_vel_mogi,
    }

    # Source for block bounding segments. Dict of length n_segments
    segsource.data = {
        "xseg": [
            np.array((segment.loc[i, "lon1"], segment.loc[i, "lon2"]))
            for i in range(len(segment))
        ],
        "yseg": [
            np.array((segment.loc[i, "lat1"], segment.loc[i, "lat2"]))
            for i in range(len(segment))
        ],
        "ssrate": list(segment["model_strike_slip_rate"]),
        "dsrate": list(segment["model_dip_slip_rate"]),
        "active_comp": list(segment["model_strike_slip_rate"]),
    }

    tdesource.data = {
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
folder_load_button_1.on_click(load_data)

##############################
# END: Load data from button #
##############################


################
# Figure setup #
################
def get_coastlines():
    COASTLINES = scipy.io.loadmat("coastlines.mat")
    COASTLINES["lon"] = COASTLINES["lon"].flatten()
    COASTLINES["lat"] = COASTLINES["lat"].flatten()
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
folder_label_2 = Div(text="folder_2: 0000000293")
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

seg_text_checkbox_2 = CheckboxGroup(labels=["slip"], active=[])
seg_color_checkbox_2 = CheckboxGroup(labels=["slip"], active=[])
tde_checkbox_2 = CheckboxGroup(labels=["tde"], active=[])


# Other controls
velocity_scaler = Slider(
    start=0.0, end=50, value=1, step=1.0, title="vel scale", width=200
)


###############
# Map objects #
###############

# Velocity colors
obs_color = RGB(r=0, g=0, b=256)
mod_color = RGB(r=256, g=0, b=0)
res_color = RGB(r=256, g=0, b=256)
rot_color = RGB(r=0, g=256, b=0)
seg_color = RGB(r=0, g=256, b=256)
tde_color = RGB(r=256, g=166, b=0)
str_color = RGB(r=0, g=128, b=128)

# Folder 1: TDE slip rates
# Plotting these first so that coastlines, segments, and stations lie above
tde_obj_1 = fig.patches(
    xs="xseg",
    ys="yseg",
    source=tdesource,
    fill_color={"field": "active_comp", "transform": slip_color_mapper},
    line_width=0,
    visible=False,
)

# Folder 1: Static segments. Always shown
seg_obj_1 = fig.multi_line(
    xs="xseg",
    ys="yseg",
    line_color="blue",
    source=segsource,
    line_width=1,
    visible=True,
)

# Folder 1: Colored line rates
seg_color_obj_1 = fig.multi_line(
    xs="xseg",
    ys="yseg",
    line_color={"field": "active_comp", "transform": slip_color_mapper},
    source=segsource,
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
    "lon", "lat", source=source, size=1, color="black", visible=False
)

# Folder 1: observed velocities
obs_vel_obj_1 = fig.segment(
    "lon",
    "lat",
    "obs_east_vel_lon",
    "obs_north_vel_lat",
    source=source,
    line_width=1,
    color=obs_color,
    alpha=0.5,
    visible=False,
)

# Folder 1: modeled velocities
mod_vel_obj_1 = fig.segment(
    "lon",
    "lat",
    "mod_east_vel_lon",
    "mod_north_vel_lat",
    source=source,
    line_width=1,
    color=mod_color,
    alpha=0.5,
    visible=False,
)

# Folder 1: residual velocities
res_vel_obj_1 = fig.segment(
    "lon",
    "lat",
    "res_east_vel_lon",
    "res_north_vel_lat",
    source=source,
    line_width=1,
    color=res_color,
    visible=False,
)

# Folder 1: rotation velocities
rot_vel_obj_1 = fig.segment(
    "lon",
    "lat",
    "rot_east_vel_lon",
    "rot_north_vel_lat",
    source=source,
    line_width=1,
    color=rot_color,
    visible=False,
)

# Folder 1: elastic velocities
seg_vel_obj_1 = fig.segment(
    "lon",
    "lat",
    "seg_east_vel_lon",
    "seg_north_vel_lat",
    source=source,
    line_width=1,
    color=seg_color,
    visible=False,
)

# Folder 1: tde velocities
tde_vel_obj_1 = fig.segment(
    "lon",
    "lat",
    "tde_east_vel_lon",
    "tde_north_vel_lat",
    source=source,
    line_width=1,
    color=tde_color,
    alpha=0.5,
    visible=False,
)

# Folder 1: strain velocities
str_vel_obj_1 = fig.segment(
    "lon",
    "lat",
    "str_east_vel_lon",
    "str_north_vel_lat",
    source=source,
    line_width=1,
    color=str_color,
    alpha=0.5,
    visible=False,
)

# Folder 1: mogi velocities
mog_vel_obj_1 = fig.segment(
    "lon",
    "lat",
    "mog_east_vel_lon",
    "mog_north_vel_lat",
    source=source,
    line_width=1,
    color=str_color,
    alpha=0.5,
    visible=False,
)

#############
# Callbacks #
#############
# Define JavaScript callbacks to toggle visibility
checkbox_callback_js = """
    plot_object.visible = cb_obj.active.includes(0);
"""

loc_checkbox_callback_1 = CustomJS(
    args={"scatter": loc_obj_1},
    code="""
    scatter.visible = cb_obj.active.includes(0);
""",
)

obs_vel_checkbox_callback_1 = CustomJS(
    args={"segment": obs_vel_obj_1},
    code="""
    segment.visible = cb_obj.active.includes(0);
""",
)

mod_vel_checkbox_callback_1 = CustomJS(
    args={"segment": mod_vel_obj_1},
    code="""
    segment.visible = cb_obj.active.includes(0);
""",
)

seg_color_checkbox_callback_1 = CustomJS(
    args={"segment": seg_color_obj_1},
    code="""
    segment.visible = cb_obj.active.includes(0);
""",
)

tde_checkbox_callback_1 = CustomJS(
    args={"segment": tde_obj_1},
    code="""
    segment.visible = cb_obj.active.includes(0);
""",
)

# JavaScript callback for velocity magnitude scaling
velocity_scaler_callback = CustomJS(
    args=dict(
        source=source, velocity_scaler=velocity_scaler, VELOCITY_SCALE=VELOCITY_SCALE
    ),
    code="""
    const velocity_scale_slider = velocity_scaler.value
    const lon = source.data.lon
    const lat = source.data.lat
    const obs_east_vel =  source.data.obs_east_vel
    const obs_north_vel = source.data.obs_north_vel
    const mod_east_vel =  source.data.mod_east_vel
    const mod_north_vel = source.data.mod_north_vel
    const res_east_vel =  source.data.res_east_vel
    const res_north_vel = source.data.res_north_vel
    const rot_east_vel =  source.data.rot_east_vel
    const rot_north_vel = source.data.rot_north_vel
    const seg_east_vel =  source.data.seg_east_vel
    const seg_north_vel = source.data.seg_north_vel
    const tde_east_vel =  source.data.tde_east_vel
    const tde_north_vel = source.data.tde_north_vel
    const str_east_vel =  source.data.str_east_vel
    const str_north_vel = source.data.str_north_vel
    const mog_east_vel =  source.data.mog_east_vel
    const mog_north_vel = source.data.mog_north_vel
    
    // Update velocities with current magnitude scaling
    let obs_east_vel_lon = [];
    let obs_north_vel_lat = [];
    let mod_east_vel_lon = [];
    let mod_north_vel_lat = [];
    let res_east_vel_lon = [];
    let res_north_vel_lat = [];
    let rot_east_vel_lon = [];
    let rot_north_vel_lat = [];
    let seg_east_vel_lon = [];
    let seg_north_vel_lat = [];
    let tde_east_vel_lon = [];
    let tde_north_vel_lat = [];
    let str_east_vel_lon = [];
    let str_north_vel_lat = [];
    let mog_east_vel_lon = [];
    let mog_north_vel_lat = [];
    for (let i = 0; i < lon.length; i++) {
        obs_east_vel_lon.push(lon[i] + VELOCITY_SCALE * velocity_scale_slider *  obs_east_vel[i]);
        obs_north_vel_lat.push(lat[i] + VELOCITY_SCALE * velocity_scale_slider * obs_north_vel[i]);
        mod_east_vel_lon.push(lon[i] + VELOCITY_SCALE * velocity_scale_slider *  mod_east_vel[i]);
        mod_north_vel_lat.push(lat[i] + VELOCITY_SCALE * velocity_scale_slider * mod_north_vel[i]);
        res_east_vel_lon.push(lon[i] + VELOCITY_SCALE * velocity_scale_slider *  res_east_vel[i]);
        res_north_vel_lat.push(lat[i] + VELOCITY_SCALE * velocity_scale_slider * res_north_vel[i]);
        rot_east_vel_lon.push(lon[i] + VELOCITY_SCALE * velocity_scale_slider *  rot_east_vel[i]);
        rot_north_vel_lat.push(lat[i] + VELOCITY_SCALE * velocity_scale_slider * rot_north_vel[i]);
        seg_east_vel_lon.push(lon[i] + VELOCITY_SCALE * velocity_scale_slider *  seg_east_vel[i]);
        seg_north_vel_lat.push(lat[i] + VELOCITY_SCALE * velocity_scale_slider * seg_north_vel[i]);
        tde_east_vel_lon.push(lon[i] + VELOCITY_SCALE * velocity_scale_slider *  tde_east_vel[i]);
        tde_north_vel_lat.push(lat[i] + VELOCITY_SCALE * velocity_scale_slider * tde_north_vel[i]);
        str_east_vel_lon.push(lon[i] + VELOCITY_SCALE * velocity_scale_slider *  str_east_vel[i]);
        str_north_vel_lat.push(lat[i] + VELOCITY_SCALE * velocity_scale_slider * str_north_vel[i]);
        mog_east_vel_lon.push(lon[i] + VELOCITY_SCALE * velocity_scale_slider *  mog_east_vel[i]);
        mog_north_vel_lat.push(lat[i] + VELOCITY_SCALE * velocity_scale_slider * mog_north_vel[i]);
    
    }

    // Package everthing back into dictionary
    // Try source.change.emit();???
    source.data = { lon, lat, obs_east_vel, obs_north_vel, obs_east_vel_lon, obs_north_vel_lat, mod_east_vel, mod_north_vel, mod_east_vel_lon, mod_north_vel_lat, res_east_vel, res_north_vel, res_east_vel_lon, res_north_vel_lat, rot_east_vel, rot_north_vel, rot_east_vel_lon, rot_north_vel_lat, seg_east_vel, seg_north_vel, seg_east_vel_lon, seg_north_vel_lat, tde_east_vel, tde_north_vel, tde_east_vel_lon, tde_north_vel_lat, str_east_vel, str_north_vel, str_east_vel_lon, str_north_vel_lat, mog_east_vel, mog_north_vel, mog_east_vel_lon, mog_north_vel_lat}
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
# loc_checkbox_1.js_on_change("active", loc_checkbox_callback_1)
# obs_vel_checkbox_1.js_on_change("active", obs_vel_checkbox_callback_1)
# mod_vel_checkbox_1.js_on_change("active", mod_vel_checkbox_callback_1)

velocity_scaler.js_on_change("value", velocity_scaler_callback)
seg_color_checkbox_1.js_on_change("active", seg_color_checkbox_callback_1)
seg_color_radio_1.js_on_change(
    "active", CustomJS(args=dict(source=segsource), code=slip_component_callback_js)
)
tde_checkbox_1.js_on_change("active", tde_checkbox_callback_1)
tde_radio_1.js_on_change(
    "active", CustomJS(args=dict(source=tdesource), code=slip_component_callback_js)
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
)

grid_layout[6, 0] = pn.Column(
    pn.pane.Bokeh(seg_color_checkbox_1),
    pn.pane.Bokeh(seg_color_radio_1),
    pn.pane.Bokeh(tde_checkbox_1),
    pn.pane.Bokeh(tde_radio_1),
)

# Placing controls for folder 2
grid_layout[0:1, 1] = pn.Column(
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
)

grid_layout[5, 0:1] = pn.Column(
    pn.pane.Bokeh(velocity_scaler),
)

# Place map
grid_layout[0:8, 2:10] = fig

# Show the app
grid_layout.show()

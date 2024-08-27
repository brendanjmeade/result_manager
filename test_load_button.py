import os
import tkinter as tk
from tkinter import filedialog

import pandas as pd
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Button
from bokeh.plotting import figure
from bokeh.layouts import column

VELOCITY_SCALE = 0.01

# Initial empty ColumnDataSource
# source = ColumnDataSource(data=dict(lon=[], lat=[]))
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

# Define the button
button = Button(label="Load Data", button_type="success")


def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    foldername = filedialog.askdirectory(
        title="load", initialdir="/Users/meade/Desktop/result_manager"
    )
    return foldername


# Define the callback function
def load_data():
    # Step 1: Read data from a local file
    foldername = select_folder()
    file_path = foldername + "/model_station.csv"
    station = pd.read_csv(file_path)

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
    }


# Link the callback function to the button
button.on_click(load_data)
print(source)

# Create a simple plot to display the data
p = figure(title="Example Plot")
p.line(x="lon", y="lat", source=source)

# Layout the button and plot together
layout = column(button, p)

# Add the layout to the current document
curdoc().add_root(layout)

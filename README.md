Graphical result viewer for model output from [celeri](https://github.com/brendanjmeade/celeri) kinematic earthquake cycle modeling software.

![https://private-user-images.githubusercontent.com/4225359/363271379-f93c61d5-2c22-4477-88a6-433db0ebf7cd.png](https://github.com/brendanjmeade/result_manager/blob/main/assets/result_manager_screenshot.png)

### Getting started
1. Installation:
```
conda config --prepend channels conda-forge
conda env create
conda activate result_manager
pip install --no-use-pep517 -e .
```

2. Obtain a Mapbox access token to enable tile-based rendering of topograph and bathymetry:
- Get a Mapbox access token at: [https://docs.mapbox.com/help/getting-started/access-tokens/]([url](https://docs.mapbox.com/help/getting-started/access-tokens/)).
- Run this command `cp mapbox_token.py.template mapbox_token.py`.
- Add your mapbox api key to 'mapbox_token.py' as string.

### Running after installation and configuration
1. Activate conda environment:
`conda activate result_manager`

2. Start the application:
`bokeh serve --show result_manager.py`


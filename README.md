Graphical result view for model output from [celeri](https://github.com/brendanjmeade/celeri) kinematic earthquake cycle modeling software.

![https://private-user-images.githubusercontent.com/4225359/363271379-f93c61d5-2c22-4477-88a6-433db0ebf7cd.png](https://github.com/brendanjmeade/result_manager/blob/main/assets/result_manager_screenshot.png)

Built by @brendanjmeade and @jploveless.


Installation:
```
conda config --prepend channels conda-forge
conda env create
conda activate result_manager
pip install --no-use-pep517 -e .
```

Activate conda environment:
`conda activate result_manager`

Start the application:
`bokeh serve --show result_manager.py`

In order to get map styling, create a Mapbox account and do the following:
Run this command 'cp mapbox_token.py.template mapbox_token.py'
Add your mapbox api key to 'mapbox_token.py'
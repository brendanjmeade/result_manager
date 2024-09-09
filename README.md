![result_manager_screenshot](https://private-user-images.githubusercontent.com/4225359/363271379-f93c61d5-2c22-4477-88a6-433db0ebf7cd.png)

Graphical result view for model output from: https://github.com/brendanjmeade/celeri.

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

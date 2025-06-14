Graphical result viewer for model output from [celeri](https://github.com/brendanjmeade/celeri) kinematic earthquake cycle modeling software.

![https://private-user-images.githubusercontent.com/4225359/363271379-f93c61d5-2c22-4477-88a6-433db0ebf7cd.png](https://github.com/brendanjmeade/result_manager/blob/main/assets/result_manager_screenshot.png)

### Getting started
`result_manager` isn't a proper python package but is easy to use:

1. Obtain a Mapbox access token to enable tile-based rendering of topograph and bathymetry:
- Get a Mapbox access token at: [https://docs.mapbox.com/help/getting-started/access-tokens/]([url](https://docs.mapbox.com/help/getting-started/access-tokens/)).
- Run this command: `cp mapbox_token.py.template mapbox_token.py`.
- Add your mapbox api key to 'mapbox_token.py' as string.

2. Create `result_manager` environment:
```
[conda|micromamba] config --prepend channels conda-forge
[conda|micromamba] env create -f environment.yml
```

3. Create a shortcut in your .bashrc or .zshrc to launch:
```
result_manager() {
  micromamba activate myenv
  cd ~/path/to/result_manager
  bokeh serve --show result_manager.py
}
```

4. Restart your shell

5. Type `result_manager`

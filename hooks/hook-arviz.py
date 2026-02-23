from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

# Collect the full static folder
datas = collect_data_files("arviz", subdir="static")

# Include example data JSON files
datas += collect_data_files("arviz", subdir="data/example_data")

# Hidden backend plot
hiddenimports = collect_submodules("arviz.plots.backends")
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("pytensor", include_py_files=True)
hiddenimports = collect_submodules("pytensor")

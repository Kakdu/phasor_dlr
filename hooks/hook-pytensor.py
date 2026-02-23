from PyInstaller.utils.hooks import collect_data_files

# include all c_code files
datas = collect_data_files("pytensor.tensor")

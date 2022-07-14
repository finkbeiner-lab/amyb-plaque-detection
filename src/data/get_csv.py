import os
import glob
import shutil
import pdb


result_data_path = "/home/vivek/Projects/amyb-plaque-detection/reports/figures/"
result_folders = glob.glob(os.path.join(result_data_path, "*"))

destination_path = "/home/vivek/Projects/amyb-plaque-detection/reports/"
csv_dir = os.path.join(os.path.dirname(destination_path), "csv_data")
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

for result_folder in result_folders:
    csv_file = glob.glob(os.path.join(result_folder, "*.csv"))
    
    if len(csv_file) == 0:
        print("Empty", result_folder)
        continue
    csv_file_name = os.path.basename(csv_file[0])

    destination_dir = os.path.join(csv_dir, csv_file_name)
    shutil.copy(csv_file[0], destination_dir)




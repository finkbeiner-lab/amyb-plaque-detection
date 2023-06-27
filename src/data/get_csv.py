import os
import glob
import shutil
import pdb


result_data_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/reports/figures/good-star-659/"
result_folders = glob.glob(os.path.join(result_data_path, "*"))

destination_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/"
#csv_dir = os.path.join(os.path.dirname(destination_path), "csv_data","Internal_dataset")
csv_dir = os.path.join(os.path.dirname(destination_path), "csv_data","Internal_dataset_latest")

#if not os.path.exists(csv_dir):
 #   os.makedirs(csv_dir)

for result_folder in result_folders:
    csv_file = glob.glob(os.path.join(result_folder, "*.csv"))
    print(csv_file)
    if len(csv_file) == 0:
        print("Empty", result_folder)
        continue
    csv_file_name = os.path.basename(csv_file[0])

    destination_dir = os.path.join(csv_dir, csv_file_name)
    shutil.copy(csv_file[0], destination_dir)




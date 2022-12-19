"""
To filter out Images for which diagonisis is AD Def.
This script uses Image and Clinacal Metadata 

"""

import pdb
import pandas as pd

clinial_df = pd.read_csv('/home/vivek/Metadata/VA_BR_Clinical_combo_matched_CMC_AMPAD_10082020.csv')
image_df = pd.read_csv('/home/vivek/Metadata/NPSAD_image_metadata.csv')

image_df = image_df[image_df['Stain'] == "Amyloid Beta"]

AD_def_df = clinial_df[clinial_df['Final.Dx'] == 'AD-Def.']

bb_no = AD_def_df['BB']


filtered_df = pd.DataFrame()

for number in bb_no:
    image_files = image_df[image_df['BB'] == number]
    image_files = image_files[image_files['BrainRegion'] == "Middle Frontal Gyrus"]
    filtered_df = filtered_df.append(image_files, ignore_index=True)

print(filtered_df)
filtered_df.to_csv("/home/vivek/Metadata/AmyB-AD-Def_cases.csv")
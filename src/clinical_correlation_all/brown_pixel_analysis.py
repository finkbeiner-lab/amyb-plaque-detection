import os
import glob
import pdb
import pandas as pd
import tqdm
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import f_oneway
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots


csv_dir = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/reports/New-Minerva-Data-output/yp2mf3i8_epoch=108-step=872.ckpt"

csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))


def is_non_zero_file(fpath):  
    if os.path.isfile(fpath):
        tmp = pd.read_csv(fpath)
        if len(tmp.columns)>1:
            return True
    return False


csv_files_df = [pd.read_csv(csv_file) for csv_file in csv_files if is_non_zero_file(csv_file)] 


len(csv_files_df)


all_plaque_objects  = pd.concat(csv_files_df, axis=0, ignore_index=True)


print(len(all_plaque_objects))

all_plaque_objects_wth_brown_pixels = all_plaque_objects[all_plaque_objects["brown_pixels"]>0]


all_plaque_objects_wth_brown_pixels["norm_brown_pixels"] = all_plaque_objects_wth_brown_pixels["brown_pixels"]/(1024*1024)


all_plaque_objects_wth_brown_pixels.groupby(["label"])["brown_pixels"].mean()

mean_values  =  all_plaque_objects_wth_brown_pixels.groupby(["label"])["norm_brown_pixels"].mean()
median_values  =  all_plaque_objects_wth_brown_pixels.groupby(["label"])["norm_brown_pixels"].median()

min_values  =  all_plaque_objects_wth_brown_pixels.groupby(["label"])["norm_brown_pixels"].min()
max_values  =  all_plaque_objects_wth_brown_pixels.groupby(["label"])["norm_brown_pixels"].max()
std_values  =  all_plaque_objects_wth_brown_pixels.groupby(["label"])["norm_brown_pixels"].std()

print("mean_values", mean_values)

print("median_values", median_values)

print("min_values", min_values)

print("max_values",max_values)

print("std_values",std_values)
median_values  =  all_plaque_objects_wth_brown_pixels.groupby(["label"])["brown_pixels"].median()
std_values  =  all_plaque_objects_wth_brown_pixels.groupby(["label"])["brown_pixels"].std()
print("median_values", median_values)
print("max_values",max_values)

labels  = all_plaque_objects_wth_brown_pixels["label"].unique()


x = [0,1]
df = all_plaque_objects_wth_brown_pixels
var = "norm_brown_pixels"

fig = make_subplots(rows=1, cols=len(labels))

for i, label in enumerate(labels):
    x = df[df["label"]==label][var]
    trace0 = go.Histogram(x=x, name = label, histnorm='percent',xbins=dict(
        start=0,
        end=0.5,
        size=0.01
    ))
    fig.append_trace(trace0, 1, i+1)

fig.write_html("/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/reports/Clinical_correlation_analysis/brown_pixel1.html")
"""
x_axis_title = "Labels"
l =[]
var_x = "label"
var_y = "norm_brown_pixels"
df = all_plaque_objects_wth_brown_pixels
var = "norm_brown_pixels"

y_axis_title ="Total Brown Pixel (Normalized by Image pixels)"
title = "Total Brown Pixel vs label"
y = labels
fig = go.Figure()
for e, i in enumerate(labels):
  fig.add_trace(go.Box(y=df[df[var_x]==i][var_y],
              boxpoints='all', # can also be outliers, or suspectedoutliers, or False
              jitter=0.3, # add some jitter for a better separation between points
              pointpos=0, # relative position of points wrt box
              name = i))
  l.append(df[df[var_x]==i][var_y].values)
  
  if e>0:
    _, p = stats.ttest_ind(df[df[var_x]==y[e-1]][var_y].values, df[df[var_x]==y[e]][var_y].values, nan_policy='omit')
    print("p value ",y[e-1]," ",y[e]," ", p)
fig.update_layout(height=900, width = 1000)
fig.update_layout(
  xaxis_title_text=x_axis_title, # xaxis label
  yaxis_title_text=y_axis_title) # yaxis label
fig.update_layout(title=title)
fig.update_layout( plot_bgcolor='white')
fig.update_xaxes( mirror=True,
  ticks='outside',
  showline=True,
  linecolor='black',
  gridcolor='lightgrey'
)
fig.update_yaxes(
  mirror=True,
  ticks='outside',
  showline=True,
  linecolor='black',
  gridcolor='lightgrey'
)

#fig.write_html("/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/reports/Clinical_correlation_analysis/brown_pixel2.html")
fig.write_image("/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/reports/Clinical_correlation_analysis/brown_pixel2.png")

print("ANOVA for all :" ,f_oneway(l[0],l[1]))
"""
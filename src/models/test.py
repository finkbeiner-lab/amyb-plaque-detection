import glob
import os
import pdb
from PIL import Image

file_path = glob.glob(os.path.join('/home/vivek/Datasets/mask_rcnn/dataset/val/images', '*.jpg'))

for f in file_path:
   
    N, temp_string, ext = 4, "amyb_", ".jpg"
    digit_repr = lambda N, i: temp_string + ''.join(['0'] * (N - len(str(i)))) + str(i)
    
    # Labels
    # fname = os.path.join(os.path.dirname(f) , digit_repr(N, os.path.basename(f).split('_')[0]) + ext)
    fname = os.path.join(os.path.dirname(f) , digit_repr(N, os.path.basename(f).split('.')[0]) + ext)

    os.rename(f, fname)
    print(fname)
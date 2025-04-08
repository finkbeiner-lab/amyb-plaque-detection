import sys
sys.path.append('../../../')
import os
import pyvips as Vips
from Reinhard import Reinhard
from openslide import OpenSlide
import pdb
from PIL import Image

# Path to the reference slide
ref_slide_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi_v2/XE07-047_1_AmyB_1/image/XE07-047_1_AmyB_1_8381x_120830y_image.png"

# Path to the target slide to be normalized
target_slide_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/test_image4.jpg"

# Output directory for the normalized image
output_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Norm_test_image4.jpg"

# Function to initialize normalization based on the reference image
def initialize_normalizer(ref_slide_path):
    print("Initializing normalization with reference slide:", ref_slide_path)
    ref_image = Vips.Image.new_from_file(ref_slide_path)
    normalizer = Reinhard()
    normalizer.fit(ref_image)
    return normalizer

# Function to normalize a single slide
def normalize_slide(target_slide_path, normalizer, output_dir):
    print("Normalizing slide:", target_slide_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #pdb.set_trace()
    # Load the target slide
    target_image = Vips.Image.new_from_file(target_slide_path)

    # Perform normalization
    normalized_image = normalizer.transform(target_image)

    # Generate the output file path
    file_name = os.path.basename(target_slide_path).replace(".mrxs", ".tif")
    output_file_path = os.path.join(output_dir, file_name)

    # Save the normalized image
    normalized_image.write_to_file(output_file_path)
    print("Normalized image saved to:", output_file_path)

# Initialize the normalizer
normalizer = initialize_normalizer(ref_slide_path)

# Normalize the single target slide
normalize_slide(target_slide_path, normalizer, output_dir)







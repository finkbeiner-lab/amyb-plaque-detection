import pdb
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
import pdb



def tile_image(image_path, tile_size):
    # Open the image
    image = Image.open(image_path)

    # Get the width and height of the image
    width, height = image.size

    # Calculate the number of tiles horizontally and vertically
    num_tiles_x = width // tile_size[0]
    num_tiles_y = height // tile_size[1]


    # Tile the image
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Calculate the tile coordinates
            tile_x = x * tile_size[0]
            tile_y = y * tile_size[1]

            # Extract the tile from the original image
            tile = image.crop((tile_x, tile_y, tile_x + tile_size[0], tile_y + tile_size[1]))

            save_path = "/mnt/new-nas/work/data/npsad_data/vivek/Datasets/Neuropil_threads/tiles/{x}_{y}_tiled.jpg"

           
            # Save the tiled image
            tile.save(save_path.format(x=x, y=y))


def dataaugmentation():
    transforms  = A.Compose([
                            A.VerticalFlip(p=0.5),
                            A.HorizontalFlip(p=0.5),
                            A.Blur(blur_limit=3),
                            A.OpticalDistortion(),
                            A.HueSaturationValue(),
                            A.RandomRotate90(),
                            A.RandomBrightnessContrast(p=0.2),
                        ])
    
    


if __name__ == '__main__':

    image_path = "/mnt/new-nas/work/data/npsad_data/vivek/Datasets/Neuropil_threads/sample.jpg"
    tile_size = (512,512)

    tile_image(image_path, tile_size)

    dataaugmentation()
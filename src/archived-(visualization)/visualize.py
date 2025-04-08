import matplotlib.pyplot as plt
import numpy as np
import pdb

def visualize_train(images, targets, wandb, img_no, visualize=False):

    if visualize:

        plt.figure(figsize=(10,10)) # specifying the overall grid size
        plt.title('Verify Training Data')

        x = images[0]
        x = x.detach().cpu()
        y = np.asarray(x)
        y = y.transpose(1,2,0)

        x1 = targets[0]['masks']
        x1 = x1.detach().cpu()
        x1 = x1.numpy()
        x1 = x1.transpose(1 ,2, 0)

        img_array = [y, x1]



        for i in range(2):
            plt.subplot(1,2,i+1)    # the number of images in the grid is 5*5 (25)
            plt.imshow(img_array[i])

        # TODO create train-data
        save_file = "../../reports/figures/train-data/{img_no}.png"
        plt.savefig(save_file.format(img_no=img_no))
        wandb.log({"train/data": wandb.Image(save_file.format(img_no=img_no))})

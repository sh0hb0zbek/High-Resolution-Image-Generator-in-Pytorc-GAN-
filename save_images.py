import matplotlib.pyplot as plt
import numpy as np


def save_images(images, plot=False, save_path=None):
    assert len(images) == 4, "You need exactly 4 images to plot"
    
    # Convert images from torch tensors to numpy arrays and transpose to (128, 128, 3)
    images = [image.permute(1, 2, 0) for image in images]
    
    fig, axes = plt.subplots(2, 2, figsize=(4, 4))
    
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i])
        ax.axis('off')  # Hide axes for better visualization
    
    plt.tight_layout()
    if plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

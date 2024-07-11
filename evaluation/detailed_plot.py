import numpy as np
import matplotlib.pyplot as plt

VERSIONS = ["v167_both"]

for VERSION in VERSIONS:
    images = np.load(f'../results/images_eval/images_eval_{VERSION}.npy', allow_pickle=True).item()

    num_rows = 5
    num_cols = 10

    plt.figure(figsize=(num_cols * 8, num_rows * 8))
    plt.suptitle(f"images {VERSION}", fontsize=80)

    plotted_images = 0

    for i, (prompt, prompt_images) in enumerate(images.items()):
        ax1 = plt.subplot(num_rows, num_cols, plotted_images + 1)
        ax1.imshow(prompt_images[0])
        ax1.set_title(prompt[36:], fontsize=45)
        ax1.axis("off")
        
        ax2 = plt.subplot(num_rows, num_cols, plotted_images + 2)
        ax2.imshow(prompt_images[1])
        ax2.set_title(prompt[36:], fontsize=45)
        ax2.axis("off")
        
        plotted_images += 2

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(f'../results/images/evaluation_images_{VERSION}.jpg', format='jpg')
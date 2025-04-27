import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches



class ImgToolkit:
    def __init__(self, dataset):
        self.dataset = dataset

    def img_to_np(self, tensor):
        inp = tensor.numpy().transpose((1, 2, 0))
        mean = np.array(self.dataset.norm_mean)
        std = np.array(self.dataset.norm_std)

        inp = std * inp + mean
        return inp

    def label_to_np(self, tensor):
        temp = tensor.numpy()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        n = len(self.dataset.class_names)
        for l in range(0, n):
            r[temp==l]=self.dataset.class_colors[l][0]
            g[temp==l]=self.dataset.class_colors[l][1]
            b[temp==l]=self.dataset.class_colors[l][2]
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:,:,0] = (r/255.0)#[:,:,0]
        rgb[:,:,1] = (g/255.0)#[:,:,1]
        rgb[:,:,2] = (b/255.0)#[:,:,2]
        return rgb

    def label_1class_to_np(self, mask, save_path=None):
        mask_rgb = mask * 255
        # Expand dimensions to create an RGB image
        mask_rgb = mask_rgb.unsqueeze(-1).repeat(1, 1, 3)
        # Convert to a NumPy array
        mask_rgb = mask_rgb.numpy()

        # Save mask if save_path is provided
        if save_path:
            mask_img = Image.fromarray(mask_rgb.astype(np.uint8))
            mask_img.save(save_path)

        return mask_rgb
    
    def view_np(self, inp):
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        plt.show()

    def __getitem__(self, index):

        # Check if dataset is a class, raise an error if so
        if isinstance(self.dataset, type):
            raise TypeError("Expected an object instance for 'dataset', but received a class. Please provide a dataset instance.")

        # Retrieve image and label tensors from the dataset
        img_tensor, label_tensor = self.dataset[index]

        # Convert tensors to numpy arrays for visualization
        view_img = self.img_to_np(img_tensor)
        view_label = self.label_to_np(label_tensor)

        return view_img, view_label

    def view(self, index):

        view_img, view_label = self.__getitem__(index)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(view_img)
        axes[0].axis('off')
        axes[0].set_title('Image')

        axes[1].imshow(view_label)
        axes[1].axis('off')
        axes[1].set_title('Label/Target')

        plt.tight_layout()
        plt.show()




    def visualize_images_with_superposition(self, img, target, alpha=0.5, save_path=None):
        """
        Visualize two images side by side and a third image superimposing both with transparency.

        Parameters:
        img: The base image to be displayed (np.ndarray)
        target: The image to be displayed as the second image and superposed. (np.ndarray)
        alpha: The transparency level for the superposed image (0 is fully transparent, 1 is fully opaque).
        save_path: Path to save the superposed image. If None, the image is not saved.
        """
        # Create a figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Display the first image (img)
        axes[0].imshow(img)
        axes[0].axis('off')  # Hide axis
        axes[0].set_title('Image')

        # Display the second image (target)
        axes[1].imshow(target)
        axes[1].axis('off')  # Hide axis
        axes[1].set_title('Target')

        # Superimpose the images with transparency in the third subplot
        superposed = img.copy()
        superposed = (alpha * target + (1 - alpha) * img).clip(0, 1)  # Combine the images with alpha
        axes[2].imshow(superposed)
        axes[2].axis('off')  # Hide axis
        axes[2].set_title('Superposed Image')

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Save the superposed image if save_path is provided
        if save_path:
            superposed_img = Image.fromarray((superposed * 255).astype(np.uint8))  # Convert to 8-bit image
            superposed_img.save(save_path)

        # Show the plot
        plt.show()


    def print_dataset_colors(self):
        """
        Print the colors of the classes in the dataset.
        
        """
        dataset=self.dataset
        class_names = dataset.class_names
        class_colors = dataset.class_colors
        
        # legend handles
        patches = [mpatches.Patch(color=[c/255.0 for c in color], label=name) 
                for name, color in zip(class_names, class_colors)]
        
        # Create a figure and axis
        plt.figure(figsize=(8, 5))
        plt.legend(handles=patches, loc='upper left', title="Class Colors")
        plt.axis('off')  # Nasconde gli assi
        plt.show()
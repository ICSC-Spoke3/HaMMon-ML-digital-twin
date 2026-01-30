import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import matplotlib.patches as mpatches



class Img:
    """Utility wrapper around a dataset providing tensor-to-numpy conversion, 
    visualization in matplotlib, and helpers to inspect segmentation masks 
    (colors, overlays, dataset length/access)."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.mean = np.asarray(getattr(self.dataset, "norm_mean", getattr(self.dataset, "mean", (0, 0, 0))))
        self.std  = np.asarray(getattr(self.dataset, "norm_std",  getattr(self.dataset, "std",  (1, 1, 1))))
    

    def __len__(self):
        # Check if dataset is a class, raise an error if so
        if isinstance(self.dataset, type):
            raise TypeError("Expected an object instance for 'dataset', but received a class. Please provide a dataset instance.")
        
        # Return the length of the dataset
        return len(self.dataset)
    
    def _to_cpu_np(self, t: torch.Tensor) -> np.ndarray:
        """Detaches, moves to cpu and converts to numpy."""
        return t.detach().cpu().numpy()

    def img_to_np(self, tensor: torch.Tensor, clip: bool = True) -> np.ndarray:
        """
        Converts a tensor to a numpy array, applying normalization if necessary."""
        inp = self._to_cpu_np(tensor).transpose(1, 2, 0)  # (H,W,C)

        mean = self.mean
        std  = self.std

        if mean is None or std is None:
            raise ValueError("Dataset does not have 'norm_mean' or 'norm_std' attributes.")
        
        inp = (inp * std) + mean
        if clip:
            inp = np.clip(inp, 0.0, 1.0)
        return inp

    def label_to_np(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Converts a label tensor to a numpy array, mapping class indices to RGB colors."""
        temp = self._to_cpu_np(tensor)
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
    
    def overlay(self, img: np.ndarray, mask: np.ndarray, alpha: float = .5) -> np.ndarray:
        """Returns an overlay of the image and mask with a given alpha."""
        img_f   = img.astype(np.float32)
        mask_f  = mask.astype(np.float32)
        if img_f.max() > 1:   img_f  /= 255.0
        if mask_f.max() > 1:  mask_f /= 255.0
        return (1-alpha)*img_f + alpha*mask_f

    
    def view_np(self, inp, figsize=(10, 10)):
        """
        Visualizes a numpy array as an image."""
        plt.figure(figsize=figsize) 
        # grayscale
        if inp.ndim == 2:
            plt.imshow(inp, cmap='gray')
        # RGB
        elif inp.ndim == 3 and inp.shape[2] == 3:
            plt.imshow(inp)
        plt.show()

    def view_img(self, img_tensor, clip=True, figsize=(10, 10)):
        """
        Visualizes an image tensor  """
        if isinstance(img_tensor, int):
            # If img_tensor is an index, retrieve the image from the dataset
            img_tensor = self.dataset[img_tensor][0]
        view_img = self.img_to_np(img_tensor, clip=clip)
        self.view_np(view_img, figsize=figsize)

    def view_label(self, label_tensor, figsize=(10, 10)):
        """
            Visualizes a label tensor  """
        if isinstance(label_tensor, int):
            # If label_tensor is an index, retrieve the label from the dataset
            label_tensor = self.dataset[label_tensor][1]
        view_label = self.label_to_np(label_tensor)
        self.view_np(view_label, figsize=figsize)
    
    def view_overlay(self, img_tensor, label_tensor=0, alpha=0.5, figsize=(10, 10)):
        """
        Visualizes an overlay of an image tensor and a label tensor.
        """
        if isinstance(img_tensor, int):
            el  = self.dataset[img_tensor]
            img_tensor = el[0]
            label_tensor = el[1]

        view_img = self.img_to_np(img_tensor)
        view_label = self.label_to_np(label_tensor)

        superposed = self.overlay(view_img, view_label, alpha=alpha)
        self.view_np(superposed, figsize=figsize)
    

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

    def view(self, index, figsize=(10, 5)):
        """Display the RGB image and its segmentation label from the dataset 
        index side by side using matplotlib."""

        view_img, view_label = self.__getitem__(index)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(view_img)
        axes[0].axis('off')
        axes[0].set_title('Image')

        axes[1].imshow(view_label)
        axes[1].axis('off')
        axes[1].set_title('Label/Target')

        plt.tight_layout()
        plt.show()




    def visualize_images_with_superposition(self, img, target=None, alpha=0.5, save_path=None):
        """
        Visualize two images side by side and a third image superimposing both with transparency.

        Parameters:
        img: np.ndarray or int - base image or dataset index
        target: np.ndarray or None - second image to superimpose, or None if img is an index
        alpha: float - transparency level for overlay (0 is fully transparent, 1 is fully opaque)
        save_path: str or None - if set, saves the overlay image to this path
        """

        # If img is an index, load from dataset
        if isinstance(img, int):
            if target is not None:
                raise ValueError("If 'img' is an index, 'target' must be None.")
            img, target = self[img]  # calls __getitem__

        # Validate input types
        if not isinstance(img, np.ndarray) or not isinstance(target, np.ndarray):
            raise TypeError("'img' and 'target' must be numpy arrays or 'img' must be an integer index.")

        # Create a figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Display the first image (img)
        axes[0].imshow(np.clip(img, 0, 1))
        axes[0].axis('off')
        axes[0].set_title('Image')

        # Display the second image (target)
        axes[1].imshow(np.clip(target, 0, 1))
        axes[1].axis('off')
        axes[1].set_title('Target')

        # Overlay
        superposed = self.overlay(img, target, alpha=alpha)
        axes[2].imshow(np.clip(superposed, 0, 1))
        axes[2].axis('off')
        axes[2].set_title('Superposed Image')

        plt.tight_layout()

        # Save the superposed image if save_path is provided
        if save_path:
            superposed_img = Image.fromarray((np.clip(superposed, 0, 1) * 255).astype(np.uint8))
            superposed_img.save(save_path)

        plt.show()

    def print_dataset_colors(self, figsize=(4, 2)):
        """
        Render a legend summarizing dataset class names and their RGB colors
        using matplotlib patches
        """
        dataset=self.dataset
        class_names = dataset.class_names
        class_colors = dataset.class_colors
        
       
        patches = [mpatches.Patch(color=[c/255.0 for c in color], label=name) 
                for name, color in zip(class_names, class_colors)]
        
        plt.figure(figsize=figsize)
        plt.legend(handles=patches, loc='upper left', title="Class Colors")
        plt.axis('off')  # Nasconde gli assi
        plt.show()
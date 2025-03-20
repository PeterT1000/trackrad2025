import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import glob

def display_medical_image(file_path=None):
    """
    Display all slices of a medical image with metadata.
    Args:
        file_path (str): Path to the medical image file. If None, will look for .mha files in the download2 directory.
    """
    try:
        if file_path is None:
            # Look for .mha files in the download2 directory
            search_path = os.path.join(os.path.dirname(__file__), "download2", "*.mha")
            available_files = glob.glob(search_path)
            
            if not available_files:
                raise FileNotFoundError("No .mha files found in the download2 directory")
            
            file_path = available_files[0]  # Use the first file found
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found at: {file_path}")

        print(f"Reading image from: {file_path}")
        
        # Read the image
        image = sitk.ReadImage(file_path)

        # Get image information
        size = image.GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()

        print("\nImage Information:")
        print("-----------------")
        print("Image Size:", size)
        print("Pixel Spacing:", spacing)
        print("Origin:", origin)
        print("Direction Cosines:", direction)

        # Convert the image to a NumPy array
        image_array = sitk.GetArrayFromImage(image)
        
        # Validate array dimensions
        if len(image_array.shape) not in [2, 3]:
            raise ValueError(f"Unexpected image dimensions: {image_array.shape}. Expected 2D or 3D image.")
        
        # Handle both 2D and 3D images
        if len(image_array.shape) == 2:
            image_array = image_array[np.newaxis, ...]  # Add a dimension for consistency
        
        # Transpose the array to get the correct orientation
        image_array = np.transpose(image_array, (0, 2, 1))
        print("\nArray Information:")
        print("-----------------")
        print("Array shape:", image_array.shape)
        print(f"Data range: [{image_array.min():.2f}, {image_array.max():.2f}]")

        # Calculate grid dimensions
        n_images = image_array.shape[0]
        n_cols = min(8, n_images)  # Use 8 columns or less if fewer images
        n_rows = math.ceil(n_images / n_cols)

        # Create a figure with subplots for all images
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows + 1))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.ravel()  # Flatten the array of axes for easier indexing

        # Add image information as title
        info_text = (f"Image Size: {size}\n"
                    f"Pixel Spacing: {spacing}\n"
                    f"Origin: {origin}\n"
                    f"Data Range: [{image_array.min():.2f}, {image_array.max():.2f}]")
        plt.suptitle(info_text, y=0.98, fontsize=10)

        # Find global min and max for consistent colormap scaling
        vmin, vmax = np.percentile(image_array, [2, 98])  # Use percentiles to avoid outliers

        # Plot each slice
        for i in range(n_images):
            im = axes[i].imshow(image_array[i], cmap='gray', vmin=vmin, vmax=vmax)
            slice_shape = image_array[i].shape
            axes[i].set_title(f'Slice {i+1}\nSize: {slice_shape[0]}x{slice_shape[1]}')
            axes[i].axis('off')

        # Hide empty subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
            axes[i].set_visible(False)

        # Add a colorbar
        plt.tight_layout()
        plt.subplots_adjust(right=0.92, top=0.90)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Intensity Values')

        plt.show()

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure SimpleITK is correctly installed (pip install SimpleITK)")
        print("2. Verify that the image file exists and is accessible")
        print("3. Check if the image file is in a valid format (.mha)")
        print("4. Ensure you have sufficient memory to load the image")

if __name__ == "__main__":
    display_medical_image()  # Will automatically find and display first .mha file in download2 directory
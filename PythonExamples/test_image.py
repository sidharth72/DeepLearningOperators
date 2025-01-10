import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def load_and_compare_images(image_path):
    """
    Load an image using both Pillow and OpenCV, convert to numpy arrays,
    and display them side by side for comparison
    """
    # Load with Pillow
    pil_img = Image.open(image_path)
    pil_array = np.array(pil_img)

    pil_array = pil_array / 255.0

    # Resize the pillow array to have 32x32x3 dimensions
    pil_array = np.array(pil_img.resize((32, 32)))
    
    # Load with OpenCV
    cv_img = cv2.imread(image_path)
    cv_array = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    cv_array = np.array(cv_array)
    cv_array = cv_array.resize((32, 32))
    

    print("Pillow array", pil_array / 255.0)
    print("OpenCV array", cv_array / 255.0)
    
    # Print array information
    print("PIL Array:")
    print(f"Shape: {pil_array.shape}")
    print(f"Value range: [{pil_array.min()}, {pil_array.max()}]")
    print(f"Mean: {pil_array.mean():.3f}")
    print("\nOpenCV Array:")
    print(f"Shape: {cv_array.shape}")
    print(f"Value range: [{cv_array.min()}, {cv_array.max()}]")
    print(f"Mean: {cv_array.mean():.3f}")
    
    # Calculate difference
    diff = np.abs(pil_array - cv_array)
    print(f"\nMax absolute difference: {diff.max()}")
    
    # Visualize the images and their difference
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(pil_array)
    ax1.set_title('Pillow -> NumPy')
    ax1.axis('off')
    
    ax2.imshow(cv_array)
    ax2.set_title('OpenCV -> NumPy')
    ax2.axis('off')
    
    ax3.imshow(diff)
    ax3.set_title('Difference')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pil_array, cv_array

# Example usage
if __name__ == "__main__":
    image_path = "data/inputs/cat6.jpg"
    pil_array, cv_array = load_and_compare_images(image_path)
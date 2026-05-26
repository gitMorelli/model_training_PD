import cv2
import numpy as np

def convert_background_to_white(image_path, output_path):
    # 1. Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image.")
        return

    # 2. Threshold the image to isolate the white squares
    # This turns anything bright into pure white (255) and everything else to black (0)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # 3. Find the external contours of the white squares
    # RETR_EXTERNAL ensures we only grab the outer borders of the squares, ignoring the symbols inside
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Create a blank black mask of the same size
    mask = np.zeros_like(img)

    # 5. Fill the detected square contours with pure white on our mask
    # Because we use cv2.FILLED, the entire inside of the square becomes white on the mask
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # 6. Create the final image
    result = img.copy()
    
    # Where the mask is 0 (which means it's the original outer background), change it to white (255)
    result[mask == 0] = 255

    # 7. Save the result
    cv2.imwrite(output_path, result)
    print(f"Success! Image saved to {output_path}")

# Example Usage:
input_filename = "data/X.png" # Path to your source image
output_filename = "data/X_white_bg.png" # Name for the output

# First, ensure you have the libraries installed:
# pip install opencv-python numpy


if __name__ == "__main__":
    # Run the function:
    convert_background_to_white(input_filename, output_filename)
import cv2
import numpy as np

def convert_background_to_white(image):
    img = image.copy()
    #convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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

    return result
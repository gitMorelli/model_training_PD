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


def get_tiles(image,coords,num_tiles):
    # Create a copy to avoid modifying the original image array
    output_img = image.copy()
    h, w = output_img.shape[:2]
    #identify all the tiles defined by the coordinates
    #build the list of UL coordinates
    new_coords = []
    x_coords = [0]+sorted(list(coords[0, :]))+[w]
    y_coords = [0]+sorted(list(coords[1, :]))+[h]
    size = len(x_coords)-1
    processed_tiles=0
    for i in range(size):
        for j in range(size):
            processed_tiles+=1
            if processed_tiles<=num_tiles:
                new_coords.append([(int(x_coords[j]), int(y_coords[i]),int(x_coords[j+1]), int(y_coords[i+1])),processed_tiles])
            else:
                new_coords.append([(int(x_coords[j]), int(y_coords[i]),int(x_coords[j+1]), int(y_coords[i+1])),-1]) #empty tiles
    return new_coords


def recolor_border_via_profiles(image, coords, black_tolerance=5):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, black_tolerance, 255, cv2.THRESH_BINARY)

    
    for tile in coords:
        x1, y1, x2, y2 = tile[0]
        tile_num = tile[1]
        tile_img = thresh[y1:y2, x1:x2]
        # Compute the sum of pixel values along rows and columns
        x_profile = np.sum(tile_img, axis=0)
        y_profile = np.sum(tile_img, axis=1)

        # 4. Find the indices where the profiles are greater than 0
        # This means there is at least one non-black pixel in that row/column
        x_content_indices = np.where(x_profile > 0)[0]
        y_content_indices = np.where(y_profile > 0)[0]

        # Handle the edge case where the image is entirely black
        if len(x_content_indices) == 0 or len(y_content_indices) == 0 or tile_num==-1:
            #set the tile to white
            image[y1:y2, x1:x2] = 255
            continue
        
        # 5. Identify the bounding box coordinates
        # The first and last indices represent the edges of the core content
        x_min, x_max = x_content_indices[0], x_content_indices[-1]
        y_min, y_max = y_content_indices[0], y_content_indices[-1]

        # 7. Create a white background of the original image size
        white_background = np.full_like(img[y1:y2,x1:x2], 255)

        # 8. Paste the core image into the exact same position on the white canvas
        white_background[y_min:y_max+1, x_min:x_max+1] = img[y1+y_min:y1+y_max+1, x1+x_min:x1+x_max+1].copy()

        image[y1:y2, x1:x2] = white_background.copy()
    return image

import json
import numpy as np
import cv2

def apply_perspective_transform(image, transformation_matrix, output_width, output_height):
    transformed_image = cv2.warpPerspective(image, transformation_matrix, (output_width, output_height))
    return transformed_image

def main():
    with open('transformation_matrix.json', 'r') as f:
        matrix_list = json.load(f)
    transformation_matrix = np.array(matrix_list, dtype=np.float32)

    resized_image = cv2.imread('resized_image.jpg')
    if resized_image is None:
        raise FileNotFoundError("Could not load 'resized_image.jpg'. Make sure the file exists.")

    output_width = 800
    output_height = 600

    warped_image = apply_perspective_transform(resized_image, transformation_matrix, output_width, output_height)

    cv2.imshow('Warped View', warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('warped_view.jpg', warped_image)
    print("Warped image saved as 'warped_view.jpg'.")

if __name__ == "__main__":
    main()

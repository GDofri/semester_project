import numpy as np
import cv2

# Curtesy of gpt-o1-preview. 28.09.2024

def generate_gradient_image(height, width, start_value=0.0, end_value=0.5):
    """
    Generates a vertical gradient image from start_value to end_value.

    Parameters:
    - height (int): Height of the image.
    - width (int): Width of the image.
    - start_value (float): Starting value of the gradient (default is 0.0).
    - end_value (float): Ending value of the gradient (default is 0.5).

    Returns:
    - image (np.ndarray): Generated gradient image.
    """
    # Create a vertical gradient from start_value to end_value
    gradient = np.linspace(start_value, end_value, height, dtype=np.float32)
    gradient = np.tile(gradient[:, np.newaxis], (1, width))
    return gradient

def is_border_pixel(x, y, width, height):
    """
    Checks if a pixel is on the border of the image.

    Parameters:
    - x (int): x-coordinate of the pixel.
    - y (int): y-coordinate of the pixel.
    - width (int): Width of the image.
    - height (int): Height of the image.

    Returns:
    - bool: True if the pixel is on the border, False otherwise.
    """
    return x == 0 or x == width - 1 or y == 0 or y == height - 1

def draw_line_on_image(image, point1, point2, color=1.0, thickness=1):
    """
    Draws a line between two border pixels on the image.

    Parameters:
    - image (np.ndarray): The image on which to draw.
    - point1 (tuple): (x, y) coordinates of the first point.
    - point2 (tuple): (x, y) coordinates of the second point.
    - color (float): Color of the line (default is 1.0).
    - thickness (int): Thickness of the line (default is 1).
    """
    height, width = image.shape
    # Validate that the specified points are on the border
    if not (is_border_pixel(point1[0], point1[1], width, height) and
            is_border_pixel(point2[0], point2[1], width, height)):
        raise ValueError("Both points must be on the border of the image.")
    # Draw a line between the two border pixels
    cv2.line(image, point1, point2, color=color, thickness=thickness)

def create_gradient_image_with_line(height, width, point1, point2,
                                    start_value=0.3, end_value=0.7,
                                    color=1.0, thickness=1):
    """
    Creates a gradient image and draws a line between two border pixels.

    Parameters:
    - height (int): Height of the image.
    - width (int): Width of the image.
    - point1 (tuple): (x, y) coordinates of the first border pixel.
    - point2 (tuple): (x, y) coordinates of the second border pixel.
    - start_value (float): Starting value of the gradient (default is 0.0).
    - end_value (float): Ending value of the gradient (default is 0.5).
    - color (float): Color of the line (default is 1.0).
    - thickness (int): Thickness of the line (default is 1).

    Returns:
    - image (np.ndarray): The final image with the gradient and line.
    """
    image = generate_gradient_image(height, width, start_value, end_value)
    draw_line_on_image(image, point1, point2, color, thickness)
    return image

# Example usage:
if __name__ == "__main__":
    # Define image dimensions
    height = 100
    width = 200

    # Specify two border pixels (change these as needed)
    point1 = (0, 50)    # Left border at row 50
    point2 = (199, 50)  # Right border at row 50

    # Create the image with the line
    image = create_gradient_image_with_line(height, width, point1, point2)

    # For visualization, scale the image to 0-255 and convert to uint8
    image_display = (image * 255).astype(np.uint8)

    # Save the image to a file
    cv2.imwrite('gradient_with_line.png', image_display)

    # Optionally, display the image
    cv2.imshow("Gradient Image with Line", image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

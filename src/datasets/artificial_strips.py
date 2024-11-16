import numpy as np
import os
import csv
from src import utils
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
def generate_sine_images(
        num_images,
        min_width,
        max_width,
        thickness,
        smoothing,
        output_dir=os.path.join(utils.get_project_root(), 'src', 'datasets', 'artificial_strips'),
        save_data=True
):
    """
    Generates images of sine waves with specified parameters.

    Parameters:
    - num_images (int): Number of images to generate.
    - min_width (int): Minimum width of the images.
    - max_width (int): Maximum width of the images.
    - thickness (int): Thickness of the sine wave.
    - smoothing (float): Standard deviation for Gaussian kernel.
    - output_dir (str): Directory to save images and CSV file.
    - save_data (bool): Whether to save images and CSV file to disk.

    Returns:
    - image_list (list of np.ndarray): List of generated image arrays.
    - df_params (pd.DataFrame): DataFrame containing parameters of each image.
    """
    h = 32  # Fixed height of the images

    if save_data:
        os.makedirs(output_dir, exist_ok=True)
        # Prepare CSV file to store image parameters
        csv_filename = os.path.join(output_dir, 'image_parameters.csv')
        csv_file = open(csv_filename, mode='w', newline='')
        fieldnames = [
            'filename',
            'width',
            'frequency',
            'phase',
            'start_height',
            'end_height'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    else:
        writer = None

    image_list = []
    param_list = []

    for i in tqdm(range(num_images), total=num_images, desc='Generating images'):
        # Random width between min_width and max_width
        width = np.random.randint(min_width, max_width + 1)

        # X-axis values
        x = np.linspace(0, width - 1, width)

        # Scale sine wave amplitude
        amplitude = h / 4  # Adjust as needed

        # Random start and end heights within the image height
        start_height = np.random.uniform(amplitude, h-amplitude)
        end_height = np.random.uniform(amplitude, h-amplitude)

        # Linearly interpolate between start_height and end_height
        base_line = np.linspace(start_height, end_height, width)

        # Random frequency and phase for the sine wave
        min_freq = 0.5  # Adjust to ensure visibility
        max_freq = 5.0
        frequency = np.random.uniform(min_freq, max_freq)
        phase = np.random.uniform(0, 2 * np.pi)

        # Generate the sine wave
        sine_wave = np.sin(2 * np.pi * frequency * x / width + phase)

        sine_wave_scaled = amplitude * sine_wave

        # Combine base line and sine wave
        y_values = base_line + sine_wave_scaled

        # Initialize a black image
        image = np.zeros((h, width), dtype=np.float32)

        # Draw the sine wave with specified thickness
        for xi in range(width):
            yi = int(y_values[xi])
            yi_thickness_range = range(
                max(0, yi - thickness),
                min(h, yi + thickness + 1)
            )
            for yj in yi_thickness_range:
                image[yj, xi] = 255.0  # Brightest part of the wave

        # Apply Gaussian smoothing
        if smoothing > 0:
            image = gaussian_filter(image, sigma=smoothing,  mode='constant', cval=-1000.0)

        # Normalize image to 0-255 and convert to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Append image to the list
        image_list.append(image)

        # Prepare parameters dictionary
        filename = f'image_{i}.npy'
        params = {
            'filename': filename,
            'width': width,
            'frequency': frequency,
            'phase': phase,
            'start_height': start_height,
            'end_height': end_height
        }
        param_list.append(params)

        if save_data:
            # Save the image as a numpy array
            filepath = os.path.join(output_dir, filename)
            np.save(filepath, image)

            # Write image parameters to CSV
            writer.writerow(params)

    if save_data:
        csv_file.close()
        print(f'\nAll images and parameters saved in "{output_dir}".')

    # Create a DataFrame from the parameters list
    df_params = pd.DataFrame(param_list)

    return image_list, df_params

# Example usage
if __name__ == '__main__':
    num_images = 20       # Define the number of images to generate
    min_width = 64        # Minimum image width
    max_width = 128       # Maximum image width
    thickness = 2         # Thickness of the sine wave
    smoothing = 1.0       # Smoothing factor for Gaussian blur

    images, df_parameters = generate_sine_images(
        num_images,
        min_width,
        max_width,
        thickness,
        smoothing,
        save_data=False  # Set to False to prevent saving data
    )
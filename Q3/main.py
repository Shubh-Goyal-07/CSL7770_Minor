# IMPORTS
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import pandas as pd
import logging
import warnings

warnings.filterwarnings('ignore')


# DEFINING CONSTANTS
FRAME_SIZE = 1024
HOP_SIZE = 512
DATASET_DIR = 'dataset'
FEATURES_FILE_NAME = 'features.csv'


# FUNCTION DEFINITIONS
def load_audio(file_path):
    """
    Load the audio file from the given path.
    Returns the audio signal and the sampling rate.
    """

    y, sr = librosa.load(file_path, sr=None)
    
    return y, sr


def extract_formants(y, sr, order):
    """
    Extract the first three formants from the audio signal.
    This function uses the Burg's method to extract the formants.
    """

    # Initialize forward and backward error arrays
    # same as the input signal but shifted by one sample
    forward_error = y[1:].copy()
    backward_error = y[:-1].copy()
    
    # Initialize arrays to store LPC coefficients
    lpc_coefs = np.zeros(order + 1)
    lpc_coefs[0] = 1.0
    previous_coefs = lpc_coefs.copy()

    # eps to make sure we don't divide by zero
    tiny = np.finfo(y.dtype).eps
    
    # Initialize energy of the signal
    energy = np.dot(forward_error, forward_error) + np.dot(backward_error, backward_error)
    
    # Burg's algorithm for LPC computation
    for i in range(order):
        # Calculate the reflection coefficient
        numerator = -2.0 * np.dot(backward_error, forward_error)
        k = numerator / (energy + tiny)

        previous_coefs, lpc_coefs = lpc_coefs.copy(), previous_coefs.copy()

        # Update LPC coefficients
        for j in range(1, i + 2):
            lpc_coefs[j] = previous_coefs[j] + k * previous_coefs[i - j + 1]
        
        # Update forward and backward errors
        temp_forward = forward_error.copy()
        forward_error = forward_error + k * backward_error
        backward_error = backward_error + k * temp_forward

        # Update energy
        energy_factor = 1.0 - k**2
        energy = energy_factor * energy
        
        # Update forward and backward errors
        # If the length of the forward and backward errors is greater than 1
        # update the energy and update the forward and backward errors
        if len(forward_error) > 1 and len(backward_error) > 1:
            energy -= forward_error[0]**2 + backward_error[-1]**2
        
        # if len(forward_error) > 1 and len(backward_error) > 1:
            forward_error = forward_error[1:]
            backward_error = backward_error[:-1]
    
    # Find roots of the polynomial defined by LPC coefficients
    roots = np.roots(lpc_coefs)
    
    # Filter the roots based on the following conditions:
    # 1. Magnitude less than 1 are valid
    # 2. Positive imaginary part (we only need one of each complex conjugate pair) are valid
    roots = roots[np.abs(roots) < 1]
    positive_im_roots = roots[np.imag(roots) > 0]
    
    # Calculate angles from the roots
    phase_angles = np.angle(positive_im_roots)
    formants_hz = phase_angles * (sr / (2 * np.pi))
    
    # Sort frequencies in ascending order
    formants_hz = np.sort(formants_hz)
    
    # Check results
    result = [0.0, 0.0, 0.0]
    for i in range(min(len(formants_hz), 3)):
        result[i] = formants_hz[i]

    return result[0], result[1], result[2]


def calculate_fundamental_frequency(y, sr):
    """
    Calculate the fundamental frequency of the audio signal.
    This function calculates the fundamental frequency using the autocorrelation method.
    """
    
    # Pad the audio signal if needed so that the last frame is of size FRAME_SIZE
    pad_length = (len(y) - FRAME_SIZE) % (HOP_SIZE)
    padded_audio = np.pad(y, (0, pad_length))

    num_frames = int((len(padded_audio) - FRAME_SIZE) / HOP_SIZE)

    all_f0 = []

    # Loop through each frame and calculate the fundamental frequency
    for i in range(num_frames):
        frame = padded_audio[i * HOP_SIZE: i * HOP_SIZE + FRAME_SIZE]

        # Calculate the autocorrelation of the frame
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[FRAME_SIZE - 1:]

        # Find the peak in the autocorrelation
        min_lag = int(sr / 400)
        max_lag = int(sr / 50)

        peak = np.argmax(corr[min_lag:max_lag]) + min_lag

        # Calculate the fundamental frequency
        f0 = sr / peak if peak > 0 else 0

        if f0 > 0:
            all_f0.append(f0)
        else:
            all_f0.append(0)

    # Return the mean fundamental frequency of all frames
    all_f0 = np.array(all_f0)

    return np.mean(all_f0)


def extract_audio_features(y, sr):
    """
    Extract the audio features from the audio signal.
    This function extracts the fundamental frequency, and the first three formants.
    """
    
    # Reduce noise in the audio signal
    y = librosa.effects.preemphasis(y)

    # Calculate the fundamental frequency
    f0 = calculate_fundamental_frequency(y, sr)
    # Extract the first three formants
    f1, f2, f3 = extract_formants(y, sr, order=16)
    
    f0 = float(f0)
    f1 = float(f1)
    f2 = float(f2)
    f3 = float(f3)

    return f0, f1, f2, f3


def save_dataset_features(features_data):
    """
    Save the extracted features to a CSV file.
    """
    
    df = pd.DataFrame.from_dict(features_data)
    df.to_csv(FEATURES_FILE_NAME)


def make_vowel_space_plot(features_data):
    """
    Make a vowel space using the first two formants (F1 and F2).
    The vowels are color coded based on the vowel type.
    """

    # defining and assigning colors to vowels
    vowels = ['a', 'e', 'i', 'o', 'u']
    cmap = plt.get_cmap('viridis', len(vowels))
    c = [ vowels.index(v) for v in features_data['vowel'] ]

    # Plotting the vowel space with legend for vowels
    plt.figure(figsize=(12, 8))
    plt.scatter(features_data['f1'], features_data['f2'], c=c, cmap=cmap, s=100, edgecolors='k')

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10) 
               for i in range(len(vowels))]
    plt.legend(handles, vowels, title="Vowel Labels")

    plt.xlabel('F1 (Hz)')
    plt.ylabel('F2 (Hz)')
    plt.title('Vowel Space')
    plt.tight_layout()
    plt.savefig('vowel_space.png')
    plt.close()


def visualize_audio_features(features_data):
    """
    Visualize the audio features extracted from the dataset using scatter plots.
    """

    # make 4 plots in a 4x1 grid to represent f0, f1, f2, f3 vs vowel
    fig, axs = plt.subplots(4, 1, figsize=(15, 50))  
    tick_font_size = 24  
    font_size = 24  
    marker_size = 200  
    vowels = ['a', 'e', 'i', 'o', 'u']
    c = [vowels.index(v) for v in features_data['vowel']]  # Assign colors

    # Increase vertical space between subplots
    plt.subplots_adjust(hspace=0.2)

    # Plot F0 vs Vowel
    axs[0].scatter(features_data['vowel'], features_data['f0'], c=c, cmap='viridis', s=marker_size)
    axs[0].set_title('F0 vs Vowel', fontsize=font_size)
    axs[0].set_xlabel('Vowel', fontsize=font_size)
    axs[0].set_ylabel('F0 (Hz)', fontsize=font_size)
    axs[0].tick_params(axis='both', labelsize=tick_font_size)
    axs[0].grid()

    # Plot F1 vs Vowel
    axs[1].scatter(features_data['vowel'], features_data['f1'], c=c, cmap='viridis', s=marker_size)
    axs[1].set_title('F1 vs Vowel', fontsize=font_size)
    axs[1].set_xlabel('Vowel', fontsize=font_size)
    axs[1].set_ylabel('F1 (Hz)', fontsize=font_size)
    axs[1].tick_params(axis='both', labelsize=tick_font_size)
    axs[1].grid()

    # Plot F2 vs Vowel
    axs[2].scatter(features_data['vowel'], features_data['f2'], c=c, cmap='viridis', s=marker_size)
    axs[2].set_title('F2 vs Vowel', fontsize=font_size)
    axs[2].set_xlabel('Vowel', fontsize=font_size)
    axs[2].set_ylabel('F2 (Hz)', fontsize=font_size)
    axs[2].tick_params(axis='both', labelsize=tick_font_size)
    axs[2].grid()

    # Plot F3 vs Vowel
    axs[3].scatter(features_data['vowel'], features_data['f3'], c=c, cmap='viridis', s=marker_size)
    axs[3].set_title('F3 vs Vowel', fontsize=font_size)
    axs[3].set_xlabel('Vowel', fontsize=font_size)
    axs[3].set_ylabel('F3 (Hz)', fontsize=font_size)
    axs[3].tick_params(axis='both', labelsize=tick_font_size)
    axs[3].grid()

    # Save the figure
    plt.savefig('audio_features.png')
    plt.tight_layout()
    plt.close()


def process_dataset():
    """
    Process the dataset and extract the features from all the audio files.
    """

    logging.info('Processing dataset...\n\n')

    features_data = {
        'file_name': [],
        'f0': [],
        'f1': [],
        'f2': [],
        'f3': [],
        'vowel': [],
    }

    # Loop through each file in the dataset and extract the features
    for dir in os.listdir(DATASET_DIR):
        for vowel_dir in os.listdir(f'{DATASET_DIR}/{dir}'):
            logging.info(f'Processing {dir}/{vowel_dir}...\n')
            for file in os.listdir(f'{DATASET_DIR}/{dir}/{vowel_dir}'):
                # Get the file path
                file_path = f'{DATASET_DIR}/{dir}/{vowel_dir}/{file}'

                # Load the audio file
                y, sr = load_audio(file_path)
                f0, f1, f2, f3 = extract_audio_features(y, sr)

                # Append the features to the data dictionary
                features_data['file_name'].append(file)
                features_data['f0'].append(f0)
                features_data['f1'].append(f1)
                features_data['f2'].append(f2)
                features_data['f3'].append(f3)
                features_data['vowel'].append(vowel_dir)

    logging.info('Saving features data...\n')
    # Save the features data to a CSV file
    save_dataset_features(features_data)

    logging.info('Making vowel space plot...\n')
    # Make a vowel space plot and save it
    make_vowel_space_plot(features_data)

    logging.info('Visualizing audio features...\n')
    # Visualize the audio features
    visualize_audio_features(features_data)

    logging.info('Processing complete!')


# MAIN FUNCTION
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, filename=f'feat_extract_logs.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filemode='w')
    
    process_dataset()
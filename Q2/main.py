# IMPORTS
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import json
import logging
import argparse
import warnings

warnings.filterwarnings('ignore')


# DEFINING CONSTANTS
FRAME_SIZE = 1024
HOP_SIZE = 512
N_MFCC = 13
LEADER_DATASET_DIR = 'Speeches_of_leaders'
FEATURES_DIR = 'features'
PLOTS_DIR = 'plots'


# FUNCTION DEFINITIONS
def check_output_dir():
    """
    Check if the output directories exist and delete them if they do.
    Create the output directories.
    """

    if os.path.exists(FEATURES_DIR):
        os.system(f'rm -rf {FEATURES_DIR}')

    if os.path.exists(PLOTS_DIR):
        os.system(f'rm -rf {PLOTS_DIR}')

    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    return True


def load_audio(file_path):
    """
    Load the audio file from the given path.
    Returns the audio signal and the sampling rate.
    """

    y, sr = librosa.load(file_path, sr=None)
    
    return y, sr


def save_json(data, file_name):
    """
    Save the audio signal as a waveform plot.
    """

    with open(f'{FEATURES_DIR}/{file_name}.json', 'w') as f:
        json.dump(data, f, indent=4, default=str)


def calculate_zcr(audio):
    """
    Calculate the zero crossing rate of the audio signal. For each frame, the zero crossing rate is calculated.
    """

    # Pad the audio signal if needed so that the last frame is of size FRAME_SIZE
    pad_length = (len(audio) - FRAME_SIZE) % (HOP_SIZE)
    padded_audio = np.pad(audio, (0, pad_length))

    num_frames = int((len(padded_audio) - FRAME_SIZE) / HOP_SIZE)

    zcr_values = []

    # For each frame, calculate the zero crossing rate
    for i in range(num_frames):
        frame = padded_audio[i * HOP_SIZE: i * HOP_SIZE + FRAME_SIZE]
        
        # Calculates the difference between consecutive samples
        # np.sign returns 1 for positive values, 0 for 0 and -1 for negative values
        # If the sign of consecutive samples is different, then there it gets counted as a zero crossing
        # Divide by 2 to normalize the value of each crossing 
        zcr = np.sum(np.abs(np.sign(frame[1:]) - np.sign(frame[:-1])) / 2) / FRAME_SIZE

        zcr_values.append(zcr)

    zcr_values = np.array(zcr_values)

    # Calculate the mean zero crossing rate
    zcr_mean = np.mean(zcr_values)

    return zcr_mean


def calculate_short_time_energy(audio):
    """
    Calculate the short time energy of the audio signal. For each frame, the energy is calculated.
    """
    
    # Pad the audio signal if needed so that the last frame is of size FRAME_SIZE 
    pad_length = (len(audio) - FRAME_SIZE) % (HOP_SIZE)
    padded_audio = np.pad(audio, (0, pad_length))

    num_frames = int((len(padded_audio) - FRAME_SIZE) / HOP_SIZE)

    energy_values = []

    # Loop through each frame and calculate the energy
    for i in range(num_frames):
        frame = padded_audio[i * HOP_SIZE: i * HOP_SIZE + FRAME_SIZE]
        
        # energy is the sum of squares of the samples in the frame
        energy = np.sum(frame ** 2)

        energy_values.append(energy)

    energy_values = np.array(energy_values)

    # Calculate the mean energy
    energy_mean = np.mean(energy_values)

    return energy_mean


def extract_audio_features(y, sr):
    """
    Extract the audio features from a given audio signal.
    """
    
    # Dictionary to store the features
    features = {}

    # 1. Zero Crossing Rate
    zcr = calculate_zcr(y)
    features['zcr'] = float(zcr)

    # 2. Short Time Energy
    energy = calculate_short_time_energy(y)
    features['ste'] = float(energy)

    # 3. MFCCs (using librosa)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_SIZE, n_fft=FRAME_SIZE)

    # Calculate the mean and standard deviation of the MFCCs
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    for i in range(N_MFCC):
        features[f'mfccs_mean_{i+1}'] = float(mfccs_mean[i])
        features[f'mfccs_std_{i+1}'] = float(mfccs_std[i])

    return features
    

def visualize_audio_features(features_all):
    """
    Visualize and save the auido features as line plots.
    """

    audio_features = {
        'zcr': [],
        'ste': [],
        'file': []
    }

    for i in range(N_MFCC):
        audio_features[f'mfccs_mean_{i+1}'] = []
        audio_features[f'mfccs_std_{i+1}'] = []

    for file, features in features_all.items():
        audio_features['zcr'].append(features['zcr'])
        audio_features['ste'].append(features['ste'])
        audio_features['file'].append(file.split('/')[-1].split('.')[0].split('_')[0].split(' ')[0])

        for i in range(N_MFCC):
            audio_features[f'mfccs_mean_{i+1}'].append(features[f'mfccs_mean_{i+1}'])
            audio_features[f'mfccs_std_{i+1}'].append(features[f'mfccs_std_{i+1}'])

    # sort as per ste
    audio_features['file'] = [x for _, x in sorted(zip(audio_features['ste'], audio_features['file']))]
    audio_features['zcr'] = [x for _, x in sorted(zip(audio_features['ste'], audio_features['zcr']))]
    audio_features['ste'] = sorted(audio_features['ste'])

    for i in range(N_MFCC):
        audio_features[f'mfccs_mean_{i+1}'] = [x for _, x in sorted(zip(audio_features['ste'], audio_features[f'mfccs_mean_{i+1}']))]
        audio_features[f'mfccs_std_{i+1}'] = [x for _, x in sorted(zip(audio_features['ste'], audio_features[f'mfccs_std_{i+1}']))]


    # Plot the zero crossing rate
    plt.figure(figsize=(18, 12))
    plt.plot(audio_features['file'], audio_features['zcr'])
    plt.title('Zero Crossing Rate')
    plt.xlabel('Frame')
    plt.ylabel('ZCR')
    plt.xticks(rotation=90)
    plt.savefig(f'{PLOTS_DIR}/zcr.png')
    plt.close()

    # Plot the short time energy
    plt.figure(figsize=(18, 12))
    plt.plot(audio_features['file'], audio_features['ste'])
    plt.title('Short Time Energy')
    plt.xlabel('Frame')
    plt.ylabel('Energy')
    plt.xticks(rotation=90)
    plt.savefig(f'{PLOTS_DIR}/ste.png')
    plt.close()

    # Plot the MFCCs
    for i in range(N_MFCC):
        plt.figure(figsize=(18, 12))
        plt.plot(audio_features['file'], audio_features[f'mfccs_mean_{i+1}'])
        plt.title(f'MFCC Mean {i+1}')
        plt.xlabel('Frame')
        plt.ylabel('MFCC')
        plt.xticks(rotation=90)
        plt.savefig(f'{PLOTS_DIR}/mfcc_mean_{i+1}.png')
        plt.close()

        plt.figure(figsize=(18, 12))
        plt.plot(audio_features['file'], audio_features[f'mfccs_std_{i+1}'])
        plt.title(f'MFCC Std {i+1}')
        plt.xlabel('Frame')
        plt.ylabel('MFCC')
        plt.xticks(rotation=90)
        plt.savefig(f'{PLOTS_DIR}/mfcc_std_{i+1}.png')
        plt.close()

    

def extract_dataset_features():
    """
    Extract the audio features from all the audio files in the given directory.
    """
    
    logging.info(f'Processing files in dir -- {LEADER_DATASET_DIR}\n\n')
    
    # Dictionary to store the features of all the files
    features_all = {}
    all_files = os.listdir(LEADER_DATASET_DIR)
    total_files = len(all_files)

    # Loop through each file and extract the features
    for i in range(total_files):
        file = all_files[i]
        logging.info(f'Processing {i+1}/{total_files} -- {file}')
        file_path = f'{LEADER_DATASET_DIR}/{file}'
        
        # Load the audio file
        y, sr = load_audio(file_path)

        # normalize the audio signal
        y = librosa.util.normalize(y)
        # Denoise the audio signal
        y = librosa.effects.preemphasis(y)

        # Extract the audio features
        features = extract_audio_features(y, sr)
        logging.info('Features extracted\n')
        
        # Save the features as a json file
        save_json(features, file[:-4])

        features_all[file_path] = features

    # Save the features of all the files
    save_json(features_all, 'ALL_FILES_FEATURES')

    logging.info('All files processed')
    logging.info(f'Features saved in dir -- {FEATURES_DIR}')

    # Visualize the audio features
    visualize_audio_features(features_all)
    logging.info('Audio feature plots saved')


# MAIN FUNCTION
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract audio features')
    parser.add_argument('--dir', type=str, help='Directory containing audio files')

    args = parser.parse_args()
    
    if args.dir:
        # Set the dataset directory
        LEADER_DATASET_DIR = args.dir
        FEATURES_DIR = f'{args.dir}_features'
        PLOTS_DIR = f'{args.dir}_plots'

    if not os.path.exists('logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename=f'logs/{LEADER_DATASET_DIR}.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filemode='w')

    # Check if the output directories exist and delete them if they do
    check_output_dir()

    # Extract the audio features from the dataset
    extract_dataset_features()

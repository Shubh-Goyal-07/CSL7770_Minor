# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import json
import logging
import warnings

warnings.filterwarnings('ignore')


# DEFINING CONSTANTS
FRAME_SIZE = 1024
HOP_SIZE = 512
DATASET_DIR = 'dataset'
CHARACTERISTICS_DIR = 'characteristics'
PLOTS_DIR = 'plots'
CURRENT_PLOT_DIR = ''


# FUNCTION DEFINITIONS
def check_output_dir():
    """
    Check if the output directories exist and delete them if they do.
    Create the output directories.
    """

    if os.path.exists(CHARACTERISTICS_DIR):
        os.system(f'rm -r {CHARACTERISTICS_DIR}')

    if os.path.exists(PLOTS_DIR):
        os.system(f'rm -r {PLOTS_DIR}')

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(CHARACTERISTICS_DIR, exist_ok=True)

    return


def load_audio(file_path):
    """
    Load the audio file at the given file path.
    Returns the audio signal and the sampling rate.
    """

    y, sr = librosa.load(file_path, sr=None)
    
    return y, sr


def save_audio_waveform(y, sr):
    """
    Save the audio signal as a waveform plot.
    """

    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(f'{CURRENT_PLOT_DIR}/waveform.png')
    plt.close()


def save_json(data, file_name):
    """
    Save the given data as a JSON file.
    """

    with open(f'{CHARACTERISTICS_DIR}/{file_name}.json', 'w') as f:
        json.dump(data, f, indent=4, default=str)


def calculate_rms(y):
    """
    Calculates the Root Mean Square (RMS) energy of the audio signal.
    Returns an array of RMS values for each frame.
    """

    # Pad the audio signal if needed so that the last frame is of size FRAME_SIZE
    pad_length = (len(y) - FRAME_SIZE) % (HOP_SIZE)
    padded_audio = np.pad(y, (0, pad_length))

    num_frames = int((len(padded_audio) - FRAME_SIZE) / HOP_SIZE)

    rms_values = []

    # Loop through each frame and calculate the RMS energy
    for i in range(num_frames):
        frame = padded_audio[i * HOP_SIZE: i * HOP_SIZE + FRAME_SIZE]

        # Calculate the RMS energy of the frame
        rms = np.sqrt(np.mean(frame**2))

        rms_values.append(rms)

    rms_values = np.array(rms_values)
    
    return rms_values


def save_rms(rms_values, sr):
    """
    Takes the RMS values and saves them as a plot.
    """

    plt.figure(figsize=(12, 4))
    plt.plot(librosa.times_like(rms_values, sr=sr, hop_length=HOP_SIZE), rms_values)
    plt.title('RMS Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Energy')
    plt.tight_layout()
    plt.savefig(f'{CURRENT_PLOT_DIR}/rms.png')
    plt.close()


def calculate_amplitude_envelope(y):
    """
    Function to calculate the amplitude envelope of the audio signal.
    Returns an array of amplitude values for each frame.
    """

    # Pad the audio signal if needed so that the last frame is of size FRAME_SIZE
    pad_length = (len(y) - FRAME_SIZE) % (HOP_SIZE)
    padded_audio = np.pad(y, (0, pad_length))

    num_frames = int((len(padded_audio) - FRAME_SIZE) / HOP_SIZE)

    amplitude_envelope = []
    
    # Loop through each frame and calculate the amplitude envelope
    for i in range(num_frames):
        frame = padded_audio[i * HOP_SIZE: i * HOP_SIZE + FRAME_SIZE]

        # find the maximum absolute amplitude in the frame
        amplitude = np.max(np.abs(frame))

        amplitude_envelope.append(amplitude)

    amplitude_envelope = np.array(amplitude_envelope)

    return amplitude_envelope


def save_amplitude_envelope(amplitude_envelope, sr):
    """
    Takes the amplitude envelope values and saves them as a plot.
    """

    plt.figure(figsize=(12, 4))
    plt.plot(librosa.times_like(amplitude_envelope, sr=sr, hop_length=HOP_SIZE), amplitude_envelope)
    plt.title('Amplitude Envelope')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(f'{CURRENT_PLOT_DIR}/amplitude.png')
    plt.close()


def calculate_pitch(y, sr):
    """
    Calculates the pitch of the audio signal. Uses the piptrack function from librosa.
    Returns the mean, max and min pitch values along with the pitch values for each frame.
    """

    # Calculate the pitch using the piptrack function
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch_values = []

    # Loop through each frame and find the pitch with the maximum magnitude
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    # Calculate the mean, max and min pitch values
    if pitch_values:
        pitch_mean = np.mean(pitch_values)
        pitch_max = np.max(pitch_values)
        pitch_min = np.min(pitch_values)
    else:
        pitch_mean = 0
        pitch_max = 0
        pitch_min = 0

    pitch_values = np.array(pitch_values)

    return pitch_mean, pitch_max, pitch_min, pitch_values


def save_pitch(pitch_values, sr):
    """
    Takes the pitch values and saves them as a plot.
    """

    plt.figure(figsize=(12, 4))
    plt.plot(librosa.times_like(pitch_values, sr=sr, hop_length=HOP_SIZE), pitch_values)
    plt.title('Pitch')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(f'{CURRENT_PLOT_DIR}/pitch.png')
    plt.close()


def calculate_frequency(y, sr):
    """
    Calculates the frequency of the audio signal using the pyin function from librosa.
    Returns the mean frequency and the frequency values for each frame.
    """
    
    # Calculate the frequency using the pyin function, with a frequency range of 65 - 2100 Hz
    freq, _, _ = librosa.pyin(y, fmin=65, fmax=2100, sr=sr)
    # Filter out negative frequencies
    freq = np.array([f if f > 0 else 0 for f in freq])
    
    mean_freq = np.mean(freq)

    return mean_freq, freq


def analyze_audio_file(y, sr):
    """
    Analyze the given audio file and return the results. And the frequency values for each frame.
    """
    
    # Save the audio waveform plot
    save_audio_waveform(y, sr)

    results = {}        # Dictionary to store the results
    
    # 1. Calculate the RMS energy
    rms_values = calculate_rms(y)
    results['rms_mean'] = float(np.mean(rms_values))
    results['rms_max'] = float(np.max(rms_values))

    # Save the RMS energy plot
    save_rms(rms_values, sr)


    # 2. Calculate the amplitude envelope
    amplitude_envelope = calculate_amplitude_envelope(y)
    amp_max = np.max(amplitude_envelope)
    amp_mean = np.mean(amplitude_envelope)

    results['amplitude_max'] = float(amp_max)
    results['amplitude_mean'] = float(amp_mean)

    # Save the amplitude envelope plot
    save_amplitude_envelope(amplitude_envelope, sr)
    

    # 3. Calculate the pitch information
    pitch_mean, pitch_max, pitch_min, pitch_values = calculate_pitch(y, sr)
    results['pitch_mean'] = float(pitch_mean)
    results['pitch_max'] = float(pitch_max)
    results['pitch_min'] = float(pitch_min)

    # Save the pitch plot
    save_pitch(pitch_values, sr)


    # 4. Calculate the frequency information
    frequency, frequency_values = calculate_frequency(y, sr)
    results['frequency'] = frequency

    return results, frequency_values


def make_save_spectrogram(y, sr, freq_vals):
    """
    Function to create and save the spectrogram of the audio signal.
    Also plots the frequency values on the spectrogram.
    """

    # Create the spectrogram using librosa
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Plot the spectrogram
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_axis_off()
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    
    # Plot the frequency values on the spectrogram
    freq_vals[freq_vals == 0] = np.nan
    ax.plot(librosa.times_like(freq_vals, sr=sr, hop_length=HOP_SIZE), freq_vals, color='cyan', linewidth=2)

    # Save the spectrogram
    plt.savefig(f'{CURRENT_PLOT_DIR}/spectrogram.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def analyze_dataset(input_dir):
    """
    Analyze the dataset in the given input directory.
    """
    
    logging.info(f'Analyzing dataset in {input_dir}...\n\n')

    # to update the current output plot directory
    global CURRENT_PLOT_DIR
    characteristics_all = {}

    # Loop through all the audio files in the input directory
    for audio_file in os.listdir(input_dir):
        if audio_file.endswith('.wav'):
            logging.info(f'Analyzing {audio_file}...')
            
            # update the current plot directory
            CURRENT_PLOT_DIR = os.path.join(PLOTS_DIR, audio_file[:-4])
            os.makedirs(CURRENT_PLOT_DIR, exist_ok=True)
            
            file_path = os.path.join(input_dir, audio_file)
            
            # Load the audio file
            y, sr = load_audio(file_path)
            
            # Analyze the audio file
            results, freq_vals = analyze_audio_file(y, sr)
            results['file_name'] = audio_file

            logging.info(f'Saving characteristics for {audio_file}...')
            # Save the results as a JSON file
            save_json(results, audio_file[:-4])
            characteristics_all[audio_file] = results

            logging.info(f'Saving spectrogram for {audio_file}...')
            make_save_spectrogram(y, sr, freq_vals)
            logging.info(f'Analysis complete for {audio_file}!\n\n')

    # save the characteristics of all files in a single json file
    save_json(characteristics_all, 'all_characteristics')

    logging.info('Analysis complete!')
    
    return 


# MAIN FUNCTION
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, filename=f'logs.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filemode='w')

    # Check if the output directories exist and create them
    check_output_dir()

    # Analyze the dataset
    analyze_dataset(DATASET_DIR)
    
import os
import logging
from tkinter import Tk, Label, Button, filedialog, messagebox, StringVar
from tkinter.ttk import Progressbar
from pydub import AudioSegment
from pydub.utils import mediainfo
from spleeter.separator import Separator
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


# Configuration
logging.basicConfig(
    filename="nasheed_tool.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# output directory
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "Processed_Files")

# Ensure the default output directory exists
if not os.path.exists(DEFAULT_OUTPUT_DIR):
    os.makedirs(DEFAULT_OUTPUT_DIR)





# Function to see if a directory is writable
def is_directory_writable(directory):
    try:
        testfile = os.path.join(directory, "test.tmp")
        with open(testfile, "w") as f:
            f.write("test")
        os.remove(testfile)
        return True
    except IOError:
        return False



# Function to get file size
def validate_file_size(file_path, max_size_mb=50):
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    if file_size_mb > max_size_mb:
        raise ValueError(f"File size exceeds the maximum allowed limit of {max_size_mb} MB.")



# Function to process and separate the audio file
def separate_audio(file_path, output_dir, progress_bar=None, status_label=None):
    try:
        if progress_bar:
            progress_bar["value"] = 10  # Start progress
        if status_label:
            status_label.set("Validating file...")

        # Validate file type
        valid_formats = ['mp3', 'wav', 'flac', 'aac']
        file_extension = file_path.split('.')[-1].lower()
        if file_extension not in valid_formats:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: {', '.join(valid_formats)}")
        
        if progress_bar:
            progress_bar["value"] = 30  # Update
        if status_label:
            status_label.set("Preparing for separation...")

        # Ensure output directory for separation
        separation_output_dir = os.path.join(output_dir, "Separated")
        if not os.path.exists(separation_output_dir):
            os.makedirs(separation_output_dir)

        if progress_bar:
            progress_bar["value"] = 50  # Update progress
        if status_label:
            status_label.set("Separating audio...")

        # Initialize Spleeter (2 stems: vocals and accompaniment)
        separator = Separator('spleeter:2stems')
        separator.separate_to_file(file_path, separation_output_dir)

        if progress_bar:
            progress_bar["value"] = 100  # Finish progress
        if status_label:
            status_label.set("Separation completed!")

        messagebox.showinfo("Success", f"Audio separated successfully!\nFiles saved in:\n{separation_output_dir}")
    except Exception as e:
        if progress_bar:
            progress_bar["value"] = 0  
        if status_label:
            status_label.set("Error occurred")
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


# Function to extract melody and rhythm features
def extract_features(audio_path):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)  
        print(f"Audio loaded. Sample rate: {sr}, Duration: {len(y)/sr:.2f} seconds")

        # Extract pitch (melody)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = [p[np.argmax(m)] for p, m in zip(pitches.T, magnitudes.T) if np.max(m) > 0]
        avg_pitch = np.mean(pitch)
        print(f"Average Pitch: {avg_pitch:.2f} Hz")

        # Extract tempo and beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        print(f"Tempo: {tempo:.2f} BPM")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        print(f"Beat Times: {beat_times}")

        # Visualize waveform
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

        # Visualize spectrogram
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="log")
        plt.title("Log-Frequency Spectrogram")
        plt.colorbar(format="%+2.0f dB")
        plt.show()

        return {
            "average_pitch": avg_pitch,
            "tempo": tempo,
            "beat_times": beat_times
        }
    except Exception as e:
        print(f"Error: {e}")
        return None
    


# Function to open file dialog
def open_file_dialog(file_path_var, progress_bar, status_label):
    file_path = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=(("Audio Files", "*.mp3 *.wav *.flac *.aac"), ("All Files", "*.*"))
    )
    if file_path:
        file_path_var.set(file_path)  # Update the file path display
        progress_bar["value"] = 10   # Update progress bar
        status_label.set("Processing...")  # Update status label
        
        # Call the audio separation process
        separate_audio(file_path, DEFAULT_OUTPUT_DIR, progress_bar, status_label)

# Function to change output directory
def change_output_directory():
    new_dir = filedialog.askdirectory(title="Select Output Directory")
    if new_dir:
        global DEFAULT_OUTPUT_DIR
        DEFAULT_OUTPUT_DIR = new_dir
        messagebox.showinfo("Output Directory Changed", f"New output directory:\n{DEFAULT_OUTPUT_DIR}")

# Main GUI setup
def main():
    root = Tk()
    root.title("Nasheed Transformation Tool")
    root.geometry("600x400")
    
    # File path variable
    file_path_var = StringVar()
    file_path_var.set("No file selected")
    
    # Status variable
    status_label_var = StringVar()
    status_label_var.set("Ready")

    # Progress bar
    progress_bar = Progressbar(root, orient="horizontal", length=500, mode="determinate")
    
    # Add GUI components
    Label(root, text="Welcome to Nasheed Transformation Tool", font=("Arial", 14)).pack(pady=10)
    Label(root, textvariable=file_path_var, font=("Arial", 10), wraplength=500, justify="center").pack(pady=5)
    progress_bar.pack(pady=10)  # Pack the progress bar here
    Label(root, textvariable=status_label_var, font=("Arial", 10), fg="green").pack(pady=5)

    Button(root, text="Select an Audio File", 
           command=lambda: open_file_dialog(file_path_var, progress_bar, status_label_var), font=("Arial", 12)).pack(pady=10)
    Button(root, text="Change Output Directory", command=change_output_directory, font=("Arial", 12)).pack(pady=10)
    Button(root, text="Exit", command=root.quit, font=("Arial", 12)).pack(pady=10)
    
    root.mainloop()
    


if __name__ == "__main__":
    main() 
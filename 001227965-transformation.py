import os
import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter, fftconvolve
import nltk
import warnings
import itertools
from tkinter import Tk, filedialog
import datetime

# CONFIGURATION 
INSTRUMENTAL_PATH = r"C:\Users\yusuf\Desktop\final year project\Separated\audio\accompaniment.wav"
OUTPUT_PATH = "nasheed_vocal_enhanced_LAST.wav"


#environment setup

if os.name == 'nt':
    import signal
    if not hasattr(signal, 'SIGKILL'):
        signal.SIGKILL = signal.SIGTERM
    os.environ['NEMO_CACHE_DIR'] = 'C:/temp/nemo_cache'


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
nltk.download('punkt', quiet=True)

# Import NeMo
from nemo.collections.tts.models import Tacotron2Model, WaveGlowModel


def verify_audio_file(audio_path):
    """Validate the input audio file"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
    try:
        librosa.load(audio_path, sr=22050)
        return True
    except Exception as e:
        raise ValueError(f"Invalid audio file: {str(e)}")


def extract_melody(audio_file):
    """Robust melody extraction"""
    try:
        y, sr = librosa.load(audio_file, sr=22050)
        f0, _, _ = librosa.pyin(y, fmin=80, fmax=1000, sr=sr)
        f0 = np.nan_to_num(f0, nan=0)

        if np.count_nonzero(f0) < 0.05 * len(f0):
            raise ValueError("Melody extraction failed: Too few detected pitches")

        melody_midi = librosa.hz_to_midi(np.clip(f0, 80, 1000))
        return np.clip(melody_midi, 0, 127), sr

    except Exception as e:
        raise RuntimeError(f"Melody extraction failed: {str(e)}")


def melody_to_vocals(melody):
    """Convert MIDI notes into structured vocal phrases"""
    vocal_phrases = []
    for note in melody:
        note = int(np.round(note))
        if 0 < note < 60:
            vocal_phrases.append("mhm hmm hmm hmm")
        elif 60 <= note < 72:
            vocal_phrases.append("aaah laa laa")
        elif note >= 72:
            vocal_phrases.append("ooooh eeeh ooooh")
        else:
            vocal_phrases.append(" ")
    return " ".join(k for k, _ in itertools.groupby(vocal_phrases))


def load_models():
    """Load AI models"""
    try:
        tacotron = Tacotron2Model.from_pretrained("tts_en_tacotron2").eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load Tacotron2: {e}")
    try:
        waveglow = WaveGlowModel.from_pretrained("tts_waveglow").eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load WaveGlow: {e}")
    return tacotron, waveglow


def generate_vocals(tacotron, waveglow, vocal_text, sr):
    """Generate vocal audio from text"""
    max_chars = 100
    chunks = [vocal_text[i:i+max_chars] for i in range(0, len(vocal_text), max_chars)]

    audio_pieces = []
    for chunk in chunks:
        with torch.no_grad():
            parsed = tacotron.parse(chunk)
            spectrogram = tacotron.generate_spectrogram(tokens=parsed)
            if spectrogram.shape[-1] < 20:
                continue
            audio_chunk = waveglow.convert_spectrogram_to_audio(spec=spectrogram)
            audio_pieces.append(audio_chunk.squeeze().cpu().numpy())

    if not audio_pieces:
        raise RuntimeError("No valid audio generated!")
    
    return np.concatenate(audio_pieces), sr


# enhance
def smooth_pitch(y, window_size=5):
    """Smooth pitch variations using a moving average filter"""
    return np.convolve(y, np.ones(window_size) / window_size, mode='same')


def high_pass_filter(y, cutoff=150, sr=22050, order=4):
    """Apply a high-pass filter to remove low-frequency noise"""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, y)


def enhance_audio(y, sr):
    """Apply multiple enhancements to the generated vocals"""
    y_smoothed = smooth_pitch(y)
    y_stretched = librosa.effects.time_stretch(y_smoothed, rate=1.02)
    y_harmonic, _ = librosa.effects.hpss(y_stretched, margin=2.5)
    y_filtered = high_pass_filter(y_harmonic, cutoff=150, sr=sr)

    reverb_decay = np.exp(-np.linspace(0, 1.2, 2048))
    y_reverbed = fftconvolve(y_filtered, reverb_decay, mode="same")

    y_louder = np.clip(y_reverbed * 1.6, -0.98, 0.98)
    return y_louder



def main():
    global INSTRUMENTAL_PATH, OUTPUT_PATH

    print("=== Nasheed Vocal Generator & Enhancer ===")

    # Ask user for input and output
    root = Tk()
    root.withdraw()

    print("[0] Please select the instrumental audio file...")
    input_path = filedialog.askopenfilename(
        title="Select Instrumental File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.aac"), ("All Files", "*.*")]
    )
    if not input_path:
        print(" No file selected. Exiting.")
        return

    print("[0] Please select a folder to save the output nasheed...")
    output_folder = filedialog.askdirectory(title="Select Output Directory")
    if not output_folder:
        print(" No output folder selected. Exiting.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"nasheed_vocal_enhanced_{timestamp}.wav"
    output_path = os.path.join(output_folder, output_filename)

    INSTRUMENTAL_PATH = input_path
    OUTPUT_PATH = output_path

    print(f"\nSelected Input: {INSTRUMENTAL_PATH}")
    print(f"Output will be saved to: {OUTPUT_PATH}\n")

    try:
        # Step 1: Verify input
        print("[1/6] Verifying input audio...")
        verify_audio_file(INSTRUMENTAL_PATH)

        # Step 2: Extract melody
        print("[2/6] Extracting melody...")
        melody_midi, sr = extract_melody(INSTRUMENTAL_PATH)

        # Step 3: Convert melody to vocal syllables
        print("[3/6] Converting melody to vocal phrases...")
        vocal_text = melody_to_vocals(melody_midi)

        # Step 4: Load AI models
        print("[4/6] Loading AI models...")
        tacotron, waveglow = load_models()

        # Step 5: Generate vocals
        print("[5/6] Generating vocals...")
        vocals, sr = generate_vocals(tacotron, waveglow, vocal_text, sr)

        # Step 6: Enhance vocals
        print("[6/6] Enhancing vocals...")
        enhanced_audio = enhance_audio(vocals, sr)

        # Save output
        sf.write(OUTPUT_PATH, enhanced_audio, sr)
        print(f"\n Success! Enhanced output saved to: {OUTPUT_PATH}")

    except Exception as e:
        print(f"\n Error: {str(e)}")


if __name__ == "__main__":
    main()
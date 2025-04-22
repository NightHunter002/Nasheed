import os
import wave
import contextlib
import openai
from pydub import AudioSegment
from pydub.generators import Sine
from better_profanity import profanity
import tkinter as tk
from tkinter import filedialog
import tempfile
import whisper

# OpenAI 
openai.api_key = "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"

# Initialize Whisper model
model = whisper.load_model("base")


def preprocess_audio(input_audio, output_path):
    """Preprocess audio to prepare it for transcription."""
    try:
        audio = AudioSegment.from_file(input_audio)

        # Convert to mono and the hz
        audio = audio.set_channels(1).set_frame_rate(16000)

        # Normalize volume
        audio = audio.normalize()

        # Export processed audio
        processed_file = os.path.join(output_path, "processed_audio.wav")
        audio.export(processed_file, format="wav")
        print(f"Preprocessed audio saved to: {processed_file}")
        return processed_file
    except Exception as e:
        print(f"Error during audio preprocessing: {e}")
        return None


def get_audio_duration(audio_file):
    """Retrieve the duration of the audio file in milliseconds."""
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return int(duration * 1000)


def transcribe_audio_whisper(audio_file):
    """Transcribe audio using Whisper."""
    try:
        result = model.transcribe(audio_file)
        print("Transcription completed successfully.")
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""


def detect_bad_words(text):
    """Detect offensive words in the text."""
    profanity.load_censor_words()
    flagged_words = [word for word in text.split() if profanity.contains_profanity(word)]
    return flagged_words


def beep_out_audio(input_audio, flagged_timestamps):
    """Mute flagged timestamps by overlaying a beep sound."""
    audio = AudioSegment.from_wav(input_audio)
    beep = Sine(1000).to_audio_segment(duration=500)  
    for start, end in flagged_timestamps:
        audio = audio.overlay(beep, position=start)
    output_file = "output_beeped.wav"
    audio.export(output_file, format="wav")
    print(f"Beeped audio saved to: {output_file}")


def replace_offensive_words_with_gpt(text, flagged_words):
    """Use GPT to replace offensive words with non-offensive alternatives."""
    for word in flagged_words:
        prompt = f"Replace the word '{word}' in this sentence with a non-offensive alternative: {text}"
        response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    )
    alternative = response['choices'][0]['message']['content'].strip()
    text = text.replace(word, alternative)
    return text


def save_text_to_file(text, output_path, file_name="output"):
    """Save text to a file."""
    file_path = os.path.join(output_path, f"{file_name}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Text saved to: {file_path}")


def process_audio(input_audio, text, choice, output_folder):
    """Process audio based on the user's choice."""
    flagged_words = detect_bad_words(text)

    if not flagged_words:
        print("No offensive words detected.")
        return

    print(f"Flagged words: {flagged_words}")

    if choice == "1":  # Beep-out offensive words
        flagged_timestamps = []
        beep_out_audio(input_audio, flagged_timestamps)

    elif choice == "2":  # Replace offensive words with GPT
        clean_text = replace_offensive_words_with_gpt(text, flagged_words)
        print("Clean text:", clean_text)
        save_text_to_file(clean_text, output_folder, "clean_lyrics")

    elif choice == "3":  # Highlight flagged words for review
        print("Flagged words highlighted. Please review the text manually.")
        save_text_to_file("\n".join(flagged_words), output_folder, "flagged_words")


def main():
    """Main program function."""
    print("=== Offensive Word Detector and Processor ===")

    # Select input WAV file
    root = tk.Tk()
    root.withdraw()
    input_audio = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")], title="Select a WAV file")
    if not input_audio:
        print("No file selected. Exiting.")
        return

    # Select output folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if not output_folder:
        print("No folder selected. Exiting.")
        return

    # Preprocess audio
    print("Preprocessing the audio file...")
    processed_audio = preprocess_audio(input_audio, output_folder)
    if not processed_audio:
        print("Audio preprocessing failed. Exiting.")
        return

    # Transcribe audio
    print("Transcribing the audio file...")
    lyrics_text = transcribe_audio_whisper(processed_audio)
    if not lyrics_text.strip():
        print("Failed to transcribe audio. Exiting.")
        return

    # Save the original transcription
    save_text_to_file(lyrics_text, output_folder, "original_lyrics")

    print("Choose an option for handling offensive words:")
    print("1. Mute offensive content (beep-out).")
    print("2. Replace offensive content with contextual substitutions.")
    print("3. Highlight flagged words for manual review.")

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice not in ["1", "2", "3"]:
        print("Invalid choice. Exiting.")
        return

    process_audio(input_audio, lyrics_text, choice, output_folder)


if __name__ == "__main__":
    main()
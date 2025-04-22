
# Music transformation to vocals only: art of Nasheed


The Nasheed Transformation Tool is designed to transform songs with instruments into nasheeds. 



## System Requirements

 Windows, macOS, or Linux
- **Python Version**: Python 3.8 or to 3.10
- Dependencies:
  - Spleeter
  - Librosa
  - NVIDIA NeMo
  - Tkinter
  - PyAudio

## Installation

1. Clone the repository:
https://github.com/NightHunter002/Nasheed.git

2. Install the required Python packages:
   pip install -r requirements.txt


4. Launch the first application to separate a song:

001227965_final_year.py


2. Select an audio file for separation using the GUI and an output to where the new file gets stored

3. It will Separate the audio into instrumental and vocal tracks.

4. Now run the second code 001227965-transformation.py in which it will ask for the instrumental part and it will also ask for the output of the result. It then takes some time and outputs an enhanced version of the vocals only of the instrumental. So it will essentially output a transformed version of the instrumental

5. Open and run the 001227965_lyrics.py where this time when running it, user will have to upload the vocals lyrics part only this time and it will detect the lyrics and any bad language. Now it will ask the user what to do with the flagged words in which the user can choose :
- to beep them by pressing 1
- to use an openai to automatically change the flagged words by pressing 2
-to manually change by pressing 3

in any case, it will still output a txt file with the flagged words


Thank you for using the Nasheed Transformation Tool. For any queries or issues, feel free to contact ym2911g@gre.ac.uk.

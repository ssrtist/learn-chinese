# Coso's Speech Game

A speech-driven educational game for practicing words, phrases, and sentences with real-time feedback, MIDI music, and engaging visuals.

-----

## Features

  - **Enhanced Speech Recognition**: Calibrates to ambient noise for improved accuracy. Uses your microphone to recognize spoken words, phrases, or sentences.
  - **Selectable Content**: Choose from multiple word lists directly from the main menu.
  - **Text-to-Speech**: Prompts and feedback are spoken aloud using gTTS.
  - **Multiple Modes**: Practice with words or phrases.
  - **Custom Content**: Easily edit `config.json` to change or add new word and phrase lists.
  - **Rich Visual Feedback**: Animated highlights, clipart for words, and progress indicators.
  - **Sound & Music**: Includes audio feedback for correct/incorrect answers, celebration melodies using MIDI, and a dance animation sequence for positive reinforcement.
  - **Clipart Support**: Shows relevant images for each word (from `assets/images/clipart/vector`).
  - **Configurable**: Toggle fullscreen/windowed mode, adjust content lists, and more.
  - **Cross-platform**: Runs on Windows, Linux, and macOS (Python 3.8+).

-----

## Installation

### Requirements

  - Python 3.8 or newer
  - [SDL2 libraries](https://www.libsdl.org/) (required by Pygame)
  - A MIDI synthesizer (most operating systems include one by default)
  - Microphone (for speech input)
  - Internet connection (for gTTS and Google Speech API)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Assets

  - **Sounds**: Place sound effects in `assets/sounds/` (e.g., `mouse_click.wav`, `nogood.wav`, `beep_short.wav`).
  - **Images**: Place word clipart in `assets/images/clipart/vector/` (filenames should contain the word).
  - **Dance Animation**: Place dance frame images in `assets/videos/dance2/`.
  - **Microphone Icon**: Place `microphone_001.png` in `assets/images/images/`.
  - **Fallback Image**: Place `unknown_001.png` in `assets/images/images/`.
  - **Main Menu Image**: Place `cover_speaking_girl.png` in `assets/images/images/`.

### Configuration

Edit `config.json` to customize word, phrase, and sentence lists. The configuration now supports multiple word lists and defines the order (e.g., "random"). The game will auto-create a default config if one is missing.

Example structure:

```json
{
  "word_list_animals": {
    "items": ["cat", "dog", "lion"],
    "order": "random"
  },
  "phrase_list": {
    "items": ["Hello world", "Good morning"],
    "order": "sequential"
  }
}
```

-----

## Running the Game

```bash
python co-talking.py
```

-----

## Controls

| Key / Action | Effect |
| :--- | :--- |
| Mouse click | Select menu options, dropdowns, and buttons |
| Alt + Enter | Toggle fullscreen/windowed mode |
| Esc | Return to menu / quit |
| Q | Quit game |
| C | Open configuration file (`config.json`) |

-----

## File Structure

```
learning_speech_game/
├── co-talking.py
├── config.json
├── requirements.txt
├── assets/
│   ├── images/
│   │   ├── images/
│   │   │   ├── cover_speaking_girl.png
│   │   │   ├── microphone_001.png
│   │   │   └── unknown_001.png
│   │   └── clipart/vector/
│   │       └── ... (word images)
│   ├── sounds/
│   │   ├── mouse_click.wav
│   │   ├── nogood.wav
│   │   └── beep_short.wav
│   └── videos/dance2/
│       └── ... (dance frames)
```

-----

## Dependencies

  - `pygame`
  - `gtts`
  - `SpeechRecognition`
  - `numpy`
  - `sounddevice`
  - `python-dotenv` (optional, for environment config)

-----

## Credits & Acknowledgements

### Core Technologies

| Technology | Purpose | Link |
| :--- | :--- | :--- |
| **Pygame** | Game engine, graphics & MIDI | [pygame.org](https://www.pygame.org) |
| **gTTS** | Text-to-speech synthesis | [gTTS Docs](https://gtts.readthedocs.io) |
| **SpeechRecognition**| Speech-to-text recognition | [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) |
| **Sounddevice** | Audio recording & playback | [pypi.org/project/sounddevice/](https://pypi.org/project/sounddevice/) |
| **NumPy** | Audio data processing | [numpy.org](https://numpy.org) |
| **SDL2** | Multimedia backend (Pygame) | [libsdl.org](https://www.libsdl.org) |

### Assets

| Asset Type | Source | License |
| :--- | :--- | :--- |
| Sound Effects | [Kenney.nl](https://kenney.nl/assets) | CC0 1.0 Universal |
| Fonts | [OpenDyslexic](https://opendyslexic.org) | SIL Open Font License |
| Icons/Clipart | [Font Awesome](https://fontawesome.com) | CC BY 4.0 |
| Dance Images | Various / Custom | Educational Use |

-----

## Special Thanks

  - Open Source Community
  - Beta Testers
  - Python Software Foundation
  - GitHub

-----

**If you use or adapt this project, please consider sharing improvements or feedback\!**
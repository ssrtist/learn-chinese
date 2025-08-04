""" 
Title: Little Speech Game - Co-talking.py
Version: 1.4
Date: 2025-07-31
Description:
 - image search algorithm for phrases
Todos:
 - add logging, and save recordings
"""
import pygame
import pygame.midi
import json
import os
import io
import random
from gtts import gTTS
import speech_recognition as sr
import threading
import time
import numpy as np
import sounddevice as sd
from datetime import datetime
import sys


# --- Global Constants and Configuration ---
DEFAULT_LANGUAGE = "zh-CN"
CONFIG_FILE_PATH = "config_zh.json"
SOUND_TYPE_FILE = "assets/sounds/mouse_click.wav"
SOUND_BEEP_FILE = "assets/sounds/beep_shorter.wav"
FULLSCREEN_RESOLUTION = (1920, 1080)
WINDOWED_RESOLUTION = (1920, 1080)
SPLASH_RESOLUTION = (640, 480)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = "red"
YELLOW = "yellow"
DARK_GRAY = "darkgray"
DARK_GREEN = "darkgreen"
DARK_RED = "darkred"
DARK_BLUE = "darkblue"
LIGHT_YELLOW = (255, 255, 200)
BOX_BG_COLOR = (220, 220, 220)
TEXT_BOX_COLOR = WHITE
PROMPT_BOX_COLOR =  BOX_BG_COLOR
HIGHLIGHT_COLOR = (255, 255, 0)
TEXT_COLOR = BLACK 
TARGET_WORDS = 5
TARGET_PHRASES = 5
FPS = 60
MUSICAL_KEYBOARD = True
CLIPART_PATH = "assets/images/clipart/vector"
STOP_APP = False

# Shared variable
RECOGNIZED_TEXT = ""
RECOGNIZED_DATA = None
RECOGNIZER_STATUS = "READY"

# recording parameters
RECORD_TIMEOUT = 3
RECORD_MAX = 10
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
PAUSE_THRESHOLD = 1
MIN_SILENCE_THRESHOLD = 200
RUN_SILENCE_THRESHOLD = MIN_SILENCE_THRESHOLD
ZCR_NOISE_THRESHOLD = 0.2  # Zero-crossing rate threshold for noise detection
ZCR_SPEECH_THRESHOLD = 0.15  # Zero-crossing rate threshold for speech detection

def calibrate_threshold(duration=3):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🎙️ Calibrating background noise level...")
    calibration_audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    ambient_noise_level = np.abs(calibration_audio).mean()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔊 Ambient noise level: {ambient_noise_level}")
    target_threshold = max(MIN_SILENCE_THRESHOLD, -(-ambient_noise_level * 1.5 // 100) * 100) # Round up to nearest 100 for better thresholding
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔊 Calibration complete. Using threshold: {target_threshold}")
    return target_threshold

def zero_crossing_rate(audio_chunk):
    # Compute ZCR: count zero-crossings divided by number of samples
    return np.mean(np.abs(np.diff(np.sign(audio_chunk)))) / 2

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting sound recording stream... ")
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=BLOCK_SIZE)
stream.start()

def record_audio(sample_rate=44100, silence_threshold=500, silence_duration=.5, timeout_duration=5, max_duration=10):
    chunk_size = BLOCK_SIZE  # Number of samples per chunk
    silence_samples = int(silence_duration * sample_rate / chunk_size) # Calculate number of chunks for silence duration
    timeout_samples = int(timeout_duration * sample_rate / chunk_size)
    max_samples = int(max_duration * sample_rate)
    skip_chunks = 10  # Number of initial chunks to skip for microphone stabilization
    speech_start_required = 5  # Number of chunks to confirm speech start
    speech_start_counter = 0
    speech_started = False
    min_speech_chunks = skip_chunks + speech_start_required  # Minimum speech chunks required for speech detection, about 0.5 seconds at SAMPLE_RATE=44100
    audio_data = []
    chunk_count = 0
    pause_counter = 0 # Count consecutive silent chunks for silence detection
    pause_reset_counter = 0
    noise_counter = 0
    
    # # Warm-up period to stabilize microphone (0.1 seconds)
    # for _ in range(int(0.1 * sample_rate / chunk_size)):
    #     stream.read(chunk_size)
    
    while pygame.mixer.get_busy(): 
        pygame.time.Clock().tick(FPS)

    stream.read(stream.read_available) # Clear any buffered audio data before starting recording
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Recording started. Speak now...")

    for _ in range(int(max_samples / chunk_size)):
        pygame.time.Clock().tick(120)
        chunk, overflow = stream.read(chunk_size)
        if overflow or chunk.size == 0:
            continue  # Skip invalid or empty chunks
        audio_data.append(chunk)
        chunk_count += 1
        if chunk_count < skip_chunks: # skip first 10 chunks to allow microphone to stabilize
            continue
        
        # Calculate RMS amplitude safely
        mean_square = np.mean(chunk.astype(np.float64)**2)
        if np.isnan(mean_square) or mean_square <= 0:
            rms = 0.0
        else:
            rms = np.sqrt(mean_square)
        
        # Calculate ZCR, use for next-gen speech detection
        zcr = zero_crossing_rate(chunk[:, 0])  # Use first channel if stereo
        if not speech_started and zcr > ZCR_NOISE_THRESHOLD:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Chunk {chunk_count}: noise detected, ZCR = {zcr:.4f}")
        
        # Check for speech (require contiguous chunks)
        if not speech_started:
            if rms >= silence_threshold and zcr < ZCR_SPEECH_THRESHOLD: # chunk is speech (not silent and not noisy)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Chunk {chunk_count}: speech_start_counter: {speech_start_counter}, RMS: {rms:.2f}, ZCR: {zcr:.4f}")
                speech_start_counter += 1
                if speech_start_counter >= speech_start_required:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Speech started at chunk {chunk_count}, RMS: {rms:.2f}, ZCR: {zcr:.4f}")
                    speech_started = True
            # else: # reset speech start counter if chunk is silent or noisy
            #     print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Chunk {chunk_count}: silence detected,reset speech_start_counter, RMS: {rms:.2f}, ZCR: {zcr:.4f}")
            #     speech_start_counter = 0  # Reset on non-speech chunk
            # If no speech and min_duration reached, stop
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Chunk {chunk_count}: No speech detected, RMS: {rms:.2f}, ZCR: {zcr:.4f}")
                
            if chunk_count >= timeout_samples: # enough silent chunks recorded
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Chunk {chunk_count}: Timeout reached, speech_started: {speech_started}, stopping recording.")
                break

        # If speech started, check for silence
        if speech_started:
            if rms < silence_threshold or zcr > ZCR_NOISE_THRESHOLD: # chunk is silent or noisy
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Chunk {chunk_count}: Silence detected, RMS: {rms:.2f}, ZCR: {zcr:.4f}")
                pause_counter += 1
                pause_reset_counter = 0
            else: # chunk is speech
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Chunk {chunk_count}: Voice detected, RMS: {rms:.2f}, ZCR: {zcr:.4f}")
                if pause_counter > 0:
                    pause_reset_counter += 1
                    if pause_reset_counter >= 5:  # Reset pause counter after 5 consecutive speech chunks, need to use constant
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Resetting pause_counter after {pause_reset_counter} consecutive speech chunks.")
                        pause_counter = 0
                        pause_reset_counter = 0
            if pause_counter >= silence_samples and chunk_count >= pause_counter + min_speech_chunks:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] pause detected, pause_counter: {pause_counter}, stopping recording.")
                break
            # if zcr > 0.2:
            #     noise_counter += 1
            # else:
            #     noise_counter = 0
            # if noise_counter >= silence_samples: # no speech detected for a while
            #     print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Noise duration exceeded, stopping recording.")
            #     break
            
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total chunks {chunk_count}, pause_counter: {pause_counter}: Recording finished.")
    
    # Convert audio data to numpy array
    if not audio_data:
        return "No audio data recorded."
    audio = np.concatenate(audio_data, axis=0)

    return sr.AudioData(audio.tobytes(), sample_rate, 2)

def listen_for_speech():
    global RECOGNIZED_TEXT, RECOGNIZED_DATA, RECOGNIZER_STATUS, STOP_APP
    
    # Initialize the speech recognizer
    recognizer = sr.Recognizer()

    beep_sound = load_sound(SOUND_BEEP_FILE)
    while not STOP_APP:
        # check the recognizer status and wait for the "LISTEN" state
        if RECOGNIZER_STATUS == "LISTEN":
            RECOGNIZER_STATUS = "LISTENING"

            # wait for any sound playback to finish
            while pygame.mixer.get_busy(): 
                pygame.time.Clock().tick(FPS)
            beep_sound.play()

            # with sr.Microphone() as source:
            try:
                audio = record_audio(silence_threshold=RUN_SILENCE_THRESHOLD, timeout_duration=RECORD_TIMEOUT, max_duration=RECORD_MAX, sample_rate=SAMPLE_RATE)
                if RECOGNIZER_STATUS == "LISTENING":
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Recognizing speech...")
                    RECOGNIZED_DATA = audio  
                    RECOGNIZED_TEXT = recognizer.recognize_google(audio, language=DEFAULT_LANGUAGE)  
                    # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Recognition complete...")
                    RECOGNIZER_STATUS = "COMPLETE" 
            except sr.WaitTimeoutError:
                # Handle timeout error
                if RECOGNIZER_STATUS == "READY":
                    RECOGNIZED_TEXT = ""
                    RECOGNIZED_DATA = None  # Clear recognized data if not in listening state
                else:
                    RECOGNIZED_TEXT = "TIMEOUT"
                    RECOGNIZER_STATUS = "ERROR"
            except sr.UnknownValueError:
                # Handle unrecognized speech
                if RECOGNIZER_STATUS == "READY":
                    RECOGNIZED_TEXT = ""
                    RECOGNIZED_DATA = None  # Clear recognized data if not in listening state
                else:
                    RECOGNIZED_TEXT = "UNRECOGNIZED"
                    RECOGNIZER_STATUS = "ERROR"
            except sr.RequestError:
                # Handle API errors
                if RECOGNIZER_STATUS == "READY":
                    RECOGNIZED_TEXT = ""
                    RECOGNIZED_DATA = None  # Clear recognized data if not in listening state
                else:
                    RECOGNIZED_TEXT = "API ERROR"
                    RECOGNIZER_STATUS = "ERROR"
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Recognition Status: {RECOGNIZER_STATUS}, Text: {RECOGNIZED_TEXT}")
        pygame.time.Clock().tick(FPS)

def has_common_word(str1, str2, exclude={"go", "to"}):
    words1 = set(str1.split()) - exclude
    words2 = set(str2.split()) - exclude
    return not words1.isdisjoint(words2)

def get_matching_files(word):
    matching_files = [file for file in os.listdir(CLIPART_PATH) if has_common_word(file.replace("_"," ").lower(), word.lower())]
    if not matching_files:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No matching files found for word: {word}")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found {len(matching_files)} matching files for word: {word}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Matching files: {matching_files}")
        return matching_files

def merge_sounds(sound1, sound2):
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    # Convert sounds to numpy arrays
    arr1 = pygame.sndarray.array(sound1)
    arr2 = pygame.sndarray.array(sound2)

    # Concatenate audio data along the time axis
    if arr1.ndim == 1:
        merged = np.concatenate((arr1, arr2))
    else:  
        merged = np.concatenate((arr1, arr2), axis=0)

    # Ensure correct data type
    merged = merged.astype(np.int16)

    # Return combined Sound object
    return pygame.sndarray.make_sound(merged)

def play_recorded_audio(audio_data):
    """Plays back the recorded audio data using Pygame."""
    if audio_data:
        try:
            wav_data = audio_data.get_wav_data()
            sound_file = io.BytesIO(wav_data)
            sound = pygame.mixer.Sound(file=sound_file)
            sound.play()
            while pygame.mixer.get_busy():
                pygame.time.Clock().tick(FPS)
        except Exception as e:
            print(f"Error playing recorded audio: {e}")

def countdown_timer(duration):
    global RECOGNIZED_TEXT
    for i in range(duration, 0, -1):
        if RECOGNIZED_TEXT:
            print("\nSpeech detected, stopping countdown.")
            break
        print(f"\rCountdown: {i} seconds remaining", end="", flush=True)
        time.sleep(1)
    print("\rCountdown complete!", flush=True)

def load_config():
    """Loads configuration from JSON file or uses default values."""
    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Configuration loaded from {CONFIG_FILE_PATH}.")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Configuration content: {config}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error loading configuration. Using default lists. {e}")
        config = {}
    return config

def load_sound(filepath):
    """Loads a sound file and handles potential errors."""
    try:
        sound = pygame.mixer.Sound(filepath)
        return sound
    except pygame.error as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error loading sound: {e}")
        return None

def toggle_fullscreen(screen, screen_width, screen_height, fullscreen):
    """Toggles between fullscreen and windowed mode."""
    fullscreen = not fullscreen
    if fullscreen:
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((screen_width, screen_height))
    return fullscreen, screen

def draw_highlight(surface, input_text, font, position, color, index, max_width, mode):
    """Draws a highlighted character for the prompt text indicating the next character, or user input text indicating a mistake"""
    # possible modes: cursor, prompt, text

    if mode == "text":
        this_text = input_text[:index]
        next_char = input_text[index - 1] # this should only be current char for error highlight
    elif mode == "prompt":
        this_text = input_text
        next_char = input_text[index - 1] # this can be next char, or current char for normal or error states
    elif mode == "cursor":
        this_text = input_text # we should only want to add cursor at the end of the line, for now
        next_char = ""
    words = this_text.split(" ")
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line + " ")
            current_line = word
    lines.append(current_line)
    line_height = font.get_height()
    spacing = 5

    # locate current_line, then current_index
    total_line = ""
    line_index = 0
    word_index = 0
    typed_line = ""
    for line in lines:
        total_line = total_line + line
        if  index <= len(total_line):
            word_index = index - (len(total_line) - len(line))
            typed_line = line[:word_index]
            break
        line_index += 1

    if mode != "cursor":
        typed_width = font.size(typed_line[:-1])[0]
        next_char_width = font.size(input_text[index - 1])[0]
    else:
        typed_width = font.size(typed_line)[0]
        next_char_width = 4

    highlight_x = position[0] + typed_width
    highlight_y = position[1] + (line_index) * (line_height + spacing)

    # Draw highlight box for either character or cursor
    if mode == "cursor" and pygame.time.get_ticks() % 1000 < 500:
        pygame.draw.rect(surface, color, (highlight_x, highlight_y, next_char_width, font.get_height()))

    if mode != "cursor":
        pygame.draw.rect(surface, color, (highlight_x, highlight_y, next_char_width, font.get_height()))
        next_char_surface =font.render(next_char, True, TEXT_COLOR)
        surface.blit(next_char_surface, (highlight_x, highlight_y))

def render_text_wrapped(text, font, color, max_width):
    """Renders text wrapped to a given width."""
    words = text.split(' ')
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        test_width, _ = font.size(test_line)
        if test_width <= max_width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    lines.append(' '.join(current_line))

    surfaces = []
    total_height = 0
    for line in lines:
        line_surface = font.render(line, True, color)
        surfaces.append(line_surface)
        total_height += line_surface.get_height() + 5

    combined_surface = pygame.Surface((max_width, total_height), pygame.SRCALPHA)
    y = 0
    for line_surface in surfaces:
        combined_surface.blit(line_surface, (0, y))
        y += line_surface.get_height() + 5

    return combined_surface

def open_config_file():
    """Opens the configuration file with the default OS editor."""
    config_path = CONFIG_FILE_PATH

    if not os.path.exists(config_path):
        default_config = {
            "word_list": ["apple", "banana", "orange", "grape", "mango"],
            "sentence_list": ["The quick brown fox jumps over the lazy dog.", "Hello world, welcome to the speech game."]
        }
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=4)

    try:
        os.startfile(config_path)
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error opening config file: {e}")

def generate_speech_sound(text):
    """Generates and returns a Pygame sound object from text using gTTS."""
    gtext = text if text else "nothing"
    buffer = io.BytesIO()
    tts = gTTS(text=gtext, lang=DEFAULT_LANGUAGE, slow=False)
    tts.write_to_fp(buffer)
    buffer.seek(0)
    sound = pygame.mixer.Sound(buffer)
    return sound

def draw_styled_text_box(surface, rect, text_surface, bg_color, padding=15, border_width=2, border_color=BLACK):
    """Draws a stylized text box with background, border, and padding."""
    pygame.draw.rect(surface, bg_color, rect, border_radius=20) # Background
    pygame.draw.rect(surface, border_color, rect, border_width, border_radius=20) # Border
    text_rect = text_surface.get_rect(topleft=(rect.x + padding, rect.y + padding)) # Position text with padding
    surface.blit(text_surface, text_rect)

def set_channel_volume(player, channel, volume):
    """Set the volume for a specific MIDI channel using Control Change (CC 7)."""
    player.write_short(0xB0 | channel, 7, volume)  # 0xB0 is the status byte for Control Change

# --- UI Elements ---
class Button:
    """Button UI element."""
    def __init__(self, x, y, text, width=200, height=50, color=DARK_GREEN):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.color = color
        self.text_color = WHITE
        self.text = text
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, screen, font):
        pygame.draw.rect(screen, self.color, self.rect)
        rendered_text = font.render(self.text, True, self.text_color)
        text_rect = rendered_text.get_rect(center=self.rect.center) # Center the text in the button
        screen.blit(rendered_text, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

class Checkbox:
    """Checkbox UI element."""
    def __init__(self, x, y, label, checked=False):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.label = label
        self.checked = checked
        self.label_font = pygame.font.Font(None, 24)

    def draw(self, screen):
        pygame.draw.rect(screen, LIGHT_YELLOW, self.rect, 2)
        if self.checked:
            pygame.draw.line(screen, GREEN, (self.rect.x+3, self.rect.y+10), (self.rect.x+8, self.rect.y+15), 2)
            pygame.draw.line(screen, GREEN, (self.rect.x+8, self.rect.y+15), (self.rect.x+17, self.rect.y+3), 2)
        label_surface = self.label_font.render(self.label, True, LIGHT_YELLOW)
        screen.blit(label_surface, (self.rect.x + 25, self.rect.y - 2))

    def toggle(self):
        self.checked = not self.checked

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# --- Game Class ---
class TalkingGame:
    """Main class to manage the Talking Game."""
    def __init__(self):
        global RUN_SILENCE_THRESHOLD
        pygame.init()

        # Fonts setup
        self.font = pygame.font.Font(None, 36)
        self.button_font = pygame.font.Font(None, 48)
        self.msg_font = pygame.font.SysFont("verdana", 52)
        self.game_font_small = pygame.font.SysFont("Microsoft YaHei", 52)
        self.game_font_large = pygame.font.SysFont("Microsoft YaHei", 96)
        # self.game_font_small = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 52)
        # self.game_font_large = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 96)

        # Splash Screen
        splash_screen = pygame.display.set_mode(SPLASH_RESOLUTION)
        pygame.display.set_caption("Welcome")
        splash_screen.fill(DARK_GRAY)
        prompt_text = self.msg_font.render("Loading assets...", True, HIGHLIGHT_COLOR)
        prompt_rect = prompt_text.get_rect(center=(SPLASH_RESOLUTION[0] // 2, SPLASH_RESOLUTION[1] // 2))
        splash_screen.blit(prompt_text, prompt_rect)
        pygame.display.flip()

        # Start listening in a separate thread
        RUN_SILENCE_THRESHOLD = calibrate_threshold()
        self.speech_thread = threading.Thread(target=listen_for_speech)
        self.speech_thread.start()

        # Load config

        # Load word lists from config
        self.config = load_config()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded config: {self.config}")
        self.word_list_keys = [key for key in self.config.keys() if key.startswith("word_list_")]
        self.selected_word_list_key = self.word_list_keys[0] if self.word_list_keys else None
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded word lists: {self.word_list_keys}")

        # Load phrases
        if self.config.get('phrase_list') is not None:
            self.phrase_list = self.config.get('phrase_list')["items"]
            self.phrase_order = self.config["phrase_list"]["order"]
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded {len(self.phrase_list)} phrases from config.")
        else:
            self.phrase_list = []
            self.phrase_order = []
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No phrase list found in config, using empty list.")

        self.start_fullscreen = False # todo: retrieve setting from config file

        # Load dance frames
        dance_folder = "assets/videos/dance2"
        # dance_folder = random.choice(["assets/videos/dance1", "assets/videos/dance2"]) 
        dance_files = sorted([
            f for f in os.listdir(dance_folder) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.dance_frames = [pygame.image.load(os.path.join(dance_folder, f)).convert_alpha() 
                for f in dance_files]
        if not self.dance_frames:
            raise SystemExit("No images found in the folder!")

        self.current_frame = 0

        # MIDI setup
        pygame.midi.init()
        output_id = pygame.midi.get_default_output_id()
        if output_id == -1:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No MIDI output device found!")
            exit()
        self.player = pygame.midi.Output(output_id)
        self.player.set_instrument(0)  # Acoustic Grand Piano
        self.velocity = 127  # Volume
        melody_volume = 100  # Volume for melody (channel 0)
        set_channel_volume(self.player, 0, melody_volume)  # Set melody channel volume

        self.melodies = [
            # twinkle twinkle little star
            [
            (60, 0.5), (60, 0.5), (67, 0.5), (67, 0.5), (69, 0.5), (69, 0.5), (67, 1),
            (65, 0.5), (65, 0.5), (64, 0.5), (64, 0.5), (62, 0.5), (62, 0.5), (60, 2)
            # (67, 0.5), (67, 0.5), (65, 0.5), (65, 0.5), (64, 0.5), (64, 0.5), (62, 1),
            # (67, 0.5), (67, 0.5), (65, 0.5), (65, 0.5), (64, 0.5), (64, 0.5), (62, 1),
            # (60, 0.5), (60, 0.5), (67, 0.5), (67, 0.5), (69, 0.5), (69, 0.5), (67, 1),
            # (65, 0.5), (65, 0.5), (64, 0.5), (64, 0.5), (62, 0.5), (62, 0.5), (60, 1)
            ],
            # jingle bells
            [
            (64, 0.3), (64, 0.3), (64, 0.6), 
            (64, 0.3), (64, 0.3), (64, 0.6), 
            (64, 0.3), (67, 0.3), (60, 0.3), (62, 0.3), (64, 0.6),
            (65, 0.3), (65, 0.3), (65, 0.3), (65, 0.3), (65, 0.3), (64, 0.3), (64, 0.3), (64, 0.3),
            (64, 0.3), (62, 0.3), (62, 0.3), (64, 0.3), (62, 0.6), (67, 0.6)
            ],
            # mary had a little lamb
            [
            (64, 0.3), (62, 0.3), (60, 0.3), (62, 0.3), (64, 0.3), (64, 0.3), (64, 0.6),
            (62, 0.3), (62, 0.3), (62, 0.6), (64, 0.3), (67, 0.3), (67, 0.6),
            (64, 0.3), (62, 0.3), (60, 0.3), (62, 0.3), (64, 0.3), (64, 0.3), (64, 0.3),
            (64, 0.3), (62, 0.3), (62, 0.3), (64, 0.3), (62, 0.6), (60, 0.6)
            ]
        ]

        self.this_melody = random.choice(self.melodies)
        self.max_index = len(self.this_melody) - 1
        self.this_index = 0
        self.note = 0
        self.note_time = 0
        self.note_start = pygame.time.get_ticks()

        self.type_sound = load_sound(SOUND_TYPE_FILE)

        self.clock = pygame.time.Clock()
        self.running = True
        self.game_mode = "menu" # menu, words, phrases 
        self.play_welcome_sound = True

        self.sounds = {}
        # load sfx defined in self.config.json from local assets
        for game_mode in self.config.keys():
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading SFX for \"{game_mode}\"...")
            for word in self.config.get(game_mode)["items"]:
                filename = f"assets/sounds/word_{word}.mp3"
                if os.path.exists(filename):
                    # load sfx
                    self.sounds[word] = load_sound(filename)
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] file, {filename} doesn't exists, generating...")
                    tts = gTTS(text=word, lang=DEFAULT_LANGUAGE, slow=False)
                    tts.save(filename)
                    pygame.time.wait(500)
                    self.sounds[word] = load_sound(filename)

        # Load sounds for prompts
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading prompt sounds...")
        if DEFAULT_LANGUAGE == "en":
            self.Sound_Welcome = generate_speech_sound("Welcome to Little Speech Game")
            self.Sound_Goodjob = generate_speech_sound("Good job! Continue?")
            self.Sound_PleaseSay = generate_speech_sound("Please say: ")
            self.Sound_NoGood = generate_speech_sound("No good! You said: ")
            self.Sound_Good = generate_speech_sound("Good! You said: ")
            self.Sound_NoHear = generate_speech_sound("I didn't hear you. ")
            self.Sound_Skipped = generate_speech_sound("Skipped!")
        elif DEFAULT_LANGUAGE == "es":
            self.Sound_Welcome = generate_speech_sound("Bienvenido al juego de habla")
            self.Sound_Goodjob = generate_speech_sound("¡Buen trabajo! ¿Continuar?")
            self.Sound_PleaseSay = generate_speech_sound("Por favor, di: ")
            self.Sound_NoGood = generate_speech_sound("¡No bueno! Dijiste: ")
            self.Sound_Good = generate_speech_sound("¡Bien! Dijiste: ")
            self.Sound_NoHear = generate_speech_sound("No te escuché. ")
            self.Sound_Skipped = generate_speech_sound("¡Omitido!")
        elif DEFAULT_LANGUAGE == "zh-CN":
            self.Sound_Welcome = generate_speech_sound("欢迎来到小语音游戏")
            self.Sound_Goodjob = generate_speech_sound("做得好！继续吗？")
            self.Sound_PleaseSay = generate_speech_sound("请说：")
            self.Sound_NoGood = generate_speech_sound("不好！你说的是：")
            self.Sound_Good = generate_speech_sound("好！你说的是：")
            self.Sound_NoHear = generate_speech_sound("我没听到你说话。")
            self.Sound_Skipped = generate_speech_sound("跳过了！")

        # Video setup
        if self.start_fullscreen:
            self.screen_width = FULLSCREEN_RESOLUTION[0]
            self.screen_height = FULLSCREEN_RESOLUTION[1]
            self.screen = pygame.display.set_mode(FULLSCREEN_RESOLUTION, pygame.FULLSCREEN)
            self.fullscreen = True
        else:
            self.screen_width = WINDOWED_RESOLUTION[0]
            self.screen_height = WINDOWED_RESOLUTION[1]
            os.environ['SDL_VIDEO_CENTERED'] = '1'
            self.screen = pygame.display.set_mode(WINDOWED_RESOLUTION, pygame.NOFRAME)
            self.fullscreen = False
        pygame.display.set_caption("Coso's Typing Game")

    def midi_keydown(self):
        note = self.this_melody[self.this_index][0]
        self.player.note_on(note, self.velocity)
        self.this_index += 1
        if self.this_index > self.max_index:
            self.this_index = 0

    def midi_play_song(self):
        song_complete = False
        self.elapsed_time = pygame.time.get_ticks() - self.note_start
        if self.elapsed_time >= self.note_time:
            self.player.note_off(self.note, 127)
            if self.this_index <= len(self.this_melody) - 1:
                self.player.note_on(self.this_melody[self.this_index][0], 127)
                self.note = self.this_melody[self.this_index][0]
                self.note_start = pygame.time.get_ticks()
                self.note_time = 1000 * self.this_melody[self.this_index][1]
                self.this_index += 1
            else:
                # play_round_complete = False
                self.this_index = 0
                self.note = 0
                self.note_time = 0
                self.Sound_Goodjob.play()
                song_complete = True
        return song_complete

    def run(self):
        """Main game loop."""
        global RECOGNIZER_STATUS, RECOGNIZED_TEXT, RECOGNIZED_DATA, STOP_APP
        while self.running:
            if self.game_mode == "menu":
                self.run_menu()
            elif self.game_mode == "view_word_set":
                self.run_view_word_set()
            elif self.game_mode == "words":
                self.run_words(self.word_list, TARGET_WORDS, self.word_order)
            elif self.game_mode == "phrase":
                self.run_words(self.phrase_list, TARGET_PHRASES, self.phrase_order)
            self.clock.tick(FPS)

        # Clean up speech recognition thread
        STOP_APP = True
        self.speech_thread.join()

        # Clean up Pygame resources
        pygame.quit()

    def run_menu(self):
        global MUSICAL_KEYBOARD
        menu_background = pygame.image.load("assets/images/images/cover_speaking_girl.png")
        menu_background_x, menu_background_y = menu_background.get_size() 
        menu_background = pygame.transform.smoothscale(menu_background, (menu_background_x * 1080 / menu_background_y, 1080))

        """Handles the main menu loop."""
        title_quit_button = Button(self.screen_width - 220, self.screen_height - 70, "Quit", 200, 50, DARK_RED)
        title_config_button = Button(self.screen_width - 220, self.screen_height - 140, "Config", 200, 50, DARK_BLUE)


        # --- Fix button positions for Words, Phrases ---
        button_width = 250
        button_height = 50
        button_spacing = 50
        center_x = self.screen_width // 10 * 7

        # Calculate starting x so that all three buttons are centered as a group
        total_width = button_width * 2 + button_spacing * 2
        button_x = center_x - total_width // 2
        button_y = self.screen_height // 2 + 50
        
        title_word_button = Button(button_x, button_y, "Words", button_width, button_height)
        title_phrase_button = Button(button_x + button_width + button_spacing, button_y, "Phrases", button_width, button_height)

        # Dropdown for word lists
        dropdown_active = False
        dropdown_rect = pygame.Rect(title_word_button.x, title_word_button.y + title_word_button.height + 5, button_width, len(self.word_list_keys) * 30)
        
        # Initialize self.word_list based on the initially selected key
        if self.selected_word_list_key:
            self.word_list = self.config[self.selected_word_list_key]["items"]
            self.word_order = self.config[self.selected_word_list_key]["order"]
        else:
            self.word_list = []
            self.word_order = "random"

        # ----------------------------------------------------------

        while self.game_mode == "menu" and self.running:
            self.screen.fill(DARK_GRAY)
            self.screen.blit(menu_background, (0, 0))
            
            title_quit_button.draw(self.screen, self.button_font)
            title_config_button.draw(self.screen, self.button_font)
            title_word_button.draw(self.screen, self.button_font)
            
            # Draw dropdown if active
            if dropdown_active:
                pygame.draw.rect(self.screen, LIGHT_YELLOW, dropdown_rect)
                for i, key in enumerate(self.word_list_keys):
                    item_rect = pygame.Rect(dropdown_rect.x, dropdown_rect.y + i * 30, dropdown_rect.width, 30)
                    text_surface = self.font.render(key.replace("word_list_", "").replace("_", " ").title(), True, BLACK)
                    self.screen.blit(text_surface, (item_rect.x + 5, item_rect.y + 5))
                    if key == self.selected_word_list_key:
                        pygame.draw.rect(self.screen, DARK_GREEN, item_rect, 2) # Highlight selected

            title_phrase_button.draw(self.screen, self.button_font) # Draw phrase button after dropdown

            prompt_text = self.game_font_large.render("Little Speech Game", True, DARK_BLUE)
            prompt_rect = pygame.Rect(20, 10, self.screen_width - 40, prompt_text.get_height() + 20)
            pygame.draw.rect(self.screen, LIGHT_YELLOW, prompt_rect.inflate(20, 10))
            self.screen.blit(prompt_text, (prompt_rect.width // 2 - prompt_text.get_width() // 2, prompt_rect.y + 10))

            pygame.display.flip()
            # Play welcome sound once
            if self.play_welcome_sound:
                self.Sound_Welcome.play()
                self.play_welcome_sound = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        open_config_file()
                    elif event.key == pygame.K_RETURN and pygame.key.get_mods() & pygame.KMOD_ALT:
                        self.fullscreen, self.screen = toggle_fullscreen(self.screen, self.screen_width, self.screen_height, self.fullscreen)
                    elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if title_config_button.is_clicked(event.pos):
                        self.type_sound.play()
                        open_config_file()
                    elif title_word_button.is_clicked(event.pos):
                        self.type_sound.play()
                        # self.game_mode = "words"
                        dropdown_active = not dropdown_active # Toggle dropdown
                    elif title_phrase_button.is_clicked(event.pos):
                        self.type_sound.play()
                        self.game_mode = "phrase"
                    elif dropdown_active and dropdown_rect.collidepoint(event.pos):
                        # Check if a dropdown item was clicked
                        self.game_mode = "words"
                        for i, key in enumerate(self.word_list_keys):
                            item_rect = pygame.Rect(dropdown_rect.x, dropdown_rect.y + i * 30, dropdown_rect.width, 30)
                            if item_rect.collidepoint(event.pos):
                                self.selected_word_list_key = key
                                # self.word_list = self.config[self.selected_word_list_key]["items"]
                                self.word_list = self.config.get(self.selected_word_list_key)["items"]
                                self.word_order = self.config.get(self.selected_word_list_key)["order"]
                                dropdown_active = False # Close dropdown after selection
                                self.type_sound.play()
                                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Selected word list: {self.selected_word_list_key}")
                                break
                    elif title_quit_button.is_clicked(event.pos):
                        self.type_sound.play()
                        self.running = False

            self.clock.tick(FPS)

    def run_words(self, item_list, item_target, item_order="random"):
        global RECOGNIZED_TEXT, RECOGNIZED_DATA, RECOGNIZER_STATUS
        """Handles the words mode loop."""
        back_button =     Button(self.screen_width - 220, self.screen_height - 70, "Back", 200, 50, DARK_RED)
        next_button =     Button(self.screen_width - 220, self.screen_height - 150, "Next", 200, 50, DARK_GREEN)
        new_game_button = Button(self.screen_width - 430, self.screen_height - 70, "More", 200, 50)

        self.game_font = self.game_font_large

        if item_order == "random":
            random.shuffle(item_list)
        item_index = 0
        word = item_list[item_index]
        # matching_files = [file for file in os.listdir(CLIPART_PATH) if f"_{word.replace(".","").replace("!","").lower()}_" in file.lower()]
        matching_files = get_matching_files(word)
        if matching_files:
            word_background = pygame.image.load(os.path.join(CLIPART_PATH, random.choice(matching_files)))
        else:
            word_background = pygame.image.load("assets/images/images/unknown_001.png")
        img_width, img_height = word_background.get_size()
        word_background = pygame.transform.smoothscale(word_background, (img_width * (1080 / img_height) / 2, 1080 / 2))

        # set up microphone image
        img_microphone_width, img_microphone_height = 200, 200
        img_microphone = pygame.image.load("assets/images/images/microphone_001.png")
        img_microphone = pygame.transform.smoothscale(img_microphone, (img_microphone_width, img_microphone_height))
        img_microphone_rect = pygame.Rect(self.screen_width // 2 - img_microphone_width // 2, self.screen_height - img_microphone_height - 40, img_microphone_width, img_microphone_height)

        word_complete = False
        play_new_word_sound = True
        play_round_complete = False
        start_time = None
        completed_words = 0
        game_over = False
        # self.midi_music_start_time = None

        # set up rectangles for text boxes
        box_width = self.screen_width * 0.8
        box_height = 150
        msg_box_rect = pygame.Rect(self.screen_width // 2 - box_width // 2, self.screen_height - box_height * 2 - 200, box_width, box_height // 3)
        word_box_rect = pygame.Rect(self.screen_width // 2 - box_width // 2, self.screen_height - box_height * 2 - 130, box_width, box_height)

        while not self.game_mode == "menu" and self.running:
            self.screen.fill(DARK_GRAY)

            # Display progress
            progress_surface = self.font.render(f"Words Completed: {completed_words}/{item_target}", True, LIGHT_YELLOW)
            progress_rect = progress_surface.get_rect(topleft=(20, 20))
            self.screen.blit(progress_surface, progress_rect)

            if not game_over:
                back_button.draw(self.screen, self.button_font)
                next_button.draw(self.screen, self.button_font)

                # Display Timer
                time_sec = (pygame.time.get_ticks() - start_time) / 1000 if start_time else 0
                timer_surface = self.font.render(f"Time: {time_sec:.2f}", True, LIGHT_YELLOW)
                timer_rect = timer_surface.get_rect(topright=(self.screen_width - 20, 20))
                self.screen.blit(timer_surface, timer_rect)

                # Display images
                self.screen.blit(word_background, (self.screen_width / 2 - word_background.get_width() / 2 - 10, 30))
                self.screen.blit(img_microphone, img_microphone_rect)

                # Display msg box
                msg_surface = self.msg_font.render("Please say:", True, YELLOW)
                self.screen.blit(msg_surface, msg_box_rect)

                # Display word in styled box
                word_surface = render_text_wrapped(word, self.game_font, TEXT_COLOR, box_width - 30) 
                draw_styled_text_box(self.screen, word_box_rect, word_surface, PROMPT_BOX_COLOR)
                
                # Display Instructions
                instruction_surface = self.font.render("Hint: say the word out loud.", True, WHITE)
                instruction_rect = instruction_surface.get_rect(left = 20, top = self.screen_height - 50)
                self.screen.blit(instruction_surface, instruction_rect)

                if word_complete:
                    pygame.draw.rect(self.screen, GREEN, word_box_rect.inflate(20, 10), 3, border_radius=20) # highlight box green if correct
                
                # Update the display
                pygame.display.flip()

                if word_complete:

                    if completed_words == item_target:
                        # completion target reached, game over.
                        game_over = True
                        self.this_index = 0
                        play_round_complete = True
                    else:
                        # Reset for next word
                        item_index +=1
                        if item_index >= len(item_list):
                            item_index = 0
                        word = item_list[item_index]
                        # Load new background image for the word
                        # matching_files = [file for file in os.listdir(CLIPART_PATH) if f"_{word.replace(".","").replace("!","").lower()}_" in file.lower()]
                        matching_files = get_matching_files(word)
                        if matching_files:
                            word_background = pygame.image.load(os.path.join(CLIPART_PATH, random.choice(matching_files)))
                        else:
                            # default background if no matching image found
                            word_background = pygame.image.load("assets/images/images/unknown_001.png")
                        img_width, img_height = word_background.get_size()
                        word_background = pygame.transform.smoothscale(word_background, (img_width * (1080 / img_height) / 2, 1080 / 2))
                        play_new_word_sound = True
                    start_time = None
                    while pygame.mixer.get_busy():
                        self.clock.tick(FPS)

                if play_new_word_sound:
                    # Play the sound prompt for the new word
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Playing prompt sound for word: {word}")
                    if self.sounds.get(word):
                        new_word_sound = self.sounds[word]
                    else:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sound for word '{word}' not found, generating...")
                        new_word_sound = generate_speech_sound(word, lang=DEFAULT_LANGUAGE)

                    new_word_prompt = merge_sounds(self.Sound_PleaseSay, new_word_sound)
                    while pygame.mixer.get_busy():
                        self.clock.tick(FPS)
                    new_word_prompt.play()

                    # turn off the highlight for the word box
                    word_complete = False

                    RECOGNIZER_STATUS = "LISTEN"
                    RECOGNIZED_TEXT = ""

                    play_new_word_sound = False

            else: # Game Over Prompt
                # play victory reward song and dance
                if play_round_complete:
                    # call midi song function until it's finished
                    play_round_complete = not self.midi_play_song()
                    
                    # paint animation frame
                    current_image = self.dance_frames[self.current_frame // 3]
                    img_rect = current_image.get_rect(center=(self.screen_width * 9 // 16, self.screen_height * 2 // 5))
                    self.screen.blit(current_image, img_rect)
                    self.current_frame = (self.current_frame + 1) % len(self.dance_frames)

                back_button.draw(self.screen, self.button_font)
                new_game_button.draw(self.screen, self.button_font)
                prompt_text = self.font.render("Good job! Continue?", True, DARK_BLUE)
                prompt_rect = prompt_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
                pygame.draw.rect(self.screen, LIGHT_YELLOW, prompt_rect.inflate(20, 10))
                self.screen.blit(prompt_text, prompt_rect)

                pygame.display.flip()

            if RECOGNIZER_STATUS == "COMPLETE":
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Recognized text: {RECOGNIZED_TEXT}")
                # if RECOGNIZED_TEXT.upper() == word.upper():
                if word.upper() in RECOGNIZED_TEXT.upper():
                    if  f"SAY {word.upper()}" not in RECOGNIZED_TEXT.upper():
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Word matched!")
                        completed_words += 1
                        word_complete = True
                        # play successful answer prompt
                        recorded_sound = pygame.mixer.Sound(file=io.BytesIO(RECOGNIZED_DATA.get_wav_data()))
                        combined_sound = merge_sounds(self.Sound_Good, recorded_sound)
                        combined_sound.play()
                        RECOGNIZED_TEXT = ""
                        RECOGNIZED_DATA = None
                        RECOGNIZER_STATUS = "READY"
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Word did not match.")
                    # play no good audio prompt
                    recorded_sound = pygame.mixer.Sound(file=io.BytesIO(RECOGNIZED_DATA.get_wav_data()))
                    combined_sound = merge_sounds(merge_sounds(self.Sound_NoGood, recorded_sound), new_word_prompt)
                    combined_sound.play()

                    RECOGNIZED_TEXT = ""
                    RECOGNIZED_DATA = None
                    RECOGNIZER_STATUS = "LISTEN"
            elif RECOGNIZER_STATUS == "ERROR":
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Speech recognition error: " + RECOGNIZED_TEXT)
                # play no good audio prompt
                if RECOGNIZED_TEXT == "UNRECOGNIZED":
                    if RECOGNIZED_DATA is not None:
                        recorded_sound = pygame.mixer.Sound(file=io.BytesIO(RECOGNIZED_DATA.get_wav_data()))
                        combined_sound = merge_sounds(merge_sounds(self.Sound_NoGood, recorded_sound), new_word_prompt)
                    else:
                        combined_sound = merge_sounds(self.Sound_NoGood, new_word_prompt)
                elif RECOGNIZED_TEXT == "TIMEOUT":
                    combined_sound = merge_sounds(self.Sound_NoHear, new_word_prompt)
                elif RECOGNIZED_TEXT == "API ERROR":
                    combined_sound = merge_sounds(self.Sound_NoHear, new_word_prompt)
                combined_sound.play()

                RECOGNIZED_DATA = None
                RECOGNIZED_TEXT = ""
                RECOGNIZER_STATUS = "LISTEN"

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.KEYDOWN:
                    # Toggle between fullscreen and windowed modes
                    if event.key == pygame.K_RETURN and pygame.key.get_mods() & pygame.KMOD_ALT:
                        self.fullscreen, self.screen = toggle_fullscreen(self.screen, self.screen_width, self.screen_height, self.fullscreen)

                    if event.key == pygame.K_ESCAPE:
                        self.game_mode = "menu" # Return to menu on ESC
                        # speech_thread.join()  # Ensure the listening thread has finished

                    if not game_over:
                        if start_time is None:
                            start_time = pygame.time.get_ticks()

                # Mouse event handlers
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # if not game_over:
                    if back_button.is_clicked(event.pos):
                        self.game_mode = "menu"
                    if next_button.is_clicked(event.pos):
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Word skipped!")
                        completed_words += 1
                        word_complete = True
                        self.Sound_Skipped.play()
                        RECOGNIZED_TEXT = ""
                        RECOGNIZED_DATA = None
                        RECOGNIZER_STATUS = "READY"
                    if game_over:
                        if new_game_button.is_clicked(event.pos):
                            self.type_sound.play()
                            completed_words = 0
                            item_index +=1
                            if item_index >= len(item_list):
                                item_index = 0
                            word = item_list[item_index]
                            # matching_files = [file for file in os.listdir(CLIPART_PATH) if f"_{word.replace(".","").replace("!","").lower()}_" in file.lower()]
                            matching_files = get_matching_files(word)
                            if matching_files:
                                word_background = pygame.image.load(os.path.join(CLIPART_PATH, random.choice(matching_files)))
                            else:
                                word_background = pygame.image.load("assets/images/images/unknown_001.png")
                            img_width, img_height = word_background.get_size()
                            word_background = pygame.transform.smoothscale(word_background, (img_width * (1080 / img_height) / 2, 1080 / 2))
                            play_new_word_sound = True
                            start_time = None
                            game_over = False
                            self.this_melody = random.choice(self.melodies)
                            self.this_index = 0
                            self.max_index = len(self.this_melody) - 1
                            while pygame.mixer.get_busy():
                                pygame.time.Clock().tick(FPS)
            self.clock.tick(FPS)

if __name__ == '__main__':
    game = TalkingGame()
    game.run()
    stream.stop()
    stream.close()

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import speech_recognition as sr
import io

def capture_and_transcribe(sample_rate=44100, silence_threshold=500, silence_duration=1, min_duration=5, max_duration=30):
    """
    Args:
        sample_rate (int): Sampling rate in Hz (default: 44100).
        silence_threshold (int): Amplitude threshold for silence/speech detection (default: 500).
        silence_duration (float): Duration of silence to stop recording after speech (default: 2 seconds).
        min_duration (float): Minimum recording duration if no speech (default: 5 seconds).
        max_duration (int): Maximum recording duration (default: 30 seconds).
    
    Returns:
        str: Transcribed text or error message.
    """
    try:
        chunk_size = 1024  # Number of samples per chunk
        silence_samples = int(silence_duration * sample_rate / chunk_size)
        min_samples = int(min_duration * sample_rate / chunk_size)
        max_samples = int(max_duration * sample_rate)
        audio_data = []
        silence_counter = 0
        chunk_count = 0
        speech_detected = False
        
        # Initialize audio stream
        stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', blocksize=chunk_size)
        
        print("Recording... Speak now.")
        stream.start()
        
        # Warm-up period to stabilize microphone (0.1 seconds)
        for _ in range(int(0.1 * sample_rate / chunk_size)):
            stream.read(chunk_size)
        
        for _ in range(int(max_samples / chunk_size)):
            chunk, overflow = stream.read(chunk_size)
            if overflow or chunk.size == 0:
                continue  # Skip invalid or empty chunks
            audio_data.append(chunk)
            chunk_count += 1
            
            # Calculate RMS amplitude safely
            mean_square = np.mean(chunk.astype(np.float64)**2)
            if np.isnan(mean_square) or mean_square <= 0:
                rms = 0.0
            else:
                rms = np.sqrt(mean_square)
            
            # Check for speech
            if rms >= silence_threshold:
                speech_detected = True
            
            # If speech detected, check for silence
            if speech_detected:
                if rms < silence_threshold:
                    silence_counter += 1
                else:
                    silence_counter = 0
                if silence_counter >= silence_samples:
                    break
            # If no speech and min_duration reached, stop
            elif chunk_count >= min_samples:
                break
                
        stream.stop()
        stream.close()
        print("Recording finished.")
        
        # Convert audio data to numpy array
        if not audio_data:
            return "No audio data recorded."
        audio = np.concatenate(audio_data, axis=0)
        
        # Create in-memory WAV buffer
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, sample_rate, audio)
        wav_buffer.seek(0)
        
        # Transcribe using speech_recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_buffer) as source:
            audio_data = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_data)
            print("Transcription: ", text)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError as e:
            return f"Transcription failed: {e}"
            
    except Exception as e:
        return f"Error during recording or transcription: {e}"

# Example usage
if __name__ == "__main__":
    result = capture_and_transcribe(min_duration=5)
    print("Result:", result)
# audio_analyzer.py
import pyaudio
import numpy as np
import time

class AudioAnalyzer:
    def __init__(self, rate=16000, chunk=1024):
        self.rate = rate
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)
        self.prev_whisper_time = 0
        self.prev_rustle_time = 0

    def analyze_audio(self):
        try:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Energy
            rms = np.sqrt(np.mean(audio_data**2))
            
            # FFT and normalization
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft), 1 / self.rate)
            magnitude = np.abs(fft)
            magnitude = magnitude[:len(magnitude)//2]
            freqs = freqs[:len(freqs)//2]
            magnitude /= np.max(magnitude) if np.max(magnitude) > 0 else 1

            # Spectral centroid (helps distinguish whisper vs ambient noise)
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)

            # Whisper band and rustling band
            low_freq_mag = np.mean(magnitude[(freqs >= 100) & (freqs <= 500)])
            mid_freq_mag = np.mean(magnitude[(freqs >= 800) & (freqs <= 3000)])

            current_time = time.time()

            # Whisper detection logic (refined)
            whisper_detected = (
                rms < 700 and
                low_freq_mag > 0.2 and
                spectral_centroid < 600 and
                (current_time - self.prev_whisper_time > 2)
            )

            # Rustling detection logic (unchanged)
            rustle_detected = (
                rms > 700 and
                mid_freq_mag > 0.3 and
                spectral_centroid > 1200 and
                (current_time - self.prev_rustle_time > 2)
            )

            if whisper_detected:
                self.prev_whisper_time = current_time
                return "💬 Whisper Detected"
            elif rustle_detected:
                self.prev_rustle_time = current_time
                return "📄 Paper Rustling Detected"
            else:
                return None
        except Exception as e:
            print(f"[ERROR] Audio processing failed: {e}")
            return None


    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

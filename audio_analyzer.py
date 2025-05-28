import pyaudio
import numpy as np
import webrtcvad
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from scipy.signal import butter, lfilter

class AudioAnalyzer:
    def __init__(self, sample_rate=16000, frame_duration=30):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # Aggressive speech detection

        # Load Silero VAD model ONCE
        self.model = load_silero_vad()

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=int(self.sample_rate * frame_duration / 1000)
        )

    def analyze_audio(self):
        audio_data = self.stream.read(
            int(self.sample_rate * self.frame_duration / 1000),
            exception_on_overflow=False
        )
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # WebRTC VAD (optional, for extra robustness)
        frame_length = int(self.sample_rate * self.frame_duration / 1000)
        speech_detected = self.vad.is_speech(audio_np[:frame_length].tobytes(), self.sample_rate)

        # Silero VAD expects float32 torch.Tensor normalized to [-1, 1]
        audio_float = audio_np.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float)

        # Use the loaded model as the second argument
        speech_timestamps = get_speech_timestamps(audio_tensor, self.model)
        silero_speech_detected = len(speech_timestamps) > 0

        if silero_speech_detected or speech_detected:
            return "🔊 Voice Detected!"
        else:
            return None

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

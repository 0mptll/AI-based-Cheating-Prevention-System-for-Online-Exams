import pyaudio
import numpy as np
import webrtcvad
import torch
from silero_vad import load_silero_vad, get_speech_timestamps


class AudioAnalyzer:
    def __init__(self, sample_rate=16000, frame_duration=100):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # capture larger window to catch whispers

        # ✅ WebRTC VAD: Max sensitivity
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # Mode 3 = most aggressive detection

        # ✅ Load Silero VAD ONCE
        self.model = load_silero_vad()

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=int(self.sample_rate * frame_duration / 1000)
        )

    def amplify_audio(self, audio_np, gain=3.0):
        """Boost the signal to help detect whispers."""
        audio_float = audio_np.astype(np.float32) / 32768.0
        amplified = audio_float * gain
        return np.clip(amplified, -1.0, 1.0)

    def analyze_audio(self):
        try:
            audio_data = self.stream.read(
                int(self.sample_rate * self.frame_duration / 1000),
                exception_on_overflow=False
            )
            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            # 🔊 WebRTC VAD (first 30ms only)
            vad_frame_samples = int(self.sample_rate * 30 / 1000)
            speech_webrtc = False
            if len(audio_np) >= vad_frame_samples:
                speech_webrtc = self.vad.is_speech(
                    audio_np[:vad_frame_samples].tobytes(), self.sample_rate
                )

            # 🔊 Silero VAD (amplified & normalized)
            amplified = self.amplify_audio(audio_np, gain=3.5)
            tensor_audio = torch.from_numpy(amplified)

            speech_timestamps = get_speech_timestamps(
                tensor_audio,
                self.model,
                threshold=0.25,  # 🔁 Lower threshold for whispers
                min_speech_duration_ms=60,  # detect even short bursts
                min_silence_duration_ms=30,
                window_size_samples=128  # smaller window increases sensitivity
            )
            speech_silero = len(speech_timestamps) > 0

            if speech_webrtc or speech_silero:
                return "🎙️ Human Voice Detected (any tone)"
            else:
                return None
        except Exception as e:
            print(f"Audio analysis error: {e}")
            return None

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

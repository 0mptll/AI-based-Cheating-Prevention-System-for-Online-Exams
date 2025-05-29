import whisper
import pyaudio
import numpy as np
import torch

def record_audio(duration=5, sample_rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)
    print("Recording...")
    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.int16))
    print("Recording complete.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    audio_data = np.hstack(frames).astype(np.float32) / 32768.0
    return audio_data

def transcribe_audio(audio_data, sample_rate=16000):
    model = whisper.load_model("base")
    audio = torch.from_numpy(audio_data)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text

if __name__ == "__main__":
    audio_data = record_audio(duration=5)
    transcription = transcribe_audio(audio_data)
    print("Transcription:", transcription)

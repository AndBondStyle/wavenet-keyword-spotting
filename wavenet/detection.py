from wavenet.training import model_from_checkpoint
from wavenet.dataset import init_feature_extractor
from numpy_ringbuffer import RingBuffer
from time import time, sleep
from pathlib import Path
import numpy as np
import librosa
import torch


class KeywordDetector:
    def __init__(
        self,
        checkpoint: Path,
        threshold: float = 0.9,
        smoothing: int = 5,
        timeout: float = 1.0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.config = model_from_checkpoint(checkpoint, device=self.device)
        self.model = self.model.eval().to(self.device)
        self.extractor = init_feature_extractor(self.config)
        self.threshold = threshold
        self.smoothing = smoothing
        self.timeout = timeout

        self.input_samples = self.config.fft_window_step * (self.model.input_size - 1)
        self.preds_buffer = RingBuffer(self.smoothing, "float32")
        self.raw_input_buffer = RingBuffer(capacity=self.input_samples, dtype="float32")
        self.raw_input_buffer.extend(np.zeros(self.input_samples, dtype="float32"))
        self.cooldown_time = 0

    def push_audio(self, signal: np.ndarray):
        self.raw_input_buffer.extend(signal)

    @torch.no_grad()
    def predict(self):
        chunk = self.raw_input_buffer[-self.input_samples :].copy()
        chunk = librosa.util.normalize(chunk)
        x = self.extractor(chunk).astype("float32")
        x = x.reshape((1, self.model.input_channels, self.model.input_size))
        x = torch.as_tensor(x).to(self.device)
        pred = self.model.forward(x)
        pred = torch.softmax(pred[0], 0).cpu().numpy()[1]
        self.preds_buffer.append(pred)
        mean = np.mean(self.preds_buffer)
        trigger = mean > self.threshold and self.cooldown_time < time()
        if trigger:
            self.cooldown_time = time() + self.timeout
        return mean, trigger


if __name__ == "__main__":
    from threading import Thread, Event
    import pyaudio

    detector = KeywordDetector(
        checkpoint="../checkpoints/v1.epoch5.pt",
        threshold=0.6,
        smoothing=3,
        timeout=1.0,
    )
    chunk_size = detector.config.fft_window_step
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paFloat32,
        rate=detector.config.sample_rate,
        frames_per_buffer=chunk_size,
        channels=1,
        input=True,
    )
    stop = Event()

    def prediction_loop():
        print("Prediction thread started")
        while not stop.is_set():
            res = detector.predict()
            if res is not None:
                pred, trigger = res
                keyword = "*** KEYWORD! ***"
                bar = "#" * int(pred * 100)
                bar += " " * (100 - len(bar))
                print(
                    f"LEVEL: {pred:.3f} "
                    f"{keyword if trigger else (' ' * len(keyword))} "
                    f"| {bar} |"
                )
            sleep(0.001)

    def audio_loop():
        print("Audio thread started")
        while not stop.is_set():
            chunk = stream.read(chunk_size)
            chunk = np.frombuffer(chunk, dtype="float32")
            detector.push_audio(chunk)
        stream.stop_stream()
        stream.close()

    prediction_thread = Thread(target=prediction_loop)
    audio_thread = Thread(target=audio_loop)
    prediction_thread.start()
    audio_thread.start()

    # fmt: off
    try: input()
    except KeyboardInterrupt: pass
    print("Terminating...")
    stop.set()
    prediction_thread.join()
    audio_thread.join()

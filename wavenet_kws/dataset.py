from dataclasses import dataclass, field, asdict
from wavenet_kws.helpers import *
from tqdm.auto import tqdm
from typing import Tuple
from pathlib import Path
from retry import retry
import pyrubberband
import numpy as np
import soundfile
import librosa
import pickle

DATASETS_PATH = Path(__file__).parent.parent / "datasets"
AUDIOSET_PATH = DATASETS_PATH / "audioset"
KEYWORDS_PATH = DATASETS_PATH / "keywords"


@dataclass
class DatasetConfig:
    """
    Dataset configuration object
    ============================
    - sample_rate - audio sampling rate
    - sample_length - length of generated samples, in seconds
    - fft_window_size - FFT window size, in samples
    - fft_window_step - FFT window step (hop), in samples
    - mfcc_num - number of extracted MFCC-features
    - mfcc_min_freq - minimum frequency cutoff
    - mfcc_max_freq - maximum frequency cutoff
    - sources - dict of probabilities to generate one of four samples:
        - positive_word - true keyword with random background
        - negative_word - fake keyword (similar word) with random background
        - random_speech - chunk of random speech
        - empty_background - empty random background
    - background_sources - dict of background probabilities
    - word_undercut_range - min & max undercut amount, in seconds. Will be applied both
    for positive and negative words.
    - word_right_margin_range - min & max distance from word end to sample end, in
    seconds (end = right edge, time axis goes right). Will be applied both for positive
    and negative words.
    - background_gain_range - min & max background gain, relative to main signal (word)
    - rubberband_ratio - probability of activating rubberband filter
    - rubberband_pitch_range - min & max pitch shift for rubberband filter
    """

    sample_rate: int = 16000
    sample_length: float = 3.0
    fft_window_size: int = 512
    fft_window_step: int = 256
    mfcc_num: int = 40
    mfcc_min_freq: int = 20
    mfcc_max_freq: int = 4000
    sources: dict = field(
        default_factory=lambda: dict(
            positive_word=0.5,
            negative_word=0.25,
            random_speech=0.15,
            empty_background=0.1,
        )
    )
    background_sources: dict = field(
        default_factory=lambda: dict(
            noise=0.6,
            music=0.4,
        )
    )
    word_undercut_range: Tuple[float, float] = (0, 0.05)
    word_right_margin_range: Tuple[float, float] = (0.4, 0.7)
    background_gain_range: Tuple[float, float] = (0.05, 0.2)
    rubberband_ratio: float = 0.5
    rubberband_pitch_range: Tuple[float, float] = (-2.0, +2.0)

    def asdict(self):
        return asdict(self)

    def sec_to_samples(self, sec: float):
        return int(sec * self.sample_rate)

    def samples_to_sec(self, samples: int):
        return samples / self.sample_rate

    def sec_to_windows(self, sec: float):
        return int(sec * self.sample_rate / self.fft_window_step)


def postprocess_source(file: Path, config: DatasetConfig):
    """
    Applies postprocessing on main source signal (positive or negative word)
    :param file: path to main signal file
    :param config: dataset config
    :return: completed source signal
    """
    main_signal, _ = librosa.load(str(file), sr=config.sample_rate)
    undercuts = [np.random.uniform(*config.word_undercut_range) for _ in range(2)]
    left, right = [config.sec_to_samples(x) for x in undercuts]
    main_signal = main_signal[left:-right]

    if np.random.random() < config.rubberband_ratio:
        pitch = np.random.uniform(*config.rubberband_pitch_range)
        main_signal = pyrubberband.pitch_shift(main_signal, config.sample_rate, pitch)

    bg_source = get_weighted_item(config.background_sources)
    bg_file = get_random_file(AUDIOSET_PATH / bg_source)
    bg_signal, _ = librosa.load(str(bg_file), sr=config.sample_rate)
    bg_signal = get_random_chunk(bg_signal, config.sample_length, config.sample_rate)
    bg_gain = np.random.uniform(*config.background_gain_range)
    bg_signal = bg_gain * librosa.util.normalize(bg_signal) * np.max(main_signal)

    right_margin = np.random.uniform(*config.word_right_margin_range)
    main_signal_end = config.sample_length - right_margin
    return merge_signals(
        (bg_signal, 0, None),
        (main_signal, None, main_signal_end),
        sample_rate=config.sample_rate,
        length=config.sample_length,
    )


@retry(tries=10)
def generate_sample(config: DatasetConfig):
    """
    Generates one sample and associated label
    :param config: dataset config
    :return: signal (np.ndarray), label (int)
    """
    source = get_weighted_item(config.sources)

    if source == "positive_word":
        file = get_random_file(KEYWORDS_PATH / "positives")
        signal = postprocess_source(file, config)
    elif source == "negative_word":
        file = get_random_file(KEYWORDS_PATH / "negatives")
        signal = postprocess_source(file, config)
    elif source == "random_speech":
        file = get_random_file(KEYWORDS_PATH / "random_speech")
        signal, _ = librosa.load(str(file), sr=config.sample_rate)
        signal = get_random_chunk(signal, config.sample_length, config.sample_rate)
    elif source == "empty_background":
        bg_source = get_weighted_item(config.background_sources)
        file = get_random_file(AUDIOSET_PATH / bg_source)
        signal, _ = librosa.load(str(file), sr=config.sample_rate)
        signal = get_random_chunk(signal, config.sample_length, config.sample_rate)
    else:
        raise RuntimeError(f"Unknown source: {source}")

    label = int(source == "positive_word")
    return signal, label


def init_feature_extractor(config: DatasetConfig):
    """
    Returns a feature extraction function: (raw audio) -> (audio features)
    :param config: dataset config (see above)
    :return: feature extraction function
    """

    def extract(signal: np.ndarray):
        return librosa.feature.mfcc(
            y=signal,
            sr=config.sample_rate,
            n_mfcc=config.mfcc_num,
            n_fft=config.fft_window_size,
            hop_length=config.fft_window_step,
            fmin=config.mfcc_min_freq,
            fmax=config.mfcc_max_freq,
        )

    return extract


def generate_dataset(path: Path, config: DatasetConfig, size=10000, save_audio=False):
    """
    Function to generate composite dataset using positive and negative keyword samples,
    random speech and AudioSet samples as background audio.
    :param path: where to store generated dataset
    :param config: dataset config (see above)
    :param size: number of items (samples) in dataset
    :param save_audio: False = only save extracted features, True = features + audio
    """
    assert not path.exists(), "Dataset already exists"
    print(f"Generating dataset -> {path}")
    path.mkdir(parents=True)
    extractor = init_feature_extractor(config)

    items = []
    for i in tqdm(range(size)):
        signal, label = generate_sample(config)
        signal = librosa.util.normalize(signal)
        features = extractor(signal)
        if save_audio:
            audio_path = path / "samples" / f"{i}.wav"
            audio_path.parent.mkdir(exist_ok=True)
            soundfile.write(str(audio_path), signal, config.sample_rate)
        items.append({"index": i, "label": label, "features": features})

    print("Saving data...")
    with open(path / "data.pkl", "wb") as file:
        dataset = {"config": config.asdict(), "items": items}
        pickle.dump(dataset, file)


if __name__ == "__main__":
    from pprint import pprint

    # To generate a new dataset, modify its config here
    PATH = DATASETS_PATH / "v1"
    SIZE = 10000
    SAVE_AUDIO = True
    CONFIG = DatasetConfig()
    # CONFIG.something = ...

    assert not PATH.exists(), f"{PATH} already exists"
    print("Preparing to generate dataset:")
    print(f"- Path: {PATH}")
    print(f"- Size: {SIZE}")
    print(f"- Save audio: {SAVE_AUDIO}")
    print("Dataset config:")
    pprint(CONFIG.asdict())

    ok = input("\nContinue? (y/n): ").lower() == "y"
    if not ok:
        print("Aborting...")
        exit(0)

    np.random.seed(42)
    generate_dataset(PATH, CONFIG, SIZE, SAVE_AUDIO)

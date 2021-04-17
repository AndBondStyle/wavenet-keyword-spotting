# WaveNet Keyword Spotting

[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Open in Colab](https://img.shields.io/badge/Open%20in%20Colab-blue?style=for-the-badge&logo=Google%20Colab)](https://colab.research.google.com/github/AndBondStyle/wavenet-keyword-spotting/blob/master/notebooks/colab_demo.ipynb)
[![University: HSE](https://img.shields.io/badge/University-HSE-blue?&style=for-the-badge)](https://www.hse.ru/)
[![Code style: black](https://img.shields.io/badge/Code%20style-black-000000.svg?&style=for-the-badge)](https://github.com/psf/black)

This is a PyTorch implementation of keyword spotting based on
[WaveNet](https://arxiv.org/pdf/1609.03499.pdf) architecture.  
To get started: jump straight
[to the code](https://github.com/AndBondStyle/wavenet-keyword-spotting/blob/master/wavenet_kws/model.py),
run demo
[notebook in Google Colab](https://colab.research.google.com/github/AndBondStyle/wavenet-keyword-spotting/blob/master/notebooks/colab_demo.ipynb)
or continue reading.

## Installation

To install the repository with all requirements, run:
```
pip install git+https://github.com/AndBondStyle/wavenet-keyword-spotting
```

Directory `wavenet_kws` will be installed as a package (`import wavenet_kws`).  
If you're planning to generate a custom dataset, you will also need to install
[ffmpeg](https://www.ffmpeg.org/) and
[rubberband](https://breakfastquay.com/rubberband/).

## Generating dataset

Custom dataset consists of 3 directories:
- `positives` - cropped samples of **true** keywords you want to detect
- `negatives` - cropped samples of **fake** keywords (e.g. similar words).
- `random_speech` - random samples of speech, not containing keyword

All files should have `.wav` extension and `16000 Hz` sample rate.

Dataset can be generated using [dataset.py](wavenet_kws/dataset.py) script (look inside for details). 
Before running that, you will also need to download
[AudioSet](https://research.google.com/audioset/) (used as background noise source)
via [audioset_download.ipynb](notebooks/audioset_download.ipynb) notebook.

## Training

Training is done using [training.py](wavenet_kws/training.py) script (look inside for details).  
An important note is that model checkpoint file also contains dataset configuration
(see [DatasetConfig](wavenet_kws/dataset.py) class), and model config (keyword arguments
used to initialize model, like `WavenetKWS(**kwargs)`). That way it's very convenient
to load models: check [`model_from_checkpoint`](wavenet_kws/training.py) for details.

## Running

There are several scripts to try your model (or
[pretrained](https://github.com/AndBondStyle/wavenet-keyword-spotting/releases/tag/v1)
one):
- [colab_demo](notebooks/colab_demo.ipynb) - notebook specially designed to run in
[Google Collab](https://colab.research.google.com/github/AndBondStyle/wavenet-keyword-spotting/blob/master/notebooks/colab_demo.ipynb)
- [live_mic_detection](notebooks/live_mic_detection.ipynb) - notebook with real-time prediction using microphone stream
- [detection.py](wavenet_kws/detection.py) - console version of above notebook, if you don't like jupyter

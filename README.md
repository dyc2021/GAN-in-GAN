## About The Project

This is the code implementation of GAN-in-GAN. 
GAN-in-GAN is a novel GAN architecture proposed by us to simultaneously optimize both spectrogram-level and audio-level representations for an end-to-end monaural speech enhancement.


### Author: Yicun Duan (20216268)
### Email: scyyd3 at nottingham dot edu dot cn

---

## Getting Started

In this section, we will showcase how to run this code.

### Prerequisites

You should install PyTorch and the following support packets:

* soundfile (v0.12.1)
  ```sh
  pip install soundfile
  ```
  
* ptflops (v0.7)
  ```sh
  pip install ptflops
  ```
  
You should also correctly deploy the VoiceBank+DEMAND dataset.
Please place the training audio samples under folder `./dataset/train/`, and 
place testing audio samples under folder `./dataset/test/`.

The correct directory structure should be like:
```bash
─dataset
  ├─json
  │  ├─test
  │  └─train
  ├─test
  │  ├─clean_testset_wav
  │  └─noisy_testset_wav
  └─train
      ├─clean_trainset_28spk_wav
      └─noisy_trainset_28spk_wav
```



### Usage

To start training, just enter the command:

```bash
python main.py
```

---
## Files

* dataset.py: it contains the `Dataset` and `DataLoader` classes
* discriminator.py: it contains the implementation of discriminators
* generator.py: it contains the implementation of generators
* json_extract.py: it generates the json files listing train data and test data
* main.py: it's the entry of this project
* train.py and train_solver.py: they contain the training code
---

## Acknowledgments

* I thank Prof. Jianfeng Ren, Prof. Xudong Jiang and Prof. Heng Yu for their patient guidance and insightful suggestions.
* The complex-valued Restormer implementation is adapted from https://github.com/leftthomas/Restormer
* The training code uses https://github.com/yuguochencuc/DB-AIAT for reference.

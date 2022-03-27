# Add noise for Language Identification

## Introduction to noise data set
Different channels of rats data set are used as noise, in which channel A,E,H is used as noise data of test set,
B,C,D,F,G channel is used as noise of training set.

At the same time, different SNR (5, 10, 15, 20) are used for noise addition, The smaller the SNR, the greater the noise

Before running, you need to change ```rats_data``` variable, change to your own rats noise data path

```
# for training data set
bash add-noise-for-lid.sh --steps 1-2 --src-train lre_train --noise_dir data/rats_noise_channel_BCDFG

# fot test set
bash add-noise-for-lid.sh --steps 2 --src-train lre_eval_3s --noise_dir data/rats_noise_channel_AEH
bash add-noise-for-lid.sh --steps 2 --src-train lre_eval_10s --noise_dir data/rats_noise_channel_AEH
bash add-noise-for-lid.sh --steps 2 --src-train lre_eval_30s --noise_dir data/rats_noise_channel_AEH
```

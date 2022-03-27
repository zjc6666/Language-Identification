# Transformer based Language Identification System
# Data preparation scripts and training pipeline for Language Identification

## Data preparation
### upsampling to 16k
Our proposed model aims to use the feature of wav2vec2 model, but the pretrained XLSR-53 wav2vec2 model is trained with 16K data. 
Therefore, in order to ensure the effect of pretrained model, all data are transformed into 16K(includde train, valid and test set).

```
## wav_scp: The wav.scp file of the dataset you want to upsample
## temp_dir: Temporary folders
## save_16k_dir: Save address of wav file after downsampling

python3 upsampling_16k.py wav_scp temp_dir save_16k_dir

egs:
python3 upsampling_16k.py data/lre17_train/wav.scp /home3/jicheng/source-data/temp/ /home3/jicheng/source-data/lre17-16k/lre_train
python3 upsampling_16k.py data/lre17_eval_3s/wav.scp /home3/jicheng/source-data/temp/ /home3/jicheng/source-data/lre17-16k/lre17_eval_3s
python3 upsampling_16k.py data/lre17_eval_10s/wav.scp /home3/jicheng/source-data/temp/ /home3/jicheng/source-data/lre17-16k/lre17_eval_10s
python3 upsampling_16k.py data/lre17_eval_30s/wav.scp /home3/jicheng/source-data/temp/ /home3/jicheng/source-data/lre17-16k/lre17_eval_30s
```


### Prepare new kaldi format file
You can use the following commands to prepare data, but you need to change ```save_16k_dir``` variable.
```
save_16k_dir=/home3/jicheng/source-data/lre17-16k/
mkdir data-16k
for x in lre_train lre17_eval_3s lre17_eval_10s lre17_eval_30s;do
  mkdir data-16k/$x
  cp data/$x/{utt2spk,spk2utt,utt2lang,wav.scp} data-16k/$x
  cat data-16k/$x/utt2spk | awk -v p=save_16k_dir '{print $1 " " p"/"$1".wav"}' > data-16k/$x/wav.scp
  utils/fix_data_dir.sh data-16k/$x
done
```

### Add Noise
In order to test the performance of the system under noisy background, all data sets are denoised.
Different channels of rats data set are used as noise, in which channel A,E,H is used as noise data of test set, B,C,D,F,G channel is used as noise of training set.

At the same time, different SNR (5, 10, 15, 20) are used for noise addition, The smaller the SNR, the greater the noise.
Before running, you need to change ```rats_data``` variable, change to your own rats noise data path.

```
cd Add-Noise

# for training data set
bash add-noise-for-lid.sh --steps 1-2 --src-train data-16k/lre_train --noise_dir data-16k/rats_noise_channel_BCDFG

# fot test set
bash add-noise-for-lid.sh --steps 2 --src-train data-16k/lre_eval_3s --noise_dir data-16k/rats_noise_channel_AEH
bash add-noise-for-lid.sh --steps 2 --src-train data-16k/lre_eval_10s --noise_dir data-16k/rats_noise_channel_AEH
bash add-noise-for-lid.sh --steps 2 --src-train data-16k/lre_eval_30s --noise_dir data-16k/rats_noise_channel_AEH
```
After run "add-noise-for-lid.sh" script, Each folder generates four additional folders
egs: 
  for lre_train, will generate lre_train_5_snrs、lre_train_10_snrs、lre_train_15_snrs、lre_train_20_snrs

Generate new wav file for noise data

## Training pipiline

## Notice

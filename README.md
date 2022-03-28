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
bash add-noise-for-lid.sh --steps 2 --src-train ../data-16k/lre_eval_3s --noise_dir ../data-16k/rats_noise_channel_AEH
bash add-noise-for-lid.sh --steps 2 --src-train ../data-16k/lre_eval_10s --noise_dir ../data-16k/rats_noise_channel_AEH
bash add-noise-for-lid.sh --steps 2 --src-train ../data-16k/lre_eval_30s --noise_dir ../data-16k/rats_noise_channel_AEH
```
After run "add-noise-for-lid.sh" script, Each folder generates four additional folders
egs: 
  for lre_train, will generate lre_train_5_snrs、lre_train_10_snrs、lre_train_15_snrs、lre_train_20_snrs

### Generate new wav file for noise data

You should change this path "/home3/jicheng/source-data/lre17-16k/" to yourself path.
```
save_16k_dir=/home3/jicheng/source-data/lre17-16k/
for x in lre_train_5srns lre_train_10srns lre_train_15srns lre_train_20srns 
        lre17_eval_3s_5_snrs lre17_eval_3s_10_snrs lre17_eval_3s_15_snrs lre17_eval_3s_20_snrs 
        lre17_eval_10s_5_snrs lre17_eval_10s_10_snrs lre17_eval_10s_15_snrs lre17_eval_10s_20_snrs 
        lre17_eval_30s_5_snrs lre17_eval_30s_10_snrs lre17_eval_30s_15_snrs lre17_eval_30s_20_snrs; do
  cat data-16k/$x/wav.scp | 
    awk -v n=$x p=$save_16k_dir '{l = length($0); a = substr($0, 0,length-3); print $2" "$3" "$4" "$5" "$6" "$7 " " p "/" n "/" $1 ".wav"}' > data-16k/$x/${x}.cmd
    bash generate_new_wav_cmd.sh $x/$x.cmd
done

for x in lre17_eval_3s lre17_eval_10s lre17_eval_30s;do
    for y in 5 10 15 20;do
        cp data-16k/$x/{utt2spk,wav.scp,utt2lang,spk2utt,reco2dur} data-16k/${x}"_"${y}"_snrs/"
        local=${save_16k_dir}"/"${x}"_"${y}"_snrs/"
        cat data-16k/${x}"_"${y}_snrs/wav.scp | awk -v p=$local '{print $1 " " p "noise-" $1 ".wav"}' > data-16k/${x}"_"${y}_snrs/new_wav.scp
        mv data-16k/${x}"_"${y}_snrs/new_wav.scp data-16k/${x}"_"${y}_snrs/wav.scp
    done
done

for x in lre17_train;do
    for y in 5 10 15 20;do
        path=${save_16k_dir}/${x}_${y}_snrs/
        snrs=_${y}_snrs
        rm data-16k/${x}_${y}_snrs/{reco2dur,spk2utt,utt2uniq,wav.scp}
        cat data-16k/${x}_${y}_snrs/utt2lang | awk -v p=$path s=${snrs} '{l=length($1);name=substr($1,7,l);print name s" " p $1".wav"}' > data-16k/${x}_${y}_snrs/wav.scp
        cat data-16k/${x}_${y}_snrs/utt2lang | awk -v p=$path s=${snrs} '{l=length($1);name=substr($1,7,l);print name p" " $2}' > data-16k/${x}_${y}_snrs/utt2spk
        cp data-16k/${x}_${y}_snrs/utt2spk data-16k/${x}_${y}_snrs/utt2lang
        utils/fix_data_dir.sh data-16k/${x}_${y}_snrs/
    done
done
```

## Training pipiline

## Notice

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
## Training pipiline

## Notice

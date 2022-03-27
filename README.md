# Transformer based Language Identification System
# Data preparation scripts and training pipeline for Language Identification

## Data preparation
### upsampling to 16k
Our proposed model aims to use the feature of wav2vec2 model, but the pretrained XLSR-53 wav2vec2 model is trained with 16K data. 
Therefore, in order to ensure the effect of pretrained model, all data are transformed into 16K(includde train, valid and test set).


## Training pipiline

## Notice

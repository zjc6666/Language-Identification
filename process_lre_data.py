import os
import json
import torch
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import s3prl.upstream.wav2vec2.hubconf as hubconf

from kaldiio import WriteHelper

def make_200ms_feat(mfccs, overlap=10, chunk_len=20):
    new_feat = 0
    feature = mfccs
    seq_len = feature.shape[0]
    step = chunk_len - overlap
    num_chunk = (seq_len - overlap) // (chunk_len - overlap)
    if num_chunk > 1:
        start = 0
        end = 0
        for id in range(num_chunk):
            end = start + chunk_len
            feat_temp = feature[start:end, :]
            feat_temp = np.hstack(feat_temp)
            start += step
            if id == 0:
                new_feat = feat_temp
            else:
                new_feat = np.vstack((new_feat, feat_temp))
        num_left = seq_len - end
        start = end - (chunk_len - num_left)
        feat_temp = feature[start:, :]
        feat_temp = np.hstack(feat_temp)
        new_feat = np.vstack((new_feat, feat_temp))
    return new_feat


# 两个文件的第一列需要相同，使用utils/fix_data_dir.sh
def wav_lang_extract(wavscp, utt2lang):
    with open(wavscp, 'r') as f:
        lines_wav = f.readlines()
    audio_list = [x.split()[-1].strip() for x in lines_wav]
    name_list = [x.split()[0].strip() for x in lines_wav]
    with open(utt2lang, 'r') as f:
        lines_utt = f.readlines()
    label_list = [x.split()[-1].strip().replace('-', '') for x in lines_utt]
    return audio_list, label_list, name_list

def feat_extract(wav2vec2lang, model, layer, save_dir, audio_list, label_list, name_list, device):

    feat_scp_path = "{}.scp".format(os.path.join(save_dir, "feats"))
    feat_ark_path = "{}.ark".format(os.path.join(save_dir, "feats"))
    with WriteHelper('ark,scp:' + feat_ark_path + "," + feat_scp_path) as writer:

        with open(wav2vec2lang, 'w') as f:
            for i in tqdm(range(len(audio_list))):
                audio = audio_list[i]
                name = name_list[i]
                print("audio: ", audio)
                print("name:", name)
                data, sr = librosa.load(audio, sr=None)
                data_ = torch.tensor(data).to(device=device, dtype=torch.float).unsqueeze(0)
                # try:
                features = model(data_)
                features = features['hidden_state_{}'.format(layer)]
                features_ = features.squeeze(0).cpu().detach().numpy()
                new_feat = make_200ms_feat(features_, overlap=0, chunk_len=20)
                # save_name = audio.replace(os.path.split(audio[0]), save_dir).replace('.wav', 'w2v_{}.npy'.format(layer))
                # np.save(save_name, new_feat)
                writer(name, new_feat)
                f.write("{} {} {}\n".format(name, label_list[i], new_feat.shape[0]))
                # except:
                #     print("Len:{} {} fail to extract".format(len(data) / sr, audio))
def feat_extract_for_long_utterance(wav2vec2lang, model, layer, save_dir, audio_list, label_list, name_list, device):
    feat_scp_path = "{}.scp".format(os.path.join(save_dir, "feats"))
    feat_ark_path = "{}.ark".format(os.path.join(save_dir, "feats"))
    with WriteHelper('ark,scp:' + feat_ark_path + "," + feat_scp_path) as writer:
        with open(wav2vec2lang, 'w') as f:
            for i in tqdm(range(len(audio_list))):
                audio = audio_list[i]
                name = name_list[i]
                print("audio: ", audio)
                print("name:", name)
                data, sr = librosa.load(audio, sr=None)

                dur_ = len(data)/sr
                num_segs = dur_ // 30
                res_ = dur_ % 30
                N = sr * 30
                print("## len(data_):", len(data), " ; ", num_segs)
                feat_ = list()
                if num_segs >= 1:
                    for ii in range(int(num_segs)):
                        temp = data[N * ii:N * (ii + 1)]
                        temp = torch.tensor(temp).to(device=device, dtype=torch.float).unsqueeze(0)
                        features = model(temp)
                        features = features['hidden_state_{}'.format(layer)]
                        feat_.append(features)
                    if res_ > 2 / 30:
                        res_feats = data[N * int(num_segs):]
                        res_feats = torch.tensor(res_feats).to(device=device, dtype=torch.float).unsqueeze(0)
                        features = model(res_feats)
                        features = features['hidden_state_{}'.format(layer)]
                        feat_.append(features)
                    new_feats = torch.cat(feat_, 1)
                    new_feats = new_feats.squeeze(0).cpu().detach().numpy()
                else:
                    data_ = torch.tensor(data).to(device=device, dtype=torch.float).unsqueeze(0)
                    features = model(data_)
                    features = features['hidden_state_{}'.format(layer)]
                    new_feats = features.squeeze(0).cpu().detach().numpy()
                new_feats = make_200ms_feat(new_feats, overlap=0, chunk_len=20)

                writer(name, new_feats)
                f.write("{} {} {}\n".format(name, label_list[i], new_feats.shape[0]))

def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--json', type=str, default='xsa_config.json')
    args = parser.parse_args()

    with open(args.json, 'r') as json_obj:
        config_proj= json.load(json_obj)

    le = LabelEncoder()
    device = torch.device('cuda:{}'.format(config_proj["optim_config"]["device"])
                          if torch.cuda.is_available() else 'cpu')
    # get pretrained SLSR-53 model
    model_path = config_proj["Input"]["userroot"] + config_proj["wav2vec_info"]["model_path"]
    model = hubconf.wav2vec2_local(ckpt=model_path)
    model.to(device)


    feat_layer = config_proj["wav2vec_info"]["layer"]
    wav_scp_train = config_proj["Input"]["userroot"] + config_proj["Input"]["train_set"] + "/wav.scp"
    utt2lang_train = config_proj["Input"]["userroot"] + config_proj["Input"]["train_set"] + "/utt2lang"
    audio_train, labels_train, name_list = wav_lang_extract(wav_scp_train, utt2lang_train)
    labels_train_index = le.fit_transform(labels_train)
    save_w2v_train_dir = wav_scp_train.replace('/wav.scp', "/wav2vec_" + config_proj["wav2vec_info"]["model_name"] + "_" + str(feat_layer) + "_layer")


    if not os.path.exists(save_w2v_train_dir):
        os.mkdir(save_w2v_train_dir)
    train_txt = config_proj["Input"]["userroot"] + config_proj["Input"]["train_set"] + "/wav2vec_" + config_proj["wav2vec_info"]["model_name"] + ".txt"
    feat_extract(wav2vec2lang=train_txt, model=model, layer=feat_layer, save_dir=save_w2v_train_dir,
                 audio_list=audio_train, label_list=labels_train_index, name_list=name_list, device=device)


    if config_proj["Input"]["valid_set"] != "none":
        wav_scp_valid = config_proj["Input"]["userroot"] + config_proj["Input"]["valid_set"] + "/wav.scp"
        utt2lang_valid = config_proj["Input"]["userroot"] + config_proj["Input"]["valid_set"] + "/utt2lang"
        audio_valid, labels_valid, name_list = wav_lang_extract(wav_scp_valid, utt2lang_valid)
        labels_valid_index = le.transform(labels_valid)
        
        save_w2v_valid_dir =wav_scp_valid.replace('/wav.scp', "/wav2vec_" + config_proj["wav2vec_info"]["model_name"] + "_" + str(feat_layer) + "_layer")
        if not os.path.exists(save_w2v_valid_dir):
            os.mkdir(save_w2v_valid_dir)

        valid_txt = config_proj["Input"]["userroot"] + config_proj["Input"]["valid_set"] + "/wav2vec_" + config_proj["wav2vec_info"]["model_name"] + ".txt"
        feat_extract(wav2vec2lang=valid_txt, model=model, layer=feat_layer, save_dir=save_w2v_valid_dir,
                     audio_list=audio_valid, label_list=labels_valid_index, name_list=name_list, device=device)

    
    test_sets = config_proj["Input"]["test_sets"].split()
    for test in test_sets:
        wav_scp_test = config_proj["Input"]["userroot"] + test + "/wav.scp"
        utt2lang_test = config_proj["Input"]["userroot"] + test + "/utt2lang"
        audio_test, labels_test, name_list = wav_lang_extract(wav_scp_test, utt2lang_test)
        print(labels_test)
        labels_test_index = le.fit_transform(labels_test)
        save_w2v_test_dir = config_proj["Input"]["userroot"] + \
                             test + "/wav2vec_" + config_proj["wav2vec_info"]["model_name"] + "_" + str(feat_layer) + "_layer" 
        if not os.path.exists(save_w2v_test_dir):
            os.mkdir(save_w2v_test_dir)
        test_txt = config_proj["Input"]["userroot"] + test + "/wav2vec_" + config_proj["wav2vec_info"]["model_name"] + ".txt"
        feat_extract_for_long_utterance(wav2vec2lang=test_txt, model=model, layer=feat_layer, save_dir=save_w2v_test_dir,
                     audio_list=audio_test, label_list=labels_test_index, name_list=name_list, device=device)

if __name__ == "__main__":
    main()



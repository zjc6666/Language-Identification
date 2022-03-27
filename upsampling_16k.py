
import os
import subprocess
import sys

def upsampling_lre(wav_scp, save_wav, save_16k):
    
    with open(wav_scp, 'r') as f:
        lines_wav = f.readlines()
    audio_path = [x.split()[-2].strip() for x in lines_wav]
 
    for audio in audio_path:
        # print(audio)
        if audio.endswith('.sph'):
            new_name = save_wav + '/' + os.path.split(audio)[-1].replace('.sph', '.wav')
            subprocess.call("sph2pipe -p " + audio + " " + new_name, shell=True)
            new_name_16k = new_name.replace(save_wav, save_16k)
            subprocess.call("sox " + new_name + " -r 16000 " + new_name_16k, shell=True)
            os.remove(new_name)
        elif audio.endswith('.flac'):
            new_name = save_16k + '/' + os.path.split(audio)[-1].replace('.flac', '.wav')
            subprocess.call("sox " + audio + " -r 16000" + new_name, shell=True)
    os.rmdir(save_wav)

wav_scp = sys.argv[1]
# save temp file, will be delete after run
save_wav = sys.argv[2]
save_16k = sys.argv[3]

upsampling_lre(wav_scp, save_wav, save_16k)

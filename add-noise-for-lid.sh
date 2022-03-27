#!/bin/bash

# updata by Zhang Jicheng


. path.sh
. cmd.sh

echo
echo "## LOG: $0 $@"
echo

# begin option
#cmd="slurm.pl --quiet"
# cmd="slurm.pl --quiet --exclude=node0[4-8]"
cmd='run.pl'
nj=10
steps=
# add noise
noise_dir=data/rats_noise_channel_BCDFG
# Data sets that need to add noise
src_train=
sampling_rate=16000
# rate of all aug data 0<rate<1
subset_noise_rate=1

# noise
noise_fg_interval=1
noise_bg_snrs=20 #15:10:5:0
# end option

. parse_options.sh || exit 1

steps=$(echo $steps | perl -e '$steps=<STDIN>;  $has_format = 0;
  if($steps =~ m:(\d+)\-$:g){$start = $1; $end = $start + 10; $has_format ++;}
        elsif($steps =~ m:(\d+)\-(\d+):g) { $start = $1; $end = $2; if($start == $end){}elsif($start < $end){ $end = $2 +1;}else{die;} $has_format ++; }
      if($has_format > 0){$steps=$start;  for($i=$start+1; $i < $end; $i++){$steps .=":$i"; }} print $steps;' 2>/dev/null)  || exit 1

if [ ! -z "$steps" ]; then
  for x in $(echo $steps|sed 's/[,:]/ /g'); do
    index=$(printf "%02d" $x);
    declare step$index=1
  done
fi

rats_data=/home3/andrew219/python_scripts/extract_rats_noise/rats_channels/

if [ ! -z $step01 ]; then 
  mkdir data/rats_channels_AEH_noise || exit 1;
  mkdir data/rats_channels_BCDFG_noise || exit 1;
  ls $rats_data/channel_{A,E,H}/*.wav > data/rats_channels_AEH_noise/rats_channels_AEH_noise_file_list.txt
  ls $rats_data/channel_{B,C,D,F,G}/*.wav > data/rats_channels_BCDFG_noise/rats_channels_BCDFG_noise_file_list.txt

  for x in rats_channels_AEH_noise rats_channels_BCDFG_noise;do
    cat $x/${x}_noise_file_list.txt | awk '{ split($0, arr, "/"); c=arr[7]; l=length(arr[8]); name=substr(arr[8], 0, l-4); print name " "name}' > $x/utt2spk
    cat $x/${x}_noise_file_list.txt | awk '{ split($0, arr, "/"); c=arr[7]; l=length(arr[8]); name=substr(arr[8], 0, l-4); print name " "$0}' > $x/wav.scp
    utils/fix_data_dir.sh cat $x/
  done
fi

if [ ! -f $src_train/reco2dur ]; then
    utils/data/get_reco2dur.sh --nj $nj  --cmd "$cmd" $src_train || exit 1;
fi
if [ ! -f $noise_dir/reco2dur ]; then
    utils/data/get_reco2dur.sh --nj $nj  --cmd "$cmd" $noise_dir || exit 1;
fi

if [ ! -z $step02 ]; then
  for noise_bg_snrs in 5 10 15 20;do
    # Augment with musan_noise
    steps/data/augment_data_dir.py --utt-prefix "noise" --modify-spk-id "true" \
      --fg-interval $noise_fg_interval --fg-snrs "$noise_bg_snrs" --fg-noise-dir "$noise_dir" \
      $src_train ${src_train}_${noise_bg_snrs}_snrs || exit 1
    echo "## LOG (step02): Make a noise version of the train '${src_train}_noise' done!"
  done
fi

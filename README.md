# VoxSim trainer

This repository contains the framework for training speaker similarity prediction models described in the paper '_VoxSim: A perceptual voice similarity dataset_'.

### Dependencies
```
pip install -r requirements.txt
```

### Data preparation

Please follow the 'Data preparation' part of the [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer) github repo to prepare VoxCeleb datasets.

### Training examples

- ECAPA-TDNN with voxsim raw scores:
```
python ./trainSpeakerNet.py --config ./configs/ECAPA_TDNN.yaml --train_list data/voxsim_train_list_raw.txt
```

- WavLM-ECAPA with voxsim mean scores:
```
python ./trainSpeakerNet.py --config ./configs/WavLM_ECAPA.yaml --train_list data/voxsim_train_list_mean.txt
```

- WavLM-ECAPA pre-trained on VoxCeleb with voxsim mean scores:
```
python ./trainSpeakerNet.py --config ./configs/WavLM_ECAPA_sv.yaml --train_list data/voxsim_train_list_mean.txt
```

You can pass individual arguments that are defined in trainSpeakerNet.py by `--{ARG_NAME} {VALUE}`.
Note that the configuration file overrides the arguments passed via command line.

### Pretrained models

A pretrained model, described in [1], can be downloaded from [here](https://drive.google.com/drive/folders/10YIeXdi1luhiwyUbKQFsm7nkH0h1lkkK?usp=drive_link).

You can check that the following script returns: `Pearson 0.83695 ...`.

```
python ./trainSpeakerNet.py --eval --model wavlm_large --save_path test/wavlm_ecapa --test_list data/voxsim_test_list.txt --eval_frames 400 --initial_model wavlm_ecapa.model
```


### Citation

Please cite [1] if you make use of the code.

[1] _VoxSim: A perceptual voice similarity dataset_
```
@inproceedings{ahn2024voxsim,
  title={VoxSim: A perceptual voice similarity dataset},
  author={Ahn, Junseok and Kim, Youkyum and Choi, Yeunju and Kwak, Doyeop and Kim, Ji-Hoon and Mun, Seongkyu and Chung, Joon Son},
  booktitle={Proc. Interspeech},
  year={2024}
}
```

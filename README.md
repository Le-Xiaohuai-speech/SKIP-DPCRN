# SKIP-DPCRN
Implementation of the DPCRN model with skipping strategy in the accepted manuscript for IEEE/ACM TASLP.

# Requirements
tensorflow>=1.14,
numpy,
matplotlib,
librosa,
sondfile.
# Run
```shell
python main.py --mode train --cuda 0 --experimentName experiment_1
```
```shell
python main.py --mode test --cuda 0 --ckpt PATH_OF_PRETRAINED_MODEL --test_dir PATH_OF_NOISY_AUDIO --output_dir PATH_OF_ENHANCED_AUDIO
```
# Reference
1. X. Le, H. Chen, K. Chen, and J. Lu, “DPCRN: Dual-Path Convolution Recurrent Network for Single Channel Speech Enhancement,” Proc. Interspeech 2021, pp. 2811–2815, 2021. (https://github.com/Le-Xiaohuai-speech/DPCRN_DNS3)

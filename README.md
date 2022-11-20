# PreQuEL: Quality Estimation of Machine Translation Outputs in Advance

<p align="center">
  <img src="PreQuEL.png" width=512px>
</p>

Paper link: https://arxiv.org/abs/2205.09178

---

## Install
```bash
git clone https://github.com/shachardon/PreQuEL.git
cd PreQuEL
pip install -e requirements.txt
```

## Quick Start - Train

To train a model, please run this command from the root directory of PreQuEL.

```bash
python run_model.py --model_type simple --epochs 3 --folds 3 --extra_finetune 3 --replace_comet --dir <output_dir> --lang de
```

- **--model_type** can be simple/combined/multiple

- **--extra_finetune <n>** combined with **--replace_comet**, train another n epochs on the DA data after training on the COMET augmented data. 

- **--lang** de for en-de, en for de-en, et for et-en, zh for en-zh. 

# torch-retriever-vc

PyTorch implementation of Retriever: Learning Content-Style Representation.

- Retriever: Learning Content-Style Representation as a Token-Level Bipartite Graph, Yin et al., 2022. [[arXiv:2202.12307](https://arxiv.org/abs/2202.12307)]

## Requirements

Tested in python 3.7.9 conda environment.

## Usage

Initialize the submodule and patch.

```bash
git submodule init --update
cd hifi-gan; patch -p0 < ../hifi-gan-diff
```

Download LibriTTS dataset from [openslr](https://openslr.org/60/)

Dump the preprocessed LibriTTS dataset.

```bash
python -m utils.libritts \
    --data-dir /datasets/LibriTTS/train-clean-360 \
    --out-dir /datasets/LibriTTS/train-clean-360-dump \
    --num-proc 8
```

To train model, run [train.py](./train.py)
```bash
python train.py \
    --data-dir /datasets/LibriTTS/train-clean-360-dump
```

To start to train from previous checkpoint, --load-epoch is available.
```bash
python train.py \
    --data-dir /datasets/LibriTTS/train-clean-360-dump \
    --load-epoch 20 \
    --config ./ckpt/t1.json
```

Checkpoint will be written on `TrainConfig.ckpt`, tensorboard summary on `TrainConfig.log`.

```bash
tensorboard --logdir ./log
```

To inference model, run [inference.py](./inference.py)

```bash
MODEL=t1
CKPT=./ckpt
CUDA_VISIBLE_DEVICES=7 python inference.py \
    --data-dir /datasets/LibriTTS/train-clean-360-dump \
    --ret-config $CKPT/$MODEL.json \
    --ret-ckpt $CKPT/$MODEL/"$MODEL"_49.ckpt \
    --hifi-config ./ckpt/hifigan/config.json \
    --hifi-ckpt ./ckpt/hifigan/g_02500000
```

Pretrained checkpoints will be relased on [releases](https://github.com/revsic/torch-retriever-vc/releases).

To use pretrained model, download files and unzip it. Followings are sample script.

```py
from config import Config
from retriever import Retriever

with open('t1.json') as f:
    config = Config.load(json.load(f))

ckpt = torch.load('t1_49.ckpt', map_location='cpu')

retriever = Retriever(config.model)
retriever.load(ckpt)
retriever.eval()
```

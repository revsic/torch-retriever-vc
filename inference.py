import argparse
import json

import librosa
import torch

from config import Config
from retriever import Retriever
from utils.hifigan import HiFiGANWrapper
from utils.libritts import DumpedLibriTTS


parser = argparse.ArgumentParser()
parser.add_argument('--ret-config')
parser.add_argument('--ret-ckpt')
parser.add_argument('--hifi-config')
parser.add_argument('--hifi-ckpt')
parser.add_argument('--data-dir')
args = parser.parse_args()


with open(args.ret_config) as f:
    config = Config.load(json.load(f))

ckpt = torch.load(args.ret_ckpt, map_location='cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

retriever = Retriever(config.model)
retriever.load(ckpt)
retriever.to(device)
retriever.eval()

hifigan = HiFiGANWrapper(args.hifi_config, args.hifi_ckpt, device)

libritts = DumpedLibriTTS(args.data_dir)
testset = libritts.split(config.train.split)

sid, text, mel, textlen, mellen = testset[:2]
# wrap
mel, mellen = torch.tensor(mel, device=device), torch.tensor(mellen, device=device)

with torch.no_grad():
    synth, aux = retriever(mel, mellen)
    # flip style
    style = aux['style'][[1, 0]]

    mixed, _ = retriever(mel, mellen, refstyle=style)
    # vocoding
    out = hifigan.forward(torch.cat([synth, mixed], dim=0))
    out = out.cpu().numpy()

for i, (wav, mlen) in enumerate(zip(out, mellen.repeat(2))):
    # unwrap
    librosa.output.write_wav(
        f'{i}.wav', wav[:mlen.item() * config.data.hop], sr=config.data.sr)

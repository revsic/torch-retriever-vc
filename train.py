import argparse
import json
import os

import git
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

import speechset
from config import Config
from retriever import Retriever
from utils.dataset import RealtimeWavDataset, WeightedRandomWrapper
from utils.hifigan import HiFiGANWrapper
from utils.wrapper import TrainingWrapper


class Trainer:
    """Retriever trainer.
    """
    LOG_IDX = 0
    LOG_AUDIO = 5
    LOG_MAXLEN = 3

    def __init__(self,
                 model: Retriever,
                 hifigan: HiFiGANWrapper,
                 dataset: RealtimeWavDataset,
                 testset: RealtimeWavDataset,
                 config: Config,
                 device: torch.device):
        """Initializer.
        Args:
            model: retriever model.
            hifigan: HiFi-GAN wrapper.
            dataset, testset: dataset.
            config: unified configurations.
            device: target computing device.
        """
        self.model = model
        self.dataset = dataset
        self.testset = testset
        self.config = config
        # for disabling auto-collation
        def identity(x): return x

        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.train.batch,
            shuffle=config.train.shuffle,
            collate_fn=identity,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory)

        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=config.train.batch,
            collate_fn=identity,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory)

        # training wrapper
        self.wrapper = TrainingWrapper(model, config, device)

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            config.train.learning_rate,
            (config.train.beta1, config.train.beta2))

        self.train_log = SummaryWriter(
            os.path.join(config.train.log, config.train.name, 'train'))
        self.test_log = SummaryWriter(
            os.path.join(config.train.log, config.train.name, 'test'))

        self.ckpt_path = os.path.join(
            config.train.ckpt, config.train.name, config.train.name)

        self.hifigan = hifigan
        # colormap
        self.cmap = np.array(plt.get_cmap('viridis').colors)

    def train(self, epoch: int = 0):
        """Train wavegrad.
        Args:
            epoch: starting step.
        """
        self.model.train()
        step = epoch * len(self.loader)
        # for restarting
        self.wrapper.lambda_cont = (step + 1) * self.config.train.cont_start
        for epoch in tqdm.trange(epoch, self.config.train.epoch):
            with tqdm.tqdm(total=len(self.loader), leave=False) as pbar:
                for it, bunch in enumerate(self.loader):
                    speeches, lengths = self.dataset.collate(bunch)
                    loss, losses, aux = self.wrapper.compute_loss(speeches, lengths)
                    # update
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    step += 1
                    pbar.update()
                    pbar.set_postfix({'loss': loss.item(), 'step': step})

                    self.wrapper.update_warmup()

                    for key, val in losses.items():
                        self.train_log.add_scalar(key, val, step)

                    with torch.no_grad():
                        grad_norm = np.mean([
                            torch.norm(p.grad).item()
                            for p in self.model.parameters() if p.grad is not None])
                        param_norm = np.mean([
                            torch.norm(p).item()
                            for p in self.model.parameters() if p.dtype == torch.float32])

                    self.train_log.add_scalar('common/grad-norm', grad_norm, step)
                    self.train_log.add_scalar('common/param-norm', param_norm, step)
                    self.train_log.add_scalar(
                        'common/learning-rate', self.optim.param_groups[0]['lr'], step)

                    if it % (len(self.loader) // 50) == 0:
                        idx = Trainer.LOG_IDX
                        with torch.no_grad():
                            # [seglen, mel]
                            mel = aux['rctor'][idx]
                            # [seglen x hop]
                            rctor = self.hifigan.forward(
                                torch.tensor(mel.T[None], device=self.wrapper.device)).squeeze(dim=0)
                        self.train_log.add_image(
                            # [3, M, T]
                            'train/gt-mel', self.mel_img(aux['mel'][idx]), step)
                        self.train_log.add_audio(
                            'train/gt-aud', aux['seg'][idx, None], step, sample_rate=self.config.model.sr)

                        self.train_log.add_image(
                            'train/synth-mel', self.mel_img(mel), step)
                        self.train_log.add_audio(
                            'train/synth-aud', rctor[None], step, sample_rate=self.config.model.sr)

            self.model.save(f'{self.ckpt_path}_{epoch}.ckpt', self.optim)

            cumul = {key: [] for key in losses}
            with torch.no_grad():
                for bunch in self.testloader:
                    speeches, lengths = self.testset.collate(bunch)
                    _, losses, _ = self.wrapper.compute_loss(speeches, lengths)
                    for key, val in losses.items():
                        cumul[key].append(val)
                # test log
                for key, val in cumul.items():
                    self.test_log.add_scalar(key, np.mean(val), step)
                # [B]
                bsize, = lengths.shape
                for i in range(Trainer.LOG_AUDIO):
                    idx = (Trainer.LOG_IDX + i) % bsize
                    # min-length
                    len_ = min(
                        lengths[idx].item(),
                        int(Trainer.LOG_MAXLEN * self.config.model.sr))
                    # [T], gt plot
                    speech = speeches[idx, :len_]
                    # [fft // 2 + 1, T / hop]
                    spec = torch.stft(
                        speech,
                        self.config.model.fft,
                        self.config.model.hop,
                        self.config.model.win,
                        self.wrapper.window,
                        return_complex=True).abs()
                    # [mel, T // hop]
                    mel = torch.matmul(self.wrapper.fbank, spec).clamp_min(1e-5).log()
                    self.test_log.add_image(
                        f'test{i}/gt-mel', self.mel_img(mel.T.cpu().numpy()), step)
                    self.test_log.add_audio(
                        f'test{i}/gt-aud', speech[None], step, sample_rate=self.config.model.sr)

                    # [1, T / hop, mel]
                    rctor, _ = self.model.forward(speech[None])
                    # [T]
                    signal = self.hifigan.forward(rctor.transpose(1, 2)).squeeze(dim=0)
                    self.test_log.add_image(
                        f'test{i}/synth-mel', self.mel_img(rctor.squeeze(dim=0).cpu().numpy()), step)
                    self.test_log.add_audio(
                        f'test{i}/synth-aud', signal[None], step, sample_rate=self.config.model.sr)
                self.model.train()

    def mel_img(self, mel: np.ndarray) -> np.ndarray:
        """Generate mel-spectrogram images.
        Args:
            audio: [np.float32; [T, mel]], audio signal.
        Returns:
            [np.float32; [3, mel, T]], mel-spectrogram in viridis color map.
        """
        # minmax norm in range(0, 1)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-7)
        # in range(0, 255)
        mel = (mel * 255).astype(np.uint8)
        # [T, M, 3]
        mel = self.cmap[mel]
        # [3, M, T], make origin lower
        mel = np.flip(mel, axis=1).transpose(2, 1, 0)
        return mel


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config', default=None)
    parser.add_argument('--load-epoch', default=None, type=int)
    parser.add_argument('--name', default=None)
    parser.add_argument('--auto-rename', default=False, action='store_true')
    parser.add_argument('--hifi-ckpt', default=None)
    parser.add_argument('--hifi-config', default=None)
    args = parser.parse_args()

    # seed setting
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # configurations
    config = Config()
    if args.config is not None:
        print('[*] load config: ' + args.config)
        with open(args.config) as f:
            config = Config.load(json.load(f))

    if args.name is not None:
        config.train.name = args.name

    log_path = os.path.join(config.train.log, config.train.name)
    # auto renaming
    if args.auto_rename and os.path.exists(log_path):
        config.train.name = next(
            f'{config.train.name}_{i}' for i in range(1024)
            if not os.path.exists(f'{log_path}_{i}'))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ckpt_path = os.path.join(config.train.ckpt, config.train.name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    # prepare datasets
    sr = config.model.sr
    dataset = RealtimeWavDataset(
        speechset.datasets.ConcatReader([
            speechset.datasets.VCTK('./datasets/VCTK-Corpus', sr),
            speechset.datasets.LibriTTS('./datasets/LibriTTS/train-clean-360', sr)]),
        device=device)
    dataset = WeightedRandomWrapper(dataset)

    testset = RealtimeWavDataset(
        speechset.datasets.LibriSpeech('./datasets/LibriSpeech/test-clean', sr),
        device=device)
    # shuffle
    rng, idxer = np.random.default_rng(0), testset.indexer
    testset.indexer = [idxer[i] for i in rng.permutation(len(idxer))]

    # model definition
    model = Retriever(config.model)
    model.to(device)

    # HiFi-GAN Wrapper
    hifigan = HiFiGANWrapper(args.hifi_config, args.hifi_ckpt, device)

    trainer = Trainer(model, hifigan, dataset, testset, config, device)

    # loading
    if args.load_epoch is not None:
        # find checkpoint
        ckpt_path = os.path.join(
            config.train.ckpt,
            config.train.name,
            f'{config.train.name}_{args.load_epoch}.ckpt')
        # load checkpoint
        ckpt = torch.load(ckpt_path)
        model.load_(ckpt, trainer.optim)
        print('[*] load checkpoint: ' + ckpt_path)
        # since epoch starts with 0
        args.load_epoch += 1

    # git configuration
    repo = git.Repo()
    config.train.hash = repo.head.object.hexsha
    with open(os.path.join(config.train.ckpt, config.train.name + '.json'), 'w') as f:
        json.dump(config.dump(), f)

    # start train
    trainer.train(args.load_epoch or 0)

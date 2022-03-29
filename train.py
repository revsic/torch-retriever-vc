import argparse
import json
import os

import git
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import Config
from retriever import Retriever
from utils.libritts import DumpedLibriTTS
from utils.wrapper import TrainingWrapper


class Trainer:
    """Retriever trainer.
    """
    LOG_IDX = 0

    def __init__(self,
                 model: Retriever,
                 dataset: DumpedLibriTTS,
                 config: Config,
                 device: torch.device):
        """Initializer.
        Args:
            model: retriever model.
            dataset: dataset.
            config: unified configurations.
            device: target computing device.
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        # train-test split
        self.testset = self.dataset.split(config.train.split)

        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.train.batch,
            shuffle=config.train.shuffle,
            collate_fn=self.dataset.collate,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory)

        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=config.train.batch,
            collate_fn=self.dataset.collate,
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

        self.cmap = np.array(plt.get_cmap('viridis').colors)

    def train(self, epoch: int = 0):
        """Train wavegrad.
        Args:
            epoch: starting step.
        """
        self.model.train()
        step = epoch * len(self.loader)
        for epoch in tqdm.trange(epoch, self.config.train.epoch):
            with tqdm.tqdm(total=len(self.loader), leave=False) as pbar:
                for it, bunch in enumerate(self.loader):
                    mel = torch.tensor(
                        self.wrapper.random_segment(bunch), device=self.device)
                    loss, losses, aux = self.wrapper.compute_loss(mel)
                    # update
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    step += 1
                    pbar.update()
                    pbar.set_postfix({'loss': loss.item(), 'step': step})

                    for key, val in losses.items():
                        self.train_log.add_scalar(f'loss/{key}', val, step)

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

                    if (it + 1) % (len(self.loader) // 50) == 0:
                        mel = mel[Trainer.LOG_IDX].cpu().numpy()
                        self.train_log.add_image(
                            # [3, M, T]
                            'train/gt', self.mel_img(mel), step)
                        self.train_log.add_image(
                            'train/synth', self.mel_img(aux['synth'][Trainer.LOG_IDX]), step)

            cumul = {key: [] for key in losses}
            with torch.inference_mode():
                for bunch in self.testloader:
                    mel = torch.tensor(
                        self.wrapper.random_segment(bunch), device=self.device)
                    _, losses, _ = self.wrapper.compute_loss(mel)
                    for key, val in losses.items():
                        cumul[key].append(val)
                # test log
                for key, val in cumul.items():
                    self.test_log.add_scalar(f'loss/{key}', np.mean(val), step)

                # wrap last bunch
                _, _, mel, _, mellen = bunch
                # [T, M], gt plot
                mel = mel[Trainer.LOG_IDX, :mellen[Trainer.LOG_IDX]]
                self.test_log.add_image('test/gt', self.mel_img(mel), step)

                # inference
                self.model.eval()
                # [1, T, M]
                synth, _ = self.model(torch.tensor(mel[None], device=device))
                self.model.train()
                # [T, M]
                synth = synth.squeeze(0).cpu().detach().numpy()
                self.test_log.add_image('test/synth', self.mel_img(synth), step)

            self.model.save(f'{self.ckpt_path}_{epoch}.ckpt', self.optim)

    def mel_img(self, mel: np.ndarray) -> np.ndarray:
        """Generate mel-spectrogram images.
        Args:
            mel: [float32; [T, M]], spectrogram.
        Returns:
            [float32; [3, M, T]], mel-spectrogram in viridis color map.
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
    parser.add_argument('--load-epoch', default=0, type=int)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--auto-rename', default=False, action='store_true')
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

    # prepare datasets
    libritts = DumpedLibriTTS(args.data_dir)

    # model definition
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Retriever(config.model)
    model.to(device)

    trainer = Trainer(model, libritts, config, device)

    # loading
    if args.load_epoch > 0:
        # find checkpoint
        ckpt_path = os.path.join(
            config.train.ckpt,
            config.train.name,
            f'{config.train.name}_{args.load_epoch}.ckpt')
        # load checkpoint
        ckpt = torch.load(ckpt_path)
        model.load(ckpt, trainer.optim)
        print('[*] load checkpoint: ' + ckpt_path)
        # since epoch starts with 0
        args.load_epoch += 1

    # git configuration
    repo = git.Repo()
    config.train.hash = repo.head.object.hexsha
    with open(os.path.join(config.train.ckpt, config.train.name + '.json'), 'w') as f:
        json.dump(config.dump(), f)

    # start train
    trainer.train(args.load_epoch)

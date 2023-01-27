from retriever.config import Config as ModelConfig


class TrainConfig:
    """Configuration for training loop.
    """
    def __init__(self, sr: int, hop: int):
        """Initializer.
        Args:
            sr: sample rate.
            hop: stft hop length.
        """
        # optimizer
        self.learning_rate = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.99

        # augment
        self.pitch_shift = 2.
        self.bins_per_octave = 12
        self.cutoff_lowpass = 60
        self.cutoff_highpass = 10000
        self.q_min = 2
        self.q_max = 5
        self.num_peak = 8
        self.g_min = -12
        self.g_max = 12

        # contents loss
        self.cont_start = 1e-5
        self.cont_end = 1
        self.num_adj = 10
        self.num_cand = 15
        self.kappa = 0.1

        # loader settings
        self.batch = 64
        self.shuffle = True
        self.num_workers = 4
        self.pin_memory = True

        # train iters
        self.epoch = 1000

        # segment length
        sec = 2
        self.seglen = sec * int(sr // hop) * hop

        # path config
        self.log = './log'
        self.ckpt = './ckpt'

        # model name
        self.name = 't1'

        # commit hash
        self.hash = 'unknown'


class Config:
    """Integrated configuration.
    """
    def __init__(self):
        self.model = ModelConfig()
        self.train = TrainConfig(self.model.sr, self.model.hop)

    def dump(self):
        """Dump configurations into serializable dictionary.
        """
        return {k: vars(v) for k, v in vars(self).items()}

    @staticmethod
    def load(dump_):
        """Load dumped configurations into new configuration.
        """
        conf = Config()
        for k, v in dump_.items():
            if hasattr(conf, k):
                obj = getattr(conf, k)
                load_state(obj, v)
        return conf


def load_state(obj, dump_):
    """Load dictionary items to attributes.
    """
    for k, v in dump_.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj

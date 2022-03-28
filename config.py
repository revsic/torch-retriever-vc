from retriever.config import Config as ModelConfig
from speechset.config import Config as DataConfig


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
        self.learning_rate = 4e-3
        self.beta1 = 0.9
        self.beta2 = 0.999

        # loss weights
        self.lambda_rec = 5
        self.lambda_vq = 0.3
        self.lambda_sc = 0.1

        # loader settings
        self.batch = 120
        self.shuffle = True
        self.num_workers = 4
        self.pin_memory = True

        # train iters
        self.epoch = 50

        # segment length
        self.seglen = int(sr // hop) * 4

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
        self.data = DataConfig(batch=None)
        self.train = TrainConfig(self.data.sr, self.data.hop)
        self.model = ModelConfig(self.data.mel)

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

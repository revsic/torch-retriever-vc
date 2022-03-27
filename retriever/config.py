class Config:
    """Retriever, module configurations.
    """
    def __init__(self, mel: int):
        """Initializer.
        Args:
            mel: size of the spectrogram.
        """
        self.mel = mel

        # prenet
        self.reduction = 2

        # model
        self.channels = 512
        self.heads = 8
        self.enc_blocks = 6
        self.ret_blocks = 6
        self.dec_blocks = 6
        self.dec_kernels = 31

        # style vectors
        self.styles = 128

        # quantizer
        self.groups = 2
        self.vectors = 100
        self.temp_max = 2
        self.temp_min = 0.5
        self.temp_factor = 0.9996

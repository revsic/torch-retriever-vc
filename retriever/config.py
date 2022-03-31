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

        # channels
        self.contexts = 512    # context channels
        self.styles = 192      # style channels
        self.prototypes = 60   # the number of the prototypes

        # encoder
        self.enc_ffn = 2048    # contexts x 4
        self.enc_heads = 8
        self.enc_blocks = 4
        self.enc_dropout = 0.1

        # style retriever
        self.ret_ffn = 512     # queries x 4 / 3
        self.ret_heads = 4
        self.ret_blocks = 3
        self.ret_dropout = 0.1

        # decoder
        self.dec_kernels = 31
        self.dec_ffn = 2048    # contexts x 4
        self.dec_heads = 8
        self.dec_blocks = 4
        self.dec_dropout = 0.1

        # detokenizer
        self.detok_ffn = 2048

        # quantizer
        self.groups = 2
        self.vectors = 100
        self.temp_max = 2
        self.temp_min = 0.01
        self.temp_factor = 0.9996

class Config:
    """Retriever, module configurations.
    """
    def __init__(self):
        self.sr = 22050
        self.w2v2_name = 'facebook/wav2vec2-large-xlsr-53'

        # channels
        self.contexts = 512    # context channels
        self.styles = 192      # style channels
        self.prototypes = 60   # the number of the prototypes

        # spec
        self.mel = 80
        self.hop = 256
        self.win = 1024
        self.fft = 1024
        self.fmin = 0
        self.fmax = 8000

        # linguistic
        self.ling_hiddens = 256
        self.ling_heads = 4
        self.ling_ffn = 512
        self.ling_blocks = 4
        self.ling_dropout = 0.5

        # style retriever
        self.ret_ffn = 512
        self.ret_heads = 4
        self.ret_blocks = 3
        self.ret_dropout = 0.1

        # decoder
        self.dec_heads = 4
        self.dec_ffn = 2048
        self.dec_blocks = 12
        self.dec_dropout = 0.1

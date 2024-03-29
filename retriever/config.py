class Config:
    """Retriever, module configurations.
    """
    # hardcoded
    ENCODEC_STRIDES = 320
    ENCODEC_DIM = 128

    def __init__(self):
        self.sr = 24000
        self.w2v2_name = 'facebook/wav2vec2-large-xlsr-53'

        # channels
        self.contexts = 512    # context channels
        self.styles = 192      # style channels
        self.prototypes = 60   # the number of the prototypes
        self.encodecs = 1024

        # time embeddings
        self.pe = 128
        self.timesteps = 8
        self.embeds = 512

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
        self.fst_heads = 4
        self.fst_ffn = 2048
        self.fst_blocks = 12
        self.fst_dropout = 0.5

        self.rst_heads = 4
        self.rst_ffn = 2048
        self.rst_blocks = 12
        self.rst_dropout = 0.5

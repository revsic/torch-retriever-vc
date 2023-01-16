class Config:
    """Retriever, module configurations.
    """
    def __init__(self):
        # channels
        self.contexts = 512    # context channels
        self.styles = 192      # style channels
        self.prototypes = 60   # the number of the prototypes

        # linguistic
        self.ling_kernels = 31
        self.ling_hiddens = 128
        self.ling_heads = 2
        self.ling_ffn = 512
        self.ling_blocks = 6
        self.ling_dropout = 0.1

        # style retriever
        self.ret_ffn = 512
        self.ret_heads = 4
        self.ret_blocks = 3
        self.ret_dropout = 0.1

        # decoder
        self.dec_ffn = 2048
        self.dec_heads = 8
        self.dec_blocks = 4
        self.dec_dropout = 0.1

        # detokenizer
        self.detok_ffn = 2048  # 4096

import argparse
from typing import Optional

import speechset


class DumpedLibriTTS(speechset.utils.DumpDataset):
    """Dumped libritts support.
    """
    def __init__(self, data_dir: str):
        """Initializer.
        Args:
            data_dir: path to the dumped dataset.
        """
        super().__init__(DumpedLibriTTS.IDWrappedAcoustic, data_dir)

    class IDWrappedAcoustic(speechset.utils.IDWrapper):
        """ID-wrapper for DumpDataset support.
        """
        def __init__(self, *args, **kwargs):
            """Pass the acoustic dataset to the IDWrapper
            """
            super().__init__(speechset.AcousticDataset(*args, **kwargs))


def dump(data_dir: str,
         out_dir: str,
         num_proc: int,
         config: Optional[speechset.Config] = None) -> int:
    """Dump preprocessed LibriTTS datasets.
    Args:
        data_dir: dataset directory.
        out_dir: path to the dumped dataset.
        num_proc: the number of the processor.
        config: dataset configuration, if provided.
    Returns:
        dataset lengths.
    """
    config = config or speechset.Config()
    libri = speechset.datasets.LibriTTS(data_dir)
    # construct multi speaker
    acoustic = speechset.utils.IDWrapper(
        speechset.AcousticDataset(libri, config))
    # dump
    return speechset.utils.mp_dump(acoustic, out_dir, num_proc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--num-proc', defulat=4, type=int)
    args = parser.parse_args()

    dump(args.data_dir, args.out_dir, args.num_proc)

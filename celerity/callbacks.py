from typing import *

from devtools import debug

class LossLogger(object):
    """Callback to write out into tensorboard

    Parameters
    ----------
    tb_writer : tensorboard.Writer
        The tensorboard writer.
    data_set: str, either 'train' or 'valid'.
        If it is the training/validation set.
    """
    def __init__(self, tb_writer, data_set) -> None:
        super().__init__()
        self.writer = tb_writer
        self.data_set = data_set

    def __call__(self, step: int, dict: Dict) -> None:
        """Update the tensorboard with the recorded scores from step

        Parameters
        ----------
        step : int
            The training step number.
        dict : Dict
            The dictionary containing the information to write out.
        """
        for scoring_method, items in dict[self.data_set].items():
            if step in items.keys():
                self.writer.add_scalars(scoring_method, {self.data_set: items[step]}, step)



from typing import Callable, List, Any, Union, Tuple, Dict
from abc import ABC, abstractclassmethod, abstractmethod
import collections

from torch.utils.data import Dataset
import torch
import mdtraj
import numpy as np
import glob
import h5py
from addict import Dict as Adict
import pydash as pyd
from devtools import debug



def loader(path: str, idx: int = None):
    if idx is None:
        return mdtraj.load(path)
    else:
        return mdtraj.load_frame(path, index=idx)


class MappableDatasetMixin(ABC):
    DEFAULT = Adict()

    @abstractmethod
    def __init__(self, options=None):
        print('in abstract init')
        if options is None:
            options = {}
        self.options = self.get_options(options)
        pass

    @classmethod
    def get_options(cls, options=None):
        if options is None:
            options = {}
        combined_options = Adict(cls.get_default_options())
        combined_options.update(Adict(options))
        # combined_options.version = __version__
        combined_options.dataset = cls.__name__
        return combined_options

    @classmethod
    def get_default_options(cls):
        return Adict(cls.DEFAULT)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self):
        pass


<<<<<<< HEAD
# class StreamingTrajectory():
#     def __init__(self, options):
#
#         self.trajectory_paths = options['trajectories']
#
#     def get_n_frame(self, i_traj: int) -> int:
#         with h5py.File(self.trajectory_paths[i_traj], 'r') as f:
#             return f['coordinates'].shape[0]
#
#     def get_frame_traj(self, i_frame_traj: Tuple[int, int]) -> mdtraj.Trajectory:
#         return mdtraj.load_frame(self.trajectory_paths[i_frame_traj[1]], index=i_frame_traj[0])
#
#     def __len__(self):
#         pass
#
#     @abstractmethod
#     def __getitem__(self):
#         pass


=======
>>>>>>> f21080441e0b7598d17f09616eb7e2e1e229e862
class StreamingTrajectory():
    def __init__(self, options):
        
        self.trajectory_paths = options['trajectories']

    def get_n_frame(self, i_traj: int) -> int:
        with h5py.File(self.trajectory_paths[i_traj], 'r') as f: 
            return f['coordinates'].shape[0]
    
    def get_frame_traj(self, i_frame_traj: Tuple[int, int]) -> mdtraj.Trajectory: 
        return mdtraj.load_frame(self.trajectory_paths[i_frame_traj[1]], index=i_frame_traj[0])



class TrajectoryDataset(Dataset, MappableDatasetMixin):

    DEFAULT = Adict(
        traj_paths_pattern='*.h5',
        stride=1,
        in_memory=False,
    )

    def __init__(self, options: Dict[Any, Any]) -> None:
        # The torch.Dataset class does not have an 'init' so don't call super() here or you'll end up calling
        # the mixin init which is useless.
        self.options = self.get_options(options)
        self.traj_paths = self.get_trajectory_paths()
        self.in_memory = self.options.in_memory
        self.stride = self.options.stride
        # Data about dataset
        self.trajs: List[np.ndarray] = None
        self.trajectory_stream:  None
        self.cum_available_frames: np.ndarray = None

        # NOTE striding is done on loading. But this means fetching a frame is different for in memory/from disk.
        # see __getitem__
        if self.in_memory:
            trajs = [mdtraj.load(traj_path) for traj_path in self.traj_paths]
            self.trajs = trajs
        else:
            self.trajectory_stream = StreamingTrajectory(dict(trajectories=self.traj_paths, atom_mask='protein'))

        self.calculate_available_frames()

        # Update options dictionary
        self.options.cum_available_frames = self.cum_available_frames
        self.options.traj_paths = self.traj_paths

    def calculate_available_frames(self):
        available_frames = []
        for i in range(len(self.traj_paths)):
            if self.in_memory:
                n_frames = self.trajs[i].n_frames
                available_frames.append(n_frames)
            else:
                n_frames = self.trajectory_stream.get_n_frame(i)
                available_frames.append(int(n_frames//self.stride))
        self.cum_available_frames = np.cumsum(available_frames)


    def get_trajectory_paths(self) -> List[str]:
        traj_paths = sorted(glob.glob(self.options.traj_paths_pattern))
        if len(traj_paths) == 0:
            raise ValueError(f"No trajectories found using {self.options.traj_paths_pattern}")
        return traj_paths

    def __len__(self):
        return int(self.cum_available_frames[-1])

    def __getitem__(self, idx):
        traj_ix = int(np.where(idx >= self.cum_available_frames)[0].shape[0])
        if traj_ix > 0:
            frame_ix = idx - self.cum_available_frames[traj_ix - 1]
        else:
            frame_ix = idx

        if self.in_memory:
            # already strided
            t_ix = int(frame_ix)
            pos_t = self.trajs[traj_ix][t_ix]
        else:
            t_ix = int(frame_ix*self.stride)
            pos_t = self.trajectory_stream.get_frame_traj((t_ix, traj_ix))

        return pos_t



class TimeLaggedDataset(Dataset, MappableDatasetMixin):

    DEFAULT = Adict(
        traj_paths_pattern='*.h5',
        lag_time=1,
        stride=1,
        in_memory=False,
        transform=None,
    )

    def __init__(self, options: Dict[Any, Any]) -> None:
        # The torch.Dataset class does not have an 'init' so don't call super() here or you'll end up calling
        # the mixin init which is useless.
        self.options = self.get_options(options)
        self.traj_paths = self.get_trajectory_paths()
        self.lag_time = self.options.lag_time
        self.in_memory = self.options.in_memory
        self.stride = self.options.stride
        # Data about dataset
        self.trajs: List[np.ndarray] = None
        self.trajectory_stream:  None
        self.cum_available_frames: np.ndarray = None

        # NOTE striding is done on loading. But this means fetching a frame is different for in memory/from disk.
        # see __getitem__
        if self.in_memory:
            trajs = [mdtraj.load(traj_path) for traj_path in self.traj_paths]
            self.trajs = trajs
        else:
            self.trajectory_stream = StreamingTrajectory(dict(trajectories=self.traj_paths, atom_mask='protein'))

        # self.calculate_output_dimension()
        self.calculate_available_frames()

        # Update options dictionary
        self.options.cum_available_frames = self.cum_available_frames
        self.options.traj_paths = self.traj_paths

    def calculate_available_frames(self):
        available_frames = []
        for i in range(len(self.traj_paths)):
            if self.in_memory:
                n_frames = self.trajs[i].n_frames
                available_frames.append(n_frames - self.lag_time)
            else:
                n_frames = self.trajectory_stream.get_n_frame(i)
                available_frames.append(int(n_frames//self.stride) - self.lag_time)
        self.cum_available_frames = np.cumsum(available_frames)


    def get_trajectory_paths(self) -> List[str]:
        traj_paths = sorted(glob.glob(self.options.traj_paths_pattern))
        if len(traj_paths) == 0:
            raise ValueError(f"No trajectories found using {self.options.traj_paths_pattern}")
        return traj_paths

    def __len__(self):
        return int(self.cum_available_frames[-1])

    def __getitem__(self, idx):
        traj_ix = int(np.where(idx >= self.cum_available_frames)[0].shape[0])
        if traj_ix > 0:
            frame_ix = idx - self.cum_available_frames[traj_ix - 1]
        else:
            frame_ix = idx

        if self.in_memory:
            # already strided
            t_ix = int(frame_ix)
            tau_ix = int(frame_ix + self.lag_time)
            pos_t = self.trajs[traj_ix][t_ix]
            pos_tau = self.trajs[traj_ix][tau_ix]
        else:
            t_ix = int(frame_ix*self.stride)
            tau_ix = int((frame_ix + self.lag_time)*self.stride)
            pos_t = self.trajectory_stream.get_frame_traj((t_ix, traj_ix))
            pos_tau = self.trajectory_stream.get_frame_traj((tau_ix, traj_ix))

        return pos_t, pos_tau


def isnumpy(elem: List[Any]) -> bool:
    return isinstance(elem, collections.abc.Iterable) and type(elem).__module__ == 'numpy'


def istensor(elem: List[Any]) -> bool:
    return isinstance(elem, collections.abc.Iterable) and type(elem).__module__ == 'torch'


def collate_mdtraj(batch: List[Any], md_transform: Callable[[mdtraj.Trajectory], np.ndarray], output: str = 'numpy') -> Union[Tuple[np.ndarray], np.ndarray]:
    if output == 'tensor':
        def transform(x):
            return torch.Tensor(md_transform(x))
    elif output == 'numpy': 
        transform = md_transform

    # Batch will have length 'batch_size'
    elem = batch[0]
    # time lagged datasets have tuples as elements of the batch
    if isinstance(elem, tuple):

        if isinstance(elem[0], mdtraj.Trajectory):
            return tuple([transform(mdtraj.join([x[i] for x in batch])) for i in range(len(elem))])
        else:
            raise ValueError(f"Unknown tuple element type {type(elem[0])}")

    # Pure trajectory
    elif isinstance(elem, mdtraj.Trajectory):
        return transform(mdtraj.join(batch))
    else:
        raise ValueError(f'Unknown batch element type: {type(elem)}')


class LoaderMixin(ABC):
    DEFAULT = Adict()

    @abstractmethod
    def __init__(self, options=None):
        if options is None:
            options = {}
        self.options = self.get_options(options)
        pass

    @classmethod
    def get_options(cls, options=None):
        if options is None:
            options = {}
        combined_options = Adict(cls.get_default_options())
        combined_options.update(Adict(options))
        combined_options.version = ''
        combined_options.feature = cls.__name__
        return combined_options

    @classmethod
    def get_default_options(cls):
        return Adict(cls.DEFAULT)


class DataLoader(torch.utils.data.DataLoader, LoaderMixin):
    """
    Thin wrapper around the torch implementation.
    Used to make some of the options more user-friendly and
    define custom samplers.
    """
    DEFAULT = Adict(
        dataset=None,
        batch_size=1,
        shuffle=False,
        output='numpy',
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        transform = None
    )

    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = self.get_options(options)
        # parse output options

        def f(batch): 
            return collate_mdtraj(batch, self.options.transform, self.options.output)

        self.options.collate_fn = f 

        # Setup args, kwargs for pytorch
        kwargs = pyd.clone(self.options)
        for opt in ['output', 'version', 'feature', 'transform']:
            _ = kwargs.pop(opt)
        dataset = kwargs.pop('dataset')

        super().__init__(dataset, **kwargs)


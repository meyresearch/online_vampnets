#!/usr/bin/env python,

from typing import List, Dict, Union
from warnings import warn
from copy import deepcopy
from pathlib import Path
from devtools import debug

import numpy as np
import click
from addict import Dict as Adict
import mdtraj.geometry as mdg
import mdtraj as md
from itertools import combinations


from abc import ABC, abstractmethod


class FeaturizerMixin(ABC):

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
        # combined_options.version = __version__
        combined_options.feature = cls.__name__
        return combined_options

    @classmethod
    def get_default_options(cls) -> Dict:
        return Adict(cls.DEFAULT)

    @abstractmethod
    def __call__(self, trajectory: md.Trajectory) -> np.ndarray:
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def get_features_labels(self):
        pass


class Dihedrals(FeaturizerMixin):
    DEFAULT = Adict(
        topology_path='topology.h5',
        which=['phi', 'psi'],
        dihedral_ixs=None,
        residue_ixs=None,
        cossin=True,
        periodic=True,
        opt=True
    )

    def __init__(self, options):
        self.options = self.get_options(options)
        self.topology = md.load(self.options.topology_path).top
        self.dihedral_ixs = self.get_indices()
        self.n_features = self.dihedral_ixs.shape[0] * 2 if self.options.cossin else self.dihedral_ixs.shape[0]
        self.features, self.labels = self.get_features_labels()

    def get_features_labels(self):
        if self.options.cossin:
            features = list(np.repeat(['cos-dihedral', 'sin-dihedral'], self.n_features/2))
            labels = np.tile(self.dihedral_ixs, (2, 1))
        else:
            features = list(np.repeat(['dihedral'], self.n_features))
            labels = self.dihedral_ixs
        labels = ["{}-{}-{}-{}".format(*label) for label in labels]
        return features, labels

    def __repr__(self):
        str = f"{self.options}"
        return str

    def get_indices(self) -> np.ndarray:
        if self.options.which is not None:
            if self.options.dihedral_ixs is not None:
                warn("ignoring `dihedral_ixs` and using `which` to specify dihedrals")
            indices = []
            for key in self.options.which:
                fn = getattr(md.geometry, "indices_" + key)
                indices.append(fn(self.topology))
            indices = np.vstack(indices)
            if self.options.selection:  # filter the residues
                # get the residue indexes of indices
                sele_indices = []
                for ind_dih in indices:
                    is_in_selection = np.any(
                        [self.topology.atom(x).residue.index in self.options.selection for x in ind_dih])
                    if is_in_selection:
                        sele_indices.append(ind_dih.reshape(1, -1))
                sele_indices = np.concatenate(sele_indices, axis=0)
            else:
                sele_indices = indices
        elif self.options.dihedral_ixs is not None:
            sele_indices = self.options.dihedral_ixs
        else:
            raise ValueError('Must specifiy either "dihedral_ixs" or "which" parameter')
        return sele_indices

    def __call__(self, frame: md.Trajectory):
        vals = md.compute_dihedrals(frame, indices=self.dihedral_ixs, periodic=self.options.periodic,
                                    opt=self.options.opt)
        if self.options.cossin:
            vals = np.concatenate([np.cos(vals), np.sin(vals)], axis=1)
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        return vals


class Contacts(FeaturizerMixin):
    DEFAULT = Adict(
        residues_ix=None,
        contacts_ix=None,
        periodic=True,
        scheme=None,
        topology_path='topology.h5', 
        offset = 2 
    )

    def __init__(self, options):
        self.options = self.get_options(options)
        self.options.contacts_ix = self.get_contacts_ix()
        self.features, self.labels = self.get_features_labels()

    def get_features_labels(self):
        features = ['contacts']*len(self.options.contacts_ix)
        labels = [f"{x}-{y}" for x, y in zip(self.options.contacts_ix[:, 0], self.options.contacts_ix[:, 1])]
        return features, labels

    def __repr__(self): 
        return f"{self.options}"


    def get_contacts_ix(self) -> Union[np.ndarray, str]:
        residues_ix = self.options.residues_ix
        if residues_ix is None:
            top = md.load(self.options.topology_path).top
            residues_ix = [x.index for x in top.residues]

        contacts_ix = np.array([x for x in combinations(residues_ix, 2) if x[1] > x[0] + self.options.offset])

        return contacts_ix

    def __call__(self, frame: md.Trajectory):
        contacts = self.options.contacts_ix
        scheme = self.options.scheme
        vals, contacts_ix = md.compute_contacts(frame, contacts=contacts, scheme=scheme)
        return vals


# class Positions(object):
#     def __init__(self, reference_path: str = None, which_pattern: str = None, which_ix: np.ndarray = None,
#                  flatten_coordinates: bool = True):
#         self.reference = mdtraj.load(reference_path) if reference_path is not None else None
#         self.flatten_coordinates = flatten_coordinates
#         if which_pattern is not None:
#             if self.reference is None:
#                 raise ValueError("need reference structure with a which_pattern")
#             self.which_ix = self.reference.top.select(which_pattern)
#         elif which_ix is not None:
#             self.which_ix = which_ix
#         else:
#             self.which_ix = None
#
#     def __call__(self, frame: mdtraj.Trajectory):
#         if self.reference is not None:
#             frame.superpose(reference=self.reference)
#
#         if self.which_ix is not None:
#             xyz = frame.atom_slice(self.which_ix).xyz
#         else:
#             xyz = frame.xyz
#
#         if self.flatten_coordinates:
#             xyz = xyz.reshape(xyz.shape[0], xyz.shape[1] * xyz.shape[2])
#
#         return xyz
#     #
    # @classmethod
    # @abstractmethod
    # def get_inputs(cls, options) -> [str]:
    #     return []
    #
    # @classmethod
    # @abstractmethod
    # def get_outputs(cls, options) -> [str]:
    #     return []
    #
    # @abstractmethod
    # def run(self):
    #     pas

#
# METHODS = ["dihedrals", "contacts", "positions"]
#
# from foam.utils import (
#     load_options,
#     load_trajectories,
#     configure_calculation,
#     data_path,
#     data_hook,
# )
# from foam.models import FeatureTrajectory
#
#
# def configure(**kwargs):
#     kwargs["command"] = "featurizer"
#     return configure_calculation(**kwargs)
#
#
# def dihedrals(
#     trajectory_path: str,
#     which: List[str] = None,
#     selection: List[int] = [],
#     indices: np.ndarray = None,
#     **kwargs
# ) -> FeatureTrajectory:
#     trajectory = md.load(trajectory_path)
#     topology = trajectory.top
#     if which is not None:
#         if indices is not None:
#             warn("ignoring `indices` and using `which` to specify dihedrals")
#         indices = []
#         for key in which:
#             fn = getattr(mdg, "indices_" + key)
#             indices.append(fn(topology))
#         indices = np.vstack(indices)
#         if selection:  # filter the residues
#             # get the residue indexes of indices
#             sele_indices = []
#
#             for ind_dih in indices:
#                 is_in_selection = np.any([topology.atom(x).residue.index in selection for x in ind_dih])
#                 if is_in_selection:
#                     sele_indices.append(ind_dih.reshape(1, -1))
#             sele_indices = np.concatenate(sele_indices, axis=0)
#     else:
#         sele_indices = indices
#
#     vals = md.compute_dihedrals(trajectory, indices=sele_indices, periodic=True, opt=True)
#     vals = np.concatenate([np.cos(vals), np.sin(vals)], axis=1)
#     features = list(np.repeat(['cos-dihedrals', 'sin-dihedrals'], vals.shape[1]/2))
#     labels = np.tile(sele_indices, (2, 1))
#     labels = ["{}-{}-{}-{}".format(*label) for label in labels]
#     assert vals.shape[1] == len(features)
#     assert vals.shape[1] == len(labels)
#     ftraj = FeatureTrajectory(data=vals, trajectory_path=trajectory_path, labels=labels, features=features)
#     return ftraj
#
#
# def contacts(
#     trajectory_path: str, scheme: str, selection: List[int] = [], func: str = "",
# **kwargs) -> FeatureTrajectory:
#     traj = md.load(trajectory_path)
#     if selection:
#         contacts_ix = np.array(list(combinations(selection, 2)))
#     else:
#         contacts_ix = "all"
#     vals, contacts_ix = md.compute_contacts(traj, contacts=contacts_ix, scheme=scheme)
#     if func == "exp_inv":
#         vals = np.exp(-vals)
#     elif func == "logistic":
#         x0, k = 4, 10
#         vals = 1 / (1 + np.exp(-(x0 - k * vals)))
#     elif func == "inv":
#         vals = 1 / vals
#
#     features = [f'{func}-contacts']*vals.shape[1]
#     labels = ["{}-{}".format(*c) for c in contacts_ix]
#     assert vals.shape[1] == len(features)
#     assert vals.shape[1] == len(labels)
#     ftraj = FeatureTrajectory(data=vals, trajectory_path=trajectory_path, labels=labels, features=features)
#
#     return ftraj
#
#
# def positions(trajectory_path: str, align_path: str, which: str, **kwargs) -> FeatureTrajectory:
#     traj = md.load(trajectory_path)
#     ref = md.load(align_path)
#     ref_ix = ref.top.select(which)
#     tar_ix = traj.top.select(which)
#     traj = traj.superpose(ref, frame=0, atom_indices=tar_ix, ref_atom_indices=ref_ix)
#     positions = traj.xyz
#     positions = positions[:, tar_ix, :]
#     vals = positions.reshape(positions.shape[0], -1)
#
#     features = ['positions']*vals.shape[1]
#     labels = [f"{ix}-{dim}" for ix in tar_ix for dim in ['x', 'y', 'z']]
#
#     assert vals.shape[1] == len(features)
#     assert vals.shape[1] == len(labels)
#
#     ftraj = FeatureTrajectory(data=vals, trajectory_path=trajectory_path,labels=labels, features=features)
#     return ftraj
#
#
# def featurize_trajectory(
#     trajectory_path: str, feature_options: Adict
# ) -> FeatureTrajectory:
#     ftrajs = []
#     # Get features and kwargs
#     kwargs_list = feature_options.kwargs.features
#     if not isinstance(kwargs_list, list):
#         kwargs_list = [kwargs_list]
#     features = [x.feature for x in kwargs_list]
#
#     # Order the features
#     feature_order = np.argsort(features)
#     features = [features[i] for i in feature_order]
#     kwargs_list = [kwargs_list[i] for i in feature_order]
#
#     # Do featurization
#     for feature, kwargs in zip(features, kwargs_list):
#         if feature == "contacts":
#             ftrajs.append(contacts(trajectory_path, **kwargs))
#         elif feature == "dihedrals":
#             ftrajs.append(dihedrals(trajectory_path, **kwargs))
#         elif feature == "positions":
#             ftrajs.append(positions(trajectory_path, **kwargs))
#         else:
#             raise NotImplementedError(f"{feature} not implemented")
#     if len(ftrajs) == 1:
#         ret = ftrajs[0]
#     else:  # combine the ftrajs to a single
#         data = []
#         labels = []
#         features = []
#         for ftraj in ftrajs:
#             data.append(ftraj.data)
#             labels.append(ftraj.labels)
#             features.append(ftraj.features)
#
#         ret = FeatureTrajectory(data=np.concatenate(data, axis=1),
#                                 trajectory_path=trajectory_path,
#                                 labels=np.concatenate(labels, axis=0),
#                                 features=np.concatenate(features, axis=0))
#     return ret
#
#
# def run(options: Adict) -> None:
#     for input_path, output_path in zip(
#         options.trajectory_paths, options.output.trajectory_paths
#     ):
#         print(f"Featurizing {Path(input_path).name}")
#         ftraj = featurize_trajectory(input_path, options)
#         ftraj.write(output_path)
#
#
# @click.command()
# @click.argument("options")
# def main(options):
#     options = load_options(options)
#     run(options)
#
#
# if __name__ == "__main__":
#     main()

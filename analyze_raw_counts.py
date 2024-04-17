# This code is associated to the paper `Scaling quantum computing with dynamic circuits`
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set

import networkx as nx
import numpy as np

from qiskit.quantum_info import Pauli, PauliList

from cgs_result import CGSResult
from cut_graph import CutGraph
from trex_manager import TREXManager


def merge_paulis(pauli_list: PauliList) -> Pauli:
    """Take a list of commuting Paulis and merge them into one."""
    pauli_exclusion = {Pauli("X"): {"Y", "Z"}, Pauli("Y"): {"X", "Z"}, Pauli("Z"): {"X", "Y"}}

    merged_paulis = ["I"] * len(pauli_list[0])
    for paulis in pauli_list:
        for idx, pauli in enumerate(paulis):
            if pauli == Pauli("I"):
                continue

            if merged_paulis[idx] not in pauli_exclusion[pauli]:
                merged_paulis[idx] = pauli.to_label()
            else:
                raise ValueError("Cannot merge Paulis")

    return Pauli(Pauli("".join(merged_paulis[::-1])))


def expectation_value(counts: Dict[str, int], mask: Set[int], normalize: bool = True):
    """Compute the expectation value of the given mask (i.e. observable)."""
    exp_val = 0.0
    for bit_str, count in counts.items():
        sign = 1
        for idx in mask:
            if bit_str[idx] == "1":
                sign *= -1

        exp_val += sign * count

    if normalize:
        exp_val /= sum(counts.values())

    return exp_val


class BaseAnalysis:
    """A base class to analyze the raw data from cut graph states."""

    def __init__(self, raw_result: CGSResult, trex_manager: TREXManager, stabilizer_groups: List):
        """Setup variables that all subclasses will use."""

        self._raw_result = raw_result  # Must be marginalized and TREX merged.
        self.trex_manager = trex_manager

        self.n_resample = 10
        self.mit_sample_size = self.trex_manager.mit_shots // self.n_resample

        self._stabilizer_to_basis = {}
        for group in stabilizer_groups:
            basis = merge_paulis(group)
            self._stabilizer_to_basis.update({stab: basis.to_label() for stab in group})

    @property
    def raw_result(self) -> CGSResult:
        """The raw result object after merging jobs."""
        return self._raw_result

    def trex_denominator(self, mask: Set[int]) -> float:
        """Compute the TREX denominator."""
        mit_resample = defaultdict(int)
        for idx in np.random.randint(0, self.trex_manager.mit_shots, self.mit_sample_size):
            mit_resample[self.trex_manager.mitigator_memory[idx]] += 1

        return expectation_value(mit_resample, mask)


class LOCCAnalysis(BaseAnalysis):
    """Compute expectation values from LOCC measured counts."""

    KEY_FIELDS = ["zne_c", "basis", "gate_qpd_full_enum_idx", "flipped_cbits"]

    def __init__(self, raw_result, trex_manager, stabilizer_groups):
        """Set extra parameters for the LOCC analysis."""
        super().__init__(raw_result, trex_manager, stabilizer_groups)
        self._num_qpd_circuits = 27
        self.gamma = 7

    def mean_std(self, observable: Pauli, filters: Optional[Dict] = None) -> Tuple:
        """Compute the mean and std for the observable with resampling."""
        # 1. Create a list of keys to enable O(1) look-up in our data.
        filters = filters or {}

        label = observable.to_label()

        keys, signs, weights = [], [], []
        for qpd_idx in range(self._num_qpd_circuits):
            zne_c = filters.get("zne_c", 1)

            basis = self._stabilizer_to_basis[observable]

            keys.append(f"zne_c{zne_c}_basis{basis}_gate_qpd_full_enum_idx{qpd_idx}")
            signs.append(self.raw_result.counts_metadata(keys[-1])["gate_qpd_sign"])
            weights.append(self.raw_result.counts_metadata(keys[-1])["gate_qpd_weight"])

        mask = {idx for idx, char in enumerate(observable.to_label()) if char != "I"}

        # 2. Compute the expectation value of the observable by putting the QPD into a dict.
        n_resample, exp_values = 10, []

        for _ in range(n_resample):
            all_counts = defaultdict(float)
            for kidx, key in enumerate(keys):
                mem = self.raw_result.memory(key)
                sample_size = len(mem) // n_resample

                for idx in np.random.randint(0, len(mem), sample_size):
                    all_counts[mem[idx]] += signs[kidx] * weights[kidx]  # a single count

            exp_val = expectation_value(dict(all_counts), mask)
            mit_val = self.trex_denominator(mask)

            exp_values.append(exp_val / mit_val)

        return np.average(exp_values), np.std(exp_values)


class LOAnalysis(BaseAnalysis):
    """Compute expectation values from LO measured counts."""

    KEY_FIELDS = ["basis", "gate_qpd_full_enum_idx", "flipped_cbits"]

    def __init__(self, raw_result, trex_manager, stabilizer_groups, cgs: CutGraph):
        """Set extra parameters for the LO analysis."""
        super().__init__(raw_result, trex_manager, stabilizer_groups)
        self._num_qpd_circuits = 36
        self.gamma = 9
        self.cgs = cgs

    def combine_counts(self, qpd_counts: List, qpd_signs: List[int], light_cones: Set) -> dict:
        """Combine counts taking the QPD into account and the LO MCM."""
        final_counts = dict()

        weights = np.ones(len(qpd_counts)) / len(qpd_counts)

        shots = sum(qpd_counts[0].values())

        for idx, counts in enumerate(qpd_counts):
            sign = qpd_signs[idx]

            for bin_str, val in counts.items():
                sign_ = sign
                qpd_coeff = val / shots * self.gamma * weights[idx]
                bin_str_split = bin_str.split(" ")

                # update the sign with the mid-circuit measurement: -1 if the measurement was 1
                for creg in light_cones:
                    for meas_bit in bin_str_split[-(creg+1)]:
                        sign_ *= (-1) ** int(meas_bit)
                final_bin_str = bin_str_split[0]
                final_counts[final_bin_str] = final_counts.get(final_bin_str, 0) + sign_ * qpd_coeff

        return final_counts

    def resampled_counts(self, key: str):
        """Computes a list of resampled dictionaries of counts."""
        mem = self.raw_result.memory(key)

        sample_size = len(mem) // self.n_resample
        resampled_dicts = []
        for _ in range(self.n_resample):
            resample = defaultdict(int)
            for idx in np.random.randint(0, len(mem), sample_size):
                resample[mem[idx]] += 1

            resampled_dicts.append(dict(Counter(resample)))

        return resampled_dicts

    def make_keys(self, observable: Pauli) -> list:
        """Make the keys to retrieve the QPD entries."""
        keys = []
        for qpd_idx in range(self._num_qpd_circuits):
            basis = self._stabilizer_to_basis[observable]
            keys.append(f"basis{basis}_gate_qpd_full_enum_idx{qpd_idx}")

        return keys

    def mean_std(self, observable: Pauli) -> Tuple:
        """Compute the mean and std for the observable with resampling."""
        # 1. Calculate the light-cones to which the stabilizer belongs
        idx_x = []
        for idx, c in enumerate(observable.to_label()[::-1]):
            if c == 'X' or c == 'Y':
                idx_x.append(idx)

        light_cones = set()
        for idx_l, lightcone in enumerate(self.cgs.cut_edges):
            for cut_edge in lightcone:
                for node in cut_edge:
                    for idx in idx_x:
                        if len(nx.shortest_path(self.cgs.graph, idx, node)) < 2:
                            light_cones.add(idx_l)

        # 2. Compute the expectation value of the observable
        keys = self.make_keys(observable)
        qpd_signs = [self.raw_result.counts_metadata(key)["gate_qpd_sign"] for key in keys]

        mask = {idx for idx, char in enumerate(observable.to_label()) if char != "I"}

        qpd_counts_resampled = [self.resampled_counts(key) for key in keys]

        exp_values = []
        for idx in range(self.n_resample):
            qpd_counts = [resampled[idx] for resampled in qpd_counts_resampled]
            final_counts = self.combine_counts(qpd_counts, qpd_signs, light_cones)

            exp_val = expectation_value(final_counts, mask, normalize=False)
            exp_values.append(exp_val / self.trex_denominator(mask))

        return np.average(exp_values), np.std(exp_values)


class StandardAnalysis(BaseAnalysis):
    """Compute expectation values from measured counts when there are no QPDs."""

    KEY_FIELDS = ["zne_c", "basis", "flipped_cbits"]

    def mean_std(self, observable: Pauli) -> Tuple:
        """Compute the mean and std for the observable with resampling."""
        key = f"zne_c1_basis{self._stabilizer_to_basis[observable]}"
        mem = self.raw_result.memory(key)
        mask = {idx for idx, char in enumerate(observable.to_label()) if char != "I"}

        sample_size = len(mem) // self.n_resample
        exp_values = []
        for _ in range(self.n_resample):
            resample = defaultdict(int)
            for idx in np.random.randint(0, len(mem), sample_size):
                resample[mem[idx]] += 1

            exp_val = expectation_value(resample, mask)

            exp_values.append(exp_val / self.trex_denominator(mask))

        return np.average(exp_values), np.std(exp_values)

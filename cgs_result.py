# This code is associated to the paper `Scaling quantum computing with dynamic circuits`
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import copy
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np

from qiskit_experiments.framework import ExperimentDecoder


def bytes_to_bin(data, num_bits: int):
    """Convert bytes to binary form."""
    val = int.from_bytes(data, "big")
    return bin(val)[2:].zfill(num_bits)


def packed_bits_to_counts(packed_bits_list: List[np.ndarray], num_bits_list: List[int]) -> List:
    """Convert packed bits to a standard counts dictionary."""
    # axes have dimensions (num_parameter_sets, num_shots, num_bytes_per_shot)
    for idx, packed_bits in enumerate(packed_bits_list):
        assert packed_bits.ndim == 3
        assert packed_bits.shape[-1] * 8 >= num_bits_list[idx]

    num_param_sets = len(packed_bits_list[0])
    num_rows = len(packed_bits_list[0][0])
    num_cregs = len(packed_bits_list)
    counts_list = []
    for idx in range(num_param_sets):
        counts = defaultdict(int)
        for idx_row in range(num_rows):
            # build string
            bin_str = []
            for idx_creg in range(num_cregs):
                bin_str.append(
                    bytes_to_bin(
                        packed_bits_list[idx_creg][idx][idx_row].tobytes(),
                        num_bits_list[idx_creg],
                    )
                )

            counts[" ".join(bin_str)] += 1

        counts_list.append(dict(counts))

    return counts_list


class CGSResult:
    """Result object tailored for cut graph states for efficient data look-up."""

    def __init__(self, key_fields: List[str]):
        """Instantiate the result container.

        Args:
            key_fields: The fields in the metadata from which to make the key.
                The entries must be present in the metadata. The values under
                these keys must be serializable.
        """
        self._key_fields = key_fields

        # Counts and corresponding metadata are both stored under the same key.
        self._data: List[Tuple[Dict, Dict]] = []

        # The mapping between key and index in the counts.
        self._index: Dict[str, int] = dict()

        # Any extra metadata that is not tied to a circuit execution
        self._metadata = {}

        # Memory for the count data.
        self._memory = dict()

    @property
    def metadata(self) -> Dict:
        """Overall metadata of the result."""
        return self._metadata

    @property
    def key_fields(self) -> List[str]:
        """Return the key fields."""
        return self._key_fields

    def keys(self):
        """Return the items in the result."""
        return self._index.keys()

    def memory(self, key: str | int) -> List:
        """Get a memory list from the counts at the given key."""
        if isinstance(key, str):
            key = self._index[key]

        if key not in self._memory:
            mem = []
            for bit_str, cnt in self.counts(key).items():
                mem += [bit_str] * int(cnt)

            self._memory[key] = mem

        return self._memory[key]

    def counts(self, key: str | int) -> Dict[str, int]:
        """Return the counts at the given key."""
        if isinstance(key, str):
            return self._data[self._index[key]][0]
        else:
            return self._data[key][0]

    def counts_metadata(self, key: str | int) -> Dict:
        """Return the metadata at the given key."""
        if isinstance(key, str):
            return self._data[self._index[key]][1]
        else:
            return self._data[key][1]

    def counts_metadata_entry(self, key: int, field: str):
        """Return a specific field of the metadata.

        This only supports int index for speed. It is used by the result processor.
        """
        return self._data[key][1][field]

    def make_key(self, metadata: Dict[str, Any]) -> str:
        """Make a key from the metadata."""
        key_entries = []
        for field in self._key_fields:
            value = metadata[field]
            key_entries.append(f"{field}{value}")

        return "_".join(key_entries)

    def add_counts(self, counts: Dict[str, int], metadata: Dict[str, Any]):
        """Add counts to the data."""
        self._index[self.make_key(metadata)] = len(self._data)
        self._data.append((counts, metadata))

    def __len__(self) -> int:
        """Length of the result, i.e., number of count objects stored."""
        return len(self._data)

    def __getitem__(self, item: str | int):
        """Return an entry of the data."""
        if isinstance(item, str):
            return self._data[self._index[item]]
        else:
            return self._data[item]

    def copy_empty_like(self) -> "CGSResult":
        """Create an empty copy of the class."""
        result = CGSResult(key_fields=copy.deepcopy(self.key_fields))
        result.metadata.update(copy.deepcopy(self._metadata))
        return result

    @classmethod
    def from_data(
        cls,
        data: dict,
        key_fields: List[str],
        creg_keys,
        creg_len,
        job_task_metadata=None,
        separate_jobs=False
    ) -> "CGSResult":
        """Create a class instance from serialized data."""
        decode = ExperimentDecoder().decode

        results = data["results"]
        metadata_list = decode(data["__config__"]["_run_metadata"])
        max_parameters_per_task = data["__config__"].get("_array_max_parameters", None)

        if job_task_metadata is None:
            job_task_metadata = decode(data["__config__"]["_array_job_task_metadata"])

        cgs_result = cls(key_fields)

        for r_idx, serialized_result in enumerate(results):
            result = decode(serialized_result)

            cgs_result.metadata.update(result[1])

            batch_metadata = {}

            # Check if metadata_list is a list of lists
            separate_metadatas = isinstance(metadata_list[0], list)

            for t_idx, task in enumerate(result[0]):
                batch_metadata[t_idx] = task["batch"]
                tasks = [task[key] for key in creg_keys]
                counts_list = packed_bits_to_counts(tasks, creg_len)
                for idx, counts in enumerate(counts_list):
                    meta_idx = idx
                    if max_parameters_per_task is not None:
                        meta_idx += max_parameters_per_task * r_idx

                    if separate_metadatas:
                        if separate_jobs:
                            m_index = r_idx
                        else:
                            m_index = t_idx
                        # assumes params in inner run loop
                        meta_idx = meta_idx % len(metadata_list[m_index])

                        metadata = copy.deepcopy(metadata_list[m_index][meta_idx])
                    else:
                        # assumes params in inner run loop
                        meta_idx = meta_idx % len(metadata_list)

                        metadata = copy.deepcopy(metadata_list[meta_idx])

                    metadata.update(copy.deepcopy(job_task_metadata[r_idx][t_idx]))

                    cgs_result.add_counts(counts, metadata)

            cgs_result.metadata.update(batch_metadata)

        return cgs_result

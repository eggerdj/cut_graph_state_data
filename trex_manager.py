# This code is associated to the paper `Scaling quantum computing with dynamic circuits`
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Dict, List
import numpy as np


class TREXManager:
    """A Class responsible for holding the trex calibration counts."""

    def __init__(self, calibration_counts: Dict[str, int], num_nodes: int):
        """The calibration counts are measured by twirling readouts with X gates."""

        if not np.all([len(bit_str) == num_nodes for bit_str in calibration_counts]):
            self.calibration_counts = {
                bin(int(k))[2:].zfill(num_nodes): v for k, v in calibration_counts.items()
            }
        else:
            self.calibration_counts = calibration_counts

        self._mitigator_memory = None
        self._mit_shots = None

    @property
    def mitigator_memory(self) -> List[str]:
        """Return the memory of the calibration counts for resampling."""
        if self._mitigator_memory is None:
            self._make_mitigator_memory()

        return self._mitigator_memory

    @property
    def mit_shots(self) -> int:
        """Return the number of shots in the mitigator."""
        if self._mit_shots is None:
            self._make_mitigator_memory()

        return self._mit_shots

    def _make_mitigator_memory(self):
        """Create a memory like variable to sample from."""
        self._mitigator_memory = []
        self._mit_shots = int(sum(self.calibration_counts.values()))
        for key, cnt in self.calibration_counts.items():
            self._mitigator_memory += [key] * int(cnt)

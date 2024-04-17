# This code is associated to the paper `Scaling quantum computing with dynamic circuits`
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from cgs_result import CGSResult


class ResultProcessor:
    """Processes a result object."""

    def __init__(self, trex_grouping: Optional[List[Tuple[int, ...]]] = None):
        """Initialize the class with the way we grouped the trex twirls."""
        self._trex_grouping = trex_grouping

    @staticmethod
    def trex_correct_counts(counts: Dict[str, int], flips: Tuple[int], all_counts: dict):
        """Corrects the counts by flipping the bits of the twirled readouts."""
        for bits, value in counts.items():
            bits = [bit for bit in bits[::-1]]

            for idx in flips:
                bits[idx] = "1" if bits[idx] == "0" else "0"

            all_counts["".join(bits[::-1])] += value

    def trex_merge_results(self, result: CGSResult) -> CGSResult:
        """Take a result, merge the trex samples, and return a new result."""
        new_result = result.copy_empty_like()
        new_result.key_fields.remove("flipped_cbits")

        for group in self._trex_grouping:
            all_counts, metadata = defaultdict(int), {}
            metadata = copy.deepcopy(result.counts_metadata(group[0]))
            metadata.pop("flipped_cbits")

            for idx, result_idx in enumerate(group):
                flips = result.counts_metadata_entry(result_idx, "flipped_cbits")
                counts = result.counts(result_idx)
                self.trex_correct_counts(counts, flips, all_counts)

            new_result.add_counts(dict(all_counts), metadata)

        return new_result

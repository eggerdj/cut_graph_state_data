# This code is associated to the paper `Scaling quantum computing with dynamic circuits`
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper method to compute witnesses."""

from typing import Dict, Optional, Tuple
import networkx as nx
import numpy as np

from qiskit.quantum_info import Pauli

from cut_graph import CutGraph


def compute_witness(
    stabilizers: Dict[str, Tuple],
    graph: nx.Graph,
    cgs: CutGraph,
    stabilizers_zne: Optional[Dict[str, Tuple]] = None,
):
    """Compute the value of the edge witnesses.

    This method computes the witness as 1/4 - (Si + Sj + SiSj) / 4.
    Note that the error bar assumes that there is no correlation between the measurement
    of the witnesses. This is an approximation. We could justify this by stating that the
    fits for the zero-noise extrapolation are independent.

    Args:
        stabilizers: A dictionary where the key is a Pauli that expresses a stabilizer. The
            value is a two-tuple of mean and standard deviation. Note that for this method
            to work all node stabilizers and their product across adges must be present.
        graph: The graph for which to compute the witnesses.
        cgs: An instance of a CutGraphState from which to get the edge stabilizers.
        stabilizers_zne: Stabilizers computed from ZNE. If this option is given the function
            will compute the witnesses by using the ZNE stabilizers for stabilizers affected by
            the cut and the stabilizers in `stabilizers` otherwise.
    """
    witnesses = {}

    for edge in graph.edges:
        edge = tuple(sorted(edge))
        edge_stabilizers = [stab.to_label() for stab in cgs.edge_stabilizers([edge])]

        if stabilizers_zne is None:
            mean = 1 / 4 - sum(stabilizers[s][0] for s in edge_stabilizers) / 4
            err = 1 / 4 * np.sqrt(sum(stabilizers[s][1] ** 2 for s in edge_stabilizers))
        else:
            stab_values = []
            for stab in edge_stabilizers:
                if cgs.touches_cut(Pauli(stab)):
                    stab_values.append(stabilizers_zne[stab])
                else:
                    stab_values.append(stabilizers[stab])

            mean = 1 / 4 - sum(val[0] for val in stab_values) / 4
            err = 1 / 4 * np.sqrt(sum(val[1] ** 2 for val in stab_values))

        witnesses[edge] = (mean, err)

    return witnesses

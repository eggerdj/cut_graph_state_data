# This code is associated to the paper `Scaling quantum computing with dynamic circuits`
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class to manipulate stabilizers on cut graphs."""

import json
from typing import List, Optional, Tuple
import networkx as nx

from qiskit.quantum_info import Pauli


class CutGraph:
    """Class to manipulate stabilizers on cut graphs."""

    def __init__(self, file_name: str):
        """Initialize CutGraphState from data in a file"""
        with open(f"data/{file_name}", "r") as fin:
            data = json.load(fin)

        self.graph = nx.from_edgelist(data["edge list"])

        self.flat_cuts = []
        self.cut_edges = data["cut edges"]
        for group in data["cut edges"]:
            self.flat_cuts += [edge for edge in group] + [edge[::-1] for edge in group]

        # These are nodes that are on the cut.
        self.nodes_on_cut = set()
        for edge in self.flat_cuts:
            self.nodes_on_cut.update(edge)

        # A variable to easily make stabilizer labels for plots.
        self._stabilizer_labels = {}

    def stabilizers(self, nodes: Optional[List[int]] = None) -> List[Pauli]:
        """Returns a list of stabilizer Pauli operators.

        A stabilizer for node `i` in the graph has an `X` operator on node `i`
        and `Z` operators on the nodes adjacent to `i` in the graph.

        Args:
            nodes: The nodes for which to return the first order stabilizers.
                If None is given we return the stabilizers for all the nodes.

        Returns:
            A list of stabilizing Pauli operators. Each Pauli operator has the same
            length as the number of nodes in the graph.
        """
        stabilizers = []

        nodes = nodes or self.graph.nodes

        for node in nodes:
            paulis = ["I"] * len(self.graph.nodes)
            paulis[node] = "X"

            for neighbor_node in self.graph.neighbors(node):
                paulis[neighbor_node] = "Z"

            stabilizer = Pauli("".join(paulis[::-1]))

            self._stabilizer_labels[stabilizer] = "$S_{" + f"{node}" + "}$"

            stabilizers.append(stabilizer)

        return stabilizers

    def edge_stabilizers(
        self, edges: Optional[List[Tuple[int, int]]] = None
    ) -> List[Pauli]:
        """Generate the stabilizers needed to measure the entanglement witness across and edge.

        For each edge, `(i, j)` this method will generate the first order
        stabilizers `S_i` and `S_j` as well as their product `S_iS_j`.

        Args:
            edges: The edges for which to generate the edge stabilizers. If None is given then
                we generate the edge stabilizers for all the edges.
        """
        stabilizers = set()

        edges = edges or self.graph.edges

        for edge in edges:
            first_order = self.stabilizers(edge)
            s1, s2 = first_order[0], first_order[1]

            num1, num2 = f"{edge[0]}", f"{edge[1]}"

            self._stabilizer_labels[s1] = "$S_{" + num1 + "}$"
            self._stabilizer_labels[s2] = "$S_{" + num2 + "}$"
            self._stabilizer_labels[s1 @ s2] = "$S_{" + num1 + "}S_{" + num2 + "}$"

            stabilizers.update([s1, s2, s1 @ s2])

        return sorted(list(stabilizers), key=lambda x: x.to_label())

    def stabilizer_label(self, stabilizer: Pauli) -> str:
        """Return a label for plotting. This works based on cached generated stabilizers."""
        return self._stabilizer_labels.get(stabilizer, None)

    def touches_cut(self, stabilizer: Pauli) -> bool:
        """Return True if one of the Paulis in the stabilizer is on the cut."""
        is_not_ids = [x_ or stabilizer.z[idx] for idx, x_ in enumerate(stabilizer.x)]
        for idx, is_not_id in enumerate(is_not_ids):
            if is_not_id and idx in self.nodes_on_cut:
                return True

        return False

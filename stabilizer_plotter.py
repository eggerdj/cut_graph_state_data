# This code is associated to the paper `Scaling quantum computing with dynamic circuits`
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Make bar plots of stabilizers and witnesses."""

from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from cut_graph import CutGraph


class StabilizerPlotter:
    """A class to plot stabilizers as bars."""

    def __init__(self, graph: CutGraph):
        """Initialize the plotter."""
        self.graph = graph

    def stabilizer_plot_order(self):
        """Group the stabilizers by their distance to the cut edges.

        The number of a stabilizer is the node number where the first-order stabilizer
        has an X Pauli operator. We compute the distance of a Stabilizer to a cut edge
        as the distance between the stabilizer node and a node with a cut edge.

        Returns:
            A list with the plot order of the first-order stabilizers, a list
            of stabilizers distances sorted in ascending order, a dictionary
            with the number of stabilizers that are at a given distance, and
            the distances of stabilizers from cuts in the same order as the
            first returned argument.
        """

        # plot the stabilizers according to their distance from a cut edge

        shortest_length = dict(nx.all_pairs_shortest_path_length(self.graph.graph))
        costs = []

        for node in self.graph.graph.nodes:
            # The following lines can error if there are no cut edges. In this case
            # we set the cost to zero for all stabilizers.
            try:
                cost = min(
                    shortest_length[node][cnode] for cnode in self.graph.nodes_on_cut
                )
            except ValueError:
                cost = 0
            costs.append((node, cost))

        costs.sort(key=lambda x: x[1])
        stb_plot_order = [_[0] for _ in costs]
        stb_distances = [_[1] for _ in costs]

        num_dists = defaultdict(int)
        for _, cost in costs:
            num_dists[cost] += 1

        dists = sorted(dist for dist in num_dists)

        return stb_plot_order, dists, num_dists, stb_distances

    @staticmethod
    def add_distance_delimiters(dists: List[int], num_dists: Dict[int, int], ax):
        """Add vertical lines to delimit the distance of the stabilizers.

        Note the `dists` and `num_dists` arguments come from the `stabilizer_plot_order`
        method defined above.

        Args:
            dists: A list of stabilizers distances sorted in ascending order.
            num_dists: A dictionary with the number of stabilizers that are at a given distance.
            ax: axis on which to plot. If None, then use the class's axis.
        """
        ylims = ax.get_ylim()
        tot_dist = 0
        for dist in dists:
            tot_dist += num_dists[dist]
            ax.vlines(tot_dist - 0.5, ylims[0], ylims[1], "k", ls="--")

        ax.set_ylim(ylims)

    def witness_plot_order(self):
        """Witnesses according to their distance to a cut."""

        stb_order, _, _, dists = self.stabilizer_plot_order()

        stb_dist_map = {stb_idx: dists[idx] for idx, stb_idx in enumerate(stb_order)}

        costs = []
        for edge in self.graph.graph.edges:
            dist = np.average([stb_dist_map[edge[0]], stb_dist_map[edge[1]]])
            edge = sorted(edge)
            costs.append((edge[0], edge[1], dist))

        costs = sorted(costs, key=lambda x: x[2])
        edge_order = [(_[0], _[1]) for _ in costs]

        edge_distances = [_[2] for _ in costs]
        num_dists = defaultdict(int)
        for cost in costs:
            num_dists[cost[-1]] += 1

        dists = sorted(dist for dist in num_dists)

        labels = {}
        for edge in edge_order:
            labels[edge] = "$\mathcal{W}_{" + f"{edge[0]}, {edge[1]}" + "}$"

        return edge_order, dists, num_dists, edge_distances, labels

    def stabilizer_witness_plot(
        self,
        stabilizer_vals,
        stabilizer_vals_zne,
        witnesses,
        witnesses_zne,
        fig_size: Tuple[int, int] = (12, 5.5),
        width: float = 0.2,
        max_x_values: int = 20,
    ):
        """Make the plot with stabilizers above and witnesses below."""
        stabilizers = self.graph.stabilizers()

        stab_map = {
            stab.to_label()[::-1].index("X"): stab.to_label() for stab in stabilizers
        }

        fig, axs = plt.subplots(2, 1, figsize=fig_size)

        ordr, dists, num_dists, _ = self.stabilizer_plot_order()
        eordr, edists, enum_dists, _, wlabels = self.witness_plot_order()

        # Define number of stabilizers that will have ZNE plotted.
        # This can error if there are no cut edges in which case there is no switch ZNE.
        try:
            num_znes = num_dists[dists[0]] + num_dists[dists[1]]
            enum_znes = enum_dists[edists[0]] + enum_dists[edists[1]]
        except IndexError:
            num_znes, enum_znes = 0, 0

        zne_ordr, zne_edge_ordr = ordr[0:num_znes], eordr[0:enum_znes]

        w = width

        settings = {
            "swaps": (-1.5, "darkblue", "dodgerblue"),
            "drop": (1.5, "darkred", "firebrick"),
            "locc": (-0.5, "darkgreen", "forestgreen"),
            "lo": (0.5, "darkorange", "orange"),
        }

        if max_x_values is not None:
            ordr = ordr[:max_x_values]
            eordr = eordr[:max_x_values]
            zne_ordr = zne_ordr[:max_x_values]
            zne_edge_ordr = zne_edge_ordr[:max_x_values]

        for tag in stabilizer_vals:
            offset, ecolor, color = settings[tag.lower()]

            # Stabilizers, solid bars, no ZNE.
            if tag.lower() == "locc":
                y_err_s = [
                    stabilizer_vals[tag]["1.0"][stab_map[idx]][1]
                    for idx in ordr[num_znes:]
                ]
                y_err_w = [witnesses[tag][e][1] for e in eordr[enum_znes:]]

                # Bars without error bars
                axs[0].bar(
                    [x + offset * w for x in range(len(zne_ordr))],
                    [stabilizer_vals[tag]["1.0"][stab_map[idx]][0] for idx in zne_ordr],
                    width=0.9 * w,
                    color=color,
                    edgecolor=color,
                    lw=0.5,
                )

                # Witnesses, solid bars.
                axs[1].bar(
                    [x + offset * w for x in range(len(zne_edge_ordr))],
                    [witnesses[tag][e][0] for e in zne_edge_ordr],
                    color=color,
                    edgecolor=color,
                    lw=0.05,
                    width=0.9 * w,
                )

                # Bars with error bars
                axs[0].bar(
                    [x + offset * w for x in range(num_znes, len(ordr))],
                    [
                        stabilizer_vals[tag]["1.0"][stab_map[idx]][0]
                        for idx in ordr[num_znes:]
                    ],
                    width=0.9 * w,
                    color=color,
                    edgecolor=color,
                    label=f"{tag}",
                    yerr=y_err_s,
                    lw=0.5,
                    error_kw={"elinewidth": 0.75},
                    capsize=2,
                )

                # Witnesses, solid bars.
                axs[1].bar(
                    [x + offset * w for x in range(enum_znes, len(eordr))],
                    [witnesses[tag][e][0] for e in eordr[enum_znes:]],
                    color=color,
                    edgecolor=color,
                    lw=0.05,
                    label=f"{tag}",
                    width=0.9 * w,
                    yerr=y_err_w,
                    error_kw={"elinewidth": 0.75},
                    capsize=2,
                )
            else:
                axs[0].bar(
                    [x + offset * w for x in range(len(ordr))],
                    [stabilizer_vals[tag]["1.0"][stab_map[idx]][0] for idx in ordr],
                    width=0.9 * w,
                    label=f"{tag}",
                    color=color,
                    edgecolor=color,
                    lw=0.5,
                    error_kw={"elinewidth": 0.75},
                    yerr=[
                        stabilizer_vals[tag]["1.0"][stab_map[idx]][1] for idx in ordr
                    ],
                    capsize=2,
                )

                # Witnesses, solid bars.
                axs[1].bar(
                    [x + offset * w for x in range(len(eordr))],
                    [witnesses[tag][e][0] for e in eordr],
                    color=color,
                    edgecolor=color,
                    lw=0.05,
                    label=f"{tag}",
                    width=0.9 * w,
                    error_kw={"elinewidth": 0.75},
                    yerr=[witnesses[tag][e][1] for e in eordr],
                    capsize=2,
                )

            # Locc is special since it has ZNE on the switch. Now plot the ZNE.
            if tag.lower() == "locc":
                axs[0].bar(
                    [x + offset * w for x in range(len(zne_ordr))],
                    [stabilizer_vals_zne[stab_map[idx]][0] for idx in zne_ordr],
                    width=0.9 * w,
                    label=f"{tag} with ZNE",
                    color=[0, 0, 0, 0],
                    edgecolor=ecolor,
                    lw=2,
                    error_kw={"elinewidth": 0.75},
                    yerr=[stabilizer_vals_zne[stab_map[idx]][1] for idx in zne_ordr],
                    capsize=2,
                    zorder=100,
                )

                axs[1].bar(
                    [x + offset * w for x in range(len(zne_edge_ordr))],
                    [witnesses_zne[tag][e][0] for e in zne_edge_ordr],
                    color=[0, 0, 0, 0],
                    edgecolor=ecolor,
                    lw=2,
                    label=f"{tag} with ZNE",
                    width=0.9 * w,
                    error_kw={"elinewidth": 0.75},
                    yerr=[witnesses_zne[tag][e][1] for e in zne_edge_ordr],
                    capsize=2,
                    zorder=100,
                )

        axs[0].set_xticks(range(len(ordr)))
        axs[1].set_xticks(range(len(eordr)))
        axs[1].set_xticklabels([wlabels[e] for e in eordr], rotation=45)
        axs[0].set_ylabel("Eigenvalue")
        axs[1].set_ylabel("Witness value")
        axs[1].set_xlabel("Witness")
        axs[0].hlines(1, -0.5, len(stabilizers) - 0.5, "k", ls=":", zorder=-10)
        axs[1].hlines(-0.5, -0.5, len(stabilizers) - 0.5, "k", ls=":", zorder=-10)
        axs[1].hlines(0.0, -0.5, len(stabilizers) - 0.5, "k", zorder=-10, lw=0.75)
        axs[0].legend(ncol=5, loc=1, framealpha=1.0, fontsize=10)

        self.add_distance_delimiters(dists, num_dists, ax=axs[0])
        self.add_distance_delimiters(edists, enum_dists, ax=axs[1])

        axs[0].set_xticklabels(["$S_{" + f"{x}" + "}$" for x in ordr])

        axs[0].set_xlim([-0.5, len(ordr) - 1.5])
        axs[1].set_xlim([-0.5, len(eordr) - 1.5])

        return fig, axs

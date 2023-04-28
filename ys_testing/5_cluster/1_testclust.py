# %%
import os

# import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
from matplotlib import pyplot as plt

# %%
class model:
    def __init__(
        self,
        n_agents,
        w_min=1e-17,
        w_0=None,
        G=None,
        additive=False,
        theta=0,
        save_every=np.inf,
        upd_w_every=np.inf,
        upd_graph_every=np.inf,
        plot_every=np.inf,
        figpath=None,
    ):  # sourcery skip: assign-if-exp
        self.N = n_agents
        self.w_min = w_min
        self.G = G
        self.additive = additive
        if self.G is not None:
            self.posi = {i: self.G.nodes[i]["pos"] for i in range(self.N)}
            self.theta = theta
        self.update_w = upd_w_every
        self.update_links = upd_graph_every
        self.every = save_every
        self.plot = plot_every
        if self.plot != np.inf:
            self.fig, self.ax = plt.subplots(dpi=150)
            self.ax.set_xlim([-0.03, 1.03])
            self.ax.set_ylim([-0.03, 1.03])
            self.figpath = figpath
            self.temppath = os.path.join(self.figpath, "temp")
            self.figpath = figpath
        # Initialize n agents with random risks and wealth between (0, 1]
        # and normalize wealth
        # n[i, 0] is the wealth and n[i, 1] is the risk of agent i
        self.n = np.random.rand(self.N, 2)
        if w_0 is not None:
            self.n[:, 0] = w_0
        else:
            self.n[:, 0] = self.n[:, 0] / (np.sum(self.n[:, 0]))
        self.gini = [self.get_gini()]
        self.n_active = [self.get_actives()]

    def get_opponents(self):
        if self.G is None:
            random_array = np.random.randint(0, self.N, self.N)
            indices = np.arange(0, self.N)
            # Create array of random numbers that are not equal to the index
            # If i=j then assign j'=i+1 (j'=0 if i=N-1)
            random_array = np.where(
                random_array == indices, (random_array + 1) % self.N, random_array
            )
        else:
            random_array = np.full(self.N, fill_value=-1)
            for i in range(self.N):
                if neighbors := list(nx.all_neighbors(self.G, i)):
                    random_array[i] = np.random.choice(neighbors)

        return random_array

    def is_valid(self, i, j):
        # Check if both agents have w > w_min
        return (self.n[i, 0] > self.w_min) and (self.n[j, 0] > self.w_min)

    def get_dw(self, i, j):
        return np.minimum(self.n[i, 0] * self.n[i, 1], self.n[j, 0] * self.n[j, 1])

    def get_gini(self):
        w = np.sort(self.n[:, 0])
        p_cumsum = np.cumsum(w) / np.sum(w)
        B = np.sum(p_cumsum) / self.N
        return 1 + 1 / self.N - 2 * B

    def get_actives(self):
        return np.sum(self.n[:, 0] > self.w_min)

    def get_liquidity():
        return

    def update_wealth(self, i, j, dw):
        self.n[i, 0] += dw
        self.n[j, 0] -= dw

    def choose_winner(self, i, j):
        raise NotImplementedError("You need to choose a valid model.")

    def update_weights(self):
        w = dict(enumerate(self.n[:, 0]))
        nx.set_node_attributes(self.G, w, "weight")

    def plot_snapshot(self, mcs):
        ## Esto se tiene que borrar y escribir mejor
        w = dict(enumerate(self.n[:, 0]))
        a = np.array(list(w.values()))

        dead_nodes = [node for node, weight in w.items() if weight < self.w_min]

        node_size = 500 * np.sqrt(a)
        node_colors = plt.cm.coolwarm(100 * a)
        edge_colors = [
            "r" if (e[0] in dead_nodes or e[1] in dead_nodes) else "black"
            for e in self.G.edges
        ]

        filename = os.path.join(self.temppath, "test_{:05d}.png".format(mcs))

        self.ax.clear()
        self.ax.set_title(f"t = {mcs}")
        nx.draw(
            self.G,
            node_size=node_size,
            width=0.2,
            pos=self.posi,
            node_color=node_colors,
            edge_color=edge_colors,
            ax=self.ax,
        )
        self.fig.savefig(filename, format="PNG")

    def MCS(self, steps):
        """
        Main MC loop
        """
        for mcs in range(steps):
            if mcs % self.plot == 0 and self.G is not None:
                self.plot_snapshot(mcs)

            j = self.get_opponents()

            for i, ji in enumerate(j):
                # Check both agents have w > w_min and node is not isolated
                if self.is_valid(i, ji) and ji != -1:
                    dw = self.get_dw(i, ji)
                    winner = self.choose_winner(i, ji)
                    dw = np.where(winner == i, dw, -dw)
                    self.update_wealth(i, ji, dw)

            # After self.update_w update weights
            if mcs % self.update_w == 0 and self.G is not None:
                self.update_weights()

            # Recompute the links if the network is dynamic
            if (mcs + 1) % self.update_links == 0 and self.G is not None:
                self.G = nx.geographical_threshold_graph(
                    self.N,
                    theta=self.theta,
                    weight=self.n[:, 0],
                    dim=2,
                    pos=self.posi,
                    additive=self.additive,
                )
            # After self.every MCS append new Gini index
            if (mcs + 1) % self.every == 0:
                self.gini.append(self.get_gini())
                self.n_active.append(self.get_actives())


# %%
class YSmodel(model):
    def __init__(self, n_agents=200, f=0, **kwargs):
        super().__init__(n_agents, **kwargs)
        # f is the social protection factor
        self.f = f

    def choose_winner(self, i, j):
        p = 0.5 + self.f * (
            (self.n[j, 0] - self.n[i, 0]) / (self.n[i, 0] + self.n[j, 0])
        )
        return np.random.choice([i, j], p=[p, 1 - p])

# Numero de nodos = Numero de agentes
n = 1000
# Pesos de los nodos = riquezas iniciales de los agentes
a_0 = np.random.rand(n)
a_0 /= np.sum(a_0)
w_0 = dict(enumerate(a_0))

f_set = [0]

MCS = 1000

upd_w = np.inf
upd_g = np.inf
plot_every = upd_w


# %%
for f in f_set:
    model = YSmodel(f=f, n_agents=n, w_min=1e-17, save_every=1)

    start = time.time()
    model.MCS(MCS)
    end = time.time()
    net_time = end - start
    print("Elapsed time: {:.2f} s.".format(net_time))


# %%

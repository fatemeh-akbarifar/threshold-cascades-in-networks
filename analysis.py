"""Paired threshold cascades on matched-size synthetic graph families."""

from pathlib import Path
import json, hashlib, platform
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FAMILIES = ["random", "small-world", "modular"]
POLICIES = ["none", "random", "degree"]


def graph(family, gid):
    seed = 10000 * FAMILIES.index(family) + gid
    rng = np.random.default_rng(seed)
    if family == "random":
        g = nx.gnm_random_graph(200, 600, seed=seed)
    elif family == "small-world":
        g = nx.watts_strogatz_graph(200, 6, 0.1, seed=seed)
    else:
        g = nx.empty_graph(200)
        for b in range(4):
            candidates = [
                (i, j)
                for i in range(50 * b, 50 * (b + 1))
                for j in range(i + 1, 50 * (b + 1))
            ]
            g.add_edges_from(
                [candidates[i] for i in rng.choice(len(candidates), 120, replace=False)]
            )
        candidates = [
            (i, j) for i in range(200) for j in range(i + 1, 200) if i // 50 != j // 50
        ]
        g.add_edges_from(
            [candidates[i] for i in rng.choice(len(candidates), 120, replace=False)]
        )
    assert (
        len(g) == 200 and g.number_of_edges() == 600 and nx.number_of_selfloops(g) == 0
    )
    return g, seed


def cascade(g, seeds, protected=(), threshold=0.2):
    nodes = list(g)
    index = {n: i for i, n in enumerate(nodes)}
    a = nx.to_numpy_array(g, nodelist=nodes, dtype=np.int16)
    deg = a.sum(axis=1)
    required = np.ceil(threshold * deg - 1e-12).astype(int)
    active = np.zeros(len(nodes), bool)
    blocked = np.zeros(len(nodes), bool)
    active[[index[n] for n in seeds]] = True
    blocked[[index[n] for n in protected]] = True
    assert 0 < threshold <= 1 and not np.any(active & blocked)
    times = np.full(len(nodes), -1, int)
    times[active] = 0
    counts = [int(active.sum())]
    for t in range(1, len(nodes) + 1):
        new = (a @ active.astype(np.int16) >= required) & (deg > 0) & ~active & ~blocked
        if not new.any():
            break
        active |= new
        times[new] = t
        counts.append(int(active.sum()))
    assert not np.any(active & blocked) and np.all(np.diff(counts) >= 0)
    return (
        counts,
        {n for n in nodes if active[index[n]]},
        dict(zip(nodes, times.tolist())),
    )


def selections(g, seed, setid, budget):
    rng = np.random.default_rng(1000000 + 100 * seed + setid)
    seeds = rng.choice(200, 4, replace=False)
    candidates = np.array([n for n in g if n not in seeds])
    random = rng.permutation(candidates)[:budget]
    tie = rng.random(200)
    degree = sorted(candidates, key=lambda n: (-g.degree(n), tie[n]))[:budget]
    return seeds, {"none": [], "random": random, "degree": degree}


def checks():
    g = nx.path_graph(5)
    assert cascade(g, [], threshold=0.5)[0] == [0]
    assert cascade(g, [0], threshold=0.5)[1] == set(range(5))
    assert cascade(g, [0], threshold=1)[1] == {0}
    assert cascade(g, [0], [2], 0.5)[1] == {0, 1}
    g.add_node(5)
    assert 5 not in cascade(g, [0], threshold=0.5)[1]
    assert 5 in cascade(g, [5], threshold=0.5)[1]
    mapping = {i: str(10 - i) for i in g}
    renamed = nx.relabel_nodes(g, mapping)
    assert cascade(renamed, [mapping[0]], threshold=0.5)[1] == {
        mapping[i] for i in cascade(g, [0], threshold=0.5)[1]
    }
    return {"hand_checks_passed": True}


def main():
    OUT.mkdir(exist_ok=True)
    validation = checks()
    rows = []
    edges = []
    sensitivity = []
    graphrows = []
    for family in FAMILIES:
        for gid in range(30):
            g, seed = graph(family, gid)
            components = list(nx.connected_components(g))
            desc = {
                "family": family,
                "graph_id": gid,
                "random_seed": seed,
                "components": len(components),
                "largest_component_fraction": max(map(len, components)) / 200,
            }
            graphrows.append(
                {
                    **desc,
                    "clustering": nx.average_clustering(g),
                    "degree_sd": float(np.std([d for _, d in g.degree])),
                }
            )
            edges.extend(
                {"family": family, "graph_id": gid, "source": a, "target": b}
                for a, b in g.edges
            )
            for budget in [10, 5, 20] if gid < 10 else [10]:
                for setid in range(5):
                    seeds, policies = selections(g, seed, setid, budget)
                    for theta in [0.1, 0.2, 0.3]:
                        for policy, protected in policies.items():
                            counts, active, times = cascade(g, seeds, protected, theta)
                            row = {
                                **desc,
                                "seed_set": setid,
                                "policy": policy,
                                "threshold": theta,
                                "budget": budget,
                                "protected_count": len(protected),
                                "final_fraction": len(active) / 200,
                                "rounds": len(counts) - 1,
                                "large_cascade": len(active) >= 100,
                                "check_passed": True,
                            }
                            (rows if budget == 10 else sensitivity).append(row)
            print(family, gid, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "cascade-runs.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "budget-sensitivity.csv", index=False)
    budget_runs = pd.concat([d[d.graph_id < 10], pd.DataFrame(sensitivity)])
    budget_graphs = (
        budget_runs.groupby(["family", "graph_id", "threshold", "policy", "budget"])[
            "final_fraction"
        ]
        .mean()
        .reset_index()
    )
    budget_summary = (
        budget_graphs.groupby(["family", "threshold", "policy", "budget"])[
            "final_fraction"
        ]
        .mean()
        .reset_index()
    )
    budget_summary.to_csv(OUT / "budget-summary.csv", index=False)
    pd.DataFrame(edges).to_csv(OUT / "graph-edges.csv", index=False)
    pd.DataFrame(graphrows).to_csv(OUT / "graph-properties.csv", index=False)
    avg = (
        d.groupby(["family", "graph_id", "threshold", "policy"])[
            ["final_fraction", "rounds", "large_cascade"]
        ]
        .mean()
        .reset_index()
    )
    avg.to_csv(OUT / "graph-level-outcomes.csv", index=False)
    rng = np.random.default_rng(42)
    summaries = []
    paired = []
    for family in FAMILIES:
        for theta in [0.1, 0.2, 0.3]:
            part = avg[(avg.family == family) & (avg.threshold == theta)]
            wide = part.pivot(
                index="graph_id", columns="policy", values="final_fraction"
            )
            idx = rng.integers(0, 30, (2000, 30))
            for policy in POLICIES:
                v = wide[policy].to_numpy()
                ci = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
                summaries.append(
                    {
                        "family": family,
                        "threshold": theta,
                        "policy": policy,
                        "mean_fraction": v.mean(),
                        "lower": ci[0],
                        "upper": ci[1],
                        "mean_rounds": part[part.policy == policy]["rounds"].mean(),
                        "large_cascade_frequency": part[
                            part.policy == policy
                        ].large_cascade.mean(),
                    }
                )
            for a, b in [("random", "none"), ("degree", "none"), ("degree", "random")]:
                v = (wide[a] - wide[b]).to_numpy()
                ci = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
                paired.append(
                    {
                        "family": family,
                        "threshold": theta,
                        "contrast": a + " minus " + b,
                        "mean_difference": v.mean(),
                        "lower": ci[0],
                        "upper": ci[1],
                    }
                )
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "family-summary.csv", index=False)
    pair = pd.DataFrame(paired)
    pair.to_csv(OUT / "paired-policy-summary.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for family, ax in zip(FAMILIES, axs):
        for policy in POLICIES:
            p = summary[(summary.family == family) & (summary.policy == policy)]
            ax.errorbar(
                p.threshold,
                p.mean_fraction,
                yerr=[p.mean_fraction - p.lower, p.upper - p.mean_fraction],
                marker="o",
                label=policy,
                capsize=3,
            )
        ax.set_title(family)
        ax.set_xlabel("Activation threshold")
        ax.set_ylim(-0.03, 1.03)
        ax.legend()
    axs[0].set_ylabel("Final active fraction (graph-level mean)")
    fig.tight_layout()
    fig.savefig(OUT / "cascade-comparison.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for family, ax in zip(FAMILIES, axs):
        for j, theta in enumerate([0.1, 0.2, 0.3]):
            w = avg[(avg.family == family) & (avg.threshold == theta)].pivot(
                index="graph_id", columns="policy", values="final_fraction"
            )
            v = w.degree - w["random"]
            ax.scatter(
                np.full(30, j) + np.linspace(-0.08, 0.08, 30), v, alpha=0.6, s=15
            )
            r = pair[
                (pair.family == family)
                & (pair.threshold == theta)
                & (pair.contrast == "degree minus random")
            ].iloc[0]
            ax.errorbar(
                j,
                r.mean_difference,
                yerr=[[r.mean_difference - r.lower], [r.upper - r.mean_difference]],
                color="k",
                fmt="s",
                capsize=4,
            )
        ax.axhline(0, color="gray", ls=":")
        ax.set_xticks(range(3), [0.1, 0.2, 0.3])
        ax.set_xlabel("Threshold")
        ax.set_title(family)
    axs[0].set_ylabel("Degree minus random: paired active fraction")
    fig.tight_layout()
    fig.savefig(OUT / "paired-protection.png", dpi=160)
    plt.close(fig)
    g, seed = graph("modular", 0)
    seeds, _ = selections(g, seed, 0, 10)
    counts, active, times = cascade(g, seeds, threshold=0.2)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    pos = nx.spring_layout(g, seed=0)
    nx.draw_networkx_edges(g, pos, ax=axs[0], alpha=0.08)
    colors = [times[n] for n in g]
    nodes = nx.draw_networkx_nodes(
        g, pos, ax=axs[0], node_size=18, node_color=colors, cmap="viridis"
    )
    fig.colorbar(nodes, ax=axs[0], label="Activation round (−1: inactive)")
    axs[0].set_axis_off()
    axs[1].step(range(len(counts)), counts, where="post")
    axs[1].set_xlabel("Update round")
    axs[1].set_ylabel("Active nodes")
    fig.suptitle("Fixed illustration: modular graph0, seed set0, threshold0.2")
    fig.tight_layout()
    fig.savefig(OUT / "illustrative-cascade.png", dpi=160)
    plt.close(fig)
    record = {
        "checks": validation,
        "primary_runs": len(d),
        "sensitivity_runs": len(sensitivity),
        "independent_graphs_per_family": 30,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "networkx": nx.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "run.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""
Meilenstein 4: Auswertung eines Laufs.

    python analyze.py runs/20260101-120000

Erzeugt im Lauf-Ordner:  fitness.png, gait.png, lineage.png
und schreibt Diagnosen auf die Konsole.

Braucht kein EvoGym -- liest nur history.json, lineage.jsonl, best.npy.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")           # kein Fenster noetig
import matplotlib.pyplot as plt
import numpy as np

from evolve import BODY, decode, genome_bounds, n_actuators, action_at, Config


# ---------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------

def load_run(rundir):
    with open(os.path.join(rundir, "history.json")) as f:
        history = json.load(f)
    records = []
    with open(os.path.join(rundir, "lineage.jsonl")) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    cfg = Config()
    cfg_path = os.path.join(rundir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = Config(**json.load(f))
    best_path = os.path.join(rundir, "best.npy")
    best = np.load(best_path) if os.path.exists(best_path) else None
    if best is None:                      # Lauf noch nicht fertig -> bestes aus dem Log
        best = np.array(max(records, key=lambda r: r["fitness"])["genome"])
    return history, records, cfg, best


# ---------------------------------------------------------------------------
# Plot 1: Fitnessverlauf
# ---------------------------------------------------------------------------

def plot_fitness(history, records, path):
    fig, ax = plt.subplots(figsize=(9, 5))

    # alle je bewerteten Individuen als Punktwolke -> zeigt die Streuung,
    # die eine reine Best-Kurve verschweigt
    gens = np.array([r["gen"] for r in records])
    fits = np.array([r["fitness"] for r in records])
    ax.scatter(gens, fits, s=6, alpha=0.18, color="#888888", label="alle Nachkommen")

    g = [h["gen"] for h in history]
    ax.plot(g, [h["best"] for h in history], lw=2, color="#1f77b4", label="bestes Individuum")
    ax.plot(g, [h["mean_parents"] for h in history], lw=1.5, ls="--",
            color="#d62728", label="Mittel der Eltern")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitnessverlauf")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Gangdiagramm des besten Individuums
# ---------------------------------------------------------------------------

def plot_gait(best, cfg, path, seconds=4.0):
    n_act = n_actuators(BODY)
    freq, amp, phase, offset = decode(best, n_act)

    t = np.arange(0, seconds, cfg.dt)
    signals = np.array([action_at(best, n_act, ti) for ti in t]).T  # (n_act, T)

    # welcher Aktuator ist horizontal (3) bzw. vertikal (4)?
    kinds = BODY[(BODY == 3) | (BODY == 4)]
    labels = [f"A{i} ({'horiz' if k == 3 else 'vert'})" for i, k in enumerate(kinds)]

    fig = plt.figure(figsize=(14, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2.4, 1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2], projection="polar")   # direkt polar anlegen

    # Koerperlayout
    ax0.imshow(BODY, cmap="viridis", vmin=0, vmax=4)
    for (r, c), v in np.ndenumerate(BODY):
        if v:
            ax0.text(c, r, str(v), ha="center", va="center", color="w", fontsize=9)
    ax0.set_title("Körper (0 leer, 1 starr,\n2 weich, 3 horiz, 4 vert)", fontsize=9)
    ax0.set_xticks([]); ax0.set_yticks([])

    # Aktuatorsignale als Heatmap -> klassisches Gangdiagramm
    im = ax1.imshow(signals, aspect="auto", cmap="RdBu_r", vmin=0.6, vmax=1.6,
                    extent=[0, seconds, n_act - 0.5, -0.5])
    ax1.set_yticks(range(n_act)); ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Zeit (s, angenommen)")
    spc = 1.0 / (freq * cfg.dt)
    ax1.set_title(f"Aktuatorsignale — Frequenz {freq:.2f}, {spc:.1f} Samples/Zyklus")
    fig.colorbar(im, ax=ax1, label="Zieldehnung")

    # Phasen auf dem Kreis -> zeigt Koordination auf einen Blick
    for i, (ph, a) in enumerate(zip(phase, amp)):
        ax2.plot([ph, ph], [0, a], lw=2)
        ax2.plot(ph, a, "o", ms=5)
    ax2.set_title("Phase & Amplitude", fontsize=9, pad=14)
    ax2.set_rlim(0, max(0.5, amp.max() * 1.1))

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Ahnenlinie des besten Individuums
# ---------------------------------------------------------------------------

def ancestry(records):
    """Kette vom besten Individuum zurueck bis zur Startpopulation."""
    by_id = {r["id"]: r for r in records}
    cur = max(records, key=lambda r: r["fitness"])
    chain = [cur]
    while cur["parent"] is not None and cur["parent"] in by_id:
        cur = by_id[cur["parent"]]
        chain.append(cur)
    return chain[::-1]


def plot_lineage(records, path):
    chain = ancestry(records)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    gens = [r["gen"] for r in chain]
    ax0.plot(gens, [r["fitness"] for r in chain], "o-", color="#2ca02c", ms=4)
    ax0.set_ylabel("Fitness")
    ax0.set_title(f"Ahnenlinie des besten Individuums ({len(chain)} Vorfahren)")
    ax0.grid(alpha=0.25)

    # Wie stark aendert sich das Genom von Eltern zu Kind?
    G = np.array([r["genome"] for r in chain])
    steps = np.linalg.norm(np.diff(G, axis=0), axis=1)
    ax1.plot(gens[1:], steps, "o-", color="#9467bd", ms=4)
    ax1.set_ylabel("Mutationsschritt (L2)")
    ax1.set_xlabel("Generation")
    ax1.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnosen
# ---------------------------------------------------------------------------

def diagnose(history, records, best, cfg_dt=0.05):
    n_act = n_actuators(BODY)
    lo, hi = genome_bounds(n_act)
    freq, amp, phase, offset = decode(best, n_act)
    print("\n" + "=" * 62)
    print("DIAGNOSE")
    print("=" * 62)

    # 1) Stagnation
    bests = [h["best"] for h in history]
    peak = int(np.argmax(bests))
    since = len(bests) - 1 - peak
    print(f"Bestwert {max(bests):.3f} in Generation {history[peak]['gen']}; "
          f"seitdem {since} Generationen ohne Verbesserung.")
    if since > len(bests) * 0.4:
        print("  -> Lange Stagnation: sigma erhoehen oder Fitness verfeinern.")

    # 2) Selektionsdruck
    by_gen = {}
    for r in records:
        by_gen.setdefault(r["gen"], []).append(r["fitness"])
    rates = []
    for h in history:
        thresh = h["mean_parents"]
        kids = by_gen.get(h["gen"], [])
        if kids:
            rates.append(np.mean([k > thresh for k in kids]))
    if rates:
        print(f"Nachkommen über dem Elternmittel: {np.mean(rates)*100:.1f} % "
              f"(zuletzt {rates[-1]*100:.1f} %)")
        if np.mean(rates) < 0.08:
            print("  -> Fast alle Mutationen schaden: sigma ist zu gross.")
        elif np.mean(rates) > 0.45:
            print("  -> Fast alle Mutationen helfen: sigma ist zu klein, "
                  "die Suche kriecht.")

    # 3) Parameter an den Schranken?
    print("\nParameter des besten Individuums:")
    groups = [("Frequenz", np.array([freq]), lo[:1], hi[:1]),
              ("Amplitude", amp, lo[1:1+n_act], hi[1:1+n_act]),
              ("Offset", offset, lo[1+2*n_act:], hi[1+2*n_act:])]
    for name, vals, l, h_ in groups:
        rel = (vals - l) / (h_ - l)
        flag = ""
        if np.any(rel > 0.97):
            flag = "  <-- an der OBERSCHRANKE, Bereich erweitern"
        elif np.any(rel < 0.03):
            flag = "  <-- an der UNTERSCHRANKE"
        print(f"  {name:10s} {np.array2string(vals, precision=2, floatmode='fixed')}{flag}")

    # 4) Koordination: laufen alle Aktuatoren synchron?
    r_vec = np.abs(np.mean(np.exp(1j * phase)))
    print(f"\nPhasenkohärenz R = {r_vec:.2f} "
          f"({'synchron, kein Gang' if r_vec > 0.85 else 'phasenversetzt -> Gangmuster'})")
    spc = 1.0 / (freq * cfg_dt)
    print(f"Samples pro Gangzyklus: {spc:.1f}")
    if spc < 8:
        print("  -> Zu grob abgetastet: der Controller wird bei diesem dt "
              "verzerrt. Frequenz-Oberschranke senken.")
    if amp.max() < 0.05:
        print("  -> Amplituden nahe null: der Roboter bewegt sich praktisch nicht.")

    # 5) Kostenbilanz
    print(f"\n{len(records)} Evaluationen insgesamt.")
    print("=" * 62)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rundir")
    args = p.parse_args()

    history, records, cfg, best = load_run(args.rundir)
    plot_fitness(history, records, os.path.join(args.rundir, "fitness.png"))
    plot_gait(best, cfg, os.path.join(args.rundir, "gait.png"))
    plot_lineage(records, os.path.join(args.rundir, "lineage.png"))
    diagnose(history, records, best, cfg.dt)
    print(f"Plots geschrieben nach {args.rundir}/")


if __name__ == "__main__":
    main()

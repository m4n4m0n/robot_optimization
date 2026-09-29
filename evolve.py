"""
Meilenstein 2+3: Fester Körper, Sinus-Controller (open-loop CPG), (mu+lambda)-ES.

Nutzung:
    python evolve.py                      # Evolution starten
    python evolve.py --replay runs/xy/best.npy   # bestes Individuum anschauen
    python evolve.py --resume runs/xy     # Lauf fortsetzen

Benoetigt Python 3.10 oder aelter:  pip install evogym --upgrade
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

# ---------------------------------------------------------------------------
# 1. KOERPER  (fest, wird in Meilenstein 5 mutierbar)
# ---------------------------------------------------------------------------
# Voxel-Codes:  0 = leer   1 = starr   2 = weich
#               3 = horizontaler Aktuator   4 = vertikaler Aktuator
#
# Bewusst symmetrisch: starrer Rumpf oben, weiche Schultern,
# vertikale Aktuatoren als "Beine" links und rechts.

BODY = np.array([
    [0, 1, 1, 1, 0],
    [1, 3, 1, 3, 1],
    [2, 1, 1, 1, 2],
    [4, 3, 0, 3, 4],
    [4, 0, 0, 0, 4],
], dtype=int)

ENV_NAME = "Walker-v0"


# ---------------------------------------------------------------------------
# 2. GENOM
# ---------------------------------------------------------------------------
# Layout:  [freq, amp_0..amp_{n-1}, phase_0..phase_{n-1}, offset_0..offset_{n-1}]
# Ein Parametertripel pro Aktuator-Voxel, Frequenz global geteilt.
# Genau das macht das Encoding morphologie-agnostisch: kommt spaeter ein
# Aktuator dazu, bringt er einfach sein eigenes Tripel mit.

# EvoGym-Aktionen sind Zieldehnungen im Bereich [0.6, 1.6] (1.0 = neutral).
ACTION_LOW, ACTION_HIGH = 0.6, 1.6


def n_actuators(body: np.ndarray) -> int:
    return int(np.sum((body == 3) | (body == 4)))


def genome_bounds(n_act: int) -> tuple[np.ndarray, np.ndarray]:
    """Untere/obere Schranke pro Genom-Dimension."""
    lo = np.concatenate([[0.5], np.full(n_act, 0.0), np.full(n_act, 0.0), np.full(n_act, 0.8)])
    hi = np.concatenate([[3.0], np.full(n_act, 0.5), np.full(n_act, 2 * np.pi), np.full(n_act, 1.4)])
    return lo, hi


def random_genome(n_act: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = genome_bounds(n_act)
    return rng.uniform(lo, hi)


def decode(genome: np.ndarray, n_act: int):
    freq = genome[0]
    amp = genome[1:1 + n_act]
    phase = genome[1 + n_act:1 + 2 * n_act]
    offset = genome[1 + 2 * n_act:1 + 3 * n_act]
    return freq, amp, phase, offset


def action_at(genome: np.ndarray, n_act: int, t: float) -> np.ndarray:
    """Open-loop Controller: reine Zeitfunktion, kein Sensorfeedback."""
    freq, amp, phase, offset = decode(genome, n_act)
    a = offset + amp * np.sin(2 * np.pi * freq * t + phase)
    return np.clip(a, ACTION_LOW, ACTION_HIGH)


# ---------------------------------------------------------------------------
# 3. EVALUATION
# ---------------------------------------------------------------------------

@dataclass
class Config:
    steps: int = 500          # Simulationsschritte pro Evaluation
    dt: float = 0.05          # angenommene Sekunden pro Schritt (fuer die Sinus-Phase)
    mu: int = 6               # Eltern
    lam: int = 24             # Nachkommen pro Generation
    generations: int = 50
    sigma: float = 0.12       # Mutationsstaerke, relativ zur Bandbreite
    seed: int = 0
    eval_seed: int = 12345    # fest -> alle Individuen sehen dieselbe Startbedingung
    workers: int = 1          # Prozesse fuer die Evaluation (1 = seriell)


_ENV_CACHE = {}


def get_env(render_mode=None):
    """Env einmal pro Prozess bauen und wiederverwenden -- gym.make ist teuer."""
    key = render_mode
    if key not in _ENV_CACHE:
        import gymnasium as gym
        import evogym.envs  # noqa: F401  (registriert die Environments)
        from evogym import get_full_connectivity

        connections = get_full_connectivity(BODY)
        _ENV_CACHE[key] = gym.make(
            ENV_NAME, body=BODY, connections=connections, render_mode=render_mode
        )
    return _ENV_CACHE[key]


def evaluate(genome: np.ndarray, cfg: Config, render=False, collect_traj=False):
    """Fitness = aufsummierter Env-Reward (bei Walker-v0 im Wesentlichen x-Distanz)."""
    env = get_env("human" if render else None)
    n_act = n_actuators(BODY)

    env.reset(seed=cfg.eval_seed)
    total = 0.0
    traj = []

    for i in range(cfg.steps):
        t = i * cfg.dt
        action = action_at(genome, n_act, t)
        _obs, reward, terminated, truncated, _info = env.step(action)
        total += float(reward)

        if collect_traj:
            traj.append(robot_points(env))
        if render:
            env.render()
        if terminated or truncated:
            break

    return (total, traj) if collect_traj else total


def robot_points(env):
    """Punktmassen des Roboters als Liste [[x, y], ...] -- Rohmaterial fuer Unity."""
    base = env.unwrapped
    pos = base.object_pos_at_time(base.get_time(), "robot")  # shape (2, N)
    return np.asarray(pos).T.round(4).tolist()


def _eval_worker(args):
    """Modulebene-Wrapper, damit Pool.map ihn picklen kann."""
    genome, cfg = args
    return evaluate(genome, cfg)


def eval_batch(genomes, cfg, pool=None):
    """Bewertet eine Liste von Genomen -- parallel, aber ergebnistreu zur Reihenfolge.

    Pool.map behaelt die Eingabereihenfolge bei, deshalb bleibt der Lauf
    bei gleichem Seed exakt reproduzierbar, egal wie viele Prozesse laufen.
    """
    if pool is None:
        return [evaluate(g, cfg) for g in genomes]
    return list(pool.map(_eval_worker, [(g, cfg) for g in genomes]))


# ---------------------------------------------------------------------------
# 4. (mu + lambda)-ES
# ---------------------------------------------------------------------------

def mutate(parent: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    lo, hi = genome_bounds(len(parent) // 3)
    step = rng.normal(0.0, cfg.sigma * (hi - lo))
    child = parent + step
    # Phase ist zyklisch -> umlaufen statt abschneiden
    n_act = (len(parent) - 1) // 3
    ps, pe = 1 + n_act, 1 + 2 * n_act
    child[ps:pe] = np.mod(child[ps:pe], 2 * np.pi)
    mask = np.ones(len(child), dtype=bool)
    mask[ps:pe] = False
    child[mask] = np.clip(child[mask], lo[mask], hi[mask])
    return child


def run_evolution(cfg: Config, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    n_act = n_actuators(BODY)
    print(f"Koerper hat {n_act} Aktuatoren -> Genom mit {3 * n_act + 1} Parametern")

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    log = open(os.path.join(outdir, "lineage.jsonl"), "a")
    next_id = 0

    # Prozesspool einmal anlegen -- jeder Worker baut sein eigenes Env beim
    # ersten Task und haelt es danach im Cache.
    pool = None
    if cfg.workers > 1:
        pool = mp.get_context("spawn").Pool(cfg.workers)
        print(f"Evaluation auf {cfg.workers} Prozessen")

    # Startpopulation
    init = [random_genome(n_act, rng) for _ in range(cfg.mu)]
    population = []  # Liste von (genome, fitness, id)
    for g, fit in zip(init, eval_batch(init, cfg, pool)):
        log.write(json.dumps({"id": next_id, "parent": None, "gen": 0,
                              "fitness": fit, "genome": g.tolist()}) + "\n")
        population.append((g, fit, next_id))
        next_id += 1

    history = []
    for gen in range(1, cfg.generations + 1):
        t0 = time.time()
        # Erst alle Nachkommen erzeugen (seriell, damit der RNG-Strom
        # unabhaengig von der Worker-Zahl bleibt), dann gebuendelt bewerten.
        children, parents = [], []
        for _ in range(cfg.lam):
            pg, _pf, pid = population[rng.integers(len(population))]
            children.append(mutate(pg, cfg, rng))
            parents.append(pid)

        fitnesses = eval_batch(children, cfg, pool)

        offspring = []
        for child, pid, fit in zip(children, parents, fitnesses):
            log.write(json.dumps({"id": next_id, "parent": pid, "gen": gen,
                                  "fitness": fit, "genome": child.tolist()}) + "\n")
            offspring.append((child, fit, next_id))
            next_id += 1

        # (mu + lambda): Eltern und Kinder konkurrieren gemeinsam
        population = sorted(population + offspring, key=lambda x: -x[1])[:cfg.mu]
        log.flush()

        best = population[0][1]
        mean = float(np.mean([f for _, f, _ in population]))
        history.append({"gen": gen, "best": best, "mean_parents": mean})
        print(f"Gen {gen:3d}  best={best:8.3f}  mean={mean:8.3f}  ({time.time() - t0:.1f}s)")

        np.savez(os.path.join(outdir, "checkpoint.npz"),
                 genomes=np.array([g for g, _, _ in population]),
                 fitness=np.array([f for _, f, _ in population]),
                 gen=gen)
        with open(os.path.join(outdir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    if pool is not None:
        pool.close()
        pool.join()
    np.save(os.path.join(outdir, "best.npy"), population[0][0])
    log.close()
    print(f"\nFertig. Bestes Individuum: {population[0][1]:.3f} -> {outdir}/best.npy")
    return population[0][0]


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=str, help="Pfad zu best.npy -- anschauen statt evolvieren")
    p.add_argument("--export", type=str, help="Trajektorie als JSON fuer Unity schreiben")
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="Prozesse fuer die Evaluation (1 = seriell)")
    p.add_argument("--benchmark", action="store_true",
                   help="Eine Evaluation messen und beenden")
    args = p.parse_args()

    cfg = Config(generations=args.generations, seed=args.seed, workers=args.workers)

    if args.benchmark:
        n_act = n_actuators(BODY)
        g = random_genome(n_act, np.random.default_rng(0))
        evaluate(g, cfg)                      # Env-Aufbau nicht mitmessen
        t0 = time.time(); evaluate(g, cfg); dt = time.time() - t0
        total = cfg.generations * cfg.lam + cfg.mu
        print(f"{dt*1000:.0f} ms pro Evaluation ({cfg.steps} Schritte)")
        print(f"-> voller Lauf seriell: {total*dt/60:.1f} min, "
              f"auf {cfg.workers} Kernen ~{total*dt/60/cfg.workers:.1f} min")
        return

    if args.replay:
        genome = np.load(args.replay)
        if args.export:
            fit, traj = evaluate(genome, cfg, render=False, collect_traj=True)
            with open(args.export, "w") as f:
                json.dump({"fitness": fit, "dt": cfg.dt,
                           "body": BODY.tolist(), "frames": traj}, f)
            print(f"Fitness {fit:.3f}, {len(traj)} Frames -> {args.export}")
        else:
            fit = evaluate(genome, cfg, render=True)
            print(f"Fitness: {fit:.3f}")
        return

    outdir = args.out or os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    run_evolution(cfg, outdir)


if __name__ == "__main__":
    main()

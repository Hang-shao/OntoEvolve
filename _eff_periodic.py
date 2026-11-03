"""动态 vs 周期全量重构：配对效应量（rank-biserial）与差值 95% CI（bootstrap）。"""
import csv, itertools, json, math, random, argparse
from collections import defaultdict

random.seed(0)


def load(base):
    acc = defaultdict(dict)
    with open(f"results/{base}/scores.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            acc[(row["seed"], row["version"], row["kclass"], row["qid"])][row["system"]] = int(row["correct"])
    return acc


def systems(base):
    keys = set()
    with open(f"results/{base}/scores.csv", encoding="utf-8") as f:
        for row in itertools.islice(csv.DictReader(f), 3000):
            keys.add(row["system"])
    return keys


def wilcoxon_paired(diffs):
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 1.0, 0.0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        r = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    wp = sum(r for r, d in zip(ranks, nz) if d > 0)
    wm = sum(r for r, d in zip(ranks, nz) if d < 0)
    rb = (wp - wm) / (wp + wm) if (wp + wm) else 0.0
    t = min(wp, wm)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd == 0:
        return 1.0, rb
    z = (t - mu) / sd
    p = math.erfc(abs(z) / math.sqrt(2))
    return min(1.0, p * 2), rb


def boot_ci(vals, n=20000):
    if not vals:
        return (float("nan"),) * 2
    m = len(vals)
    ms = sorted(sum(random.choices(vals, k=m)) / m for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


CLASSES = ["stable", "new", "redefined"]
NAME = {"stable": "stable", "new": "new", "redefined": "redef"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periodic", default="exp_qa_periodic")
    ap.add_argument("--baseline", default="exp_qa")
    ap.add_argument("--tag", default="P=5")
    a = ap.parse_args()

    print("systems:", systems(a.baseline), systems(a.periodic))
    ld, lp = load(a.baseline), load(a.periodic)
    print(f"### {a.tag}  dynamic vs periodic (paired by item)")
    for k in CLASSES:
        for v in ("a", "b"):
            item, seed_diff = [], defaultdict(lambda: [0, 0, 0])
            for key, dd in ld.items():
                s, vv, kk, _q = key
                if kk != k or vv != v:
                    continue
                pp = lp.get(key)
                if not pp or "dynamic" not in dd or "periodic" not in pp:
                    continue
                item.append(dd["dynamic"] - pp["periodic"])
                c = seed_diff[s]
                c[0] += dd["dynamic"] - pp["periodic"]
                c[1] += 1
            if not item:
                continue
            p, rb = wilcoxon_paired(item)
            per_seed = [c[0] / c[1] for c in seed_diff.values()]
            mean = sum(per_seed) / len(per_seed)
            lo, hi = boot_ci(per_seed)
            print(f"{NAME[k]:>5} v{v}: delta={mean:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
                  f"rb={rb:+.3f} p={p:.3g} n={len(item)} seeds={len(per_seed)}")


if __name__ == "__main__":
    main()

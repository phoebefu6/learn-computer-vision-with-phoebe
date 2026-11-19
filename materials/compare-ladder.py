"""Compare the python canon with the node canon BY VALUE (never a text diff).

    python3 -I materials/compare-ladder.py canon-python.json canon-node.json [--plant]

Integers (correct counts, confusion cells, class counts, curve points) must match exactly;
floats must match to 1e-6. --plant perturbs one python float by 0.1 percent and one count by 1
first, and the script must then FAIL: a comparer that cannot fail proves nothing.
"""
import json
import sys


def walk(a, b, path, bad, counter):
    if path.endswith("._ms") or path == "._ms":
        return
    if isinstance(a, dict):
        keys = set(a) | set(b)
        for k in sorted(keys):
            if k == "_ms":
                continue
            if k not in a or k not in b:
                bad.append(f"{path}.{k}: missing on one side")
                continue
            walk(a[k], b[k], f"{path}.{k}", bad, counter)
    elif isinstance(a, list):
        if len(a) != len(b):
            bad.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", bad, counter)
    else:
        counter[0] += 1
        if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool):
            if a != b:
                bad.append(f"{path}: {a} vs {b}")
        elif abs(float(a) - float(b)) > 1e-6:
            bad.append(f"{path}: {a} vs {b}")


def main():
    py = json.load(open(sys.argv[1]))
    nd = json.load(open(sys.argv[2]))
    if "--plant" in sys.argv:
        py["rungs"]["conv"]["acc"] *= 1.001
        py["curve"]["random"][3] += 1
    bad, counter = [], [0]
    walk(py, nd, "", bad, counter)
    print(f"compared {counter[0]} values, {len(bad)} mismatches")
    for b in bad[:20]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

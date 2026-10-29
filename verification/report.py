"""Deterministischer Bericht der Code-Verifikation: verification/REPORT.md + report.json (sortiert, ohne Zeitstempel)."""
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
CATS = {"test_golden": "Golden-Regression", "test_properties": "Property/Fuzz (Hypothesis, derandomisiert)",
        "test_differential": "Differential/Metamorph", "test_hyp_stats": "Statistik-Bausteine (Experimentcode)"}


def junit():
    root = ET.parse(HERE / "junit.xml").getroot()
    by = defaultdict(Counter)
    for tc in root.iter("testcase"):
        mod = tc.get("classname", "").split(".")[-1]
        cat = CATS.get(mod, "Unit")
        st = "failed" if tc.find("failure") is not None or tc.find("error") is not None else \
             "skipped" if tc.find("skipped") is not None else "passed"
        by[cat][st] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(by.items())}


def mutmut():
    p = HERE / "mutmut_results.txt"
    if not p.exists():
        return {}
    by = defaultdict(Counter)
    for line in p.read_text().splitlines():
        m = re.match(r"\s*sbench\.([a-z_]+)\.\S+:\s*(\w[\w ]*)$", line)
        if m:
            by[m.group(1)][m.group(2).strip()] += 1
    out = {}
    for mod, c in sorted(by.items()):
        killed = c.get("killed", 0) + c.get("timeout", 0)
        total = sum(c.values()) - c.get("skipped", 0)
        out[mod] = {**dict(sorted(c.items())), "score": round(killed / total, 3) if total else None}
    return out


def main():
    rep = {"tests": junit(), "mutation": mutmut()}
    (HERE / "report.json").write_text(json.dumps(rep, indent=1, sort_keys=True))
    L = ["# Code-Verifikation", "", "Erzeugt mit `verification/run.sh`. Prüft den Code, nicht die Modellaussagen "
         "(die stehen getrennt in `experiments/`).", "", "## Tests", "", "| Kategorie | bestanden | fehlgeschlagen |",
         "| --- | --- | --- |"]
    for k, v in rep["tests"].items():
        L.append(f"| {k} | {v.get('passed', 0)} | {v.get('failed', 0)} |")
    L += ["", "## Mutation-Tests (mutmut)", "", "Score = getötet / alle bewerteten Mutanten; Abstürze (segfault) zählen nicht als getötet.", "",
          "| Modul | getötet | überlebt | ohne Test | Absturz | Score |", "| --- | --- | --- | --- | --- | --- |"]
    for k, v in rep["mutation"].items():
        L.append(f"| {k} | {v.get('killed', 0) + v.get('timeout', 0)} | {v.get('survived', 0)} | "
                 f"{v.get('no tests', 0)} | {v.get('segfault', 0)} | {v.get('score')} |")
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

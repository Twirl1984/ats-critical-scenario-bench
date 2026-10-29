"""Ergebnisse als zusätzliche Blätter in die Arbeitsmappe 'Sensormodelle_Gesamtkette.xlsx' schreiben."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .suite import ENVS, SUITES

HEAD = PatternFill("solid", fgColor="DDE6F0")


def _sheet(wb, title, header, rows, widths=None, note=None):
    if title in wb.sheetnames:
        del wb[title]
    ws = wb.create_sheet(title)
    ws.append(header)
    for c in ws[1]:
        c.font, c.fill = Font(bold=True), HEAD
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for r in rows:
        ws.append(r)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for i, w in enumerate(widths or [], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    if note:
        ws.append([])
        ws.append([note])
    ws.freeze_panes = "A2"
    return ws


def export(src_xlsx: str, out_xlsx: str, results="results_suite"):
    st = json.loads(Path(results, "suite_study.json").read_text())
    tr = json.loads(Path(results, "trace_study.json").read_text())
    wb = openpyxl.load_workbook(src_xlsx)
    f = lambda x, d=1: None if x is None else round(x, d)

    # 1 Traceability
    rows = [[t["req_id"], t["title"], t["risk"], ", ".join(t["test_scenarios"]), json.dumps(t["thresholds"], ensure_ascii=False) if t["thresholds"] else "",
             t["bench"], t["phenomena"], t["coverage"], t["note"]] for t in tr["trace"]]
    rows += [[n.split(" ")[0], n, "", "", "", "–", "–", "nicht abbildbar", "kein Quer-/Fahrer-/Übernahmemodell in der 1D-Kette"] for n in tr["not_covered"]]
    _sheet(wb, "Traceability R157", ["Anforderung", "Titel", "Risiko", "Testszenarien (Quelle: r157_alks.json)", "Schwellen", "Bench-Szenario",
                                     "Sensorphänomene", "Abdeckung", "Bemerkung"], rows, [14, 34, 9, 34, 30, 28, 28, 14, 50],
           "Mapping Anforderung→Szenario ist Ingenieurentscheidung; Anforderungstexte aus dem compliance-Repo (r157_alks.json), nicht gegen UN R157 im Original geprüft.")

    # 2 Abdeckung
    sn = list(SUITES)
    rows = []
    for r in st["rows"]:
        rows.append([r["problem"], r["env"], r["n_valid_pool"]] + [f(r["suites"][s]["rate"], 2) for s in sn])
    _sheet(wb, "Abdeckung Szenario x Sensor", ["Szenario (Prior)", "Umgebung", "gültige Szenarien im Pool"] + [f"Fehlerrate % – {s}" for s in sn],
           rows, [18, 14, 12] + [14] * len(sn),
           f"Fehlerrate = Anteil gültiger Kollisionen (plausibel + vermeidbar) an {st['n']} Prior-Szenarien. 'ideal' = Logikgrenze der AEB-Stufung. "
           "Prior = Autobahn-Wertebereich (15–40 m/s), nicht ODD-beschränkt. Alle Sensorparameter sind Platzhalter (Blatt 'Annahmen').")

    # 3 Ursachen / Upgrade
    rows = []
    for r in st["rows"]:
        for s, v in r["suites"].items():
            if s == "ideal" or v["excess"] <= 0.5:
                continue
            ab = v.get("ablation_pp") or {}
            top = sorted(ab.items(), key=lambda kv: -kv[1])[:2]
            rows.append([r["problem"], r["env"], s, f(v["rate"], 2), f(v["excess"], 2),
                         "; ".join(f"{k} (−{x:.1f} pp)" for k, x in top if x > 0) or "nicht ablationsgeprüft",
                         v.get("range_factor_needed", "")])
    _sheet(wb, "Ursachen und Upgrade-Bedarf", ["Szenario", "Umgebung", "Suite", "Fehlerrate %", "Mehrfehler ggü. ideal (pp)",
                                               "dominante Phänomene (Ablation, vermiedene Fehler)", "nötiger Reichweitenfaktor (r50, range_max)"],
           rows, [18, 14, 20, 11, 14, 60, 18],
           "Ablation: Phänomen wird für alle Sensoren der Suite idealisiert, nur Szenarien mit Fehler werden neu gerechnet. Reichweitenfaktor '>3.0' = Reichweite allein genügt nicht.")

    # 4 Budgets
    rows = []
    for b in tr["budgets"]:
        q = lambda k: [f(b[k][x], 2) for x in ("p05", "p50", "p95")]
        rows.append([b["problem"], b["n_valid"], *q("dT_quantiles"), *q("ds_quantiles"), *q("R_quantiles"),
                     f(100 * b["share_dT_below"]["0.2"], 1), f(100 * b["share_R_above"]["46"], 1), f(100 * b["share_R_unavoidable_by_logic"], 1)])
    _sheet(wb, "Budgets Rueckrechnung", ["ODD-Szenario", "n gültig", "ΔT* p05 [s]", "ΔT* p50", "ΔT* p95", "Δs* p05 [m]", "Δs* p50", "Δs* p95",
                                         "R* p05 [m]", "R* p50", "R* p95", "Anteil ΔT*<0,2 s [%]", "Anteil R*>46 m [%]", "R* nicht erreichbar (Logik) [%]"],
           rows, [24, 9] + [11] * 12,
           "ΔT*: tolerierbare Zusatzlatenz, Δs*: tolerierbare Distanzüberschätzung, R*: nötige Erkennungsreichweite (idealer Sensor, 0,2 s Latenz). "
           "p05/p95 nur über endliche Werte; inf = nie Kollision im Suchbereich. 46 m = R157-7.1.1 min_forward_detection_range_m.")
    rc = tr["range_curve"]
    _sheet(wb, "Reichweite Stehendes Hindernis", ["v_ego [m/s]", "R* ideal-Sensor [m]", "R* mit 0,2 s Latenz [m]", "Physik-Minimum [m]"],
           [[f(r["v_ego"], 1), f(r["R_ideal"], 1), f(r["R_lat"], 1), f(r["R_physik"], 1)] for r in rc], [12, 18, 20, 18],
           "µ=0,8. Physik-Minimum = v·0,3 s + v²/(2µg) (0,2 s Aktuatorlag + 0,1 s Reaktion, volle Bremsung). R* (ACC im Beharrungszustand) liegt nahe am Physik-Minimum; die Latenz (0,2 s) kostet im Mittel ca. 25 % mehr Reichweite.")

    # 5 Annahmen
    rows = []
    for name, suite in SUITES.items():
        for sp in suite:
            if name in ("ideal",) or name == sp.name or name.startswith("radar+kamera+lidar"):
                d = asdict(sp)
                rows.append([sp.name] + [d[k] for k in ("hz", "latency", "range_max", "r50", "width", "pd0", "sig0", "sig1", "sig2", "sig_rate", "ghost", "p_det", "stat_mult")])
    seen, uniq = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0]); uniq.append(r)
    uniq += [[], ["Umgebung", "Faktor auf r50 je Sensorart"]] + [[e.name, json.dumps(e.mult)] for e in ENVS.values()]
    _sheet(wb, "Annahmen Stufe 1", ["Sensor", "Takt [Hz]", "Latenz [s]", "range_max [m]", "r50 [m]", "Breite w [m]", "Pd0", "σ0 [m]", "σ1 [-]",
                                    "σ2 [1/m]", "σ Rate [m/s]", "Geisterziel/Scan", "p_det", "Stationär-Pd-Faktor"], uniq, [12] + [11] * 13,
           "ALLE WERTE SIND PLATZHALTER (Annahmen des Assistenten), nicht aus Datenblatt/Messung/Literatur belegt. Aussagen sind strukturell, die Größenordnung folgt erst nach Identifikation (Stufe 1).")
    wb.save(out_xlsx)


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "results_suite")

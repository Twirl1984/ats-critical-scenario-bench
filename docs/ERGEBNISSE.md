# Ergebnisse Abdeckungsstudie (Sensor-Stufe 1) und Rückrechnung

Reproduzierbar mit `sbench suitestudy` und `sbench trace` (Rohdaten: `results_suite/*.json`, Excel: `results_suite/Sensormodelle_Gesamtkette_Abdeckung.xlsx`).

> **Wichtig:** Alle Sensorparameter (`src/sbench/suite.py`) sind **Platzhalter-Annahmen**, nicht aus Datenblatt, Messung oder Literatur belegt.
> Die Aussagen sind strukturell (welches Phänomen begrenzt), die Größenordnungen folgen erst nach Identifikation der Parameter aus Messdaten.

## Was die Kette pro Szenario abbildet

| Szenario | Abgebildet | Sensorphänomene, die tragen | Nicht abgebildet |
|---|---|---|---|
| Bremsender Vorausfahrender | ja | Reichweite/Pd, Rate (Doppler), Latenz | Mehrziel, Verdeckung |
| Cut-in | teilweise | Sichtfeld (Erkennungszeitpunkt), Latenz | echte Quergeometrie, Winkelfehler |
| Stehendes Hindernis (Stauende) | ja | Stationärfilter (Radar), Reichweite, Latenz | Reflexions-/Multipath-Effekte |
| Cut-out/Reveal, Fußgänger, Kurve, Querführung | nein | – | 1D-Längsmodell |

## Befunde (Prior = Autobahn-Wertebereich, 800 Szenarien je Zelle, nur gültige Kollisionen)

- **Bremsender Vorausfahrender:** Radar allein deckt ab (0 %). Kamera/Lidar allein kippen mit Nebel/Regen stark (Nebel: Kamera 20 %, Lidar 35 %). Jede Fusion mit Radar → 0 %.
- **Stehendes Hindernis:** Der Stationärfilter des Radars dominiert (Radar allein 72 %, davon ca. 54 pp durch den Filter). Kamera+Radar+Lidar 12 % (ideal: 8 % = Logikgrenze). Dort bringt mehr Latenzreduktion am meisten (−5,9 pp), dann Stationärfilter (−2,8 pp).
- **Cut-in:** Ein früheres Erkennen (Sichtfeld) und Latenz zählen mehr als Reichweite; Reichweitenfaktor allein genügt nirgends.
- **Phantombremsen:** Mit konservativer Min-Fusion steigt die Rate mit der Sensorzahl (Radar 5,0 %, Radar+Kamera+Lidar 7,2 %, Lidar 0 %) – bei angenommener Geisterzielrate ohne Tracker. Hier wäre 2-aus-3-Fusion der nächste Schritt.

## Rückrechnung (ODD ≤ 60 km/h, Lead-Verzögerung ≤ 6 m/s², R157-Anforderungsdatei)

- **Stehendes Hindernis, µ=0,8:** nötige Erkennungsreichweite R* = 22 m (idealer Sensor) bzw. 29 m (0,2 s Latenz) bei 16,7 m/s; Physik-Minimum 23 m. R157-7.1.1 fordert 46 m – im ODD nicht der bindende Wert (p95 von R* = 25 m bei 0,2 s Latenz).
- **Cut-in ODD:** 3 % der gültigen Szenarien sind durch die AEB-Logik nicht vermeidbar (obwohl per Oracle vermeidbar), 2,5 % haben ΔT* < 0,2 s. Das sind die unfallkritischen Szenarien für die Latenz-Auslegung.
- Tolerierbare Zusatzlatenz ΔT* im Median 0,9–1,3 s für bremsenden Vorausfahrenden und Stehendes Hindernis.

## Bekannte Grenzen

- Sensor-Fehler unabhängig je Scan (Abhängigkeiten ignoriert), kein Tracker, Fusion = Minimum.
- Geisterziele: 1 Scan, ohne Bestätigung.
- Prior der Studie = Autobahn-Wertebereich (15–40 m/s), die Rückrechnung = ODD ≤ 60 km/h.
- Das Mapping R157-Anforderung → Szenario ist Ingenieurentscheidung, nicht aus dem Regelwerk abgeleitet.
- Die AEB-Stufe 1 (−4 m/s²) liegt unter der EM-Schwelle 5 m/s² (R157-5.3.1): Interpretation gegen das Regelwerk prüfen.

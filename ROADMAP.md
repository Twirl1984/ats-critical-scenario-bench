# Roadmap (status 29 Oct 2026)

Open work, in the order I would do it. Nothing here is implemented yet unless marked.

## 1. Traceability requirement → system element → scenario → test artefact

Basis: the requirement schema of the `digital_loop` pipeline (req_id, clause, ODD parameters, test scenarios, thresholds, pass criteria; currently UN R157) and its credibility assessment.

- **Scenario levels** after Menzel, Bagschik and Maurer (IEEE IV 2018): functional → logical (`Problem` with parameter ranges) → concrete (sampled `u` with seed).
- **ODD tagging** along the PEGASUS 6-layer model (road, infrastructure, temporary changes, objects, environment, digital information). [not yet checked against a primary PEGASUS source]
- **Process and metadata** along the Credible Simulation Process and SSP Traceability 1.0.0 (Modelica Association; drafted in SET Level): phases analysis → requirements → design → implementation → execution → evaluation → fulfilment; resource kinds requirement, specification, model, parameter, testcase, result, rationale.
- **Applicable frame for a driverless shuttle:** Implementing Regulation (EU) 2022/1426 (Annex II performance, Annex III assessment incl. Part 4 credibility of the virtual toolchain), ISO 22737, ISO 21448. UN R157 only as an analogue (ALKS: motorway, no pedestrians).
- Integrity tests: no dangling IDs, every logical scenario carries ODD tags, every requirement is covered or carries an explicit "not covered" reason.

## 2. AI-Act side track: validation of the learned sub-model

For vehicles type-approved under Regulation 2019/2144 the AI Act applies only via Art. 6(1), Art. 102–109 and Art. 112 (AI Act Art. 2(2)); Art. 15 (accuracy, robustness, cybersecurity) enters through the amendment of 2019/2144 Art. 11 by AI Act Art. 109, not directly. Planned: map the learned slip map to Art. 9/10/15-style evidence (declared accuracy metric, robustness under perturbation = experiments H1/H1b, data governance of the training set).

## 3. Localisation over mobile networks: fault injection, spoofing, jamming

Cellular (5G) positioning and GNSS with network assistance as an additional source in the world model. Spoofing = correlated, adversarial bias that drifts within plausibility limits; jamming = loss with age limit and degraded fallback. This is where gap 1 (common-cause errors) and gap 6 (localisation) of the main document meet a security question.

## 4. Differential test against the Rust world model

`world.py` is a port of `world.rs` (private see–think–act stack). A small standalone harness that replays identical event traces through both is planned; the property tests already found one boundary difference (float time vs. `Duration`), fixed with a 1 ns tolerance.

## 5. Model fixes from the gap analysis

Oracle limited to the platform brake limit (hook `oracle_a_max` exists, default unchanged), 2-out-of-3 fusion, correlated sensor errors, measured latency distributions.

## 6. Where the learned part becomes binding (exploratory, follow-up to H1b)

Sweep the IMU-bias tolerance and the sensitivity range to locate the threshold at which the Jacobian of the learned slip map starts to cost latency margin; a confirmatory claim needs its own pre-registration.

## 7. Measurement data: possible sources (not contacted, nothing promised)

- **Schaeffler:** announced in 2023 with VDL Groep a Level-4 shuttle for public transport (Mobileye driving system; street tests planned from 2025, series from 2030 in Born). No 2025/26 status found, neither progress nor termination; to be asked directly. Schaeffler supplies drive, brake/steer-by-wire and corner-module technology, which is exactly where gap 4 (actuator: dead time, pressure build-up) and the slip model of this bench sit. Schaeffler is also the industry partner of CoDeFeL (corrosion work), which could be a route for a joint use case.
- **2getthere / ZF:** timestamps along sensor → fusion → brake, sensor vs. reference, braking ramps (see main document; external request, NDA and publication rights to clarify).

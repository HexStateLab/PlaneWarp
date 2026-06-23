============================================================================================
 Comprehensive plane_warp Bench — STIM circuit-level noise, 5 rounds
 Panel per circuit:  plain --decode   vs   --cap-auto at two calibrated rates
 LER = logical-observable error over the shown shots, [Wilson 95% CI].  * = best setting.
============================================================================================

── CZ-based  (basis-matched) ──  grid 6×6 (n=36), p_g=5.0e-04, p_meas=1e-03, 5 rounds, 2000 shots
     baseline (no correction)         1.65%  [ 1.18,  2.31]
  *  plain                            0.20%  [ 0.08,  0.51]  corrects
     cap-auto 0.015 (cap=2)           0.25%  [ 0.11,  0.58]  corrects
     cap-auto 0.030 (cap=3)           0.20%  [ 0.08,  0.51]  corrects

── phenomenological (data+meas) ──  grid 6×6 (n=36), p_g=8.0e-04, p_meas=1e-03, 5 rounds, 2000 shots
     baseline (no correction)         1.10%  [ 0.73,  1.66]
  *  plain                            0.00%  [ 0.00,  0.19]  corrects
     cap-auto 0.015 (cap=2)           0.00%  [ 0.00,  0.19]  corrects
     cap-auto 0.030 (cap=3)           0.00%  [ 0.00,  0.19]  corrects

── correlated-pair ──  grid 6×6 (n=36), p_g=5.0e-04, p_meas=1e-03, 5 rounds, 2000 shots
     baseline (no correction)         1.45%  [ 1.01,  2.07]
  *  plain                            0.20%  [ 0.08,  0.51]  corrects
     cap-auto 0.015 (cap=2)           0.20%  [ 0.08,  0.51]  corrects
     cap-auto 0.030 (cap=3)           0.20%  [ 0.08,  0.51]  corrects

── asymmetric (10x hot sub) ──  grid 6×6 (n=36), p_g=5.0e-04, p_meas=1e-03, 5 rounds, 1000 shots
     baseline (no correction)        13.00%  [11.06, 15.23]
  *  plain                            4.30%  [ 3.21,  5.74]  corrects
     cap-auto 0.015 (cap=2)           4.60%  [ 3.47,  6.08]  corrects
     cap-auto 0.030 (cap=3)           4.30%  [ 3.21,  5.74]  corrects

── CNOT-based (basis-mismatched) ──  grid 6×6 (n=36), p_g=5.0e-04, p_meas=1e-03, 5 rounds, 1000 shots
     baseline (no correction)         1.90%  [ 1.22,  2.95]
     plain                           44.10%  [41.05, 47.19]  WORSE
  *  cap-auto 0.015 (cap=2)           2.00%  [ 1.30,  3.07]  =baseline
     cap-auto 0.030 (cap=3)           2.90%  [ 2.03,  4.13]  =baseline

── CZ-based  (basis-matched) ──  grid 20×20 (n=400), p_g=4.0e-04, p_meas=1e-03, 5 rounds, 300 shots
     baseline (no correction)         3.67%  [ 2.06,  6.45]
  *  plain                            1.67%  [ 0.71,  3.84]  =baseline
     cap-auto 0.015 (cap=11)          1.67%  [ 0.71,  3.84]  =baseline
     cap-auto 0.030 (cap=19)          1.67%  [ 0.71,  3.84]  =baseline

── CNOT-based (basis-mismatched) ──  grid 20×20 (n=400), p_g=2.0e-04, p_meas=1e-03, 5 rounds, 300 shots
     baseline (no correction)         2.33%  [ 1.13,  4.74]
     plain                           53.33%  [47.68, 58.90]  WORSE
  *  cap-auto 0.015 (cap=11)          2.33%  [ 1.13,  4.74]  =baseline
     cap-auto 0.030 (cap=19)          2.33%  [ 1.13,  4.74]  =baseline

============================================================================================
 SUMMARY — best setting per circuit
============================================================================================
  circuit                           grid     p_g  baseline  best setting               LER  result
  ─────────────────────────────── ────── ───────  ────────  ────────────────────── ───────  ──────────
  CZ-based  (basis-matched)          6×6 5.0e-04     1.65%  plain                    0.20%  corrects
  phenomenological (data+meas)       6×6 8.0e-04     1.10%  plain                    0.00%  corrects
  correlated-pair                    6×6 5.0e-04     1.45%  plain                    0.20%  corrects
  asymmetric (10x hot sub)           6×6 5.0e-04    13.00%  plain                    4.30%  corrects
  CNOT-based (basis-mismatched)      6×6 5.0e-04     1.90%  cap-auto 0.015 (cap=2)   2.00%  =baseline
  CZ-based  (basis-matched)        20×20 4.0e-04     3.67%  plain                    1.67%  =baseline
  CNOT-based (basis-mismatched)    20×20 2.0e-04     2.33%  cap-auto 0.015 (cap=11)   2.33%  =baseline

────────────────────────────────────────────────────────────────────────────────────────────
 How to read this
────────────────────────────────────────────────────────────────────────────────────────────
 • Basis-matched / trustworthy syndromes (CZ, phenom, correlated, asymmetric):
   plain decode corrects below baseline. The cap only ever abstains here, so it
   ties or slightly trails plain — engaging it on clean noise is a mild tax.
 
 • Basis-mismatched syndrome (CNOT): plain decode is catastrophic (~46%) because it
   trusts a syndrome that reports the wrong basis. cap-auto recognises the implausibly
   heavy correction and abstains, recovering to ≈baseline. That is damage control, not
   correction: a useless syndrome cannot be turned into uplift, so 'best' = baseline.
 
 • Net: the decoder corrects wherever the syndrome is honest, and the cap is the safety
   net that stops it being fooled where the syndrome is not. Gate the cap on a trust
   signal; never leave it on for data-dominated noise.
 
 • Scope: this harness is single-frame — it decodes the last round's syndrome against the
   final data observable. That under-serves measurement-dominated noise (phenom), where the
   right tool is a spacetime/multi-round decode over the full syndrome history; testing that
   fairly needs a detector-based harness and is out of scope here.

# Values copied from the verified v9 (paper_final_v9) and v10 (paper_final_v10) analysis tables, alpha = 0.05 unless stated.
BB = ["ConvNeXt-T", "ResNet-50", "DINOv2-S"]
# worst-scenario miss rate per group (v9 TA2), order of methods below
METHODS = ["Fixed threshold", "Source CRC", "Weighted CRC", "Unlabelled correction", "Ordinal sets", "Entropy abstention", "PAC label-shift sets", "Always label", "SHARC (CRC)", "SHARC (exact)"]
WORST = {  # group -> backbone -> list aligned with METHODS
 "Prevalence shift": {"ConvNeXt-T": [.049, .048, .070, .045, .044, .047, .087, .042, .048, .047],
                      "ResNet-50":  [.066, .049, .068, .048, .073, .066, .126, .046, .049, .049],
                      "DINOv2-S":   [.063, .049, .061, .046, .048, .063, .138, .045, .049, .049]},
 "Appearance shift": {"ConvNeXt-T": [.233, .236, .101, .662, .197, .233, .520, .046, .049, .038],
                      "ResNet-50":  [.339, .289, .111, .182, .347, .325, .857, .048, .048, .036],
                      "DINOv2-S":   [.662, .624, .041, .038, .491, .610, .821, .046, .046, .038]},
 "Real sites": {"ConvNeXt-T": [.048, .048, .075, .034, .046, .042, .058, .045, .045, .024],
                "ResNet-50":  [.078, .058, .071, .050, .092, .078, .095, .045, .045, .024],
                "DINOv2-S":   [.074, .063, .083, .052, .064, .074, .083, .048, .048, .024]},
}
# controlled scenarios, ConvNeXt (v9 E1), miss by severity 0..3 (0 = no shift)
SEV = {
 "Blur":         {"Source CRC": [.048, .116, .230, .236], "Weighted CRC": [.044, .050, .075, .039], "Unlabelled correction": [.045, .070, .269, .662], "SHARC (CRC)": [.047, .045, .042, .040], "SHARC (exact)": [.047, .022, .021, .020]},
 "Contrast loss":{"Source CRC": [.048, .065, .082, .112], "Weighted CRC": [.044, .028, .015, .019], "Unlabelled correction": [.045, .031, .023, .048], "SHARC (CRC)": [.047, .043, .041, .043], "SHARC (exact)": [.047, .023, .020, .020]},
 "Darkening":    {"Source CRC": [.048, .051, .061, .083], "Weighted CRC": [.044, .036, .036, .053], "Unlabelled correction": [.045, .036, .040, .075], "SHARC (CRC)": [.047, .043, .041, .043], "SHARC (exact)": [.047, .022, .019, .020]},
 "Colour cast":  {"Source CRC": [.048, .059, .085, .117], "Weighted CRC": [.044, .021, .001, .001], "Unlabelled correction": [.045, .025, .002, .003], "SHARC (CRC)": [.047, .043, .043, .042], "SHARC (exact)": [.047, .021, .020, .020]},
}
# v10 deployment grid, ConvNeXt: rows prevalence 0.4, 1, 3; columns none + 12 degradations
COLS = ["none", "blur1", "blur2", "blur3", "contrast1", "contrast2", "contrast3", "dark1", "dark2", "dark3", "colorcast1", "colorcast2", "colorcast3"]
SRC_MISS = {  # source-only miss rate does not depend on prevalence (same degradation)
 "none": .049, "blur1": .119, "blur2": .235, "blur3": .244, "contrast1": .067, "contrast2": .084, "contrast3": .115,
 "dark1": .053, "dark2": .063, "dark3": .086, "colorcast1": .062, "colorcast2": .087, "colorcast3": .120}
FLAG = {0.4: {"none": .016}, 1.0: {"none": .032}, 3.0: {"none": .030}}   # all degraded cells flagged in 100% of runs
LABELS = {0.4: {"none": 5.5, "blur1": 352.8, "blur2": 353.6, "blur3": 351.4, "contrast1": 353.8, "contrast2": 351.0, "contrast3": 350.6, "dark1": 353.8, "dark2": 352.6, "dark3": 352.9, "colorcast1": 348.7, "colorcast2": 354.4, "colorcast3": 352.3},
          1.0: {"none": 6.6, "blur1": 200.1, "blur2": 200.7, "blur3": 200.3, "contrast1": 200.9, "contrast2": 200.2, "contrast3": 200.8, "dark1": 201.1, "dark2": 201.9, "dark3": 201.0, "colorcast1": 199.2, "colorcast2": 201.2, "colorcast3": 202.4},
          3.0: {"none": 3.0, "blur1": 104.2, "blur2": 103.5, "blur3": 102.7, "contrast1": 103.4, "contrast2": 103.0, "contrast3": 103.2, "dark1": 104.4, "dark2": 104.3, "dark3": 103.0, "colorcast1": 104.4, "colorcast2": 104.1, "colorcast3": 103.5}}
# v10 by setting, ConvNeXt (TD2): labels_mean, miss, refer
SETTINGS = ["No shift", "Prevalence only", "Appearance", "Mixed", "Concept"]
POL = {
 "Source only":      {"labels": [0, 0, 0, 0, 0],                  "miss": [.049, .049, .111, .111, .125], "refer": [.430, .470, .431, .466, .430]},
 "SHARC":            {"labels": [6.6, 4.3, 200.8, 228.0, 4.5],    "miss": [.049, .049, .043, .043, .122], "refer": [.433, .471, .625, .654, .437]},
 "Always label":     {"labels": [200.6, 228.6, 200.8, 228.0, 158.3], "miss": [.044, .042, .043, .043, .043], "refer": [.494, .532, .625, .654, .707]},
 "Fixed 100 labels": {"labels": [100, 100, 100, 100, 100],       "miss": [.034, .021, .034, .021, .035], "refer": [.620, .821, .725, .867, .766]},
}
# label budget (v9 TA1b real sites, TA1c prevalence), alpha -> list of (prevalence, mean units inspected)
BUDGET = {0.02: [(.406, 282.1), (.449, 253.3), (.224, 509.8), (.464, 313.4), (.104, 702.2)],
          0.05: [(.406, 111.9), (.449, 100.1), (.500, 88.9), (.626, 71.7), (.306, 146.3), (.224, 200.7), (.464, 102.4), (.104, 351.2)],
          0.10: [(.406, 55.1), (.449, 48.9), (.500, 43.4), (.626, 34.8), (.306, 72.2), (.224, 99.1), (.464, 48.6), (.104, 191.0)]}

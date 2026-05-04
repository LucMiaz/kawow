# Ionization-Stratified Model Performance on S01 and S02

This report stratifies the S01 logP/logKow and S02 logKoa benchmark sets into approximate acid, neutral, and basic subgroups using rule-based SMARTS detection with rough literature-like pKa anchors. The pKa values are approximate screening estimates for subgrouping, not formal microstate pKa predictions.

## Subgroup Counts

| Dataset | Ionization class | N |
|---|---:|---:|
| S01 | acid | 309 |
| S01 | basic | 575 |
| S01 | neutral | 2486 |
| S02 | acid | 53 |
| S02 | basic | 199 |
| S02 | neutral | 1731 |

## Performance Summary

| Dataset | Endpoint | Ionization class | Best model by RMSE | RMSE | R2 |
|---|---|---|---|---:|---:|
| S01 | logKow | acid | smarts_mixed | 0.435 | 0.896 |
| S01 | logKow | basic | smarts_mixed | 0.406 | 0.949 |
| S01 | logKow | neutral | smarts_mixed | 0.398 | 0.966 |
| S02 | logKoa | acid | smarts_mixed | 0.664 | 0.921 |
| S02 | logKoa | basic | smarts_mixed | 0.675 | 0.956 |
| S02 | logKoa | neutral | smarts_mixed | 0.508 | 0.971 |

## S01 logKow

| Ionization class | Model | N | RMSE | MAE | Bias | R2 |
|---|---|---:|---:|---:|---:|---:|
| acid | smarts_mixed | 308 | 0.435 | 0.330 | -0.040 | 0.896 |
| acid | mqg | 308 | 0.470 | 0.353 | 0.107 | 0.878 |
| acid | kawow | 309 | 0.576 | 0.462 | -0.255 | 0.823 |
| acid | smarts | 308 | 0.688 | 0.578 | 0.405 | 0.739 |
| basic | smarts_mixed | 546 | 0.406 | 0.308 | -0.005 | 0.949 |
| basic | kawow | 575 | 0.687 | 0.523 | -0.008 | 0.869 |
| basic | mqg | 546 | 0.749 | 0.553 | 0.330 | 0.827 |
| basic | smarts | 546 | 1.000 | 0.699 | 0.075 | 0.692 |
| neutral | smarts_mixed | 2486 | 0.398 | 0.306 | 0.006 | 0.966 |
| neutral | kawow | 2461 | 0.623 | 0.446 | 0.042 | 0.917 |
| neutral | mqg | 2484 | 0.734 | 0.556 | -0.100 | 0.884 |
| neutral | smarts | 2486 | 0.742 | 0.504 | 0.106 | 0.881 |

## S02 logKoa

| Ionization class | Model | N | RMSE | MAE | Bias | R2 |
|---|---|---:|---:|---:|---:|---:|
| acid | smarts_mixed | 53 | 0.664 | 0.506 | -0.270 | 0.921 |
| acid | kawow | 53 | 0.880 | 0.665 | -0.548 | 0.861 |
| acid | mqg | 53 | 0.914 | 0.765 | -0.728 | 0.850 |
| acid | smarts | 53 | 1.752 | 1.632 | -1.615 | 0.450 |
| basic | smarts_mixed | 199 | 0.675 | 0.474 | -0.005 | 0.956 |
| basic | mqg | 199 | 0.705 | 0.529 | -0.104 | 0.952 |
| basic | kawow | 199 | 1.035 | 0.729 | 0.090 | 0.897 |
| basic | smarts | 199 | 3.070 | 1.988 | 0.444 | 0.093 |
| neutral | smarts_mixed | 1731 | 0.508 | 0.377 | 0.009 | 0.971 |
| neutral | kawow | 1704 | 0.642 | 0.467 | 0.009 | 0.951 |
| neutral | mqg | 1730 | 0.691 | 0.528 | 0.025 | 0.946 |
| neutral | smarts | 1731 | 1.013 | 0.620 | -0.078 | 0.883 |

## Largest Residuals

The tables below list the five largest absolute residuals within each subgroup/model slice where at least one valid prediction exists.

### S01 logKow acid

#### kawow

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| 2,6-Dihydroxybenzoic acid | 4.50 | n/a | 2.200 | 0.276 | -1.924 | carboxylic_acid:1@4.5;phenol:2@10.0 | - |
| Diflunisal | 4.50 | n/a | 4.320 | 2.473 | -1.847 | carboxylic_acid:1@4.5;phenol:1@10.0 | - |
| N-Phenylanthranilic acid | 4.50 | 5.00 | 4.360 | 2.613 | -1.747 | carboxylic_acid:1@4.5 | aniline_like:2@5.0 |
| 4-Carboxyphenylisothiocyanate | 4.50 | n/a | 3.520 | 1.946 | -1.574 | carboxylic_acid:1@4.5 | - |
| 3,5-Diiodosalicylic acid | 4.50 | n/a | 4.560 | 3.000 | -1.560 | carboxylic_acid:1@4.5;phenol:1@10.0 | - |

#### smarts

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Aspartic acid | 4.50 | n/a | -3.700 | -0.480 | 3.220 | carboxylic_acid:1@4.5 | - |
| 5-Nitrosalicylic acid | 4.50 | 5.00 | 2.340 | 0.500 | -1.840 | carboxylic_acid:1@4.5;phenol:1@10.0 | aniline_like:1@5.0 |
| 2,2-Diphenylpropionic acid | 4.50 | n/a | 2.690 | 4.340 | 1.650 | carboxylic_acid:1@4.5 | - |
| Latamoxef | 4.50 | 7.00 | -0.580 | 1.020 | 1.600 | carboxylic_acid:2@4.5;tetrazole:1@4.8;phenol:1@10.0;imide:1@8.5 | imidazole_like:4@7.0 |
| Prostaglandin F2A | 4.50 | n/a | 2.280 | 3.720 | 1.440 | carboxylic_acid:1@4.5 | - |

#### smarts_mixed

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Diflunisal | 4.50 | n/a | 4.320 | 2.920 | -1.400 | carboxylic_acid:1@4.5;phenol:1@10.0 | - |
| Flufenamic acid | 4.50 | 5.00 | 5.250 | 3.897 | -1.353 | carboxylic_acid:1@4.5 | aniline_like:2@5.0 |
| N-Phenylanthranilic acid | 4.50 | 5.00 | 4.360 | 3.058 | -1.302 | carboxylic_acid:1@4.5 | aniline_like:2@5.0 |
| Niflumic acid | 4.50 | 5.20 | 4.430 | 3.162 | -1.268 | carboxylic_acid:1@4.5 | aniline_like:2@5.0;pyridine_like:1@5.2 |
| 3,5-Diiodosalicylic acid | 4.50 | n/a | 4.560 | 3.312 | -1.248 | carboxylic_acid:1@4.5;phenol:1@10.0 | - |

#### mqg

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| 1,2,3-Propanetricarboxylic acid | 4.50 | n/a | -1.720 | 0.416 | 2.136 | carboxylic_acid:3@4.5 | - |
| Aspartic acid | 4.50 | n/a | -3.700 | -1.981 | 1.719 | carboxylic_acid:1@4.5 | - |
| 4-Amino-3,5,6-trichloro-2-pyridinecarboxlic acid | 4.50 | 5.20 | 0.300 | 1.975 | 1.675 | carboxylic_acid:1@4.5 | aniline_like:1@5.0;pyridine_like:1@5.2 |
| Cefacetrile | 4.50 | n/a | -0.450 | 1.037 | 1.487 | carboxylic_acid:1@4.5 | - |
| (9Z)-Octadecenoic acid | 4.50 | n/a | 7.640 | 6.355 | -1.285 | carboxylic_acid:1@4.5 | - |

### S01 logKow neutral

#### kawow

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| 1-Azido-4-chlorobenzene | n/a | n/a | -3.710 | 2.733 | 6.443 | - | - |
| Reglone | n/a | 5.20 | -4.600 | 1.608 | 6.208 | - | pyridine_like:2@5.2 |
| O,O'-Diisopropyl-O''-3,4-dimethoxyphenylglyoxylonitrile oximino thiophosphate | n/a | n/a | 0.471 | 4.112 | 3.641 | - | - |
| O,O'-Diethyl-O''-3,4-dimethoxyphenylglyoxylonitrile oximino thiophosphate | n/a | n/a | -0.187 | 3.265 | 3.452 | - | - |
| Tetraethoxysilane | n/a | n/a | 0.040 | 3.422 | 3.382 | - | - |

#### smarts

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Ampicillin | n/a | n/a | -2.170 | 1.930 | 4.100 | - | - |
| Cefaclor | n/a | n/a | -1.790 | 2.160 | 3.950 | - | - |
| Epivir | n/a | 5.20 | -0.930 | 2.870 | 3.800 | - | aniline_like:1@5.0;pyridine_like:2@5.2 |
| Ftorafur | n/a | 5.20 | -0.270 | 3.350 | 3.620 | - | pyridine_like:1@5.2 |
| Baclofen | n/a | n/a | -0.960 | 2.630 | 3.590 | - | - |

#### smarts_mixed

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Fimepinostat | n/a | 5.20 | -1.490 | 0.656 | 2.146 | - | aniline_like:2@5.0;pyridine_like:5@5.2 |
| DIDP | n/a | n/a | 7.700 | 9.458 | 1.758 | - | - |
| Dinitramine | n/a | 5.00 | 4.300 | 2.752 | -1.548 | - | aniline_like:4@5.0 |
| Water | n/a | n/a | -1.380 | 0.166 | 1.546 | - | - |
| 5-Dimethylamino-1,3,6-trimethyluracil | n/a | 5.20 | 0.990 | -0.429 | -1.419 | - | aniline_like:1@5.0;pyridine_like:2@5.2 |

#### mqg

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Diethylene glycol dimethyl ether | n/a | n/a | -0.360 | 3.056 | 3.416 | - | - |
| Heptane | n/a | n/a | 4.660 | 1.339 | -3.321 | - | - |
| Diethyleneglycol | n/a | n/a | -1.980 | 1.339 | 3.319 | - | - |
| Cyclooctane | n/a | n/a | 4.450 | 1.312 | -3.138 | - | - |
| Reglone | n/a | 5.20 | -4.600 | -1.644 | 2.956 | - | pyridine_like:2@5.2 |

### S01 logKow basic

#### kawow

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Carbosulfan | n/a | 10.00 | 2.200 | 5.655 | 3.455 | - | aliphatic_amine:1@10.0 |
| Triflumizole | n/a | 7.00 | 1.400 | 4.246 | 2.846 | - | imidazole_like:2@7.0 |
| (L)-Arginine | n/a | 10.00 | -1.652 | -4.244 | -2.592 | - | aliphatic_amine:4@10.0 |
| Guanidinoacetic acid | n/a | 10.00 | -1.110 | -3.557 | -2.447 | - | aliphatic_amine:3@10.0 |
| Azimsulfuron | 4.80 | 7.00 | 2.100 | -0.273 | -2.373 | tetrazole:1@4.8 | aniline_like:1@5.0;pyridine_like:2@5.2;imidazole_like:6@7.0 |

#### smarts

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Triazolam | n/a | 7.00 | 5.500 | 0.090 | -5.410 | - | imidazole_like:3@7.0 |
| Alprazolam | n/a | 7.00 | 4.900 | -0.440 | -5.340 | - | imidazole_like:3@7.0 |
| Ornithine | n/a | 10.00 | -4.410 | -0.340 | 4.070 | - | aliphatic_amine:1@10.0 |
| Tetramethylthiuram disulfide | n/a | 10.00 | 1.730 | -2.210 | -3.940 | - | aliphatic_amine:2@10.0 |
| Brotizolam | n/a | 7.00 | 2.790 | -1.100 | -3.890 | - | imidazole_like:3@7.0 |

#### smarts_mixed

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Antipyrine | n/a | 7.00 | 0.230 | 1.869 | 1.639 | - | imidazole_like:2@7.0 |
| Brotizolam | n/a | 7.00 | 2.790 | 4.279 | 1.489 | - | imidazole_like:3@7.0 |
| Mesoridazine | n/a | 10.00 | 3.050 | 4.492 | 1.442 | - | aliphatic_amine:1@10.0;aniline_like:2@5.0 |
| Triflumizole | n/a | 7.00 | 1.400 | 2.800 | 1.400 | - | imidazole_like:2@7.0 |
| Metosulam | n/a | 7.00 | 2.500 | 3.809 | 1.309 | - | aniline_like:1@5.0;imidazole_like:3@7.0 |

#### mqg

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Formylhydrazine | n/a | 10.00 | -2.050 | 0.803 | 2.853 | - | aliphatic_amine:1@10.0 |
| Diethanolamine | n/a | 10.00 | -1.430 | 1.339 | 2.769 | - | aliphatic_amine:1@10.0 |
| 6-Amino-1-hexanol | n/a | 10.00 | -0.010 | 2.686 | 2.696 | - | aliphatic_amine:1@10.0 |
| Piperazine | n/a | 10.00 | -1.170 | 1.312 | 2.482 | - | aliphatic_amine:2@10.0 |
| Formamidoxime | n/a | 10.00 | -1.640 | 0.803 | 2.443 | - | aliphatic_amine:1@10.0 |

### S02 logKoa acid

#### kawow

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Chlorethephon | 2.00 | n/a | 10.410 | 8.026 | -2.384 | phosphonic_acid:1@2.0 | - |
| Carboxymethylcellulose | 4.50 | 5.20 | 12.160 | 9.945 | -2.215 | carboxylic_acid:1@4.5 | aniline_like:1@5.0;pyridine_like:1@5.2 |
| Sorbic acid | 4.50 | n/a | 7.110 | 5.573 | -1.537 | carboxylic_acid:1@4.5 | - |
| Mecoprop | 4.50 | n/a | 9.740 | 8.224 | -1.516 | carboxylic_acid:1@4.5 | - |
| m-Anthranilic acid | 4.50 | 5.00 | 9.890 | 8.382 | -1.508 | carboxylic_acid:1@4.5 | aniline_like:1@5.0 |

#### smarts

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Carboxymethylcellulose | 4.50 | 5.20 | 12.160 | 8.930 | -3.230 | carboxylic_acid:1@4.5 | aniline_like:1@5.0;pyridine_like:1@5.2 |
| Sorbic acid | 4.50 | n/a | 7.110 | 4.360 | -2.750 | carboxylic_acid:1@4.5 | - |
| Mecoprop | 4.50 | n/a | 9.740 | 7.010 | -2.730 | carboxylic_acid:1@4.5 | - |
| Trichloroacetic acid | 4.50 | n/a | 7.360 | 4.690 | -2.670 | carboxylic_acid:1@4.5 | - |
| Bromoacetic acid | 4.50 | n/a | 6.690 | 4.120 | -2.570 | carboxylic_acid:1@4.5 | - |

#### smarts_mixed

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Carboxymethylcellulose | 4.50 | 5.20 | 12.160 | 10.360 | -1.800 | carboxylic_acid:1@4.5 | aniline_like:1@5.0;pyridine_like:1@5.2 |
| Salicylic acid | 4.50 | n/a | 7.440 | 8.847 | 1.407 | carboxylic_acid:1@4.5;phenol:1@10.0 | - |
| Mecoprop | 4.50 | n/a | 9.740 | 8.419 | -1.321 | carboxylic_acid:1@4.5 | - |
| m-Anthranilic acid | 4.50 | 5.00 | 9.890 | 8.644 | -1.246 | carboxylic_acid:1@4.5 | aniline_like:1@5.0 |
| Trichloroacetic acid | 4.50 | n/a | 7.360 | 6.143 | -1.217 | carboxylic_acid:1@4.5 | - |

#### mqg

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Formic acid | 4.50 | n/a | 4.060 | 1.896 | -2.164 | carboxylic_acid:1@4.5 | - |
| Sulfometuron | 4.50 | 5.20 | 15.410 | 13.559 | -1.851 | carboxylic_acid:1@4.5 | aniline_like:1@5.0;pyridine_like:2@5.2 |
| 4-Amino-3,5,6-trichloro-2-pyridinecarboxlic acid | 4.50 | 5.20 | 11.500 | 9.674 | -1.826 | carboxylic_acid:1@4.5 | aniline_like:1@5.0;pyridine_like:1@5.2 |
| Chlorethephon | 2.00 | n/a | 10.410 | 8.795 | -1.615 | phosphonic_acid:1@2.0 | - |
| Malic acid | 4.50 | n/a | 9.900 | 8.453 | -1.447 | carboxylic_acid:2@4.5 | - |

### S02 logKoa neutral

#### kawow

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Fludioxonil | n/a | n/a | 11.610 | 7.370 | -4.240 | - | - |
| 5-Methylisoxazol-3-one | n/a | n/a | 7.110 | 3.632 | -3.478 | - | - |
| Lenacil | n/a | 5.20 | 12.210 | 8.988 | -3.222 | - | pyridine_like:1@5.2 |
| Dicarbasulf | n/a | n/a | 9.120 | 11.976 | 2.856 | - | - |
| Tris(2-isopropylphenyl) phosphate | n/a | n/a | 11.780 | 14.340 | 2.560 | - | - |

#### smarts

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Dicarbasulf | n/a | n/a | 9.120 | 16.560 | 7.440 | - | - |
| N-Methyl perfluorooctanesulfonamidoethylacrylate | n/a | n/a | 7.870 | 15.180 | 7.310 | - | - |
| 5-Methylisoxazol-3-one | n/a | n/a | 7.110 | 0.190 | -6.920 | - | - |
| Furathiocarb | n/a | n/a | 11.860 | 18.480 | 6.620 | - | - |
| N-Ethyl perfluorooctane sulfonamidoethanol | n/a | n/a | 7.700 | 14.230 | 6.530 | - | - |

#### smarts_mixed

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| 5-Methylisoxazol-3-one | n/a | n/a | 7.110 | 4.573 | -2.537 | - | - |
| 4-Chloro-6-nitro-m-cresol | 10.00 | 5.00 | 5.580 | 7.689 | 2.109 | phenol:1@10.0 | aniline_like:1@5.0 |
| Carbon tetrafluoride | n/a | n/a | -0.950 | 1.109 | 2.059 | - | - |
| Lenacil | n/a | 5.20 | 12.210 | 10.311 | -1.899 | - | pyridine_like:1@5.2 |
| 4-s-Butyl-2-nitrophenol | 10.00 | 5.00 | 6.150 | 8.042 | 1.892 | phenol:1@10.0 | aniline_like:1@5.0 |

#### mqg

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| 1,5-Pentanediol | n/a | n/a | 7.380 | 3.793 | -3.587 | - | - |
| Sorbitol | n/a | n/a | 14.280 | 11.086 | -3.194 | - | - |
| Decabromodiphenyl ether | n/a | n/a | 16.120 | 13.001 | -3.119 | - | - |
| Tetramethylene glycol | n/a | n/a | 6.650 | 3.793 | -2.857 | - | - |
| Carbon tetrafluoride | n/a | n/a | -0.950 | 1.738 | 2.688 | - | - |

### S02 logKoa basic

#### kawow

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Fenpyroximate | n/a | 7.00 | 9.320 | 13.732 | 4.412 | - | imidazole_like:2@7.0 |
| Tiabendazole | n/a | 7.00 | 12.510 | 8.384 | -4.126 | - | imidazole_like:2@7.0 |
| Thiacloprid | n/a | 10.00 | 13.150 | 9.228 | -3.922 | - | aliphatic_amine:1@10.0;pyridine_like:1@5.2 |
| Benfuracarb | n/a | 10.00 | 10.290 | 13.563 | 3.273 | - | aliphatic_amine:1@10.0 |
| Clothianidin | n/a | 13.50 | 13.590 | 10.339 | -3.251 | - | guanidine:1@13.5;aliphatic_amine:3@10.0;imidazole_like:1@7.0 |

#### smarts

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| HMX | n/a | 10.00 | 12.550 | -4.940 | -17.490 | - | aliphatic_amine:8@10.0 |
| RDX | n/a | 10.00 | 9.440 | -3.340 | -12.780 | - | aliphatic_amine:6@10.0 |
| Benfuracarb | n/a | 10.00 | 10.290 | 21.400 | 11.110 | - | aliphatic_amine:1@10.0 |
| Carbosulfan | n/a | 10.00 | 10.170 | 20.930 | 10.760 | - | aliphatic_amine:1@10.0 |
| 1,4-Dinitrobutane | n/a | 10.00 | 4.610 | 12.700 | 8.090 | - | aliphatic_amine:2@10.0 |

#### smarts_mixed

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Tiabendazole | n/a | 7.00 | 12.510 | 9.309 | -3.201 | - | imidazole_like:2@7.0 |
| 5-Amino-1H-1,2,4-triazole | n/a | 7.00 | 8.090 | 5.287 | -2.803 | - | aniline_like:1@5.0;imidazole_like:2@7.0 |
| Prochloraz | n/a | 7.00 | 10.160 | 12.147 | 1.987 | - | imidazole_like:2@7.0 |
| Fluquinconazole | n/a | 7.00 | 11.580 | 13.355 | 1.775 | - | pyridine_like:2@5.2;imidazole_like:3@7.0 |
| Initium | n/a | 7.00 | 13.730 | 12.025 | -1.705 | - | aniline_like:1@5.0;pyridine_like:1@5.2;imidazole_like:3@7.0 |

#### mqg

| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |
|---|---:|---:|---:|---:|---:|---|---|
| Colamine | n/a | 10.00 | 6.320 | 2.814 | -3.506 | - | aliphatic_amine:1@10.0 |
| Thiacloprid | n/a | 10.00 | 13.150 | 11.283 | -1.867 | - | aliphatic_amine:1@10.0;pyridine_like:1@5.2 |
| 1,1'-Dimethyl-3,3',4,4',5,5'-hexabromo-2,2'-bipyrrole | n/a | 7.00 | 11.560 | 9.747 | -1.813 | - | imidazole_like:2@7.0 |
| Clothianidin | n/a | 13.50 | 13.590 | 11.852 | -1.738 | - | guanidine:1@13.5;aliphatic_amine:3@10.0;imidazole_like:1@7.0 |
| Ethylenediamine | n/a | 10.00 | 4.490 | 2.814 | -1.676 | - | aliphatic_amine:2@10.0 |

## Notes

- Acid/basic labels are based on rule-matched functional groups and estimated pKa values at pH 7.
- Zwitterions often remain classed as neutral when the estimated net charge is near zero.
- SMARTS mixed is consistently the strongest model across these subgrouped benchmarks.
- Plain SMARTS degrades strongly for ionizable chemistry, especially S02 basic compounds.


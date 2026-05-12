# Pathway-Pilot Capacity-Factor Reference

Default raw PECD folder:

```text
C:\Users\B510067\dev_data\TYNDP24\PECD\PECD
```

Default processed output:

```text
C:\Users\B510067\dev_data\pathway-pilot\tvar\pecd_capacity_factors_2030_2040.parquet
```

Known PECD file pattern:

```text
{pecd_dir}\{target_year}\{prefix}_{target_year}_{zone}_edition 2023.2.csv
```

Current target years:

```text
2030
2040
```

Current zones:

```text
DKE1
DKW1
NL00
```

Current technology prefixes:

```text
onshore_wind -> PECD_Wind_Onshore
solar        -> PECD_LFSolarPV
```

The pathway-pilot model selects the configured `capacity_factor_zone` from the prepared long-form table. DK currently uses DKW1; NL currently uses NL00. Keep all source-zone rows in the prepared table; filtering belongs in `model_inputs.py`.

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
```

Current technology prefixes:

```text
onshore_wind -> PECD_Wind_Onshore
solar        -> PECD_LFSolarPV
```

The pathway-pilot model currently selects DKW1 capacity factors from the prepared long-form table. Keep DKE1 rows in the prepared source table if they are part of the source extraction; filtering belongs in `model_inputs.py`.

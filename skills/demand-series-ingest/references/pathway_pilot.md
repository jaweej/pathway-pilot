# Pathway-Pilot Demand Reference

Default raw TYNDP demand folder:

```text
C:\Users\B510067\dev_data\TYNDP24\Demand\Demand Profiles\NT\Electricity demand profiles
```

Default processed output:

```text
C:\Users\B510067\dev_data\pathway-pilot\tvar\tyndp_demand_profiles_2030_2040.parquet
```

Known workbook pattern:

```text
{target_year}_National Trends.xlsx
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

Expected input columns after `header=7`:

```text
Date
Month
Day
Hour
<weather-year columns>
```

The pathway-pilot model sums the configured demand zones from `model_cases[active_model]` in `config/model_config.yaml`. DK uses DKE1 and DKW1; NL uses NL00. Keep source-zone rows separate in the prepared demand table.

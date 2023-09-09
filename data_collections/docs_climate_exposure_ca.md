# Overview

| Properties             | Value                                                                                                                                                                                                                                                                                                     |
|:-----------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Description            | This dataset includes information about the average weather and climate  conditions during the summer months of 2020. It covers factors like  temperature, wildfire smoke, wind speed, and relative humidity, sourced  from PRISM and NOAA, and population density aggregated at the census tract  level. |
| Spatial Coverage       | California                                                                                                                                                                                                                                                                                                |
| Spatial Resolution     | census tract                                                                                                                                                                                                                                                                                              |
| Temporal Coverage      | 2020                                                                                                                                                                                                                                                                                                      |
| Temporal Resolution    | annual                                                                                                                                                                                                                                                                                                    |
| Original Data Sources  | https://prism.oregonstate.edu and https://www.ncei.noaa.gov/access/monitoring/wind/                                                                                                                                                                                                                       |
| Data Processing Code   | `notebooks/104_climate_exposure.ipynb`                                                                                                                                                                                                                                                                    |
| Data Location          | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SYNPBS                                                                                                                                                                                                                           |
| Number of Variables    | 32                                                                                                                                                                                                                                                                                                        |
| Number of Observations | 8616                                                                                                                                                                                                                                                                                                      |
| Missing Cells          | 0                                                                                                                                                                                                                                                                                                         |
| Missing cells (%)      | 0.0%                                                                                                                                                                                                                                                                                                      |
| Duplicate Rows         | 0                                                                                                                                                                                                                                                                                                         |
| Duplicate Rows (%)     | 0.0%                                                                                                                                                                                                                                                                                                      |
| Total Size In Memory   | 2.1 MB                                                                                                                                                                                                                                                                                                    |

# Variables

| Variable Name | Description |
| ------------- | ----------- |
| `GEOID` | Unique geographic identifier of the area. |
| `POPULATION_2020` | Total population of the area in 2020. |
| `POP20_SQMI` | Population density per square mile in 2020. |
| `avg_temp_jun` | Average temperature in the area for the month of June. |
| `avg_temp_jul` | Average temperature in the area for the month of July. |
| `avg_temp_aug` | Average temperature in the area for the month of August. |
| `avg_temp_sep` | Average temperature in the area for the month of September. |
| `avg_temp_oct` | Average temperature in the area for the month of October. |
| `avg_rhum_jun` | Average relative humidity in the area for the month of June. |
| `avg_rhum_jul` | Average relative humidity in the area for the month of July. |
| `avg_rhum_aug` | Average relative humidity in the area for the month of August. |
| `avg_rhum_sep` | Average relative humidity in the area for the month of September. |
| `avg_rhum_oct` | Average relative humidity in the area for the month of October. |
| `avg_wnd_jun` | Average wind speed in the area for the month of June. |
| `avg_wnd_jul` | Average wind speed in the area for the month of July. |
| `avg_wnd_aug` | Average wind speed in the area for the month of August. |
| `avg_wnd_sep` | Average wind speed in the area for the month of September. |
| `avg_wnd_oct` | Average wind speed in the area for the month of October. |
| `avg_smoke_pm_jun` | Average particulate matter from smoke in the area for the month of June. |
| `avg_smoke_pm_jul` | Average particulate matter from smoke in the area for the month of July. |
| `avg_smoke_pm_aug` | Average particulate matter from smoke in the area for the month of August. |
| `avg_smoke_pm_sep` | Average particulate matter from smoke in the area for the month of September. |
| `avg_smoke_pm_oct` | Average particulate matter from smoke in the area for the month of October. |
| `EP_AGE17` | The estimated percentage of the population aged 17 or younger. |
| `EP_AGE65` | The estimated percentage of the population aged 65 or older. |
| `EP_NOINT` | The estimated percentage of households without an internet subscription. |
| `EP_POV150` | The estimated percentage of the population living at 150% of the poverty level or below. |
| `EP_UNEMP` | The estimated percentage of the labor force that is unemployed. |
| `EP_MINRTY` | The estimated percentage of the population identified as a racial or ethnic minority. |
| `EP_LIMENG` | The estimated percentage of the population with limited English proficiency. |
| `EP_UNINSUR` | The estimated percentage of the population without health insurance. |
| `RPL_THEMES` | Overall percentile ranking. |

# Correlations

![](figs/corr_climate_exposure_ca.png)

# Sample

|      GEOID |   avg_tmax_jun |   avg_tmax_jul |   avg_tmax_aug |   avg_tmax_sep |   avg_tmax_oct |   avg_rhum_jun |   avg_rhum_jul |   avg_rhum_aug |   avg_rhum_sep |   avg_rhum_oct |   avg_smoke_pm_jun |   avg_smoke_pm_jul |   avg_smoke_pm_aug |   avg_smoke_pm_sep |   avg_smoke_pm_oct |   avg_wnd_jun |   avg_wnd_jul |   avg_wnd_aug |   avg_wnd_sep |   avg_wnd_oct |   POPULATION_2020 |   POP20_SQMI |   EP_POV150 |   EP_UNEMP |   EP_UNINSUR |   EP_AGE65 |   EP_AGE17 |   EP_LIMENG |   EP_MINRTY |   RPL_THEMES |   E_NOINT |
|-----------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|------------------:|-------------:|------------:|-----------:|-------------:|-----------:|-----------:|------------:|------------:|-------------:|----------:|
| 6.0014e+09 |        24.04   |        23.9242 |        26.9435 |          27.18 |        25.721  |        78.0967 |        84.3613 |        77.3323 |        71.41   |        55.3065 |                  0 |           0.133076 |            6.6703  |            25.2394 |            5.94424 |       4.23107 |       4.09597 |       3.59107 |       2.12102 |       2.64452 |              3038 |       1133.6 |         6.8 |        1   |          0.5 |       25.3 |       18.6 |         0   |        27.3 |       0.0255 |       320 |
| 6.0014e+09 |        23.1767 |        22.3339 |        23.9565 |          26.17 |        25.5952 |        79.64   |        84.2    |        80.3677 |        74.6667 |        62.9    |                  0 |           0.14501  |            6.92154 |            25.341  |            6.17948 |       4.24232 |       4.10167 |       3.5977  |       2.13196 |       2.6516  |              2001 |       8700   |         7   |        7.9 |          0.4 |       21.7 |       16.3 |         1.3 |        32.8 |       0.1978 |        30 |
| 6.0014e+09 |        23.1767 |        22.3339 |        23.9565 |          26.17 |        25.5952 |        79.64   |        84.2    |        80.3677 |        74.6667 |        62.9    |                  0 |           0.14501  |            6.92154 |            25.341  |            6.17948 |       4.24565 |       4.10315 |       3.59958 |       2.13571 |       2.65343 |              5504 |      12800   |         8.6 |        3.4 |          0.3 |       16.4 |       17   |         0.3 |        39   |       0.356  |       300 |
| 6.0014e+09 |        23.1767 |        22.3339 |        23.9565 |          26.17 |        25.5952 |        79.64   |        84.2    |        80.3677 |        74.6667 |        62.9    |                  0 |           0.14501  |            6.92154 |            25.341  |            6.17948 |       4.24649 |       4.10468 |       3.60054 |       2.13374 |       2.65534 |              4112 |      14685.7 |        12   |        2.5 |          2.6 |       10.8 |       21.1 |         1.3 |        35.5 |       0.1596 |        72 |
| 6.0014e+09 |        23.1767 |        22.3339 |        23.9565 |          26.17 |        25.5952 |        79.64   |        84.2    |        80.3677 |        74.6667 |        62.9    |                  0 |           0.14501  |            6.92154 |            25.341  |            6.17948 |       4.25036 |       4.1075  |       3.60318 |       2.13536 |       2.65884 |              3644 |      15843.5 |        12.8 |        6.4 |          2.7 |       15.1 |       11.5 |         1.1 |        55.3 |       0.1312 |       207 |

Generated with `notebooks/201_make_data_dict.ipynb`.

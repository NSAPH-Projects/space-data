# Overview

| Properties             | Value                                                                                                                                                                                                                                                           |
|:-----------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Description            | The dataset includes election outcomes, specifically the percentage of the population in each county that voted for the Democratic Party in 2020. It also contains various aggregated data, such as demographics and employment statistics, from 2019 and 2020. |
| Spatial Coverage       | Continental USA                                                                                                                                                                                                                                                 |
| Spatial Resolution     | county                                                                                                                                                                                                                                                          |
| Temporal Coverage      | 2019 and 2020                                                                                                                                                                                                                                                   |
| Temporal Resolution    | annual                                                                                                                                                                                                                                                          |
| Original Data Sources  | https://github.com/evangambit/JsonOfCounties                                                                                                                                                                                                                    |
| Data Processing Code   | `notebooks/101_election_data.ipynb`                                                                                                                                                                                                                             |
| Data Location          | https://drive.google.com/drive/folders/1De4e2DJhP1gzL5r-l9dXw4Jpa48imDH5?usp=sharing                                                                                                                                                                            |
| Number of Variables    | 48                                                                                                                                                                                                                                                              |
| Number of Observations | 3109                                                                                                                                                                                                                                                            |
| Missing Cells          | 3109                                                                                                                                                                                                                                                            |
| Missing cells (%)      | 100.0%                                                                                                                                                                                                                                                          |
| Duplicate Rows         | 0                                                                                                                                                                                                                                                               |
| Duplicate Rows (%)     | 0.0%                                                                                                                                                                                                                                                            |
| Total Size In Memory   | 1.49097 MB                                                                                                                                                                                                                                                      |

# Variables

| Variable Name | Description |
| ------------- | ----------- |
| `fips` | Federal Information Processing Standard code, a unique identifier for counties and county equivalents in the United States. |
| `noaa_prcp` | Average precipitation recorded by the National Oceanic and Atmospheric Administration. |
| `noaa_snow` | Average snowfall recorded by the National Oceanic and Atmospheric Administration. |
| `noaa_temp` | Average temperature recorded by the National Oceanic and Atmospheric Administration. |
| `noaa_altitude` | Altitude of the location as recorded by the National Oceanic and Atmospheric Administration. |
| `noaa_temp_jan` | Average temperature in January as recorded by the National Oceanic and Atmospheric Administration. |
| `noaa_temp_apr` | Average temperature in April as recorded by the National Oceanic and Atmospheric Administration. |
| `noaa_temp_jul` | Average temperature in July as recorded by the National Oceanic and Atmospheric Administration. |
| `noaa_temp_oct` | Average temperature in October as recorded by the National Oceanic and Atmospheric Administration. |
| `cs_male` | Male population count. |
| `cs_female` | Female population count. |
| `population/2019` | Total population of the area in 2019. |
| `cdc_suicides` | Number of suicide cases as recorded by the Centers for Disease Control and Prevention. |
| `cdc_homicides` | Number of homicide cases as recorded by the Centers for Disease Control and Prevention. |
| `cdc_vehicle_deaths` | Number of vehicle-related death cases as recorded by the Centers for Disease Control and Prevention. |
| `bls_labor_force` | Total labor force count as recorded by the Bureau of Labor Statistics. |
| `bls_employed` | Number of employed people as recorded by the Bureau of Labor Statistics. |
| `bls_unemployed` | Number of unemployed people as recorded by the Bureau of Labor Statistics. |
| `life-expectancy` | Average life expectancy of individuals in the area. |
| `cdc_police_deaths_total` | Total number of deaths caused by police as recorded by the Centers for Disease Control and Prevention. |
| `cdc_police_deaths_unarmed` | Number of deaths of unarmed individuals caused by police as recorded by the Centers for Disease Control and Prevention. |
| `police_deaths` | Total number of police deaths in the area. |
| `avg_income` | Average income of individuals in the area. |
| `cs_ed_below_highschool` | Population count with an educational attainment level below high school. |
| `cs_ed_highschool` | Population count with high school as their highest educational attainment level. |
| `cs_ed_some_college` | Population count with some college education but without a degree. |
| `cs_ed_above_college` | Population count with an educational attainment level above college. |
| `poverty-rate` | Poverty rate in the area. |
| `bls_living_wage` | Living wage as defined by the Bureau of Labor Statistics. |
| `bls_food_costs` | Average food cost as defined by the Bureau of Labor Statistics. |
| `bls_medical_costs` | Average medical cost as defined by the Bureau of Labor Statistics. |
| `bls_housing_costs` | Average housing cost as defined by the Bureau of Labor Statistics. |
| `bls_tax_costs` | Average tax cost as defined by the Bureau of Labor Statistics. |
| `health_poor_health_pct` | Percentage of population reporting poor health. |
| `health_smokers_pct` | Percentage of population that are smokers. |
| `health_obese_pct` | Percentage of population that are obese. |
| `health_phy_inactive_pct` | Percentage of population that are physically inactive. |
| `health_children_poverty_pct` | Percentage of children living in poverty. |
| `health_80th_perc_income_pct` | Income at the 80th percentile. |
| `health_20th_perc_income_pct` | Income at the 20th percentile. |
| `cs_age-to-25` | Population count under the age of 25. |
| `cs_age-25-65` | Population count between the ages of 25 and 65. |
| `cs_age-over-65` | Population count over the age of 65. |
| `cs_white` | Population count identifying as White. |
| `cs_black` | Population count identifying as Black. |
| `cs_asian` | Population count identifying as Asian. |
| `cs_hispanic` | Population count identifying as Hispanic. |
| `election_dem_pct` | Percentage of votes received by the Democratic party in the last election. |

# Correlations

![](figs/corr_election_us.png)

# Sample

|   qd_mean_pm25 |   cs_poverty |   cs_hispanic |   cs_black |   cs_white |   cs_native |   cs_asian |   cs_ed_below_highschool |   cs_household_income |   cs_median_house_value |   cs_other |   cs_population_density |   cdc_mean_bmi |   cdc_pct_cusmoker |   cdc_pct_sdsmoker |   cdc_pct_fmsmoker |   cdc_pct_nvsmoker |   cdc_pct_nnsmoker |   gmet_mean_tmmn |   gmet_mean_summer_tmmn |   gmet_mean_winter_tmmn |   gmet_mean_tmmx |   gmet_mean_summer_tmmx |   gmet_mean_winter_tmmx |   gmet_mean_rmn |   gmet_mean_summer_rmn |   gmet_mean_winter_rmn |   gmet_mean_rmx |   gmet_mean_summer_rmx |   gmet_mean_winter_rmx |   gmet_mean_sph |   gmet_mean_summer_sph |   gmet_mean_winter_sph |   cms_mortality_pct |   cms_white_pct |   cms_black_pct |   cms_others_pct |   cms_hispanic_pct |   cms_female_pct |   Notes | County             |   Deaths |   Population |   Crude Rate |   cdc_mortality_pct |   bin_NORTHEAST |   bin_SOUTH |   bin_WEST |
|---------------:|-------------:|--------------:|-----------:|-----------:|------------:|-----------:|-------------------------:|----------------------:|------------------------:|-----------:|------------------------:|---------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-----------------:|------------------------:|------------------------:|-----------------:|------------------------:|------------------------:|----------------:|-----------------------:|-----------------------:|----------------:|-----------------------:|-----------------------:|----------------:|-----------------------:|-----------------------:|--------------------:|----------------:|----------------:|-----------------:|-------------------:|-----------------:|--------:|:-------------------|---------:|-------------:|-------------:|--------------------:|----------------:|------------:|-----------:|
|        11.8556 |    0.0858086 |    0.00280528 |  0.132013  |   0.852475 |  0.00874587 | 0.00264026 |                 0.2533   |                 37351 |                  133900 | 0.00132013 |                 89.4209 |        3208.39 |          0.146341  |          0.0243902 |           0.243902 |           0.560976 |          0.0243902 |          283.863 |                 295.446 |                 273.041 |          297.262 |                 307.294 |                 284.525 |         40.4298 |                45.2537 |                43.9528 |         88.9721 |                95.9595 |                85.7228 |      0.00941711 |              0.0167348 |             0.00400174 |          0          |        0.59589  |        0.40411  |       0          |                  0 |         0.554795 |     nan | Autauga County, AL |      148 |         6546 |       2260.9 |             22.6092 |               0 |           1 |          0 |
|        10.4379 |    0.0533287 |    0.0140393  |  0.051535  |   0.918627 |  0.00541566 | 0.00224215 |                 0.160952 |                 40104 |                  177200 | 0.00814074 |                110.575  |        3249.75 |          0.147059  |          0.0392157 |           0.245098 |           0.568627 |          0         |          285.874 |                 296.672 |                 275.402 |          298.158 |                 306.452 |                 287.357 |         42.3423 |                50.0248 |                43.4265 |         90.2833 |                95.1494 |                88.1623 |      0.0102607  |              0.017282  |             0.00478393 |          0.0227273  |        0.63843  |        0.353306 |       0.00826446 |                  0 |         0.539256 |     nan | Baldwin County, AL |      640 |        30568 |       2093.7 |             20.9369 |               0 |           1 |          0 |
|        11.5042 |    0.19443   |    0.00945875 |  0.333421  |   0.656069 |  0          | 0          |                 0.40515  |                 22143 |                   88200 | 0.00105097 |                 31.3027 |        2953.69 |          0.0714047 |          0.050425  |           0.216834 |           0.564502 |          0.0111988 |          284.135 |                 295.378 |                 273.83  |          297.649 |                 307.03  |                 285.585 |         40.0521 |                46.0208 |                43.7785 |         90.7641 |                96.7203 |                86.7272 |      0.00970464 |              0.0169258 |             0.00434312 |          0          |        0.690141 |        0.28169  |       0.028169   |                  0 |         0.507042 |     nan | Barbour County, AL |      102 |         3909 |       2609.4 |             26.0936 |               0 |           1 |          0 |
|        11.8869 |    0.113087  |    0.00106686 |  0.121977  |   0.866287 |  0          | 0          |                 0.388691 |                 24875 |                   81200 | 0.0106686  |                 36.3165 |        3255.29 |          0.103448  |          0.0229885 |           0.310345 |           0.551724 |          0.0114943 |          283.539 |                 295.315 |                 272.578 |          296.797 |                 307.042 |                 283.907 |         41.5986 |                47.2497 |                43.8774 |         89.5762 |                96.4847 |                85.8461 |      0.00939423 |              0.0169568 |             0.00383068 |          0.0215264  |        0.655577 |        0.340509 |       0.00391389 |                  0 |         0.571429 |     nan | Bibb County, AL    |       73 |         2906 |       2512   |             25.1204 |               0 |           1 |          0 |
|        11.6592 |    0.104793  |    0.0094363  |  0.0105538 |   0.968463 |  0.00583561 | 0          |                 0.359449 |                 25857 |                  113700 | 0.00571145 |                 87.9251 |        3500.33 |          0.0833333 |          0         |           0.25     |           0.638889 |          0.0277778 |          282.61  |                 294.704 |                 271.343 |          295.556 |                 306.452 |                 281.915 |         42.4458 |                48.2832 |                46.0712 |         88.7252 |                96.3753 |                84.9593 |      0.00895414 |              0.0165894 |             0.0035027  |          0.00621118 |        0.701863 |        0.298137 |       0          |                  0 |         0.608696 |     nan | Blount County, AL  |      202 |         8439 |       2393.6 |             23.9365 |               0 |           1 |          0 |


Generated with `notebooks/201_make_data_dict.ipynb`.

# Simulated Medicare toy dataset

## Notes on dataset creation

Dataset was aggregated from `sim_medicare.parquet`, a single-treatment single-outcome dataset with an entirely synthetic outcome. Aggregations were performed using `notebooks/zip2county.ipynb`, and the `graph.graphml` came from the preexisting `data_collections/air_pollution_mortality_us/graph.graphml`.
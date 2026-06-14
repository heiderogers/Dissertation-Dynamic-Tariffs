# Who Benefits from Dynamic Electricity Tariffs?
## Household Switching Costs and Adoption Constraints in Germany

MPhil Dissertation - Environmental Policy, Department of Land Economy, University of Cambridge, July 2026

Driven by the importance of residential flexibility for the energy transition, this dissertation examines the household economics and adoption constraints of dynamic electricity tariffs in Germany. It begins with an exploratory examination of electricity prices based on generation type, highlighting the merit order effect, the role of renewables in price setting, and resulting price volatility. It then addresses two core research questions: what break-even switching cost makes dynamic tariffs economically worthwhile for different household archetypes, and what adoption trajectory emerges when diffusion is constrained by smart meter rollout and EV penetration.

This repository contains all Python scripts used for the data cleaning, simulation, and analysis in my dissertation. The scripts are numbered in the order they appear in the thesis: 01_data_cleaning.py builds the master hourly dataset from raw SMARD, DWD, and BDEW inputs; 02_exploratory_analysis.py produces the descriptive charts and statistics; 03_price_regression.py runs the OLS regression for Table 1; 04_simulation_inputs.py constructs the tariff price series and household load profiles; 05_simulation.py computes hourly bills for all archetype × tariff × scenario combinations; 06_breakeven.py derives C* break-even switching costs; and 07_bass_model.py projects dynamic tariff adoption using a Bass diffusion model constrained by EV penetration and smart meter rollout.

All modelling assumptions, parameter values, and data sources are documented in the script docstrings. Raw data files are included in data_raw/.

## Data sources

- 01_day_ahead_prices.csv — SMARD.de, EPEX Spot Day-Ahead DE/LU: https://www.smard.de/home/downloadcenter/download-marktdaten/
- 02_realised_generation.csv — SMARD.de, Realised generation: https://www.smard.de/home/downloadcenter/download-marktdaten/
- 03_realised_consumption.csv — SMARD.de, Realised consumption: https://www.smard.de/home/downloadcenter/download-marktdaten/
- 04_temperature_potsdam.txt — DWD CDC, Station Potsdam (ID 03987): https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/
- 05_bdew_standard_load_profiles.xlsx — BDEW H0 standard load profile (March 2025): https://www.bdew.de/energie/standardlastprofile-strom/

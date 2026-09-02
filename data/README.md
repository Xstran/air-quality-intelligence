# Data

This project combines PM2.5 air-quality observations with meteorological data for Manchester Piccadilly between January 2021 and 5 November 2023.

## Sources

### Air Quality
Hourly PM2.5 observations were obtained from the UK-AIR Manchester Piccadilly monitoring station (MAN3).

### Meteorology
Daily meteorological variables were obtained from the Open-Meteo Historical Weather API using ERA5 data.

## Data Processing

PM2.5 observations were checked for:

- Missing and invalid dates
- Duplicate dates
- Missing hourly measurements
- Invalid numeric values
- Negative concentrations
- Zero values

Daily PM2.5 means were retained only when at least 18 of the expected 24 hourly measurements were available.

Short interpolated gaps were used for exploratory analysis only and were not used as modelling targets or PM2.5 lag features.

The final modelling dataset contained 1,010 observations after quality control, lag construction and calendar-continuity checks.

## Data Availability

Large raw datasets are not stored directly in this repository.

Instructions for obtaining and preparing the source data will be added as the project is made fully reproducible.

# SmartBizLQ — Sample Data Sources

## 1. Sales Sample Dataset (`smartbizlq_sample_200rows.csv`)

- **Description**: Synthetic e-commerce sales data with 200 rows.
- **Columns**: order_id, order_date, customer_id, region, country, product_category,
  product_id, product_name, quantity, unit_price, revenue, payment_type, channel,
  month, quarter, year.
- **Source**: Generated for the original SmartBizLQ internship project.
- **Licence**: No restrictions (synthetic data).

## 2. Northern Ireland Energy Demand Dataset (`ni_energy_demand.csv`)

- **Description**: Hourly electricity system demand for Northern Ireland merged with
  Belfast weather data (temperature, wind speed), plus time features (hour, day of week, month).
- **Source**:
  - **Demand data**: EirGrid / SONI Smart Grid Dashboard
    - URL: https://www.smartgriddashboard.com/
    - Copyright © EirGrid plc & SONI Ltd. Published for market transparency.
    - Small samples used under fair dealing for research and technical demonstration.
  - **Weather data**: Open-Meteo Historical Weather API (ERA5 reanalysis)
    - URL: https://open-meteo.com/
    - Licence: CC-BY 4.0 (Copernicus Climate Change Service / ECMWF)
    - Attribution: "Weather data from Open-Meteo.com, based on ERA5 reanalysis
      data from ECMWF / Copernicus Climate Change Service."
- **Refresh**: Run `python data/download_energy_data.py --start 2023-01-01 --end 2023-12-31`
  to re-download from source.
- **Licence for redistribution**: The cached sample included in this repository is used
  under CC-BY 4.0 (weather) and fair dealing (demand). Attribution is provided above.

## 3. Telco Customer Churn Dataset (`telco_customer_churn.csv`)

- **Description**: 7,043 telecom customers with demographics, services, contract, charges and a Yes/No churn flag.
- **Source**: IBM, https://github.com/IBM/telco-customer-churn-on-icp4d (`data/Telco-Customer-Churn.csv`)
- **Licence**: Apache License 2.0.

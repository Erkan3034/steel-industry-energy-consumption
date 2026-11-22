# Data

The raw dataset is **not** committed. Download it and place it here:

```
data/raw/Steel_industry_data.csv
```

- Kaggle: https://www.kaggle.com/datasets/csafrit2/steel-industry-energy-consumption
- UCI: https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption

```bash
kaggle datasets download -d csafrit2/steel-industry-energy-consumption -p data/raw --unzip
```

**Schema:** 35,040 rows (2018, 15-minute intervals), 11 columns:
`date, Usage_kWh, Lagging_Current_Reactive.Power_kVarh, Leading_Current_Reactive_Power_kVarh, CO2(tCO2), Lagging_Current_Power_Factor, Leading_Current_Power_Factor, NSM, WeekStatus, Day_of_week, Load_Type`

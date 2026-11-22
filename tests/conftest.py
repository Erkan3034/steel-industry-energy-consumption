import numpy as np
import pandas as pd
import pytest

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LOADS = ["Light_Load", "Medium_Load", "Maximum_Load"]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Synthetic frame with the same schema as Steel_industry_data.csv."""
    rng = np.random.default_rng(0)
    n = 600
    nsm = (np.arange(n) % 96) * 900
    day = [DAYS[(i // 96) % 7] for i in range(n)]
    lag_kvarh = rng.uniform(0, 60, n)
    lead_kvarh = rng.uniform(0, 20, n)
    lag_pf = rng.uniform(40, 100, n)
    lead_pf = rng.uniform(40, 100, n)
    usage = 1.5 * lag_kvarh + 0.3 * lag_pf + 5 * np.sin(2 * np.pi * nsm / 86400) + rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2018-01-01", periods=n, freq="15min").strftime("%d/%m/%Y %H:%M"),
            "Usage_kWh": usage,
            "Lagging_Current_Reactive.Power_kVarh": lag_kvarh,
            "Leading_Current_Reactive_Power_kVarh": lead_kvarh,
            "CO2(tCO2)": usage * 0.0005,
            "Lagging_Current_Power_Factor": lag_pf,
            "Leading_Current_Power_Factor": lead_pf,
            "NSM": nsm,
            "WeekStatus": ["Weekend" if d in ("Saturday", "Sunday") else "Weekday" for d in day],
            "Day_of_week": day,
            "Load_Type": [LOADS[i % 3] for i in range(n)],
        }
    )


@pytest.fixture
def raw_csv(tmp_path, raw_df):
    path = tmp_path / "steel.csv"
    raw_df.to_csv(path, index=False)
    return path

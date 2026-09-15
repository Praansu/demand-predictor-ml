"""Generate synthetic parking occupancy data for testing.

Creates realistic-ish parking data with:
- Daily/weekly cycles
- Holiday spikes
- Random noise
- Optional special events
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays


def generate_parking_data(
    start_date: str = "2023-01-01",
    days: int = 365,
    base_capacity: int = 200,
    noise_std: float = 15.0,
    event_prob: float = 0.02,
    event_multiplier: float = 2.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic hourly parking occupancy data."""
    np.random.seed(seed)

    start = pd.Timestamp(start_date)
    timestamps = pd.date_range(start, periods=days * 24, freq="h")

    # US holidays for 2023
    us_holidays = holidays.US(years=[2023, 2024])

    data = []
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek  # 0=Mon, 6=Sun
        is_weekend = day_of_week >= 5
        is_holiday = ts.date() in us_holidays

        # Base occupancy pattern (hourly cycle)
        # Low at night, peaks at 10am and 6pm
        if 6 <= hour <= 10:
            base = 0.6 + 0.3 * np.sin((hour - 6) * np.pi / 4)
        elif 11 <= hour <= 14:
            base = 0.85
        elif 15 <= hour <= 19:
            base = 0.75 + 0.2 * np.sin((hour - 15) * np.pi / 4)
        else:
            base = 0.15 + 0.1 * np.sin((hour - 20) * np.pi / 4)

        # Weekend adjustment
        if is_weekend:
            base *= 0.7  # Less commuter traffic
            if 10 <= hour <= 14:
                base *= 1.3  # But more shopping/leisure

        # Holiday adjustment
        if is_holiday:
            base *= 0.5  # Much quieter

        # Scale to capacity
        occupancy = base * base_capacity

        # Add noise
        occupancy += np.random.normal(0, noise_std)
        occupancy = max(0, min(base_capacity, occupancy))  # Clip to capacity

        # Random special events (concerts, games, etc.)
        is_event = np.random.random() < event_prob
        if is_event and 16 <= hour <= 22:
            occupancy = min(base_capacity, occupancy * event_multiplier)

        data.append(
            {
                "timestamp": ts,
                "occupancy": round(occupancy),
                "capacity": base_capacity,
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": int(is_weekend),
                "is_holiday": int(is_holiday),
                "is_event": int(is_event),
            }
        )

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic parking data")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--days", type=int, default=365, help="Number of days to generate"
    )
    parser.add_argument("--capacity", type=int, default=200, help="Parking capacity")
    parser.add_argument("--output", default="data/parking.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = generate_parking_data(
        start_date=args.start,
        days=args.days,
        base_capacity=args.capacity,
        seed=args.seed,
    )

    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} rows -> {args.output}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Occupancy range: {df['occupancy'].min()} - {df['occupancy'].max()}")
    print(f"Mean occupancy: {df['occupancy'].mean():.1f}")


if __name__ == "__main__":
    main()

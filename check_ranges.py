import pandas as pd

df = pd.read_csv('data/area_history.csv', index_col='timestamp', parse_dates=True)

targets = ['humidity', 'temperature', 'NO2', 'O3', 'PM10', 'PM2_5', 'co']

print(f"\n{'Target':<12} {'Min':>8} {'Max':>8} {'Range':>8}")
print("-" * 40)
for t in targets:
    mn  = df[t].min()
    mx  = df[t].max()
    rng = mx - mn
    print(f"{t:<12} {mn:>8.4f} {mx:>8.4f} {rng:>8.4f}")
print(df.shape)
print(df.index.min(), df.index.max())
print(df[targets].describe())
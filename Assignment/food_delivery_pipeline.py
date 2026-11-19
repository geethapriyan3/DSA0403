# ============================================================
# DSA04 - CO1 Assignment 1 : Food-delivery delay analysis (Hyderabad)
# Run order: python food_delivery_pipeline.py   (Python 3.10+, seed = 42)
# ============================================================
import json, sqlite3, warnings, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix)
warnings.filterwarnings("ignore")
SEED = 42
PEAK_HOURS = [12, 13, 20, 21, 22]
LATE_MIN = 15
BLUE, ORANGE, GREY, TEAL = "#1f4e79", "#e07b00", "#7f8c8d", "#2a9d8f"
sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
plt.rcParams.update({"axes.titleweight": "bold", "axes.titlesize": 11, "figure.dpi": 110})
OUT, R = {}, {}          # OUT = console outputs per stage, R = values/tables for the report

def show(stage, title, obj):
    s = obj if isinstance(obj, str) else obj.to_string()
    OUT.setdefault(stage, []).append(f">>> {title}\n{s}")
    print(f"\n[{stage}] {title}\n{s}")

def tbl(df, index=True, nd=2):
    d = df.reset_index() if index else df.copy()
    rows = []
    for r in d.itertuples(index=False):
        rows.append([("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))) for v in r])
    return {"cols": [str(c) for c in d.columns], "rows": rows}

def savefig(name):
    plt.tight_layout(); plt.savefig(f"figs/{name}.png", dpi=160, bbox_inches="tight"); plt.close()

# ------------------------------------------------------------
# 0. DATASET PREPARATION  (script supplied in the question paper)
# ------------------------------------------------------------
rng = np.random.default_rng(2026)
hubs = pd.DataFrame({
    'hub_id': [f'H{i:03d}' for i in range(1, 41)],
    'zone': rng.choice(['Urban', 'Suburban', 'Outskirts'], 40, p=[0.5, 0.3, 0.2]),
    'avg_distance_km': rng.normal(4.5, 1.5, 40).round(1).clip(1),
    'num_riders': rng.integers(6, 30, 40),
    'kitchen_type': rng.choice(['Cloud', 'Restaurant', 'Premium'], 40, p=[0.6, 0.3, 0.1]),
})
hubs.loc[rng.choice(40, 3, replace=False), 'avg_distance_km'] = np.nan
dates = pd.date_range('2025-01-01', periods=365)
weather = pd.DataFrame({
    'date': dates.strftime('%Y-%m-%d'),
    'rain_mm': rng.gamma(0.4, 12, 365).round(1),
    'temp_c': rng.normal(30, 3, 365).round(1),
})
weather.to_json('data/weather.json', orient='records')
n = 5000
orders = pd.DataFrame({
    'order_id': np.arange(1, n + 1),
    'hub_id': rng.choice(hubs['hub_id'], n),
    'date': rng.choice(dates.strftime('%d-%m-%Y'), n),
    'hour': rng.integers(10, 24, n),
})
d = pd.to_datetime(orders['date'], format='%d-%m-%Y')
true_type = np.where(d.dt.dayofweek >= 5, 'Weekend', 'Weekday')
orders['day_type'] = [rng.choice([t, t.lower(), t.upper()]) for t in true_type]
rain = d.dt.strftime('%Y-%m-%d').map(weather.set_index('date')['rain_mm'])
dist = orders['hub_id'].map(hubs.set_index('hub_id')['avg_distance_km']).fillna(4.5)
peak = orders['hour'].isin([12, 13, 20, 21, 22]).astype(int)
wkend = (true_type == 'Weekend').astype(int)
orders['order_value'] = (250 + 120 * peak + 40 * wkend + 2 * rain + rng.normal(0, 40, n)).round()
orders['delay_min'] = (4 + 0.3 * rain + 5 * peak + 0.8 * dist + rng.normal(0, 2.5, n)).round(1)
orders.loc[rng.choice(n, 150, replace=False), 'order_value'] = np.nan
orders.loc[rng.choice(n, 20, replace=False), 'delay_min'] = 180
orders.loc[rng.choice(n, 8, replace=False), 'order_value'] = -100
orders.loc[rng.choice(n, 6, replace=False), 'delay_min'] = -5
orders = pd.concat([orders, orders.sample(50, random_state=1)])
orders.to_csv('data/orders.csv', index=False)
hubs.to_csv('data/hubs.csv', index=False)

# ============================================================
# STAGE 1(b) : RETRIEVING DATA  (CSV, JSON -> SQLite -> SQL)
# ============================================================
orders_raw = pd.read_csv('data/orders.csv')
hubs_raw = pd.read_csv('data/hubs.csv')
weather_raw = pd.read_json('data/weather.json', convert_dates=False)   # keep date as text

con = sqlite3.connect('data/food_delivery.db')
orders_raw.to_sql('orders', con, if_exists='replace', index=False)
hubs_raw.to_sql('hubs', con, if_exists='replace', index=False)
weather_raw.to_sql('weather', con, if_exists='replace', index=False)
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)
show('1b', 'Tables created in SQLite database', tables)

q1 = """SELECT hour, COUNT(*) AS n_orders, ROUND(AVG(delay_min),2) AS avg_delay_min
        FROM orders WHERE delay_min BETWEEN 0 AND 100
        GROUP BY hour ORDER BY hour"""
q2 = """SELECT h.zone, COUNT(*) AS n_orders, ROUND(AVG(o.order_value),1) AS avg_order_value,
               ROUND(AVG(CASE WHEN o.delay_min BETWEEN 0 AND 100 THEN o.delay_min END),2) AS avg_delay_min
        FROM orders o JOIN hubs h ON o.hub_id = h.hub_id
        GROUP BY h.zone ORDER BY avg_delay_min DESC"""
q3 = """SELECT w.date, w.rain_mm, COUNT(*) AS n_orders,
               ROUND(AVG(CASE WHEN o.delay_min BETWEEN 0 AND 100 THEN o.delay_min END),2) AS avg_delay
        FROM orders o JOIN weather w
          ON w.date = substr(o.date,7,4)||'-'||substr(o.date,4,2)||'-'||substr(o.date,1,2)
        GROUP BY w.date, w.rain_mm ORDER BY w.rain_mm DESC LIMIT 5"""
sq1, sq2, sq3 = [pd.read_sql_query(q, con) for q in (q1, q2, q3)]
con.close()
show('1b', 'SQL 1 - orders and average delay per hour', sq1)
show('1b', 'SQL 2 - JOIN orders with hubs, summary by zone', sq2)
show('1b', 'SQL 3 - JOIN orders with weather (date reformatted in SQL), 5 wettest days', sq3)
R['sql'] = {'q1': q1, 'q2': q2, 'q3': q3, 't1': tbl(sq1, False), 't2': tbl(sq2, False), 't3': tbl(sq3, False)}

# ---- first data audit ----
audit_rows = []
for name, df in [('orders', orders_raw), ('hubs', hubs_raw), ('weather', weather_raw)]:
    miss = df.isna().sum()
    audit_rows.append({'Dataset': name, 'Rows': len(df), 'Columns': df.shape[1],
                       'Missing cells': int(miss.sum()),
                       'Duplicate rows': int(df.duplicated().sum()),
                       'Columns with missing values': ', '.join(f"{c} ({v})" for c, v in miss.items() if v) or 'none'})
audit = pd.DataFrame(audit_rows)
show('1b', 'Data audit summary', audit)
for name, df in [('orders', orders_raw), ('hubs', hubs_raw), ('weather', weather_raw)]:
    show('1b', f'{name}.shape / dtypes', f"shape = {df.shape}\n{df.dtypes.to_string()}")
    show('1b', f'{name}.describe()', df.describe().round(2))
dtypes_tbl = pd.DataFrame({'orders': orders_raw.dtypes.astype(str), 'hubs': hubs_raw.dtypes.astype(str),
                           'weather': weather_raw.dtypes.astype(str)}).fillna('-')
desc_orders = orders_raw[['hour', 'order_value', 'delay_min']].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
desc_hubs = hubs_raw[['avg_distance_km', 'num_riders']].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
desc_w = weather_raw[['rain_mm', 'temp_c']].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
R['audit'] = tbl(audit, False)
R['desc_raw'] = {'orders': tbl(desc_orders), 'hubs': tbl(desc_hubs), 'weather': tbl(desc_w)}
raw_issues = {
    'rows_raw': len(orders_raw), 'dups': int(orders_raw.duplicated().sum()),
    'miss_value': int(orders_raw.order_value.isna().sum()),
    'neg_value': int((orders_raw.order_value < 0).sum()),
    'neg_delay': int((orders_raw.delay_min < 0).sum()),
    'delay_180': int((orders_raw.delay_min == 180).sum()),
    'miss_dist': int(hubs_raw.avg_distance_km.isna().sum()),
    'daytype_labels': orders_raw.day_type.value_counts().to_dict(),
    'max_delay_raw': float(orders_raw.delay_min.max()), 'min_value_raw': float(orders_raw.order_value.min()),
}
show('1b', 'day_type labels (raw)', orders_raw.day_type.value_counts())
R['raw_issues'] = raw_issues

# ---- process diagram (Figure 1) ----
fig, ax = plt.subplots(figsize=(11, 5.6)); ax.axis('off'); ax.set_xlim(0, 11); ax.set_ylim(0, 5.6)
steps = [
 ("1. Set the research goal", "Predict delay, explain order value,\nfind zones/slots needing riders\n(BQ1, BQ2, BQ3)"),
 ("2. Retrieve data", "orders.csv + hubs.csv + weather.json\n-> pandas -> SQLite -> SQL JOIN"),
 ("3. Data preparation", "Cleanse (dups, NaN, outliers),\nintegrate (3 tables), transform\n(features, encoding, scaling)"),
 ("4. Data analysis", "Statistics, 6+ charts, pivots,\nt-test (rainy vs dry),\n3 insights"),
 ("5. Build the models", "Regression (delay_min) and\nclassification (late > 15 min),\nCV + overfitting check"),
 ("6. Present & automate", "Executive summary, dashboard,\njoblib model + prediction tool,\nrecommendations"),
]
pos = [(0.2, 3.0), (3.95, 3.0), (7.7, 3.0), (7.7, 0.5), (3.95, 0.5), (0.2, 0.5)]
w, h = 3.1, 1.9
for (t, s), (x, y) in zip(steps, pos):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc="#eaf1f8", ec=BLUE, lw=1.6, zorder=2))
    ax.add_patch(plt.Rectangle((x, y + h - 0.5), w, 0.5, fc=BLUE, ec=BLUE, zorder=3))
    ax.text(x + w / 2, y + h - 0.25, t, ha='center', va='center', color='white', fontsize=10.5, fontweight='bold', zorder=4)
    ax.text(x + w / 2, y + (h - 0.5) / 2, s, ha='center', va='center', fontsize=9, zorder=4)
def arrow(a, b): ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2), zorder=5)
arrow((3.3, 3.95), (3.95, 3.95)); arrow((7.05, 3.95), (7.7, 3.95)); arrow((9.25, 3.0), (9.25, 2.4))
arrow((7.7, 1.45), (7.05, 1.45)); arrow((3.95, 1.45), (3.3, 1.45))
ax.annotate("", xy=(5.5, 3.0), xytext=(5.5, 2.4), arrowprops=dict(arrowstyle="<|-|>", color=GREY, lw=1.2, ls="--"))
ax.text(5.6, 2.7, "iterate: re-clean / re-engineer\nwhen models or charts expose problems", fontsize=8, color=GREY, va='center')
ax.set_title("Figure 1: Data-science process applied to the Hyderabad food-delivery project", fontsize=12, pad=6)
savefig('fig01_process')

# ============================================================
# STAGE 2 : CLEANSING, INTEGRATING, TRANSFORMING
# ============================================================
o = orders_raw.copy()
clog = []                                          # cleansing log rows
# (a) data types / date formats
o['date'] = pd.to_datetime(o['date'], format='%d-%m-%Y')
clog.append(['Date format', 'date stored as text dd-mm-yyyy (weather uses yyyy-mm-dd)', len(o), 'Parsed with pd.to_datetime(format="%d-%m-%Y")', 'datetime64'])
# (b) duplicates
n_dup = int(o.duplicated().sum()); rows0 = len(o)
o = o.drop_duplicates().reset_index(drop=True)
assert o.order_id.is_unique
clog.append(['Duplicate orders', f'{n_dup} exact duplicate rows (same order_id)', rows0, 'drop_duplicates(); order_id now unique', len(o)])
# (c) labels
lab_before = o.day_type.nunique()
o['day_type'] = o['day_type'].str.strip().str.capitalize()
clog.append(['Inconsistent labels', f'day_type had {lab_before} spellings (WEEKDAY/weekday/Weekday ...)', lab_before, 'str.strip().str.capitalize()', o.day_type.nunique()])
o['is_weekend'] = (o['date'].dt.dayofweek >= 5).astype(int)
label_mismatch = int(((o.day_type == 'Weekend').astype(int) != o.is_weekend).sum())
o['is_peak'] = o['hour'].isin(PEAK_HOURS).astype(int)
# (d) impossible order_value
neg_v = int((o.order_value < 0).sum())
o.loc[o.order_value < 0, 'order_value'] = np.nan
# (e) missing order_value -> group-wise median (peak x weekend)
miss_v = int(o.order_value.isna().sum())
o['order_value'] = o.groupby(['is_peak', 'is_weekend'])['order_value'].transform(lambda s: s.fillna(s.median()))
clog.append(['Impossible order_value', f'{neg_v} orders with value = -100 INR', neg_v, 'Set to NaN, then treated with the missing-value rule below', 0])
clog.append(['Missing order_value', f'{miss_v} missing values (incl. the {neg_v} reset above)', miss_v, 'Group-wise median by (is_peak, is_weekend), because order value depends on both', int(o.order_value.isna().sum())])
# (f) impossible delay
neg_d = int((o.delay_min < 0).sum()); rows1 = len(o)
o = o[o.delay_min >= 0].copy()
clog.append(['Impossible delay_min', f'{neg_d} orders with delay = -5 min', neg_d, 'Rows removed (target variable - never invent target values)', 0])
# (g) outliers in delay_min (IQR)
q1_, q3_ = o.delay_min.quantile([.25, .75]); iqr = q3_ - q1_
f15 = q3_ + 1.5 * iqr; f3 = q3_ + 3 * iqr
n15, n3 = int((o.delay_min > f15).sum()), int((o.delay_min > f3).sum())
show('2', 'IQR outlier rule on delay_min', f"Q1={q1_:.2f}  Q3={q3_:.2f}  IQR={iqr:.2f}\nupper fence 1.5xIQR = {f15:.2f}  -> {n15} rows above\nupper fence 3.0xIQR = {f3:.2f}  -> {n3} rows above\nvalues above 3xIQR fence: {[float(v) for v in sorted(o.delay_min[o.delay_min > f3].unique())]}")
o = o[o.delay_min <= f3].copy()
clog.append(['Extreme outliers in delay_min', f'{n3} orders at 180 min (fence {f3:.1f} min, 3xIQR); {n15 - n3} more lie between the 1.5x and 3x fences and are genuine heavy-rain delays', n3, 'Rows above the 3xIQR fence removed; moderate high values kept', int((o.delay_min > f3).sum())])
o = o.reset_index(drop=True)
# (h) hubs: missing avg_distance_km -> zone median
h = hubs_raw.copy()
miss_d = int(h.avg_distance_km.isna().sum())
h['avg_distance_km'] = h.groupby('zone')['avg_distance_km'].transform(lambda s: s.fillna(s.median()))
clog.append(['Missing avg_distance_km (hubs)', f'{miss_d} of 40 hubs have no distance', miss_d, 'Zone-wise median (hub table is tiny; removal would lose whole hubs)', int(h.avg_distance_km.isna().sum())])
# (i) weather
wx = weather_raw.copy(); wx['date'] = pd.to_datetime(wx['date'], format='%Y-%m-%d')
assert wx.date.is_unique and wx.isna().sum().sum() == 0
show('2', 'Cleansing log', pd.DataFrame(clog, columns=['Issue', 'Finding', 'Before', 'Treatment', 'After']))
R['clog'] = {'cols': ['Issue', 'Finding / treatment', 'Before', 'After'],
             'rows': [[c[0], f"{c[1]}. {c[3]}.", str(c[2]), str(c[4])] for c in clog]}
R['clean_counts'] = {'n_raw': rows0, 'n_dup': n_dup, 'neg_v': neg_v, 'miss_v': miss_v, 'neg_d': neg_d, 'n15': n15, 'n3': n3,
                     'f15': float(f15), 'f3': float(f3), 'q1': float(q1_), 'q3': float(q3_), 'iqr': float(iqr),
                     'label_mismatch': label_mismatch, 'n_final': len(o), 'miss_d': miss_d}
imp_check = o.groupby(['is_peak', 'is_weekend']).order_value.median().round(1)
show('2', 'order_value median per (is_peak,is_weekend) used for imputation', imp_check)

# ---- integration ----
n_before = len(o)
# demonstration: merging on the un-harmonised date text silently matches nothing
bad = orders_raw.drop_duplicates().merge(weather_raw, on='date', how='left')
unmatched_text = int(bad.rain_mm.isna().sum())
m = o.merge(h, on='hub_id', how='left', validate='many_to_one')
m = m.merge(wx, on='date', how='left', validate='many_to_one')
n_after = len(m)
show('2', 'Integration row counts',
     f"orders (clean) rows : {n_before}\nafter LEFT JOIN hubs + weather : {n_after}\n"
     f"rows lost/duplicated : {n_after - n_before}\nunmatched hub rows : {int(m.zone.isna().sum())}, unmatched weather rows : {int(m.rain_mm.isna().sum())}\n"
     f"(merge on raw text dates would leave {unmatched_text} of {len(bad)} rows without weather)")
assert n_after == n_before and m.zone.notna().all() and m.rain_mm.notna().all()
R['integration'] = {'before': n_before, 'after': n_after, 'unmatched_text': unmatched_text, 'bad_total': len(bad)}
schema = pd.DataFrame({'Column': m.columns, 'Source': ['orders' if c in o.columns else ('hubs' if c in h.columns else 'weather') for c in m.columns], 'dtype': m.dtypes.astype(str).values})
schema['Source'] = schema.apply(lambda r: 'orders' if r.Column in ['order_id','hub_id','date','hour','day_type','order_value','delay_min','is_weekend','is_peak'] else r.Source, axis=1)
R['schema'] = tbl(schema, False)

# ---- transformation ----
m['is_rainy'] = (m.rain_mm > 2.5).astype(int)
m['month'] = m.date.dt.month
m['weekday'] = m.date.dt.day_name()
m['rain_cat'] = pd.cut(m.rain_mm, [-0.01, 2.5, 10, 30, np.inf], labels=['Dry (<=2.5 mm)', 'Light (2.5-10)', 'Moderate (10-30)', 'Heavy (>30)']).astype(str)
def slot(hr): return 'Morning (10-11)' if hr < 12 else 'Lunch peak (12-13)' if hr < 14 else 'Afternoon (14-19)' if hr < 20 else 'Dinner peak (20-22)' if hr < 23 else 'Late night (23)'
SLOTS = ['Morning (10-11)', 'Lunch peak (12-13)', 'Afternoon (14-19)', 'Dinner peak (20-22)', 'Late night (23)']
m['time_slot'] = m.hour.map(slot)
m['is_late'] = (m.delay_min > LATE_MIN).astype(int)
skew_rain, skew_val, skew_delay = m.rain_mm.skew(), m.order_value.skew(), m.delay_min.skew()
m['log_rain'] = np.log1p(m.rain_mm)
skew_lograin = m.log_rain.skew()
onehot_demo = pd.get_dummies(m[['zone', 'kitchen_type']], dtype=int)
show('2', 'Skewness', f"rain_mm skew = {skew_rain:.2f}  ->  log1p(rain_mm) skew = {skew_lograin:.2f}\norder_value skew = {skew_val:.2f}\ndelay_min skew = {skew_delay:.2f}")
show('2', 'One-hot columns created', ', '.join(onehot_demo.columns))
R['skew'] = {'rain': float(skew_rain), 'lograin': float(skew_lograin), 'value': float(skew_val), 'delay': float(skew_delay)}
R['onehot_cols'] = list(onehot_demo.columns)
R['rain_cat_counts'] = m.rain_cat.value_counts().to_dict()

# daily hub-level aggregation (BQ3)
daily = m.groupby(['hub_id', 'date']).agg(zone=('zone', 'first'), num_riders=('num_riders', 'first'),
        avg_distance_km=('avg_distance_km', 'first'), rain_mm=('rain_mm', 'first'),
        n_orders=('order_id', 'count'), avg_delay=('delay_min', 'mean'), late_share=('is_late', 'mean'),
        peak_share=('is_peak', 'mean'), avg_order_value=('order_value', 'mean')).reset_index()
daily['orders_per_rider'] = daily.n_orders / daily.num_riders
daily.to_csv('data/daily_hub_level.csv', index=False)
show('2', 'daily hub-level table (head)', daily.head(5).round(2))
R['daily'] = {'shape': list(daily.shape), 'mean_orders': float(daily.n_orders.mean()), 'max_orders': int(daily.n_orders.max()),
              'share_single': float((daily.n_orders == 1).mean()), 'head': tbl(daily.head(5)[['hub_id', 'date', 'zone', 'n_orders', 'avg_delay', 'late_share', 'orders_per_rider']].assign(date=lambda x: x.date.dt.strftime('%Y-%m-%d')), False)}

transform_tbl = pd.DataFrame([
 ['is_peak', 'hour in {12,13,20,21,22}', 'Lunch/dinner rush flag given in the brief'],
 ['is_weekend', 'date.dayofweek >= 5 (replaces messy day_type)', 'Derived from the true date, so labels cannot disagree'],
 ['is_rainy', 'rain_mm > 2.5', 'Threshold given for the hypothesis test'],
 ['month, weekday', 'date.dt.month / day_name()', 'Seasonality and day-of-week patterns'],
 ['rain_cat', 'Dry / Light / Moderate / Heavy bins', 'Readable rain grading for charts and managers'],
 ['time_slot', 'Morning, Lunch peak, Afternoon, Dinner peak, Late night', 'Slot-level view for BQ3'],
 ['is_late', 'delay_min > 15', 'Target of the classification model'],
 ['log_rain', f'log1p(rain_mm): skew {skew_rain:.2f} -> {skew_lograin:.2f}', 'Rain is highly right-skewed (many dry days, few storms)'],
 ['One-hot: zone, kitchen_type', 'OneHotEncoder (fit on train only)', 'Nominal categories with no natural order'],
 ['Scaling: hour, distance, riders, rain, temp', 'StandardScaler (fit on train only)', 'Needed by Logistic Regression/KNN, makes coefficients comparable'],
], columns=['Feature / step', 'Rule', 'Reason'])
R['transform'] = tbl(transform_tbl, False)

# ---- validation (assert-based) ----
checks = []
def check(name, cond):
    assert cond, name; checks.append([name, 'PASS'])
check('No missing values in any column', m.isna().sum().sum() == 0)
check('No duplicate orders (order_id unique)', m.order_id.is_unique and not m.duplicated().any())
check('hour only between 10 and 23', m.hour.between(10, 23).all())
check('day_type labels are only Weekday / Weekend', set(m.day_type) == {'Weekday', 'Weekend'})
check('day_type agrees with calendar date', label_mismatch == 0)
check('order_value > 0 and delay_min >= 0', (m.order_value > 0).all() and (m.delay_min >= 0).all())
check('delay_min within plausible range (<= 3xIQR fence)', m.delay_min.max() <= f3)
check(f'Row count preserved by merge ({n_before})', len(m) == n_before)
check('zone labels are Urban / Suburban / Outskirts', set(m.zone) == {'Urban', 'Suburban', 'Outskirts'})
check('every hub_id found in the hubs table', m.hub_id.isin(h.hub_id).all())
show('2', 'Assert-based validation', pd.DataFrame(checks, columns=['Check', 'Result']))
R['checks'] = tbl(pd.DataFrame(checks, columns=['Check', 'Result']), False)
m.to_csv('data/cleaned_orders.csv', index=False)
R['clean_shape'] = list(m.shape)
before_after = pd.DataFrame({
  'Metric': ['Rows (orders)', 'Missing order_value', 'Negative order_value', 'Negative delay_min', 'delay_min = 180 (outlier)', 'day_type spellings', 'Hubs without distance', 'Max delay_min (min)'],
  'Before': [len(orders_raw), int(orders_raw.order_value.isna().sum()), int((orders_raw.order_value < 0).sum()), int((orders_raw.delay_min < 0).sum()), int((orders_raw.delay_min == 180).sum()), orders_raw.day_type.nunique(), int(hubs_raw.avg_distance_km.isna().sum()), orders_raw.delay_min.max()],
  'After': [len(m), int(m.order_value.isna().sum()), int((m.order_value < 0).sum()), int((m.delay_min < 0).sum()), int((m.delay_min == 180).sum()), m.day_type.nunique(), int(h.avg_distance_km.isna().sum()), m.delay_min.max()]})
R['before_after'] = tbl(before_after, False, 1)
show('2', 'Before / after', before_after)

# ============================================================
# STAGE 3 : ANALYSIS AND EXPLORATION
# ============================================================
m['peak_label'] = np.where(m.is_peak == 1, 'Peak', 'Off-peak')
def summary(col):
    rows = []
    def add(label, s):
        q = s.quantile([.25, .5, .75])
        rows.append([label, len(s), s.mean(), q[.5], s.std(), s.min(), q[.25], q[.75], s.max()])
    add('Overall', m[col])
    for g in ['zone', 'kitchen_type', 'peak_label', 'day_type']:
        for k, s in m.groupby(g)[col]: add(f'{g}: {k}', s)
    return pd.DataFrame(rows, columns=['Group', 'n', 'Mean', 'Median', 'Std', 'Min', 'Q1', 'Q3', 'Max'])
s_val, s_del = summary('order_value'), summary('delay_min')
show('3', 'Descriptive statistics - order_value (INR)', s_val.round(1))
show('3', 'Descriptive statistics - delay_min (min)', s_del.round(2))
R['stats_val'], R['stats_del'] = tbl(s_val, False, 1), tbl(s_del, False, 2)

pv_zone_slot = m.pivot_table(index='zone', columns='time_slot', values='delay_min', aggfunc='mean')[SLOTS].round(2)
pv_late = (m.pivot_table(index='zone', columns='time_slot', values='is_late', aggfunc='mean')[SLOTS] * 100).round(1)
pv_cnt = m.pivot_table(index='zone', columns='time_slot', values='order_id', aggfunc='count')[SLOTS]
pv_hour = m.groupby('hour').agg(orders=('order_id', 'count'), avg_delay=('delay_min', 'mean'), late_pct=('is_late', lambda s: s.mean() * 100), avg_value=('order_value', 'mean')).round(2)
show('3', 'Pivot: mean delay (min) by zone x time slot', pv_zone_slot)
show('3', 'Pivot: late-order % by zone x time slot', pv_late)
show('3', 'groupby: orders, delay, late %, value by hour', pv_hour)
R['pv_zone_slot'], R['pv_late'], R['pv_cnt'] = tbl(pv_zone_slot), tbl(pv_late), tbl(pv_cnt, nd=0)
R['pv_hour'] = tbl(pv_hour)
zone_tbl = m.groupby('zone').agg(hubs=('hub_id', 'nunique'), orders=('order_id', 'count'), avg_delay=('delay_min', 'mean'), late_pct=('is_late', lambda s: s.mean() * 100), avg_value=('order_value', 'mean'), avg_dist=('avg_distance_km', 'mean'), riders_per_hub=('num_riders', 'mean')).round(2)
R['zone_tbl'] = tbl(zone_tbl)
show('3', 'Zone summary', zone_tbl)

# six + extra charts
# Fig 2 histogram
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(m.delay_min, bins=40, color=BLUE, edgecolor='white')
ax.axvline(LATE_MIN, color=ORANGE, ls='--', lw=2); ax.text(LATE_MIN + .4, ax.get_ylim()[1] * .9, f'Late threshold\n(15 min)', color=ORANGE, fontsize=9)
ax.set(title='Figure 2: Distribution of delivery delay', xlabel='Delay beyond promised time (minutes)', ylabel='Number of orders')
savefig('fig02_hist_delay')
# Fig 3 box plot by rain category
cat_order = ['Dry (<=2.5 mm)', 'Light (2.5-10)', 'Moderate (10-30)', 'Heavy (>30)']
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=m, x='rain_cat', y='delay_min', order=cat_order, color="#9ec1e3", ax=ax, fliersize=2)
ax.set(title='Figure 3: Delay by rainfall category', xlabel='Daily rainfall category (mm)', ylabel='Delay (minutes)')
savefig('fig03_box_rain')
# Fig 4 bar by hour
fig, ax = plt.subplots(figsize=(7.5, 4))
cols = [ORANGE if hh in PEAK_HOURS else BLUE for hh in pv_hour.index]
ax.bar(pv_hour.index, pv_hour.avg_delay, color=cols)
ax.set(title='Figure 4: Average delay by order hour (peak hours in orange)', xlabel='Hour of day (24-h clock)', ylabel='Average delay (minutes)'); ax.set_xticks(pv_hour.index)
savefig('fig04_bar_hour')
# Fig 5 time series
ts = m.groupby('date').agg(delay=('delay_min', 'mean'), rain=('rain_mm', 'first'))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ts.index, ts.delay.rolling(7, min_periods=3).mean(), color=BLUE, lw=2, label='Avg delay (7-day rolling)')
ax.set(title='Figure 5: Daily average delay through 2025', xlabel='Date', ylabel='Average delay (minutes)')
ax2 = ax.twinx(); ax2.bar(ts.index, ts.rain, color=TEAL, alpha=.35, width=1, label='Rainfall'); ax2.set_ylabel('Rainfall (mm)'); ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)
fig.autofmt_xdate(); savefig('fig05_timeseries')
# Fig 6 scatter
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(m.rain_mm, m.delay_min, s=7, alpha=.35, color=BLUE)
sl, ic, rv, pvv, se = stats.linregress(m.rain_mm, m.delay_min)
xs = np.linspace(0, m.rain_mm.max(), 50); ax.plot(xs, ic + sl * xs, color=ORANGE, lw=2.2, label=f'Trend: +{sl:.2f} min per mm (r = {rv:.2f})')
ax.set(title='Figure 6: Delay against daily rainfall', xlabel='Daily rainfall (mm)', ylabel='Delay (minutes)'); ax.legend()
savefig('fig06_scatter')
R['rain_fit'] = {'slope': float(sl), 'intercept': float(ic), 'r': float(rv)}
# Fig 7 heatmap
corr_cols = ['order_value', 'delay_min', 'hour', 'is_peak', 'is_weekend', 'rain_mm', 'temp_c', 'avg_distance_km', 'num_riders']
corr = m[corr_cols].corr()
fig, ax = plt.subplots(figsize=(7.2, 5.8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Pearson correlation (r)'}, annot_kws={'size': 8})
ax.set_title('Figure 7: Correlation heat map of numeric variables'); plt.xticks(rotation=40, ha='right')
savefig('fig07_corr')
R['corr_key'] = {k: float(corr.loc[k[0], k[1]]) for k in [('delay_min', 'is_peak'), ('delay_min', 'rain_mm'), ('delay_min', 'avg_distance_km'), ('delay_min', 'num_riders'), ('delay_min', 'temp_c'), ('delay_min', 'is_weekend'), ('order_value', 'is_peak'), ('order_value', 'is_weekend'), ('order_value', 'rain_mm'), ('order_value', 'delay_min'), ('order_value', 'avg_distance_km')]}
R['corr_key'] = {f'{a}|{b}': v for (a, b), v in [(tuple(k.split('|')), v) for k, v in {f'{k[0]}|{k[1]}': float(corr.loc[k[0], k[1]]) for k in [('delay_min', 'is_peak'), ('delay_min', 'rain_mm'), ('delay_min', 'avg_distance_km'), ('delay_min', 'num_riders'), ('delay_min', 'temp_c'), ('delay_min', 'is_weekend'), ('order_value', 'is_peak'), ('order_value', 'is_weekend'), ('order_value', 'rain_mm'), ('order_value', 'delay_min'), ('order_value', 'avg_distance_km')]}.items()]}
show('3', 'Key correlations', pd.Series(R['corr_key']).round(3))
# Fig 8 heat map: late rate zone x slot
fig, ax = plt.subplots(figsize=(9, 3.9))
sns.heatmap(pv_late, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Late orders (%)'})
ax.set(title='Figure 8: Late-order share (%) by zone and time slot', xlabel='Time slot', ylabel='Zone')
ax.tick_params(axis='x', rotation=0, labelsize=9); ax.tick_params(axis='y', rotation=0)
savefig('fig08_late_zone_slot')
# Fig 9 order value drivers
vv = m.groupby(['peak_label', 'day_type']).order_value.mean().unstack()
fig, ax = plt.subplots(figsize=(6.2, 4))
vv.loc[['Off-peak', 'Peak']].plot(kind='bar', ax=ax, color=[BLUE, ORANGE], rot=0)
for c in ax.containers: ax.bar_label(c, fmt='%.0f', fontsize=8)
ax.set(title='Figure 9: Average order value by time of day and day type', xlabel='Time of day', ylabel='Average order value (INR)'); ax.legend(title='Day type')
savefig('fig09_value_drivers')
R['value_means'] = {f'{a}|{b}': float(vv.loc[a, b]) for a in vv.index for b in vv.columns}

# hypothesis test
rainy, dry = m.delay_min[m.is_rainy == 1], m.delay_min[m.is_rainy == 0]
tt = stats.ttest_ind(rainy, dry, equal_var=False, alternative='greater')
sp = np.sqrt(((len(rainy) - 1) * rainy.var() + (len(dry) - 1) * dry.var()) / (len(rainy) + len(dry) - 2))
cohen = (rainy.mean() - dry.mean()) / sp
diff = rainy.mean() - dry.mean()
se_d = np.sqrt(rainy.var() / len(rainy) + dry.var() / len(dry)); ci = (diff - 1.96 * se_d, diff + 1.96 * se_d)
show('3', 'Welch t-test: delay on rainy (>2.5 mm) vs dry days',
     f"rainy: n={len(rainy)} mean={rainy.mean():.2f}\ndry:   n={len(dry)} mean={dry.mean():.2f}\n"
     f"difference = {diff:.2f} min  (95% CI {ci[0]:.2f} to {ci[1]:.2f})\nt = {tt.statistic:.2f}, one-sided p = {tt.pvalue:.3g}, Cohen's d = {cohen:.2f}")
R['ttest'] = {'n_r': len(rainy), 'n_d': len(dry), 'm_r': float(rainy.mean()), 'm_d': float(dry.mean()), 'diff': float(diff), 'ci': [float(ci[0]), float(ci[1])], 't': float(tt.statistic), 'p': float(tt.pvalue), 'd': float(cohen),
              'late_r': float(m.is_late[m.is_rainy == 1].mean() * 100), 'late_d': float(m.is_late[m.is_rainy == 0].mean() * 100)}
# order-value drivers (BQ1) and delay drivers (BQ2) via multiple regression
Xv = m[['is_peak', 'is_weekend', 'rain_mm']]; lv = LinearRegression().fit(Xv, m.order_value)
R['value_reg'] = {'coef': dict(zip(Xv.columns, map(float, lv.coef_))), 'intercept': float(lv.intercept_), 'r2': float(lv.score(Xv, m.order_value))}
Xd = m[['is_peak', 'rain_mm', 'avg_distance_km', 'num_riders', 'is_weekend']]; ld = LinearRegression().fit(Xd, m.delay_min)
R['delay_reg'] = {'coef': dict(zip(Xd.columns, map(float, ld.coef_))), 'intercept': float(ld.intercept_), 'r2': float(ld.score(Xd, m.delay_min))}
show('3', 'BQ1 regression coefficients (order_value)', pd.Series(R['value_reg']['coef']).round(2))
show('3', 'BQ2 regression coefficients (delay_min)', pd.Series(R['delay_reg']['coef']).round(3))
pk = m.groupby('peak_label').agg(delay=('delay_min', 'mean'), late=('is_late', 'mean'), value=('order_value', 'mean'))
R['peak_cmp'] = {k: {c: float(pk.loc[k, c]) for c in pk.columns} for k in pk.index}
cat_tbl = m.groupby('rain_cat').agg(orders=('order_id', 'count'), delay=('delay_min', 'mean'), late=('is_late', lambda s: s.mean() * 100)).loc[cat_order].round(2)
R['rain_cat_tbl'] = tbl(cat_tbl)
m['dist_band'] = pd.cut(m.avg_distance_km, [0, 3.5, 5, 6.5, 20], labels=['<=3.5 km', '3.5-5 km', '5-6.5 km', '>6.5 km']).astype(str)
dist_tbl = m.groupby('dist_band').agg(hubs=('hub_id', 'nunique'), orders=('order_id', 'count'), delay=('delay_min', 'mean'), late=('is_late', lambda s: s.mean() * 100)).loc[['<=3.5 km', '3.5-5 km', '5-6.5 km', '>6.5 km']].round(2)
R['dist_tbl'] = tbl(dist_tbl)
show('3', 'Delay by distance band', dist_tbl)
dd = daily
R['daily_corr'] = {'opr_delay': float(dd[['orders_per_rider', 'avg_delay']].corr().iloc[0, 1]), 'riders_delay': float(m[['num_riders', 'delay_min']].corr().iloc[0, 1])}
zs = m.groupby(['zone', 'time_slot']).agg(late=('is_late', 'mean'), n=('order_id', 'count'), delay=('delay_min', 'mean')).reset_index()
top = zs.sort_values('late', ascending=False).head(3)
R['top_cells'] = [[r.zone, r.time_slot, float(r.late * 100), int(r.n), float(r.delay)] for r in top.itertuples()]
R['late_overall'] = float(m.is_late.mean() * 100)
R['n_hubs'] = int(m.hub_id.nunique())
R['value_overall'] = float(m.order_value.mean()); R['delay_overall'] = float(m.delay_min.mean())
R['orders_per_hour'] = {int(k): int(v) for k, v in pv_hour.orders.items()}
# rider load per zone & slot (orders per rider per day)
rl = m.groupby('zone').agg(orders=('order_id', 'count')).join(h.groupby('zone').num_riders.sum().rename('riders'))
rl['orders_per_rider_per_day'] = rl.orders / rl.riders / 365
R['rider_load'] = tbl(rl.round(3))

# ============================================================
# STAGE 4 : MODEL BUILDING
# ============================================================
num_f = ['hour', 'is_peak', 'is_weekend', 'rain_mm', 'temp_c', 'avg_distance_km', 'num_riders']
cat_f = ['zone', 'kitchen_type']
X = m[num_f + cat_f]; y_reg = m['delay_min']; y_clf = m['is_late']
X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(X, y_reg, y_clf, test_size=0.20, random_state=SEED)
split_info = {'train': len(X_tr), 'test': len(X_te), 'late_all': float(y_clf.mean()), 'late_train': float(yc_tr.mean()), 'late_test': float(yc_te.mean()),
              'n_late': int(y_clf.sum())}
show('4', 'Train/test split', f"train {len(X_tr)}  test {len(X_te)}  | late share: all {y_clf.mean():.3%}, train {yc_tr.mean():.3%}, test {yc_te.mean():.3%}")
def prep(): return ColumnTransformer([('num', StandardScaler(), ['hour', 'rain_mm', 'temp_c', 'avg_distance_km', 'num_riders']),
                                      ('bin', 'passthrough', ['is_peak', 'is_weekend']),
                                      ('cat', OneHotEncoder(handle_unknown='ignore'), cat_f)])
def rm(y, p): return float(np.sqrt(mean_squared_error(y, p)))
reg_models = {
  'Baseline (mean value)': DummyRegressor(strategy='mean'),
  'Linear Regression': LinearRegression(),
  'Decision Tree (depth 5)': DecisionTreeRegressor(max_depth=5, random_state=SEED),
  'Random Forest (300 trees)': RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
}
reg_fit, reg_rows = {}, []
kf = KFold(5, shuffle=True, random_state=SEED)
for name, est in reg_models.items():
    pipe = Pipeline([('prep', prep()), ('model', est)]).fit(X_tr, yr_tr)
    reg_fit[name] = pipe
    ptr, pte = pipe.predict(X_tr), pipe.predict(X_te)
    cv = -cross_val_score(Pipeline([('prep', prep()), ('model', est)]), X_tr, yr_tr, cv=kf, scoring='neg_root_mean_squared_error')
    reg_rows.append([name, mean_absolute_error(yr_te, pte), rm(yr_te, pte), r2_score(yr_te, pte), rm(yr_tr, ptr), r2_score(yr_tr, ptr), cv.mean(), cv.std()])
reg_df = pd.DataFrame(reg_rows, columns=['Model', 'Test MAE', 'Test RMSE', 'Test R2', 'Train RMSE', 'Train R2', 'CV RMSE (5-fold)', 'CV std'])
base_rmse = reg_df.loc[0, 'Test RMSE']
reg_df['RMSE reduction vs baseline (%)'] = (1 - reg_df['Test RMSE'] / base_rmse) * 100
show('4', 'Regression model comparison (delay_min)', reg_df.round(3))
R['reg_cmp'] = tbl(reg_df.assign(Model=reg_df.Model), False, 3)
best_reg_name = reg_df.iloc[1:].sort_values('Test RMSE').iloc[0].Model
lr_row = reg_df[reg_df.Model == 'Linear Regression'].iloc[0]; rf_row = reg_df[reg_df.Model.str.startswith('Random')].iloc[0]
R['reg_summary'] = {'base_rmse': float(base_rmse), 'best': best_reg_name, 'lr_rmse': float(lr_row['Test RMSE']), 'rf_rmse': float(rf_row['Test RMSE']),
                    'lr_red': float(lr_row['RMSE reduction vs baseline (%)']), 'rf_red': float(rf_row['RMSE reduction vs baseline (%)']),
                    'lr_r2': float(lr_row['Test R2']), 'rf_r2': float(rf_row['Test R2']), 'lr_mae': float(lr_row['Test MAE']), 'rf_mae': float(rf_row['Test MAE']),
                    'target': float(base_rmse * 0.8)}
# overfitting demo: unconstrained tree
dt_full = Pipeline([('prep', prep()), ('model', DecisionTreeRegressor(random_state=SEED))]).fit(X_tr, yr_tr)
R['overfit_tree'] = {'train_rmse': rm(yr_tr, dt_full.predict(X_tr)), 'test_rmse': rm(yr_te, dt_full.predict(X_te)), 'depth': int(dt_full.named_steps['model'].get_depth())}
show('4', 'Overfitting demo (unrestricted tree)', str(R['overfit_tree']))

# classification
clf_models = {
  'Baseline (always on-time)': DummyClassifier(strategy='most_frequent'),
  'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
  'Logistic Regression (balanced)': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED),
  'Decision Tree (depth 5)': DecisionTreeClassifier(max_depth=5, random_state=SEED),
  'KNN (k = 15)': KNeighborsClassifier(n_neighbors=15),
}
clf_fit, clf_rows, cms = {}, [], {}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for name, est in clf_models.items():
    pipe = Pipeline([('prep', prep()), ('model', est)]).fit(X_tr, yc_tr)
    clf_fit[name] = pipe
    pte, ptr = pipe.predict(X_te), pipe.predict(X_tr)
    cv = cross_val_score(Pipeline([('prep', prep()), ('model', est)]), X_tr, yc_tr, cv=skf, scoring='f1')
    cms[name] = confusion_matrix(yc_te, pte)
    clf_rows.append([name, accuracy_score(yc_te, pte), precision_score(yc_te, pte, zero_division=0), recall_score(yc_te, pte), f1_score(yc_te, pte), accuracy_score(yc_tr, ptr), f1_score(yc_tr, ptr), cv.mean(), cv.std()])
clf_df = pd.DataFrame(clf_rows, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Train accuracy', 'Train F1', 'CV F1 (5-fold)', 'CV std'])
show('4', 'Classification model comparison (late > 15 min)', clf_df.round(3))
R['clf_cmp'] = tbl(clf_df, False, 3)
R['cms'] = {k: v.tolist() for k, v in cms.items()}
R['split'] = split_info
# confusion matrices figure (Logistic plain vs balanced)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, nm in zip(axs, ['Logistic Regression', 'Logistic Regression (balanced)']):
    sns.heatmap(cms[nm], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, xticklabels=['On-time', 'Late'], yticklabels=['On-time', 'Late'], annot_kws={'size': 13})
    ax.set(title=nm, xlabel='Predicted', ylabel='Actual')
fig.suptitle('Figure 10: Confusion matrices on the 20% test set', fontweight='bold', fontsize=11)
savefig('fig10_confusion')
# feature names, importance
feat_names = list(reg_fit['Linear Regression'].named_steps['prep'].get_feature_names_out())
clean_names = [f.split('__')[1] for f in feat_names]
coefs = pd.Series(reg_fit['Linear Regression'].named_steps['model'].coef_, index=clean_names)
imps = pd.Series(reg_fit['Random Forest (300 trees)'].named_steps['model'].feature_importances_, index=clean_names)
log_coefs = pd.Series(clf_fit['Logistic Regression (balanced)'].named_steps['model'].coef_[0], index=clean_names)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
cs = coefs.sort_values(); axs[0].barh(cs.index, cs.values, color=[ORANGE if v > 0 else BLUE for v in cs.values])
axs[0].set(title='Linear Regression coefficients', xlabel='Change in delay (min) per 1 SD / per 1 unit')
isr = imps.sort_values(); axs[1].barh(isr.index, isr.values, color=TEAL); axs[1].set(title='Random Forest feature importance', xlabel='Importance (share of variance reduction)')
fig.suptitle('Figure 11: What drives delivery delay?', fontweight='bold', fontsize=11)
savefig('fig11_importance')
R['coefs'] = {k: float(v) for k, v in coefs.sort_values(ascending=False).items()}
R['imps'] = {k: float(v) for k, v in imps.sort_values(ascending=False).items()}
R['log_coefs'] = {k: float(v) for k, v in log_coefs.sort_values(ascending=False).items()}
R['lr_intercept'] = float(reg_fit['Linear Regression'].named_steps['model'].intercept_)
show('4', 'Linear regression coefficients', coefs.round(3).sort_values(ascending=False))
show('4', 'Random forest importances', imps.round(3).sort_values(ascending=False))
# actual vs predicted fig
fig, ax = plt.subplots(figsize=(5.2, 4.6))
pl = reg_fit['Linear Regression'].predict(X_te)
ax.scatter(yr_te, pl, s=8, alpha=.4, color=BLUE); lim = [0, max(yr_te.max(), pl.max()) + 1]; ax.plot(lim, lim, color=ORANGE, lw=2)
ax.set(title='Figure 12: Actual vs predicted delay (test set)', xlabel='Actual delay (minutes)', ylabel='Predicted delay (minutes)')
savefig('fig12_actual_pred')
# final selection
final_reg = 'Linear Regression'; final_clf = 'Logistic Regression (balanced)'
R['final'] = {'reg': final_reg, 'clf': final_clf}
# leakage demonstration: scaler fitted on ALL data vs train only
sc_all = StandardScaler().fit(X[['rain_mm']]); sc_tr = StandardScaler().fit(X_tr[['rain_mm']])
R['leak'] = {'mean_all': float(sc_all.mean_[0]), 'mean_tr': float(sc_tr.mean_[0]), 'std_all': float(sc_all.scale_[0]), 'std_tr': float(sc_tr.scale_[0])}
# speed
import time
t0 = time.perf_counter(); reg_fit['Linear Regression'].predict(X_te); t_lr = time.perf_counter() - t0
t0 = time.perf_counter(); reg_fit['Random Forest (300 trees)'].predict(X_te); t_rf = time.perf_counter() - t0
R['speed'] = {'lr_ms': t_lr * 1000, 'rf_ms': t_rf * 1000}

# ============================================================
# STAGE 5 : SAVE MODEL, PREDICTION TOOL, DASHBOARD
# ============================================================
final_pipe_reg = reg_fit[final_reg]; final_pipe_clf = clf_fit[final_clf]
bundle = {
  'regressor': final_pipe_reg, 'classifier': final_pipe_clf,
  'features': num_f + cat_f, 'late_threshold': LATE_MIN, 'peak_hours': PEAK_HOURS,
  'hubs': h.set_index('hub_id')[['zone', 'avg_distance_km', 'num_riders', 'kitchen_type']].to_dict('index'),
  'known_rain': wx.assign(date=wx.date.dt.strftime('%Y-%m-%d')).set_index('date')['rain_mm'].to_dict(),
  'temp_by_month': wx.groupby(wx.date.dt.month).temp_c.mean().round(1).to_dict(),
  'default_rain': float(wx.rain_mm.median()), 'trained_on_rows': len(X_tr), 'random_state': SEED}
joblib.dump(bundle, 'models/delay_model_bundle.joblib'); joblib.dump(final_pipe_reg, 'models/delay_regressor.joblib')
size_kb = os.path.getsize('models/delay_model_bundle.joblib') / 1024
show('5', 'Saved model', f"models/delay_model_bundle.joblib  ({size_kb:.0f} KB)")
R['bundle_kb'] = size_kb
from delay_predictor import predict_delay
tests = [('H001', '14-03-2025', 21, None), ('H017', '2025-07-19', 13, None), ('H033', '12-12-2026', 15, 25.0), ('H008', '2026-01-10', 22, 0.0)]
test_rows = []
for hub_, dt_, hr_, rn_ in tests:
    res = predict_delay(hub_, dt_, hr_, rain_mm=rn_)
    test_rows.append([hub_, dt_, str(hr_), f"{res['rain_mm']:.1f} mm ({'weather file' if res['rain_source'] == 'weather file' else 'user input'})", res['zone'], f"{res['distance_km']}", f"{res['predicted_delay_min']:.1f}", f"{res['p_late']*100:.0f}%", res['warning']])
    show('5', f'Tool test {hub_} {dt_} {hr_}h', json.dumps(res, indent=1, default=str))
R['tests'] = {'cols': ['hub_id', 'date', 'hour', 'rain_mm used', 'zone', 'dist km', 'Predicted delay (min)', 'P(late)', 'Warning'], 'rows': test_rows}
# hub-level truth for validation of tool: compare to actual rows of the data
chk = m[(m.hub_id == 'H001')].head(3)
# dashboard
fig = plt.figure(figsize=(12, 8.4)); gs = fig.add_gridspec(2, 2, hspace=.42, wspace=.28)
a1 = fig.add_subplot(gs[0, 0]); a1.bar(pv_hour.index, pv_hour.avg_delay, color=[ORANGE if hh in PEAK_HOURS else BLUE for hh in pv_hour.index])
a1.set(title=f"1. Peak hours add ~{R['peak_cmp']['Peak']['delay'] - R['peak_cmp']['Off-peak']['delay']:.1f} min of delay", xlabel='Order hour', ylabel='Avg delay (min)'); a1.set_xticks(pv_hour.index)
a2 = fig.add_subplot(gs[0, 1]); a2.bar(cat_tbl.index, cat_tbl.delay, color=[BLUE, '#4a7fb0', '#d9902f', ORANGE])
a2.set(title=f"2. Heavy rain: {cat_tbl.delay.iloc[-1]:.1f} min vs {cat_tbl.delay.iloc[0]:.1f} min when dry", xlabel='Daily rainfall category (mm)', ylabel='Avg delay (min)'); a2.tick_params(axis='x', labelsize=8)
a3 = fig.add_subplot(gs[1, 0]); sns.heatmap(pv_late, annot=True, fmt='.0f', cmap='YlOrRd', ax=a3, cbar=False, annot_kws={'size': 8}); a3.set(title='3. Late orders (%) by zone and slot', xlabel='', ylabel=''); a3.tick_params(axis='x', labelsize=7, rotation=20); a3.tick_params(axis='y', rotation=0, labelsize=8)
a4 = fig.add_subplot(gs[1, 1]); a4.scatter(yr_te, pl, s=6, alpha=.35, color=BLUE); a4.plot(lim, lim, color=ORANGE, lw=2)
a4.set(title=f"4. Model: avg error {R['reg_summary']['lr_mae']:.1f} min (R² = {R['reg_summary']['lr_r2']:.2f})", xlabel='Actual delay (min)', ylabel='Predicted delay (min)')
fig.suptitle('Figure 13: Management dashboard - causes of late deliveries and prediction accuracy', fontweight='bold', fontsize=12.5, y=.97)
plt.savefig('figs/fig13_dashboard.png', dpi=160, bbox_inches='tight'); plt.close()

# recommendation evidence numbers
R['reco'] = {
  'peak_delay': R['peak_cmp']['Peak']['delay'], 'off_delay': R['peak_cmp']['Off-peak']['delay'],
  'peak_late': R['peak_cmp']['Peak']['late'] * 100, 'off_late': R['peak_cmp']['Off-peak']['late'] * 100,
  'heavy_delay': float(cat_tbl.delay.iloc[-1]), 'dry_delay': float(cat_tbl.delay.iloc[0]), 'heavy_late': float(cat_tbl.late.iloc[-1]), 'dry_late': float(cat_tbl.late.iloc[0]),
  'far_delay': float(dist_tbl.delay.iloc[-1]), 'near_delay': float(dist_tbl.delay.iloc[0]), 'far_late': float(dist_tbl.late.iloc[-1]), 'near_late': float(dist_tbl.late.iloc[0]),
  'far_hubs': int(dist_tbl.hubs.iloc[-1]), 'km_slope': R['delay_reg']['coef']['avg_distance_km']}
pk_o = m[m.is_peak == 1]; upl = R['delay_reg']['coef']['is_peak']
R['scenario'] = {'uplift': upl, 'late_before': float(pk_o.is_late.mean() * 100),
                 'late_half': float(((pk_o.delay_min - upl / 2) > LATE_MIN).mean() * 100),
                 'late_full': float(((pk_o.delay_min - upl) > LATE_MIN).mean() * 100)}
# What-if scenario: new competitor -> fewer peak orders / delivery-fee change; assumption sensitivity: rain scenario
sens = []
for rn in [0, 10, 30, 50]:
    for hr_ in [11, 21]:
        r_ = predict_delay('H001', '2026-05-05', hr_, rain_mm=float(rn)); sens.append([f'{rn} mm', f'{hr_}:00', f"{r_['predicted_delay_min']:.1f}", f"{r_['p_late']*100:.0f}%"])
R['sens'] = {'cols': ['Rain assumed', 'Hour', 'Predicted delay (min)', 'P(late)'], 'rows': sens}
R['versions'] = {'python': sys.version.split()[0], 'pandas': pd.__version__, 'numpy': np.__version__, 'sklearn': __import__('sklearn').__version__, 'scipy': __import__('scipy').__version__, 'matplotlib': matplotlib.__version__, 'seaborn': sns.__version__}
json.dump({'R': R, 'OUT': OUT}, open('report_data.json', 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o), indent=1)
print("\nDONE")

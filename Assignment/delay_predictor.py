"""Prediction tool for the Hyderabad food-delivery delay model.
Usage (function):  from delay_predictor import predict_delay
                   predict_delay('H001', '14-03-2025', 21)
Usage (command line): python delay_predictor.py H001 2025-03-14 21 [rain_mm]
"""
import sys, os
import joblib, pandas as pd

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'delay_model_bundle.joblib')
_B = None

def _bundle():
    global _B
    if _B is None:
        _B = joblib.load(_PATH)
    return _B

def _parse(date_str):
    for fmt in ('%d-%m-%Y', '%Y-%m-%d'):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            pass
    raise ValueError("date must be dd-mm-yyyy or yyyy-mm-dd")

def predict_delay(hub_id, date, hour, rain_mm=None):
    b = _bundle()
    if hub_id not in b['hubs']:
        raise ValueError(f"unknown hub_id {hub_id}")
    if not 10 <= int(hour) <= 23:
        raise ValueError("hour must be between 10 and 23")
    d = _parse(date)
    hub = b['hubs'][hub_id]
    key = d.strftime('%Y-%m-%d')
    if rain_mm is not None:
        rain, src = float(rain_mm), 'user input'
    elif key in b['known_rain']:
        rain, src = b['known_rain'][key], 'weather file'
    else:
        rain, src = b['default_rain'], 'median rainfall (no forecast given)'
    temp = b['temp_by_month'][d.month]
    row = pd.DataFrame([{'hour': int(hour), 'is_peak': int(int(hour) in b['peak_hours']),
                         'is_weekend': int(d.dayofweek >= 5), 'rain_mm': rain, 'temp_c': temp,
                         'avg_distance_km': hub['avg_distance_km'], 'num_riders': hub['num_riders'],
                         'zone': hub['zone'], 'kitchen_type': hub['kitchen_type']}])[b['features']]
    delay = float(b['regressor'].predict(row)[0])
    p_late = float(b['classifier'].predict_proba(row)[0, 1])
    warn = ('LATE RISK - add riders / extend promise' if (p_late >= 0.5 or delay > b['late_threshold'])
            else 'ON-TIME expected')
    return {'hub_id': hub_id, 'date': key, 'hour': int(hour), 'zone': hub['zone'],
            'distance_km': round(float(hub['avg_distance_km']), 2), 'rain_mm': rain, 'rain_source': src,
            'predicted_delay_min': round(delay, 2), 'p_late': round(p_late, 3), 'warning': warn}

if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    r = predict_delay(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else None)
    for k, v in r.items():
        print(f"{k:>20}: {v}")

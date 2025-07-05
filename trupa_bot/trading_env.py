import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import ta
import gym
from gym import spaces
from stable_baselines3.common.monitor import Monitor

# === ✅ Fetch Only Recent 200 Candles ===
def fetch_data_twelvedata(symbol="XAU/USD", interval="1h", outputsize=200):
    apikey = os.getenv("TWELVE_DATA_API_KEY")
    if not apikey:
        raise ValueError("TWELVE_DATA_API_KEY environment variable is not set.")

    current_end = datetime.utcnow()
    interval_hours = int(interval.replace("h", ""))
    current_start = current_end - timedelta(hours=outputsize * interval_hours)

    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol}"
        f"&interval={interval}"
        f"&start_date={current_start.strftime('%Y-%m-%d %H:%M:%S')}"
        f"&end_date={current_end.strftime('%Y-%m-%d %H:%M:%S')}"
        f"&apikey={apikey}&format=JSON"
    )

    print(f"Fetching: {current_start} to {current_end}")
    response = requests.get(url)
    data = response.json()

    if 'values' not in data:
        raise Exception(f"API Error: {data}")

    df = pd.DataFrame(data['values'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)

    df['change'] = df['close'].pct_change().fillna(0)

    return df

# === ✅ Add Indicators ===
def add_indicators(df):
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    if 'volume' in df.columns and df['volume'].nunique() > 1:
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    else:
        df['obv'] = 0

    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()
    df['vwap'] = df['close'].expanding().mean()

    high = df['high'].max()
    low = df['low'].min()
    diff = high - low
    df['fib_0'] = low
    df['fib_236'] = high - 0.236 * diff
    df['fib_382'] = high - 0.382 * diff
    df['fib_500'] = high - 0.500 * diff
    df['fib_618'] = high - 0.618 * diff
    df['fib_100'] = high

    return df.dropna().reset_index(drop=True)

# === ✅ Trading Env (same as before) ===
class GoldTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, phase=5):
        super().__init__()
        print("✅ Using GoldTradingEnv with phase =", phase)
        self.df = df.reset_index(drop=True)
        self.phase = phase

        self.phase_params = {
            1: (1.0, -1.0, float('inf'), 0.0),
            2: (2.0, -1.0, 20, -0.05),
            3: (1.0, -2.0, 10, -0.05),
            4: (1.0, -3.0, 5, -0.2),
            5: (1.0, -4.0, 0, -0.5),
        }
        self.tp_reward, self.sl_penalty, self.entry_threshold, self.entry_penalty = self.phase_params[self.phase]

        self.tp_price_move = 4.0
        self.sl_price_move = 3.0

        # Observation space updated to (12,) to match the shape of the observation
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.current_step = 0
        self.position = None
        self.entry_waiting_steps = 0
        self.done = False

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row['open'], row['high'], row['low'], row['close'], row['ema_50'],
            row['rsi'], row['obv'], row['bb_width'], row['vwap'],
            row['fib_0'], row['fib_618']  # 12 features now
        ], dtype=np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = None
        self.entry_waiting_steps = 0
        self.done = False
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        reward = 0.0
        info = {}

        row = self.df.iloc[self.current_step]
        high = row['high']
        low = row['low']
        close = row['close']

        # === No position yet — handle entry ===
        if self.position is None:
            self.entry_waiting_steps += 1
            if self.entry_waiting_steps > self.entry_threshold:
                reward += self.entry_penalty

            if action in [0, 1]:
                self.position = {
                    'type': 'buy' if action == 0 else 'sell',
                    'entry_price': close,  # Entry at Candle N close
                    'steps_open': 0
                }
                self.entry_waiting_steps = 0

        # === Position is open — start checking from Candle N+1 ===
        else:
            self.position['steps_open'] += 1
            entry_price = self.position['entry_price']

            if self.position['type'] == 'buy':
                tp_price = entry_price + self.tp_price_move
                sl_price = entry_price - self.sl_price_move

                if low <= sl_price:
                    reward = self.sl_penalty
                    self.position = None
                elif high >= tp_price:
                    reward = self.tp_reward
                    self.position = None

            elif self.position['type'] == 'sell':
                tp_price = entry_price - self.tp_price_move
                sl_price = entry_price + self.sl_price_move

                if high >= sl_price:
                    reward = self.sl_penalty
                    self.position = None
                elif low <= tp_price:
                    reward = self.tp_reward
                    self.position = None

        self.current_step += 1
        self.done = self.current_step >= len(self.df) - 1
        obs = self._get_observation()
        return obs, reward, self.done, False, info

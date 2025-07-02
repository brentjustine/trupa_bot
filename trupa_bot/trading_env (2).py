# 📊 Import Libraries
import gym
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from gym import spaces
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

def fetch_data_twelvedata(symbol="XAU/USD", interval="1h", outputsize=500, apikey=None):
    if apikey is None:
        apikey = os.getenv("TWELVE_DATA_API_KEY")

    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={apikey}&format=JSON"
    response = requests.get(url)
    data = response.json()

    if 'values' not in data:
        raise Exception(f"API Error: {data}")

    df = pd.DataFrame(data['values'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df

# === Helper Function to Add Technical Indicators ===
def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['bb_bbm'] = ta.volatility.BollingerBands(df['close']).bollinger_mavg()

    # Drop rows with NaN values created bay indicators
    df = df.dropna().reset_index(drop=True)
    return df

class GoldTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df):
        super(GoldTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)

        # ⚙️ Core Settings
        self.initial_balance = 20.0
        self.window_size = 1
        self.max_trade_duration = 50
        self.leverage = 10
        self.tp_threshold = 0.005
        self.sl_threshold = 0.0025
        self.trade_cost = 0.02
        self.max_drawdown = 0.5

        # 🎮 Action & Observation Space
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.feature_size = df.shape[1] - 1  # Exclude datetime
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.feature_size + 3,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.position_duration = 0
        self.current_step = self.np_random.integers(0, len(self.df) - 100)
        self.last_manual_close_step = -10
        return self._get_obs(), {}

    def _get_obs(self):
        state = self.df.iloc[self.current_step].drop(['datetime']).values.astype(np.float32)
        position = np.array([self.position], dtype=np.float32)
        norm_balance = np.array([self.balance / self.initial_balance], dtype=np.float32)

        unrealized = 0.0
        if self.position != 0:
            current_price = self.df.iloc[self.current_step]['close']
            unrealized = (current_price - self.entry_price) / self.entry_price * self.position
        unrealized = np.array([unrealized], dtype=np.float32)

        return np.concatenate([state, position, norm_balance, unrealized])

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        price = self.df.iloc[self.current_step]['close']

        # 🚀 Open Position
        if action in [1, 2] and self.position == 0:
            self.position = 1 if action == 1 else -1
            self.entry_price = price
            self.position_duration = 0
            reward -= self.trade_cost  # Entry cost

        # 💣 Manual Close
        elif action == 0 and self.position != 0:
            pnl = (price - self.entry_price) / self.entry_price * self.position
            step_reward = pnl * 100 * self.leverage
            reward += step_reward
            self.balance += step_reward
            self.balance = np.round(self.balance, 4)

            # Bonus for quick profitable trades
            if pnl > 0.002 and self.position_duration <= 5:
                reward += 0.2
            elif self.position_duration <= 5:
                reward -= 0.05  # discourage frequent small losses

            self.last_manual_close_step = self.current_step
            self.position = 0
            self.entry_price = 0
            self.position_duration = 0

        # 🔁 Auto Close via TP/SL or max duration
        if self.position != 0:
            pnl = (price - self.entry_price) / self.entry_price * self.position
            self.position_duration += 1

            # 💡 Reward shaping during holding
            if pnl > 0:
                reward += pnl * 10  # small positive signal
            else:
                reward += pnl * 5  # scaled negative feedback


            if pnl >= self.tp_threshold or pnl <= -self.sl_threshold or self.position_duration >= self.max_trade_duration:
                step_reward = pnl * 100 * self.leverage
                reward += step_reward
                self.balance += step_reward
                self.balance = np.round(self.balance, 4)
                self.position = 0
                self.entry_price = 0
                self.position_duration = 0
            else:
                reward -= 0.02  # higher holding cost to discourage over-holding


        # 🛑 Drawdown-Based Termination
        if self.balance < self.initial_balance * (1 - self.max_drawdown):
            reward -= (self.initial_balance - self.balance) / self.initial_balance * 5
            terminated = True

        # 📉 End of data
        self.current_step += 1
        if self.current_step >= len(self.df) - 1 or self.balance <= 0:
            if self.position != 0:
                pnl = (price - self.entry_price) / self.entry_price * self.position
                step_reward = pnl * 100 * self.leverage
                reward += step_reward
                self.balance += step_reward
                self.position = 0
                self.entry_price = 0
                self.position_duration = 0
            terminated = True
            if self.balance < self.initial_balance:
                drawdown = (self.initial_balance - self.balance) / self.initial_balance
                reward -= drawdown * 5


        # 🎯 Reward Bonus for large profits
        if reward > 0.5:
            reward += 0.2

        reward = np.round(reward, 4)


        info["action_mask"] = [int(self.position == 0)] * 3
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step} | Balance: {self.balance:.2f} | Position: {self.position} | Entry: {self.entry_price:.2f}")

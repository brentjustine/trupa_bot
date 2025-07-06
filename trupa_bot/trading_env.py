import os
import numpy as np
import pandas as pd
import ta
import gym
from gym import Env, spaces
from gym.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import requests
import datetime  # Ensure datetime is imported
import os
import requests
import pandas as pd

def fetch_data_twelvedata(
    symbol="XAU/USD",
    interval="1h",
    outputsize=200,
    apikey=None
):
    # Use the API key, either provided or from environment variable
    apikey = apikey or os.getenv("TWELVE_DATA_API_KEY")
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={apikey}&format=JSON"
    )
    
    # Fetch the data from the API
    response = requests.get(url)
    data = response.json()

    # Check for 'values' in the response
    if 'values' not in data:
        raise Exception(f"API Error: {data}")

    # Create the DataFrame
    df = pd.DataFrame(data['values'])

    # Ensure 'datetime' column is parsed correctly
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Convert 'close' to numeric, in case there are any unexpected string values
    if 'close' in df.columns:
        df['close'] = pd.to_numeric(df['close'], errors='coerce')  # Convert 'close' to float
    
    # Ensure 'change' exists or calculate it
    if 'change' not in df.columns:
        print("Calculating 'change' column...")  # Debugging line
        df['change'] = df['close'].pct_change() * 100  # Calculate percentage change
        df['change'] = df['change'].fillna(0)  # Handle NaN values by filling with 0 or a suitable strategy
    
    # Ensure all relevant columns are numeric
    for col in ['open', 'high', 'low', 'close', 'change']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Ensure all columns are numeric

    # Final check for missing required columns
    required_columns = ['open', 'high', 'low', 'close', 'change']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Drop the 'datetime' column as requested
    df = df.drop(columns=['datetime'], errors='ignore')

    # Return the processed DataFrame
    return df

# Other functions remain unchanged
def add_indicators(df):
    df['ema_50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close=df['close'])
    df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()
    df['vwap'] = df['close'].expanding().mean()

    # Fibonacci levels
    window = 100
    high_roll = df['high'].rolling(window).max()
    low_roll = df['low'].rolling(window).min()
    diff = high_roll - low_roll
    df['fib_0'] = low_roll
    df['fib_236'] = high_roll - 0.236 * diff
    df['fib_382'] = high_roll - 0.382 * diff
    df['fib_500'] = high_roll - 0.500 * diff
    df['fib_618'] = high_roll - 0.618 * diff
    df['fib_100'] = high_roll

    return df.dropna().reset_index(drop=True)


class TPSSLTradingEnv(Env):
    def __init__(self, df: pd.DataFrame, max_episode_steps: int = 500, delay_threshold: int = 50):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.numeric_df = self.df.select_dtypes(include=[np.number])
        self.max_episode_steps = max_episode_steps
        self.delay_threshold = delay_threshold  # Delay threshold for penalty

        self.action_space = spaces.Discrete(3)  # 0=Buy, 1=Sell, 2=Wait
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.numeric_df.shape[1],), dtype=np.float32
        )

        self.tp_value = 4.0  # Take Profit value
        self.sl_value = 3.0  # Stop Loss value

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_winrates = []

        self._validate_indicators()
        self._reset_env()

    def _validate_indicators(self):
        required = ['rsi', 'macd', 'macd_signal', 'ema_50', 'close', 'high', 'low']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required indicators: {missing}")

    def _reset_env(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.current_reward_total = 0
        self.position = None
        self.entry_price = 0.0
        self.position_hold_steps = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.wait_steps = 0  # Track how long the agent has been waiting

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_env()
        return self._get_observation(), {}

    def _get_observation(self):
        return self.numeric_df.iloc[self.current_step].values.astype(np.float32)

    def step(self, action: int):
        self.steps_in_episode += 1
        reward = 0.0
        terminated = truncated = False

        price = self.df.loc[self.current_step, 'close']
        next_row = self.df.iloc[self.current_step + 1]
        next_high, next_low = next_row['high'], next_row['low']
        rsi, macd, macd_signal, ema_50 = self.df.loc[self.current_step, ['rsi', 'macd', 'macd_signal', 'ema_50']]

        def indicator_alignment(direction: str):
            score = 0.0
            if direction == "long":
                if rsi < 30: score += 0.25
                if macd > macd_signal: score += 0.25
                if price > ema_50: score += 0.25
            elif direction == "short":
                if rsi > 70: score += 0.25
                if macd < macd_signal: score += 0.25
                if price < ema_50: score += 0.25
            return score

        if self.position is None:
            if action == 0:  # Buy
                self.position = "long"
                self.entry_price = price
                reward += indicator_alignment("long")
            elif action == 1:  # Sell
                self.position = "short"
                self.entry_price = price
                reward += indicator_alignment("short")
            elif action == 2:  # Wait
                self.wait_steps += 1
                if self.wait_steps > self.delay_threshold:
                    reward -= 0.1  # Penalty for waiting too long
        else:
            self.position_hold_steps += 1
            if self.position == "long":
                if next_low <= self.entry_price - self.sl_value:
                    reward = -1  # Stop loss hit
                    self._end_trade()
                elif next_high >= self.entry_price + self.tp_value:
                    reward = 1 + indicator_alignment("long")  # Take profit hit
                    self._end_trade(win=True)
                else:
                    reward -= 0.01 * self.position_hold_steps  # Penalty for holding too long
            elif self.position == "short":
                if next_high >= self.entry_price + self.sl_value:
                    reward = -1  # Stop loss hit
                    self._end_trade()
                elif next_low <= self.entry_price - self.tp_value:
                    reward = 1 + indicator_alignment("short")  # Take profit hit
                    self._end_trade(win=True)
                else:
                    reward -= 0.01 * self.position_hold_steps  # Penalty for holding too long

        reward = float(np.clip(reward, -1.0, 1.0))
        self.current_reward_total += reward
        self.current_step += 1

        if self.current_step >= len(self.df) - 2:
            terminated = True
        if self.steps_in_episode >= self.max_episode_steps:
            truncated = True
        if terminated or truncated:
            self._finalize_episode()

        return self._get_observation(), reward, terminated, truncated, {}

    def _end_trade(self, win=False):
        self.position = None
        self.total_trades += 1
        if win:
            self.winning_trades += 1

    def _finalize_episode(self):
        self.episode_rewards.append(self.current_reward_total)
        self.episode_lengths.append(self.steps_in_episode)
        winrate = (self.winning_trades / self.total_trades) if self.total_trades else 0.0
        self.episode_winrates.append(winrate * 100.0)

    def render(self):
        print(f"Step: {self.current_step} | Position: {self.position} | Entry: {self.entry_price:.2f}")

    def close(self):
        pass

    def get_winrate_history(self):
        return self.episode_winrates.copy()

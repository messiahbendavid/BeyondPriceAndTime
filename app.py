# -*- coding: utf-8 -*-
"""
Beyond Price and Time - ETF & Liquid Stocks Edition
Copyright © 2026 Truth Communications LLC. All Rights Reserved.

Real-time trading signals with stasis detection
Merit Score: Multi-timeframe stasis confluence indicator
"""

import time
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import webbrowser
from enum import Enum
import copy
import json

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import pandas as pd

import websocket
import ssl
import requests

# ============================================================================
# API KEY
# ============================================================================

POLYGON_API_KEY = "PnzhJOXEJO7tSpHr0ct2zjFKi6XO0yGi"

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # ETFs (19) + Top 100 Most Liquid Stocks = 119 symbols
    symbols: List[str] = field(default_factory=lambda: [
        # ==================== ETFs (19) ====================
        # Major Index ETFs
        "SPY", "QQQ", "IWM", "DIA",
        # Sector ETFs
        "XLF", "XLE", "XLU", "XLK", "XLP", "XLB", "XLV", "XLI", "XLY", "XLC", "XLRE",
        # Thematic ETFs
        "KRE", "SMH", "XBI", "GDX",
        
        # ==================== TOP 100 LIQUID STOCKS ====================
        # Mega Cap Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "ADBE",
        # Tech continued
        "CRM", "AMD", "INTC", "CSCO", "QCOM", "IBM", "NOW", "INTU", "AMAT", "MU",
        "NFLX", "PYPL", "SHOP", "SQ", "UBER", "ABNB", "COIN", "SNAP", "ROKU", "PLTR",
        # Finance
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V",
        "MA", "COF", "USB", "PNC", "TFC",
        # Healthcare
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "VRTX", "REGN", "MRNA",
        # Consumer
        "WMT", "HD", "COST", "TGT", "LOW", "NKE", "SBUX", "MCD", "DIS", "CMCSA",
        # Industrial & Energy
        "CAT", "BA", "GE", "HON", "UNP", "RTX", "LMT", "DE", "UPS", "FDX",
        "XOM", "CVX", "COP", "SLB", "EOG",
        # Other High Volume
        "T", "VZ", "TMUS", "PG", "KO", "PEP", "PM", "MO", "CL", "MMM",
        "F", "GM", "RIVN", "LCID", "NIO",
    ])
    
    # ETF symbols for identification
    etf_symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "DIA",
        "XLF", "XLE", "XLU", "XLK", "XLP", "XLB", "XLV", "XLI", "XLY", "XLC", "XLRE",
        "KRE", "SMH", "XBI", "GDX",
    ])
    
    # Your specified thresholds
    thresholds: List[float] = field(default_factory=lambda: [
        0.000625, 0.00125, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10
    ])
    
    display_bits: int = 20
    update_interval_ms: int = 500
    cache_refresh_interval: float = 0.25
    history_days: int = 5
    
    polygon_api_key: str = POLYGON_API_KEY
    polygon_ws_url: str = "wss://socket.polygon.io/stocks"
    polygon_rest_url: str = "https://api.polygon.io"
    
    volumes: Dict[str, float] = field(default_factory=dict)
    week52_data: Dict[str, Dict] = field(default_factory=dict)
    
    min_tradable_stasis: int = 3

config = Config()
# Remove duplicates while preserving order
config.symbols = list(dict.fromkeys(config.symbols))

# ============================================================================
# ENUMS
# ============================================================================

class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class SignalStrength(Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"

# ============================================================================
# DATA FETCHERS
# ============================================================================

def fetch_52_week_data() -> Dict[str, Dict]:
    print("📊 Fetching 52-week high/low data...")
    week52_data = {}
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    for i, symbol in enumerate(config.symbols):
        try:
            url = (
                f"{config.polygon_rest_url}/v2/aggs/ticker/{symbol}/range/1/day/"
                f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
                f"?adjusted=true&sort=asc&limit=365&apiKey={config.polygon_api_key}"
            )
            
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results') and len(data['results']) > 0:
                    highs = [bar['h'] for bar in data['results']]
                    lows = [bar['l'] for bar in data['results']]
                    high_val = max(highs)
                    low_val = min(lows)
                    week52_data[symbol] = {
                        'high': high_val,
                        'low': low_val,
                        'range': high_val - low_val,
                    }
                else:
                    week52_data[symbol] = {'high': None, 'low': None, 'range': None}
            else:
                week52_data[symbol] = {'high': None, 'low': None, 'range': None}
            
            if (i + 1) % 20 == 0:
                print(f"   📈 Processed {i + 1}/{len(config.symbols)}...")
            
            time.sleep(0.12)
        except Exception as e:
            week52_data[symbol] = {'high': None, 'low': None, 'range': None}
    
    print(f"✅ 52-week data loaded for {len(week52_data)} symbols\n")
    return week52_data

def fetch_volume_data() -> Dict[str, float]:
    print("📊 Fetching volume data...")
    volumes = {}
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=45)
    
    for i, symbol in enumerate(config.symbols):
        try:
            url = (
                f"{config.polygon_rest_url}/v2/aggs/ticker/{symbol}/range/1/day/"
                f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
                f"?adjusted=true&sort=desc&limit=30&apiKey={config.polygon_api_key}"
            )
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results') and len(data['results']) > 0:
                    total_volume = sum(bar['v'] for bar in data['results'])
                    volumes[symbol] = (total_volume / len(data['results'])) / 1_000_000
                else:
                    volumes[symbol] = 50.0
            else:
                volumes[symbol] = 50.0
            
            if (i + 1) % 20 == 0:
                print(f"   📈 Processed {i + 1}/{len(config.symbols)}...")
            
            time.sleep(0.12)
        except:
            volumes[symbol] = 50.0
    
    print(f"✅ Volume data loaded for {len(volumes)} symbols\n")
    return volumes

def fetch_historical_bars(symbol: str, days: int = 5) -> List[Dict]:
    bars = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        url = (
            f"{config.polygon_rest_url}/v2/aggs/ticker/{symbol}/range/1/minute/"
            f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            f"?adjusted=true&sort=asc&limit=50000&apiKey={config.polygon_api_key}"
        )
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                for bar in data['results']:
                    bars.append({
                        'timestamp': datetime.fromtimestamp(bar['t'] / 1000),
                        'close': bar['c'],
                    })
    except:
        pass
    
    return bars

def calculate_52week_percentile(price: float, symbol: str) -> Optional[float]:
    if symbol not in config.week52_data:
        return None
    data = config.week52_data[symbol]
    if data is None:
        return None
    high = data.get('high')
    low = data.get('low')
    range_val = data.get('range')
    if high is None or low is None or range_val is None or range_val <= 0:
        return None
    percentile = ((price - low) / range_val) * 100
    return max(0.0, min(100.0, percentile))

def is_etf(symbol: str) -> bool:
    return symbol in config.etf_symbols

# ============================================================================
# STASIS INFO
# ============================================================================

@dataclass
class StasisInfo:
    start_time: datetime
    start_price: float
    peak_stasis: int = 1
    
    def get_duration(self) -> timedelta:
        return datetime.now() - self.start_time
    
    def get_duration_str(self) -> str:
        duration = self.get_duration()
        total_seconds = int(duration.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            return f"{total_seconds // 60}m {total_seconds % 60}s"
        else:
            return f"{total_seconds // 3600}h {(total_seconds % 3600) // 60}m"
    
    def get_start_date_str(self) -> str:
        return self.start_time.strftime("%m/%d %H:%M")
    
    def get_price_change_pct(self, current_price: float) -> float:
        if self.start_price == 0:
            return 0.0
        return (current_price - self.start_price) / self.start_price * 100

# ============================================================================
# BITSTREAM
# ============================================================================

@dataclass
class BitEntry:
    bit: int
    price: float
    timestamp: datetime

class Bitstream:
    def __init__(self, symbol: str, threshold: float, initial_price: float, volume: float):
        self.symbol = symbol
        self.threshold = threshold
        self.initial_price = initial_price
        self.volume = volume
        self.is_etf = is_etf(symbol)
        
        self.reference_price = initial_price
        self.current_live_price = initial_price
        self.last_price_update = datetime.now()
        
        self._update_bands()
        
        self.bits: deque = deque(maxlen=500)
        
        self.current_stasis = 0
        self.last_bit = None
        self.direction = None
        self.signal_strength = None
        
        self.stasis_info: Optional[StasisInfo] = None
        
        self.total_bits = 0
        self._lock = threading.Lock()
    
    def _update_bands(self):
        self.band_width = self.threshold * self.reference_price
        self.upper_band = self.reference_price + self.band_width
        self.lower_band = self.reference_price - self.band_width
    
    def process_price(self, price: float, timestamp: datetime) -> List[int]:
        with self._lock:
            self.current_live_price = price
            self.last_price_update = timestamp
            
            generated_bits = []
            
            if self.lower_band < price < self.upper_band:
                return generated_bits
            
            if self.band_width <= 0:
                return generated_bits
            
            x = int((price - self.reference_price) / self.band_width)
            
            if x > 0:
                for _ in range(x):
                    self.bits.append(BitEntry(1, price, timestamp))
                    generated_bits.append(1)
                    self.total_bits += 1
                self.reference_price = price
                self._update_bands()
            elif x < 0:
                for _ in range(abs(x)):
                    self.bits.append(BitEntry(0, price, timestamp))
                    generated_bits.append(0)
                    self.total_bits += 1
                self.reference_price = price
                self._update_bands()
            
            if generated_bits:
                self._update_stasis(timestamp)
            
            return generated_bits
    
    def _update_stasis(self, timestamp: datetime):
        if len(self.bits) < 2:
            self.current_stasis = len(self.bits)
            self.last_bit = self.bits[-1].bit if self.bits else None
            self.direction = None
            self.signal_strength = None
            return
        
        bits_list = list(self.bits)
        
        stasis_count = 1
        stasis_start_idx = len(bits_list) - 1
        
        for i in range(len(bits_list) - 1, 0, -1):
            if bits_list[i].bit != bits_list[i-1].bit:
                stasis_count += 1
                stasis_start_idx = i - 1
            else:
                break
        
        prev_stasis = self.current_stasis
        self.current_stasis = stasis_count
        self.last_bit = bits_list[-1].bit
        
        if prev_stasis < 2 and stasis_count >= 2:
            if 0 <= stasis_start_idx < len(bits_list):
                first_bit = bits_list[stasis_start_idx]
                self.stasis_info = StasisInfo(
                    start_time=first_bit.timestamp,
                    start_price=first_bit.price,
                    peak_stasis=stasis_count,
                )
        elif stasis_count >= 2 and self.stasis_info is not None:
            if stasis_count > self.stasis_info.peak_stasis:
                self.stasis_info.peak_stasis = stasis_count
        elif prev_stasis >= 2 and stasis_count < 2:
            self.stasis_info = None
        
        if self.current_stasis >= 2:
            self.direction = Direction.LONG if self.last_bit == 1 else Direction.SHORT
            if self.current_stasis >= 10:
                self.signal_strength = SignalStrength.VERY_STRONG
            elif self.current_stasis >= 7:
                self.signal_strength = SignalStrength.STRONG
            elif self.current_stasis >= 5:
                self.signal_strength = SignalStrength.MODERATE
            elif self.current_stasis >= 3:
                self.signal_strength = SignalStrength.WEAK
            else:
                self.signal_strength = None
        else:
            self.direction = None
            self.signal_strength = None
    
    def is_tradable(self) -> bool:
        with self._lock:
            return (
                self.current_stasis >= config.min_tradable_stasis and
                self.direction is not None and
                self.volume > 1.0
            )
    
    def get_snapshot(self, live_price: Optional[float] = None) -> Dict:
        with self._lock:
            current_price = live_price if live_price is not None else self.current_live_price
            
            anchor_price = None
            stasis_start_str = "—"
            stasis_duration_str = "—"
            duration_seconds = 0
            stasis_price_change_pct = None
            
            if self.stasis_info is not None:
                anchor_price = self.stasis_info.start_price
                stasis_start_str = self.stasis_info.get_start_date_str()
                stasis_duration_str = self.stasis_info.get_duration_str()
                duration_seconds = self.stasis_info.get_duration().total_seconds()
                stasis_price_change_pct = self.stasis_info.get_price_change_pct(current_price)
            
            take_profit = None
            stop_loss = None
            risk_reward = None
            distance_to_tp_pct = None
            distance_to_sl_pct = None
            
            if self.direction is not None and self.current_stasis >= 2:
                if self.direction == Direction.LONG:
                    take_profit = self.upper_band
                    stop_loss = self.lower_band
                    reward = take_profit - current_price
                    risk = current_price - stop_loss
                else:
                    take_profit = self.lower_band
                    stop_loss = self.upper_band
                    reward = current_price - take_profit
                    risk = stop_loss - current_price
                
                if risk > 0 and reward > 0:
                    risk_reward = reward / risk
                elif risk > 0 and reward <= 0:
                    risk_reward = 0.0
                else:
                    risk_reward = None
                
                if current_price > 0:
                    distance_to_tp_pct = (abs(take_profit - current_price) / current_price) * 100
                    distance_to_sl_pct = (abs(stop_loss - current_price) / current_price) * 100
            
            week52_percentile = calculate_52week_percentile(current_price, self.symbol)
            recent_bits = [b.bit for b in list(self.bits)[-15:]]
            
            return {
                'symbol': self.symbol,
                'is_etf': self.is_etf,
                'threshold': self.threshold,
                'threshold_pct': self.threshold * 100,
                'stasis': self.current_stasis,
                'total_bits': self.total_bits,
                'recent_bits': recent_bits,
                'current_price': current_price,
                'anchor_price': anchor_price,
                'direction': self.direction.value if self.direction else None,
                'signal_strength': self.signal_strength.value if self.signal_strength else None,
                'is_tradable': (
                    self.current_stasis >= config.min_tradable_stasis and
                    self.direction is not None and
                    self.volume > 1.0
                ),
                'stasis_start_str': stasis_start_str,
                'stasis_duration_str': stasis_duration_str,
                'duration_seconds': duration_seconds,
                'stasis_price_change_pct': stasis_price_change_pct,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'risk_reward': risk_reward,
                'distance_to_tp_pct': distance_to_tp_pct,
                'distance_to_sl_pct': distance_to_sl_pct,
                'week52_percentile': week52_percentile,
                'volume': self.volume,
            }

# ============================================================================
# PRICE FEED - REAL-TIME
# ============================================================================

class PolygonPriceFeed:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_prices: Dict[str, float] = {}
        self.is_running = False
        self.ws = None
        self.ws_thread = None
        self.message_count = 0
        self.last_update_time: Dict[str, datetime] = {}
        
        for symbol in config.symbols:
            self.current_prices[symbol] = None
    
    def start(self):
        self.is_running = True
        self.ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self.ws_thread.start()
        print("✅ Real-time WebSocket starting...")
        return True
    
    def stop(self):
        self.is_running = False
        if self.ws:
            self.ws.close()
    
    def _ws_loop(self):
        while self.is_running:
            try:
                self._connect()
            except Exception as e:
                print(f"❌ WebSocket error: {e}")
                if self.is_running:
                    time.sleep(5)
    
    def _connect(self):
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if isinstance(data, list):
                    for msg in data:
                        self._process(msg)
                else:
                    self._process(data)
            except:
                pass
        
        def on_open(ws):
            print("✅ Real-time WebSocket connected!")
            ws.send(json.dumps({"action": "auth", "params": config.polygon_api_key}))
        
        def on_close(ws, code, msg):
            print(f"WebSocket closed: {code}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        self.ws = websocket.WebSocketApp(
            config.polygon_ws_url,
            on_open=on_open,
            on_message=on_message,
            on_close=on_close,
            on_error=on_error
        )
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    
    def _process(self, msg: Dict):
        if msg.get('ev') == 'status':
            status = msg.get('status')
            print(f"📊 Status: {status} - {msg.get('message', '')}")
            if status == 'auth_success':
                self._subscribe()
        elif msg.get('ev') in ['A', 'AM', 'T', 'Q']:
            symbol = msg.get('sym', '') or msg.get('S', '')
            price = msg.get('c') or msg.get('vw') or msg.get('p') or msg.get('bp')
            if price and symbol in config.symbols:
                with self.lock:
                    self.current_prices[symbol] = float(price)
                    self.last_update_time[symbol] = datetime.now()
                    self.message_count += 1
    
    def _subscribe(self):
        if self.ws:
            # Subscribe in batches
            for i in range(0, len(config.symbols), 50):
                batch = config.symbols[i:i+50]
                symbols_str = ",".join([f"A.{s}" for s in batch])
                self.ws.send(json.dumps({
                    "action": "subscribe",
                    "params": symbols_str
                }))
                time.sleep(0.1)
            print(f"📡 Subscribed to {len(config.symbols)} symbols (REAL-TIME)")
    
    def get_all_prices(self) -> Dict[str, float]:
        with self.lock:
            return {k: v for k, v in self.current_prices.items() if v is not None}
    
    def get_status(self) -> Dict:
        with self.lock:
            connected = sum(1 for v in self.current_prices.values() if v is not None)
            return {
                'total': len(config.symbols),
                'connected': connected,
                'message_count': self.message_count
            }

price_feed = PolygonPriceFeed()

# ============================================================================
# BITSTREAM MANAGER
# ============================================================================

class BitstreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.streams: Dict[Tuple[str, float], Bitstream] = {}
        self.is_running = False
        
        self.cached_data: List[Dict] = []
        self.cached_merit_scores: Dict[str, Dict] = {}
        self.cache_lock = threading.Lock()
        
        self.initialized = False
        self.backfill_complete = False
        self.backfill_progress = 0
    
    def backfill(self):
        print("\n" + "=" * 60)
        print("📜 BACKFILLING HISTORICAL DATA")
        print("=" * 60)
        
        historical_data = {}
        
        for i, symbol in enumerate(config.symbols):
            bars = fetch_historical_bars(symbol, config.history_days)
            if bars:
                historical_data[symbol] = bars
            
            self.backfill_progress = int((i + 1) / len(config.symbols) * 100)
            
            if (i + 1) % 20 == 0:
                print(f"   📊 {i + 1}/{len(config.symbols)} ({self.backfill_progress}%)")
            
            time.sleep(0.12)
        
        print(f"\n✅ Historical data: {len(historical_data)} symbols")
        
        with self.lock:
            for symbol, bars in historical_data.items():
                if not bars:
                    continue
                
                initial_price = bars[0]['close']
                volume = config.volumes.get(symbol, 50.0)
                
                for threshold in config.thresholds:
                    key = (symbol, threshold)
                    self.streams[key] = Bitstream(symbol, threshold, initial_price, volume)
                    
                    for bar in bars:
                        self.streams[key].process_price(bar['close'], bar['timestamp'])
        
        self.initialized = True
        self.backfill_complete = True
        
        tradable = sum(1 for s in self.streams.values() if s.is_tradable())
        print(f"✅ Bitstreams: {len(self.streams)} | Tradable: {tradable}")
        print("=" * 60 + "\n")
    
    def start(self):
        self.is_running = True
        threading.Thread(target=self._process_loop, daemon=True).start()
        threading.Thread(target=self._cache_loop, daemon=True).start()
    
    def _process_loop(self):
        while self.is_running:
            time.sleep(0.05)
            if not self.backfill_complete:
                continue
            
            prices = price_feed.get_all_prices()
            timestamp = datetime.now()
            
            with self.lock:
                for symbol, price in prices.items():
                    for threshold in config.thresholds:
                        key = (symbol, threshold)
                        if key in self.streams:
                            self.streams[key].process_price(price, timestamp)
    
    def _calculate_merit_scores(self, snapshots: List[Dict]) -> Dict[str, Dict]:
        merit_data = defaultdict(lambda: {
            'stasis_levels': 0,
            'total_stasis': 0,
            'weighted_score': 0,
            'directions': [],
            'dominant_direction': None,
            'direction_alignment': 0,
            'avg_stasis': 0,
            'max_stasis': 0,
            'levels_detail': [],
            'thresholds_in_stasis': [],
        })
        
        symbol_snapshots = defaultdict(list)
        for snap in snapshots:
            symbol_snapshots[snap['symbol']].append(snap)
        
        for symbol, snaps in symbol_snapshots.items():
            tradable_levels = []
            all_stasis = []
            directions = []
            thresholds_in_stasis = []
            
            for snap in snaps:
                stasis = snap['stasis']
                direction = snap['direction']
                threshold = snap['threshold']
                
                all_stasis.append(stasis)
                
                if stasis >= config.min_tradable_stasis and direction:
                    tradable_levels.append({
                        'threshold': threshold,
                        'stasis': stasis,
                        'direction': direction
                    })
                    directions.append(direction)
                    thresholds_in_stasis.append(threshold)
            
            stasis_levels = len(tradable_levels)
            total_stasis = sum(all_stasis)
            max_stasis = max(all_stasis) if all_stasis else 0
            
            tradable_stasis_sum = sum(t['stasis'] for t in tradable_levels)
            level_multiplier = 1 + (stasis_levels * 0.5) if stasis_levels > 1 else 1
            weighted_score = tradable_stasis_sum * level_multiplier
            
            if directions:
                long_count = directions.count('LONG')
                short_count = directions.count('SHORT')
                dominant = 'LONG' if long_count >= short_count else 'SHORT'
                alignment = max(long_count, short_count) / len(directions) * 100
                
                if alignment == 100 and stasis_levels >= 3:
                    weighted_score *= 1.25
            else:
                dominant = None
                alignment = 0
            
            avg_stasis = tradable_stasis_sum / stasis_levels if stasis_levels > 0 else 0
            
            merit_data[symbol] = {
                'stasis_levels': stasis_levels,
                'total_stasis': total_stasis,
                'weighted_score': round(weighted_score, 1),
                'directions': directions,
                'dominant_direction': dominant,
                'direction_alignment': round(alignment, 0),
                'avg_stasis': round(avg_stasis, 1),
                'max_stasis': max_stasis,
                'levels_detail': tradable_levels,
                'thresholds_in_stasis': thresholds_in_stasis,
            }
        
        return dict(merit_data)
    
    def _cache_loop(self):
        while self.is_running:
            time.sleep(config.cache_refresh_interval)
            if not self.initialized:
                continue
            
            live_prices = price_feed.get_all_prices()
            snapshots = []
            
            with self.lock:
                for stream in self.streams.values():
                    live_price = live_prices.get(stream.symbol)
                    snapshots.append(stream.get_snapshot(live_price))
            
            merit_scores = self._calculate_merit_scores(snapshots)
            
            with self.cache_lock:
                self.cached_data = snapshots
                self.cached_merit_scores = merit_scores
    
    def get_data(self) -> List[Dict]:
        with self.cache_lock:
            return copy.deepcopy(self.cached_data)
    
    def get_merit_scores(self) -> Dict[str, Dict]:
        with self.cache_lock:
            return copy.deepcopy(self.cached_merit_scores)

manager = BitstreamManager()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_bits(bits: List[int]) -> str:
    return "".join(str(b) for b in bits) if bits else "—"

def format_rr(rr: Optional[float]) -> str:
    if rr is None:
        return "—"
    if rr <= 0:
        return "0:1"
    return f"{rr:.2f}:1" if rr < 10 else f"{rr:.0f}:1"

def format_band(threshold_pct: float) -> str:
    if threshold_pct < 0.1:
        return f"{threshold_pct:.4f}%"
    elif threshold_pct < 1:
        return f"{threshold_pct:.3f}%"
    else:
        return f"{threshold_pct:.2f}%"

def get_table_data() -> pd.DataFrame:
    data = manager.get_data()
    merit_scores = manager.get_merit_scores()
    
    if not data:
        return pd.DataFrame()
    
    rows = []
    for d in data:
        symbol = d['symbol']
        merit = merit_scores.get(symbol, {})
        
        chg_str = "—"
        if d['stasis_price_change_pct'] is not None:
            sign = "+" if d['stasis_price_change_pct'] >= 0 else ""
            chg_str = f"{sign}{d['stasis_price_change_pct']:.2f}%"
        
        w52_str = "—"
        if d['week52_percentile'] is not None:
            w52_str = f"{d['week52_percentile']:.0f}%"
        
        merit_score = merit.get('weighted_score', 0)
        stasis_levels = merit.get('stasis_levels', 0)
        
        if stasis_levels > 0:
            merit_str = f"{merit_score:.0f} ({stasis_levels})"
        else:
            merit_str = "—"
        
        # Type indicator
        type_str = "ETF" if d['is_etf'] else "STK"
        
        rows.append({
            'Type': type_str,
            'Symbol': symbol,
            'Merit': merit_str,
            'Merit_Val': merit_score,
            'Levels': stasis_levels,
            'Band': format_band(d['threshold_pct']),
            'Band_Val': d['threshold'],
            'Stasis': d['stasis'],
            'Dir': d['direction'] or '—',
            'Str': d['signal_strength'] or '—',
            'Current': f"${d['current_price']:.2f}" if d['current_price'] else "—",
            'Current_Val': d['current_price'] or 0,
            'Anchor': f"${d['anchor_price']:.2f}" if d['anchor_price'] else "—",
            'Anchor_Val': d['anchor_price'] or 0,
            'TP': f"${d['take_profit']:.2f}" if d['take_profit'] else "—",
            'TP_Val': d['take_profit'] or 0,
            'SL': f"${d['stop_loss']:.2f}" if d['stop_loss'] else "—",
            'SL_Val': d['stop_loss'] or 0,
            'R:R': format_rr(d['risk_reward']),
            'RR_Val': d['risk_reward'] if d['risk_reward'] is not None else -1,
            '→TP': f"{d['distance_to_tp_pct']:.3f}%" if d['distance_to_tp_pct'] else "—",
            '→TP_Val': d['distance_to_tp_pct'] or 0,
            '→SL': f"{d['distance_to_sl_pct']:.3f}%" if d['distance_to_sl_pct'] else "—",
            '→SL_Val': d['distance_to_sl_pct'] or 0,
            'Started': d['stasis_start_str'],
            'Duration': d['stasis_duration_str'],
            'Dur_Val': d['duration_seconds'],
            'Chg': chg_str,
            'Chg_Val': d['stasis_price_change_pct'] if d['stasis_price_change_pct'] else 0,
            '52W': w52_str,
            '52W_Val': d['week52_percentile'] if d['week52_percentile'] is not None else -1,
            'Bits': d['total_bits'],
            'Recent': format_bits(d['recent_bits']),
            'Tradable': '✅' if d['is_tradable'] else '',
            'Is_Tradable': d['is_tradable'],
            'Is_ETF': d['is_etf'],
        })
    
    return pd.DataFrame(rows)

# ============================================================================
# DASH APP
# ============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;600;700&display=swap');

body { 
    background-color: #0a0a0a !important; 
}

.title-font {
    font-family: 'Orbitron', sans-serif !important;
}

.data-font {
    font-family: 'Roboto Mono', monospace !important;
}

h1, h2, h3, h4, h5, h6, .btn, label, .nav-link, .card-header {
    font-family: 'Orbitron', sans-serif !important;
}

td, input, .form-control, pre, code {
    font-family: 'Roboto Mono', monospace !important;
}

.Select-control, .Select-menu-outer, .Select-option, .Select-value-label {
    font-family: 'Roboto Mono', monospace !important;
    font-size: 11px !important;
}
"""

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Beyond Price and Time"
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>''' + CUSTOM_CSS + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    # Header - FIXED LAYOUT
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(src='/assets/logo.png', style={'height': '45px', 'marginRight': '12px'}),
                html.Div([
                    html.H2("BEYOND PRICE AND TIME", className="text-success mb-0 title-font",
                           style={'fontSize': '22px', 'fontWeight': '700', 'letterSpacing': '2px'}),
                    html.P("REAL-TIME STASIS DETECTION", className="text-info title-font",
                          style={'fontSize': '9px', 'letterSpacing': '1px', 'marginBottom': '0'}),
                ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], width=5),
        dbc.Col([
            html.Div(id='connection-status', className="text-end")
        ], width=3),
        dbc.Col([
            html.Div(id='stats-summary', className="text-end", style={'fontSize': '10px'})
        ], width=4)
    ], className="mb-2 mt-2", style={'alignItems': 'center'}),
    
    # Stats Bar
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div(id='stats-display', className="text-center")
                ], style={'padding': '6px'})
            ], style={'backgroundColor': '#1a2a3a', 'border': '1px solid #00ff88'})
        ])
    ], className="mb-2"),
    
    # Filters - FIXED LAYOUT with proper spacing
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("ALL", id="btn-all", color="secondary", outline=True, size="sm", 
                          className="title-font", style={'letterSpacing': '1px', 'fontSize': '9px', 'padding': '4px 8px'}),
                dbc.Button("TRADE", id="btn-tradable", color="success", outline=True, size="sm", 
                          active=True, className="title-font", style={'letterSpacing': '1px', 'fontSize': '9px', 'padding': '4px 8px'}),
            ], size="sm")
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-type',
                        options=[{'label': 'ALL', 'value': 'ALL'},
                                {'label': 'ETF', 'value': 'ETF'},
                                {'label': 'STK', 'value': 'STK'}],
                        value='ALL', clearable=False, style={'fontSize': '10px', 'minWidth': '70px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-symbol', 
                        options=[{'label': 'ALL', 'value': 'ALL'}] + 
                        [{'label': s, 'value': s} for s in config.symbols],
                        value='ALL', clearable=False, style={'fontSize': '10px', 'minWidth': '80px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-threshold', 
                        options=[{'label': 'ALL BANDS', 'value': 'ALL'}] + 
                        [{'label': format_band(t*100), 'value': t} for t in config.thresholds],
                        value='ALL', clearable=False, style={'fontSize': '10px', 'minWidth': '90px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Input(id='filter-stasis', type='number', value=3, min=0,
                     placeholder="Min",
                     style={'width': '55px', 'fontSize': '10px', 'fontFamily': 'Roboto Mono', 'padding': '5px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-direction',
                        options=[{'label': 'DIR', 'value': 'ALL'}, 
                                {'label': 'LONG', 'value': 'LONG'},
                                {'label': 'SHORT', 'value': 'SHORT'}],
                        value='ALL', clearable=False, style={'fontSize': '10px', 'minWidth': '70px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-rr',
                        options=[{'label': 'R:R', 'value': -1},
                                {'label': '≥1', 'value': 1}, 
                                {'label': '≥2', 'value': 2},
                                {'label': '≥3', 'value': 3}],
                        value=-1, clearable=False, style={'fontSize': '10px', 'minWidth': '60px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-merit',
                        options=[{'label': 'MERIT', 'value': 0},
                                {'label': '≥10', 'value': 10},
                                {'label': '≥25', 'value': 25},
                                {'label': '≥50', 'value': 50}],
                        value=0, clearable=False, style={'fontSize': '10px', 'minWidth': '70px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-duration',
                        options=[{'label': 'DUR', 'value': 0}, 
                                {'label': '5m+', 'value': 300},
                                {'label': '15m+', 'value': 900}, 
                                {'label': '1h+', 'value': 3600}],
                        value=0, clearable=False, style={'fontSize': '10px', 'minWidth': '60px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-rows',
                        options=[{'label': '50', 'value': 50}, 
                                {'label': '100', 'value': 100},
                                {'label': '250', 'value': 250},
                                {'label': 'ALL', 'value': 5000}],
                        value=100, clearable=False, style={'fontSize': '10px', 'minWidth': '60px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-sort',
                        options=[{'label': 'MERIT↓', 'value': 'merit'},
                                {'label': 'STASIS↓', 'value': 'stasis'}, 
                                {'label': 'R:R↓', 'value': 'rr'}],
                        value='merit', clearable=False, style={'fontSize': '10px', 'minWidth': '80px'})
        ], width="auto"),
    ], className="mb-2", style={'flexWrap': 'nowrap', 'overflowX': 'auto'}),
    
    # Table
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='main-table',
                columns=[
                    {'name': '✓', 'id': 'Tradable', 'sortable': True},
                    {'name': 'T', 'id': 'Type', 'sortable': True},
                    {'name': 'SYM', 'id': 'Symbol', 'sortable': True},
                    {'name': 'MERIT', 'id': 'Merit', 'sortable': True},
                    {'name': 'BAND', 'id': 'Band', 'sortable': True},
                    {'name': 'STS', 'id': 'Stasis', 'sortable': True},
                    {'name': 'DIR', 'id': 'Dir', 'sortable': True},
                    {'name': 'STR', 'id': 'Str', 'sortable': True},
                    {'name': 'CURRENT', 'id': 'Current', 'sortable': True},
                    {'name': 'ANCHOR', 'id': 'Anchor', 'sortable': True},
                    {'name': 'TP', 'id': 'TP', 'sortable': True},
                    {'name': 'SL', 'id': 'SL', 'sortable': True},
                    {'name': 'R:R', 'id': 'R:R', 'sortable': True},
                    {'name': '→TP', 'id': '→TP', 'sortable': True},
                    {'name': '→SL', 'id': '→SL', 'sortable': True},
                    {'name': 'START', 'id': 'Started', 'sortable': True},
                    {'name': 'DUR', 'id': 'Duration', 'sortable': True},
                    {'name': 'CHG', 'id': 'Chg', 'sortable': True},
                    {'name': '52W', 'id': '52W', 'sortable': True},
                    {'name': 'BITS', 'id': 'Recent', 'sortable': False},
                ],
                sort_action='native',
                sort_mode='multi',
                sort_by=[{'column_id': 'Merit', 'direction': 'desc'}],
                style_table={'height': '68vh', 'overflowY': 'auto'},
                style_cell={
                    'backgroundColor': '#1a1a2e', 
                    'color': 'white',
                    'padding': '3px 5px', 
                    'fontSize': '10px',
                    'fontFamily': 'Roboto Mono, monospace', 
                    'whiteSpace': 'nowrap',
                    'textAlign': 'right',
                    'minWidth': '40px',
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'Type'}, 'textAlign': 'center', 'fontWeight': '600', 'minWidth': '35px'},
                    {'if': {'column_id': 'Symbol'}, 'textAlign': 'left', 'fontWeight': '700', 'color': '#00ff88'},
                    {'if': {'column_id': 'Merit'}, 'textAlign': 'center', 'fontWeight': '700'},
                    {'if': {'column_id': 'Dir'}, 'textAlign': 'center'},
                    {'if': {'column_id': 'Str'}, 'textAlign': 'center'},
                    {'if': {'column_id': 'Tradable'}, 'textAlign': 'center'},
                    {'if': {'column_id': 'Recent'}, 'textAlign': 'left', 'minWidth': '90px'},
                ],
                style_header={
                    'backgroundColor': '#2a2a4e', 
                    'color': '#00ff88',
                    'fontWeight': '700', 
                    'fontSize': '9px',
                    'fontFamily': 'Orbitron, sans-serif',
                    'borderBottom': '2px solid #00ff88',
                    'textAlign': 'center',
                    'letterSpacing': '0.5px',
                },
                style_data_conditional=[
                    # Type coloring
                    {'if': {'filter_query': '{Type} = "ETF"', 'column_id': 'Type'}, 'color': '#00ffff'},
                    {'if': {'filter_query': '{Type} = "STK"', 'column_id': 'Type'}, 'color': '#ffaa00'},
                    # Merit score coloring
                    {'if': {'filter_query': '{Merit_Val} >= 100', 'column_id': 'Merit'}, 
                     'color': '#ff00ff', 'fontWeight': 'bold', 'backgroundColor': '#3d2a4d'},
                    {'if': {'filter_query': '{Merit_Val} >= 50 && {Merit_Val} < 100', 'column_id': 'Merit'}, 
                     'color': '#ffff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Merit_Val} >= 25 && {Merit_Val} < 50', 'column_id': 'Merit'}, 
                     'color': '#00ffff', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Merit_Val} >= 10 && {Merit_Val} < 25', 'column_id': 'Merit'}, 
                     'color': '#88ff88'},
                    # High merit row highlighting
                    {'if': {'filter_query': '{Merit_Val} >= 50'}, 'backgroundColor': '#2d3a4d'},
                    # Stasis coloring
                    {'if': {'filter_query': '{Stasis} >= 10'}, 'backgroundColor': '#2d4a2d'},
                    {'if': {'filter_query': '{Stasis} >= 7 && {Stasis} < 10'}, 'backgroundColor': '#2a3a2a'},
                    # Direction coloring
                    {'if': {'filter_query': '{Dir} = "LONG"', 'column_id': 'Dir'}, 'color': '#00ff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Dir} = "SHORT"', 'column_id': 'Dir'}, 'color': '#ff4444', 'fontWeight': 'bold'},
                    # Strength coloring
                    {'if': {'filter_query': '{Str} = "VERY_STRONG"', 'column_id': 'Str'}, 'color': '#ffff00'},
                    {'if': {'filter_query': '{Str} = "STRONG"', 'column_id': 'Str'}, 'color': '#ffaa00'},
                    # Price columns
                    {'if': {'column_id': 'Current'}, 'color': '#00ffff', 'fontWeight': '600'},
                    {'if': {'column_id': 'Anchor'}, 'color': '#ffaa00'},
                    {'if': {'column_id': 'TP'}, 'color': '#00ff00'},
                    {'if': {'column_id': 'SL'}, 'color': '#ff4444'},
                    # R:R coloring
                    {'if': {'filter_query': '{RR_Val} >= 2', 'column_id': 'R:R'}, 'color': '#00ff00', 'fontWeight': '600'},
                    {'if': {'filter_query': '{RR_Val} >= 1 && {RR_Val} < 2', 'column_id': 'R:R'}, 'color': '#88ff88'},
                    # Change coloring
                    {'if': {'filter_query': '{Chg} contains "+"', 'column_id': 'Chg'}, 'color': '#00ff00'},
                    {'if': {'filter_query': '{Chg} contains "-"', 'column_id': 'Chg'}, 'color': '#ff4444'},
                    # 52W coloring
                    {'if': {'filter_query': '{52W_Val} >= 0 && {52W_Val} <= 20', 'column_id': '52W'}, 'color': '#00ff00'},
                    {'if': {'filter_query': '{52W_Val} >= 80', 'column_id': '52W'}, 'color': '#ff4444'},
                    # Alternating rows
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#151520'},
                ]
            )
        ])
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span("🟢 REAL-TIME", className="title-font", style={'color': '#00ff88', 'fontSize': '9px', 'fontWeight': 'bold'}),
                html.Span(" | ", style={'fontSize': '8px'}),
                html.Span("MERIT", style={'color': '#ff00ff', 'fontSize': '8px', 'fontWeight': 'bold'}),
                html.Span("=Score(Lvls) | ", style={'fontSize': '8px'}),
                html.Span("ETF", style={'color': '#00ffff', 'fontSize': '8px'}),
                html.Span("/", style={'fontSize': '8px'}),
                html.Span("STK", style={'color': '#ffaa00', 'fontSize': '8px'}),
                html.Span(" | ", style={'fontSize': '8px'}),
                html.Span("TP", style={'color': '#00ff00', 'fontSize': '8px'}),
                html.Span("/", style={'fontSize': '8px'}),
                html.Span("SL", style={'color': '#ff4444', 'fontSize': '8px'}),
            ], className="text-center text-muted mt-1"),
            html.Hr(style={'borderColor': '#333', 'margin': '5px 0'}),
            html.P("© 2026 TRUTH COMMUNICATIONS LLC",
                  className="text-muted text-center mb-0 title-font",
                  style={'fontSize': '8px', 'letterSpacing': '1px'}),
        ])
    ]),
    
    dcc.Store(id='view-mode', data='tradable'),
    dcc.Interval(id='refresh-interval', interval=config.update_interval_ms, n_intervals=0)
    
], fluid=True, className="p-2", style={'backgroundColor': '#0a0a0a'})

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('btn-all', 'active'), Output('btn-tradable', 'active'), Output('view-mode', 'data')],
    [Input('btn-all', 'n_clicks'), Input('btn-tradable', 'n_clicks')],
    [State('view-mode', 'data')]
)
def toggle_view(n1, n2, current):
    ctx = callback_context
    if not ctx.triggered:
        return False, True, 'tradable'
    btn = ctx.triggered[0]['prop_id'].split('.')[0]
    if btn == 'btn-all':
        return True, False, 'all'
    return False, True, 'tradable'

@app.callback(
    Output('connection-status', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_status(n):
    if not manager.backfill_complete:
        return html.Span(f"⏳ LOADING {manager.backfill_progress}%", 
                        className="text-warning title-font", style={'fontSize': '11px'})
    
    status = price_feed.get_status()
    if status['connected'] == 0:
        return html.Span("🔴 CONNECTING", className="text-warning", style={'fontSize': '11px'})
    elif status['connected'] < status['total']:
        return html.Span(f"🟡 {status['connected']}/{status['total']}", 
                        className="text-info data-font", style={'fontSize': '11px'})
    return html.Span(f"🟢 {status['connected']}/{status['total']} | {status['message_count']:,}", 
                    className="text-success data-font", style={'fontSize': '11px'})

@app.callback(
    Output('stats-summary', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_stats_summary(n):
    if not manager.backfill_complete:
        return ""
    
    data = manager.get_data()
    if not data:
        return ""
    
    tradable = sum(1 for d in data if d['is_tradable'])
    etf_count = len(config.etf_symbols)
    stock_count = len(config.symbols) - etf_count
    
    return html.Span([
        html.Span(f"ETF:{etf_count} ", style={'color': '#00ffff'}),
        html.Span(f"STK:{stock_count} ", style={'color': '#ffaa00'}),
        html.Span(f"SIGNALS:{tradable}", style={'color': '#00ff88', 'fontWeight': 'bold'}),
    ], className="data-font")

@app.callback(
    Output('stats-display', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_stats(n):
    if not manager.backfill_complete:
        return html.Span(f"⏳ LOADING {len(config.symbols)} symbols... {manager.backfill_progress}%", 
                        className="text-warning title-font")
    
    data = manager.get_data()
    merit_scores = manager.get_merit_scores()
    
    if not data:
        return html.Span("LOADING...", className="text-muted title-font")
    
    tradable = [d for d in data if d['is_tradable']]
    with_rr = [d for d in tradable if d['risk_reward'] is not None and d['risk_reward'] > 0]
    avg_rr = np.mean([d['risk_reward'] for d in with_rr]) if with_rr else 0
    long_count = sum(1 for d in tradable if d['direction'] == 'LONG')
    short_count = sum(1 for d in tradable if d['direction'] == 'SHORT')
    max_stasis = max([d['stasis'] for d in data]) if data else 0
    
    top_merit = max(merit_scores.values(), key=lambda x: x['weighted_score'], default={'weighted_score': 0})
    top_merit_symbol = [k for k, v in merit_scores.items() if v['weighted_score'] == top_merit['weighted_score']]
    top_merit_symbol = top_merit_symbol[0] if top_merit_symbol else "—"
    
    multi_level = sum(1 for m in merit_scores.values() if m['stasis_levels'] >= 2)
    
    return html.Div([
        html.Span("🎯 TRADABLE: ", className="title-font", style={'fontSize': '10px'}),
        html.Span(f"{len(tradable)}", className="data-font text-success", style={'fontSize': '11px', 'fontWeight': '600'}),
        html.Span("  🏆 TOP: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{top_merit_symbol}({top_merit['weighted_score']:.0f})", 
                 className="data-font text-warning", style={'fontSize': '11px', 'fontWeight': '600'}),
        html.Span("  📊 MULTI: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{multi_level}", className="data-font text-info", style={'fontSize': '11px'}),
        html.Span("  📈 L: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{long_count}", className="data-font text-success", style={'fontSize': '11px'}),
        html.Span("  📉 S: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{short_count}", className="data-font text-danger", style={'fontSize': '11px'}),
        html.Span("  ⚡ MAX: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{max_stasis}", className="data-font text-warning", style={'fontSize': '11px'}),
        html.Span("  R:R: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{avg_rr:.1f}", className="data-font text-info", style={'fontSize': '11px'}),
    ])

@app.callback(
    Output('main-table', 'data'),
    [Input('refresh-interval', 'n_intervals'),
     Input('view-mode', 'data'),
     Input('filter-type', 'value'),
     Input('filter-symbol', 'value'),
     Input('filter-threshold', 'value'),
     Input('filter-stasis', 'value'),
     Input('filter-direction', 'value'),
     Input('filter-rr', 'value'),
     Input('filter-merit', 'value'),
     Input('filter-duration', 'value'),
     Input('filter-rows', 'value'),
     Input('filter-sort', 'value')]
)
def update_table(n, view_mode, type_filter, sym, thresh, stasis, direction, rr, merit, duration, rows, sort):
    df = get_table_data()
    if df.empty:
        return []
    
    if view_mode == 'tradable':
        df = df[df['Is_Tradable'] == True]
    
    if type_filter == 'ETF':
        df = df[df['Is_ETF'] == True]
    elif type_filter == 'STK':
        df = df[df['Is_ETF'] == False]
    
    if sym != 'ALL':
        df = df[df['Symbol'] == sym]
    if thresh != 'ALL':
        df = df[df['Band_Val'] == thresh]
    if stasis and stasis > 0:
        df = df[df['Stasis'] >= stasis]
    if direction != 'ALL':
        df = df[df['Dir'] == direction]
    if rr is not None and rr >= 0:
        df = df[(df['RR_Val'].notna()) & (df['RR_Val'] >= rr)]
    if merit is not None and merit > 0:
        df = df[df['Merit_Val'] >= merit]
    if duration and duration > 0:
        df = df[df['Dur_Val'] >= duration]
    
    if sort == 'merit':
        df = df.sort_values(['Merit_Val', 'Stasis'], ascending=[False, False])
    elif sort == 'stasis':
        df = df.sort_values(['Stasis', 'Merit_Val'], ascending=[False, False])
    elif sort == 'rr':
        df = df.sort_values(['RR_Val', 'Merit_Val'], ascending=[False, False])
    
    df = df.head(rows)
    
    drop_cols = ['Band_Val', 'Current_Val', 'Anchor_Val', 'TP_Val', 'SL_Val', 
                 'RR_Val', '→TP_Val', '→SL_Val', 'Dur_Val', 'Chg_Val', '52W_Val', 
                 'Is_Tradable', 'Merit_Val', 'Levels', 'Bits', 'Is_ETF']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    return df.to_dict('records')

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  BEYOND PRICE AND TIME")
    print("  ETFs + Top 100 Liquid Stocks")
    print("  © 2026 Truth Communications LLC. All Rights Reserved.")
    print("=" * 70)
    
    print(f"\n📊 Total Symbols: {len(config.symbols)}")
    print(f"   ETFs: {len(config.etf_symbols)}")
    print(f"   Stocks: {len(config.symbols) - len(config.etf_symbols)}")
    
    print(f"\n📏 Thresholds ({len(config.thresholds)}):")
    for t in config.thresholds:
        print(f"   • {t*100:.4f}%")
    
    print(f"\n🔢 Total Bitstreams: {len(config.symbols) * len(config.thresholds)}")
    
    print("\n" + "=" * 70)
    print("📅 FETCHING 52-WEEK DATA")
    print("=" * 70)
    config.week52_data = fetch_52_week_data()
    
    print("=" * 70)
    print("📊 FETCHING VOLUME DATA")
    print("=" * 70)
    config.volumes = fetch_volume_data()
    
    manager.backfill()
    
    price_feed.start()
    manager.start()
    
    print("\n✅ Server: http://127.0.0.1:8050")
    print("\n📋 FEATURES:")
    print(f"   • {len(config.etf_symbols)} ETFs + {len(config.symbols) - len(config.etf_symbols)} liquid stocks")
    print("   • 10 threshold levels (0.0625% to 10%)")
    print(f"   • {len(config.symbols) * len(config.thresholds)} total bitstreams")
    print("   • Merit score for multi-level confluence")
    print("   • Type filter (ETF/STK)")
    print("=" * 70 + "\n")
    
    threading.Thread(target=lambda: (time.sleep(2), webbrowser.open('http://127.0.0.1:8050')), daemon=True).start()
    
    app.run(debug=False, host='127.0.0.1', port=8050)
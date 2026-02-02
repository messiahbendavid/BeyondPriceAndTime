# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 20:08:24 2026

@author: brcum
"""
# -*- coding: utf-8 -*-
"""
BEYOND PRICE AND TIME - JACKPOT EDITION
Copyright © 2026 Truth Communications LLC. All Rights Reserved.

🎰 GAMIFIED TRADING SIGNALS WITH STASIS DETECTION
Finding the "777" alignment across all price levels

The Edge: Detect when stasis aligns in the SAME DIRECTION across ALL levels
Like hitting 7-7-7 in slots or 21 in blackjack - the ultimate vector alignment
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
import os
import random

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table, ALL, MATCH
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
# GAMIFICATION CONSTANTS
# ============================================================================

# Jackpot Tiers (like casino payouts)
JACKPOT_TIERS = {
    'GRAND_JACKPOT': {'min_levels': 8, 'min_alignment': 100, 'emoji': '🎰💎🎰', 'color': '#ff00ff', 'multiplier': '1000x'},
    'MEGA_JACKPOT': {'min_levels': 6, 'min_alignment': 100, 'emoji': '🎰🎰🎰', 'color': '#ffff00', 'multiplier': '100x'},
    'SUPER_JACKPOT': {'min_levels': 5, 'min_alignment': 100, 'emoji': '💰💰💰', 'color': '#00ffff', 'multiplier': '50x'},
    'JACKPOT': {'min_levels': 4, 'min_alignment': 100, 'emoji': '🍀🍀🍀', 'color': '#00ff88', 'multiplier': '25x'},
    'BIG_WIN': {'min_levels': 3, 'min_alignment': 100, 'emoji': '⭐⭐⭐', 'color': '#88ff88', 'multiplier': '10x'},
    'WIN': {'min_levels': 2, 'min_alignment': 100, 'emoji': '✨✨', 'color': '#aaffaa', 'multiplier': '5x'},
    'NEAR_MISS': {'min_levels': 2, 'min_alignment': 75, 'emoji': '🎯', 'color': '#ffaa00', 'multiplier': '2x'},
}

# Achievement Badges
ACHIEVEMENTS = {
    'PERFECT_10': {'desc': '10 levels aligned', 'emoji': '🏆', 'rarity': 'LEGENDARY'},
    'SEVEN_SEVEN_SEVEN': {'desc': '7+ levels, 7+ stasis', 'emoji': '🎰', 'rarity': 'EPIC'},
    'BLACKJACK': {'desc': '21+ total stasis aligned', 'emoji': '🃏', 'rarity': 'RARE'},
    'FULL_HOUSE': {'desc': '5 levels same direction', 'emoji': '🏠', 'rarity': 'UNCOMMON'},
    'TRIPLE_THREAT': {'desc': '3 high-threshold alignments', 'emoji': '🔥', 'rarity': 'COMMON'},
}

# Rarity Colors
RARITY_COLORS = {
    'LEGENDARY': '#ff8000',  # Orange
    'EPIC': '#a335ee',       # Purple
    'RARE': '#0070dd',       # Blue
    'UNCOMMON': '#1eff00',   # Green
    'COMMON': '#ffffff',     # White
}

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    symbols: List[str] = field(default_factory=lambda: [
        # ==================== ETFs (19) ====================
        "SPY", "QQQ", "IWM", "DIA",
        "XLF", "XLE", "XLU", "XLK", "XLP", "XLB", "XLV", "XLI", "XLY", "XLC", "XLRE",
        "KRE", "SMH", "XBI", "GDX",
        
        # ==================== TOP 100 LIQUID STOCKS ====================
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "ADBE",
        "CRM", "AMD", "INTC", "CSCO", "QCOM", "IBM", "NOW", "INTU", "AMAT", "MU",
        "NFLX", "PYPL", "SHOP", "SQ", "UBER", "ABNB", "COIN", "SNAP", "ROKU", "PLTR",
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V",
        "MA", "COF", "USB", "PNC", "TFC",
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "VRTX", "REGN", "MRNA",
        "WMT", "HD", "COST", "TGT", "LOW", "NKE", "SBUX", "MCD", "DIS", "CMCSA",
        "CAT", "BA", "GE", "HON", "UNP", "RTX", "LMT", "DE", "UPS", "FDX",
        "XOM", "CVX", "COP", "SLB", "EOG",
        "T", "VZ", "TMUS", "PG", "KO", "PEP", "PM", "MO", "CL", "MMM",
        "F", "GM", "RIVN", "LCID", "NIO",
    ])
    
    etf_symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "DIA",
        "XLF", "XLE", "XLU", "XLK", "XLP", "XLB", "XLV", "XLI", "XLY", "XLC", "XLRE",
        "KRE", "SMH", "XBI", "GDX",
    ])
    
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
    alerts_file: str = "price_alerts.json"

config = Config()
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
# JACKPOT CALCULATOR
# ============================================================================

def calculate_jackpot_status(merit_data: Dict) -> Dict:
    """
    Calculate the jackpot status for a symbol based on alignment.
    
    This is the CORE of the gamification - detecting when all threshold levels
    are aligned like hitting 7-7-7 on a slot machine.
    """
    levels = merit_data.get('stasis_levels', 0)
    alignment = merit_data.get('direction_alignment', 0)
    total_stasis = merit_data.get('total_stasis', 0)
    long_levels = merit_data.get('long_levels', 0)
    short_levels = merit_data.get('short_levels', 0)
    weighted_score = merit_data.get('weighted_score', 0)
    max_stasis = merit_data.get('max_stasis', 0)
    
    result = {
        'tier': None,
        'tier_name': 'NO SIGNAL',
        'emoji': '⬜',
        'color': '#333333',
        'multiplier': '0x',
        'is_jackpot': False,
        'achievements': [],
        'slot_display': ['⬜', '⬜', '⬜'],
        'heat_level': 0,  # 0-100
        'vector_strength': 0,  # Combined directional force
    }
    
    if levels == 0:
        return result
    
    # Calculate vector strength (directional force)
    if alignment == 100:
        result['vector_strength'] = levels * 10 + total_stasis
    else:
        # Partial alignment reduces vector strength
        result['vector_strength'] = (levels * 10 + total_stasis) * (alignment / 100) * 0.5
    
    # Calculate heat level (0-100)
    heat = min(100, (levels / 10) * 50 + (alignment / 100) * 30 + min(20, max_stasis * 2))
    result['heat_level'] = int(heat)
    
    # Determine jackpot tier
    for tier_name, tier_info in JACKPOT_TIERS.items():
        if levels >= tier_info['min_levels'] and alignment >= tier_info['min_alignment']:
            result['tier'] = tier_name
            result['tier_name'] = tier_name.replace('_', ' ')
            result['emoji'] = tier_info['emoji']
            result['color'] = tier_info['color']
            result['multiplier'] = tier_info['multiplier']
            result['is_jackpot'] = 'JACKPOT' in tier_name
            break
    
    # Generate slot machine display
    direction = merit_data.get('dominant_direction', None)
    if direction == 'LONG':
        base_symbol = '🟢'
        alt_symbol = '🔴'
    elif direction == 'SHORT':
        base_symbol = '🔴'
        alt_symbol = '🟢'
    else:
        base_symbol = '⬜'
        alt_symbol = '⬜'
    
    # Slot display based on alignment
    if alignment == 100 and levels >= 3:
        result['slot_display'] = [base_symbol, base_symbol, base_symbol]
    elif alignment >= 75:
        result['slot_display'] = [base_symbol, base_symbol, alt_symbol]
    elif alignment >= 50:
        result['slot_display'] = [base_symbol, alt_symbol, base_symbol]
    else:
        result['slot_display'] = [base_symbol, alt_symbol, alt_symbol]
    
    # Check for achievements
    # PERFECT_10: All 10 levels aligned
    if levels >= 10 and alignment == 100:
        result['achievements'].append('PERFECT_10')
    
    # SEVEN_SEVEN_SEVEN: 7+ levels with 7+ max stasis
    if levels >= 7 and max_stasis >= 7 and alignment == 100:
        result['achievements'].append('SEVEN_SEVEN_SEVEN')
    
    # BLACKJACK: 21+ total stasis aligned
    if total_stasis >= 21 and alignment == 100:
        result['achievements'].append('BLACKJACK')
    
    # FULL_HOUSE: 5 levels same direction
    if levels >= 5 and alignment == 100:
        result['achievements'].append('FULL_HOUSE')
    
    # TRIPLE_THREAT: 3+ high-threshold alignments
    thresholds = merit_data.get('thresholds_in_stasis', [])
    high_thresholds = [t for t in thresholds if t >= 0.02]
    if len(high_thresholds) >= 3 and alignment == 100:
        result['achievements'].append('TRIPLE_THREAT')
    
    return result


def get_heat_color(heat_level: int) -> str:
    """Convert heat level (0-100) to a color gradient."""
    if heat_level >= 90:
        return '#ff0000'  # Red hot
    elif heat_level >= 75:
        return '#ff4400'  # Orange-red
    elif heat_level >= 60:
        return '#ff8800'  # Orange
    elif heat_level >= 45:
        return '#ffcc00'  # Yellow-orange
    elif heat_level >= 30:
        return '#ffff00'  # Yellow
    elif heat_level >= 15:
        return '#88ff00'  # Yellow-green
    else:
        return '#00ff88'  # Green (cool)


def get_slot_reel_html(symbols: List[str], spinning: bool = False) -> html.Div:
    """Create a slot machine reel display."""
    return html.Div([
        html.Div(s, style={
            'display': 'inline-block',
            'width': '30px',
            'height': '30px',
            'lineHeight': '30px',
            'textAlign': 'center',
            'fontSize': '20px',
            'border': '2px solid #ffd700',
            'borderRadius': '5px',
            'margin': '0 2px',
            'backgroundColor': '#1a1a2e',
            'boxShadow': '0 0 10px rgba(255,215,0,0.3)',
        }) for s in symbols
    ], style={'display': 'inline-block'})


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
        self.is_etf = symbol in config.etf_symbols
        
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


# ============================================================================
# PRICE FEED
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
        self.cached_jackpots: Dict[str, Dict] = {}
        self.cache_lock = threading.Lock()
        
        self.initialized = False
        self.backfill_complete = False
        self.backfill_progress = 0
        
        # Jackpot history for celebrations
        self.jackpot_history: List[Dict] = []
        self.recent_jackpots: deque = deque(maxlen=10)
    
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
            
            # Calculate jackpot status for each symbol
            jackpots = {}
            for symbol, merit_data in merit_scores.items():
                jackpot_status = calculate_jackpot_status(merit_data)
                jackpots[symbol] = jackpot_status
                
                # Track new jackpots
                if jackpot_status['is_jackpot']:
                    jackpot_entry = {
                        'symbol': symbol,
                        'tier': jackpot_status['tier'],
                        'levels': merit_data.get('stasis_levels', 0),
                        'direction': merit_data.get('dominant_direction'),
                        'timestamp': datetime.now(),
                    }
                    
                    # Check if this is a new jackpot
                    existing = [j for j in self.recent_jackpots if j['symbol'] == symbol]
                    if not existing or existing[-1]['tier'] != jackpot_status['tier']:
                        self.recent_jackpots.append(jackpot_entry)
            
            with self.cache_lock:
                self.cached_data = snapshots
                self.cached_merit_scores = merit_scores
                self.cached_jackpots = jackpots
    
    def _calculate_merit_scores(self, snapshots: List[Dict]) -> Dict[str, Dict]:
        """
        Merit Score = Alignment Confidence Score
        
        The higher the merit, the more "aligned" the vector forces are.
        Perfect alignment across all threshold levels = JACKPOT
        """
        
        THRESHOLD_WEIGHTS = {
            0.000625: 1.0,
            0.00125:  1.5,
            0.0025:   2.0,
            0.005:    3.0,
            0.01:     5.0,
            0.02:     8.0,
            0.03:     12.0,
            0.04:     16.0,
            0.05:     20.0,
            0.10:     30.0,
        }
        
        def get_threshold_weight(threshold: float) -> float:
            if threshold in THRESHOLD_WEIGHTS:
                return THRESHOLD_WEIGHTS[threshold]
            return max(1.0, threshold * 200)
        
        def calculate_level_weight(stasis: int, threshold: float) -> float:
            stasis_factor = stasis ** 1.3
            threshold_factor = get_threshold_weight(threshold)
            return stasis_factor * threshold_factor
        
        merit_data = defaultdict(lambda: {
            'stasis_levels': 0,
            'total_stasis': 0,
            'weighted_score': 0,
            'directions': [],
            'dominant_direction': None,
            'direction_alignment': 0,
            'conflict_penalty': 0,
            'alignment_bonus': 0,
            'avg_stasis': 0,
            'max_stasis': 0,
            'max_threshold_in_stasis': 0,
            'long_levels': 0,
            'short_levels': 0,
            'long_weight': 0,
            'short_weight': 0,
            'levels_detail': [],
            'thresholds_in_stasis': [],
            'net_direction': None,
            'confidence': 0,
            'highest_aligned_threshold': 0,
        })
        
        symbol_snapshots = defaultdict(list)
        for snap in snapshots:
            symbol_snapshots[snap['symbol']].append(snap)
        
        for symbol, snaps in symbol_snapshots.items():
            long_levels = []
            short_levels = []
            all_stasis = []
            all_thresholds_in_stasis = []
            
            for snap in snaps:
                stasis = snap['stasis']
                direction = snap['direction']
                threshold = snap['threshold']
                
                all_stasis.append(stasis)
                
                if stasis >= config.min_tradable_stasis and direction:
                    weight = calculate_level_weight(stasis, threshold)
                    
                    level_data = {
                        'threshold': threshold,
                        'threshold_pct': threshold * 100,
                        'stasis': stasis,
                        'direction': direction,
                        'weight': weight,
                        'threshold_weight': get_threshold_weight(threshold),
                    }
                    
                    all_thresholds_in_stasis.append(threshold)
                    
                    if direction == 'LONG':
                        long_levels.append(level_data)
                    else:
                        short_levels.append(level_data)
            
            long_weight = sum(level['weight'] for level in long_levels)
            short_weight = sum(level['weight'] for level in short_levels)
            long_stasis_sum = sum(level['stasis'] for level in long_levels)
            short_stasis_sum = sum(level['stasis'] for level in short_levels)
            
            total_levels = len(long_levels) + len(short_levels)
            total_weight = long_weight + short_weight
            
            max_long_threshold = max([l['threshold'] for l in long_levels]) if long_levels else 0
            max_short_threshold = max([l['threshold'] for l in short_levels]) if short_levels else 0
            max_threshold_in_stasis = max(max_long_threshold, max_short_threshold)
            
            if total_levels == 0:
                merit_data[symbol] = {
                    'stasis_levels': 0,
                    'total_stasis': sum(all_stasis),
                    'weighted_score': 0,
                    'directions': [],
                    'dominant_direction': None,
                    'direction_alignment': 0,
                    'conflict_penalty': 0,
                    'alignment_bonus': 0,
                    'avg_stasis': 0,
                    'max_stasis': max(all_stasis) if all_stasis else 0,
                    'max_threshold_in_stasis': 0,
                    'long_levels': 0,
                    'short_levels': 0,
                    'long_weight': 0,
                    'short_weight': 0,
                    'levels_detail': [],
                    'thresholds_in_stasis': [],
                    'net_direction': None,
                    'confidence': 0,
                    'highest_aligned_threshold': 0,
                }
                continue
            
            if long_weight > short_weight:
                dominant_direction = 'LONG'
                net_weight = long_weight - short_weight
                aligned_levels = long_levels
                opposing_levels = short_levels
                aligned_weight = long_weight
                opposing_weight = short_weight
                highest_aligned_threshold = max_long_threshold
            elif short_weight > long_weight:
                dominant_direction = 'SHORT'
                net_weight = short_weight - long_weight
                aligned_levels = short_levels
                opposing_levels = long_levels
                aligned_weight = short_weight
                opposing_weight = long_weight
                highest_aligned_threshold = max_short_threshold
            else:
                dominant_direction = None
                net_weight = 0
                aligned_levels = []
                opposing_levels = long_levels + short_levels
                aligned_weight = 0
                opposing_weight = total_weight
                highest_aligned_threshold = 0
            
            if total_weight > 0:
                alignment_pct = (max(long_weight, short_weight) / total_weight) * 100
            else:
                alignment_pct = 0
            
            base_score = net_weight
            
            num_aligned = len(aligned_levels)
            num_opposing = len(opposing_levels)
            
            if alignment_pct == 100 and num_aligned >= 2:
                threshold_multiplier = 1 + (get_threshold_weight(highest_aligned_threshold) / 30)
                alignment_bonus = base_score * (0.2 * (num_aligned - 1)) * threshold_multiplier
                
                if num_aligned >= 3:
                    alignment_bonus *= 1.3
                if num_aligned >= 4:
                    alignment_bonus *= 1.2
                if num_aligned >= 5:
                    alignment_bonus *= 1.2
                    
            elif alignment_pct >= 75:
                alignment_bonus = base_score * (0.1 * (num_aligned - 1))
            elif alignment_pct >= 60:
                alignment_bonus = base_score * 0.05
            else:
                alignment_bonus = 0
            
            if num_opposing > 0 and total_weight > 0:
                conflict_ratio = opposing_weight / total_weight
                max_opposing_threshold = max([l['threshold'] for l in opposing_levels]) if opposing_levels else 0
                high_threshold_conflict = max_opposing_threshold >= 0.01
                
                if conflict_ratio > 0.4:
                    conflict_penalty = base_score * 0.6
                    if high_threshold_conflict:
                        conflict_penalty *= 1.3
                elif conflict_ratio > 0.25:
                    conflict_penalty = base_score * 0.35
                    if high_threshold_conflict:
                        conflict_penalty *= 1.2
                elif conflict_ratio > 0.1:
                    conflict_penalty = base_score * 0.15
                else:
                    conflict_penalty = base_score * 0.05
            else:
                conflict_penalty = 0
            
            high_threshold_bonus = 0
            for level in aligned_levels:
                if level['threshold'] >= 0.05:
                    high_threshold_bonus += level['weight'] * 0.25
                elif level['threshold'] >= 0.02:
                    high_threshold_bonus += level['weight'] * 0.1
                elif level['threshold'] >= 0.01:
                    high_threshold_bonus += level['weight'] * 0.05
            
            max_stasis = max(all_stasis) if all_stasis else 0
            max_aligned_stasis = max([l['stasis'] for l in aligned_levels]) if aligned_levels else 0
            
            if max_aligned_stasis >= 10:
                stasis_depth_bonus = base_score * 0.2
            elif max_aligned_stasis >= 7:
                stasis_depth_bonus = base_score * 0.1
            elif max_aligned_stasis >= 5:
                stasis_depth_bonus = base_score * 0.05
            else:
                stasis_depth_bonus = 0
            
            weighted_score = (
                base_score 
                + alignment_bonus 
                + high_threshold_bonus 
                + stasis_depth_bonus 
                - conflict_penalty
            )
            
            weighted_score = max(0, weighted_score)
            
            if dominant_direction and aligned_levels:
                alignment_component = (alignment_pct / 100) * 40
                stasis_component = min(max_aligned_stasis / 10, 1.0) * 30
                threshold_component = min(get_threshold_weight(highest_aligned_threshold) / 20, 1.0) * 30
                confidence = alignment_component + stasis_component + threshold_component
            else:
                confidence = 0
            
            all_directions = [l['direction'] for l in long_levels + short_levels]
            all_levels = sorted(long_levels + short_levels, key=lambda x: x['threshold'], reverse=True)
            
            merit_data[symbol] = {
                'stasis_levels': total_levels,
                'total_stasis': sum(all_stasis),
                'weighted_score': round(weighted_score, 1),
                'directions': all_directions,
                'dominant_direction': dominant_direction,
                'direction_alignment': round(alignment_pct, 0),
                'conflict_penalty': round(conflict_penalty, 1),
                'alignment_bonus': round(alignment_bonus + high_threshold_bonus + stasis_depth_bonus, 1),
                'avg_stasis': round((long_stasis_sum + short_stasis_sum) / total_levels, 1) if total_levels > 0 else 0,
                'max_stasis': max_stasis,
                'max_threshold_in_stasis': max_threshold_in_stasis,
                'max_threshold_pct': max_threshold_in_stasis * 100,
                'long_levels': len(long_levels),
                'short_levels': len(short_levels),
                'long_weight': round(long_weight, 1),
                'short_weight': round(short_weight, 1),
                'levels_detail': all_levels,
                'thresholds_in_stasis': sorted(all_thresholds_in_stasis, reverse=True),
                'net_direction': dominant_direction,
                'confidence': round(confidence, 0),
                'highest_aligned_threshold': highest_aligned_threshold,
                'highest_aligned_threshold_pct': highest_aligned_threshold * 100,
            }
        
        return dict(merit_data)
    
    def get_data(self) -> List[Dict]:
        with self.cache_lock:
            return copy.deepcopy(self.cached_data)
    
    def get_merit_scores(self) -> Dict[str, Dict]:
        with self.cache_lock:
            return copy.deepcopy(self.cached_merit_scores)
    
    def get_jackpots(self) -> Dict[str, Dict]:
        with self.cache_lock:
            return copy.deepcopy(self.cached_jackpots)
    
    def get_recent_jackpots(self) -> List[Dict]:
        return list(self.recent_jackpots)


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
    jackpots = manager.get_jackpots()
    
    if not data:
        return pd.DataFrame()
    
    rows = []
    for d in data:
        symbol = d['symbol']
        merit = merit_scores.get(symbol, {})
        jackpot = jackpots.get(symbol, {})
        
        chg_str = "—"
        if d['stasis_price_change_pct'] is not None:
            sign = "+" if d['stasis_price_change_pct'] >= 0 else ""
            chg_str = f"{sign}{d['stasis_price_change_pct']:.2f}%"
        
        w52_str = "—"
        if d['week52_percentile'] is not None:
            w52_str = f"{d['week52_percentile']:.0f}%"
        
        merit_score = merit.get('weighted_score', 0)
        stasis_levels = merit.get('stasis_levels', 0)
        alignment = merit.get('direction_alignment', 0)
        
        # Jackpot display
        jackpot_tier = jackpot.get('tier_name', 'NO SIGNAL')
        jackpot_emoji = jackpot.get('emoji', '⬜')
        heat_level = jackpot.get('heat_level', 0)
        
        if stasis_levels > 0:
            merit_str = f"{merit_score:.0f}"
        else:
            merit_str = "—"
        
        type_str = "ETF" if d['is_etf'] else "STK"
        
        rows.append({
            'Type': type_str,
            'Symbol': symbol,
            'Jackpot': jackpot_emoji,
            'Jackpot_Tier': jackpot_tier,
            'Heat': heat_level,
            'Merit': merit_str,
            'Merit_Val': merit_score,
            'Levels': stasis_levels,
            'Align': f"{alignment:.0f}%" if stasis_levels > 0 else "—",
            'Align_Val': alignment,
            'Band': format_band(d['threshold_pct']),
            'Band_Val': d['threshold'],
            'Stasis': d['stasis'],
            'Dir': d['direction'] or '—',
            'Str': d['signal_strength'] or '—',
            'Current': f"${d['current_price']:.2f}" if d['current_price'] else "—",
            'Current_Val': d['current_price'] or 0,
            'Anchor': f"${d['anchor_price']:.2f}" if d['anchor_price'] else "—",
            'TP': f"${d['take_profit']:.2f}" if d['take_profit'] else "—",
            'SL': f"${d['stop_loss']:.2f}" if d['stop_loss'] else "—",
            'R:R': format_rr(d['risk_reward']),
            'RR_Val': d['risk_reward'] if d['risk_reward'] is not None else -1,
            '→TP': f"{d['distance_to_tp_pct']:.3f}%" if d['distance_to_tp_pct'] else "—",
            '→SL': f"{d['distance_to_sl_pct']:.3f}%" if d['distance_to_sl_pct'] else "—",
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
            'Is_Jackpot': jackpot.get('is_jackpot', False),
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# CUSTOM CSS - GAMIFIED THEME
# ============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

body { 
    background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0a0a0f 100%) !important;
    background-attachment: fixed !important;
}

/* Animated background */
body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(ellipse at 20% 80%, rgba(255, 0, 255, 0.1) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(0, 255, 136, 0.1) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(0, 255, 255, 0.05) 0%, transparent 70%);
    pointer-events: none;
    z-index: -1;
}

.title-font {
    font-family: 'Orbitron', sans-serif !important;
}

.arcade-font {
    font-family: 'Press Start 2P', cursive !important;
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

/* Jackpot animations */
@keyframes jackpot-glow {
    0%, 100% { 
        box-shadow: 0 0 5px #ffd700, 0 0 10px #ffd700, 0 0 15px #ffd700;
    }
    50% { 
        box-shadow: 0 0 10px #ffd700, 0 0 20px #ffd700, 0 0 30px #ffd700, 0 0 40px #ffd700;
    }
}

@keyframes mega-jackpot {
    0%, 100% { 
        box-shadow: 0 0 10px #ff00ff, 0 0 20px #ff00ff, 0 0 30px #ff00ff;
        transform: scale(1);
    }
    50% { 
        box-shadow: 0 0 20px #ff00ff, 0 0 40px #ff00ff, 0 0 60px #ff00ff, 0 0 80px #ff00ff;
        transform: scale(1.02);
    }
}

@keyframes pulse-green {
    0%, 100% { background-color: rgba(0, 255, 136, 0.2); }
    50% { background-color: rgba(0, 255, 136, 0.4); }
}

@keyframes pulse-red {
    0%, 100% { background-color: rgba(255, 68, 68, 0.2); }
    50% { background-color: rgba(255, 68, 68, 0.4); }
}

@keyframes slot-spin {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(0); }
}

@keyframes rainbow-border {
    0% { border-color: #ff0000; }
    17% { border-color: #ff8800; }
    33% { border-color: #ffff00; }
    50% { border-color: #00ff00; }
    67% { border-color: #0088ff; }
    83% { border-color: #8800ff; }
    100% { border-color: #ff0000; }
}

@keyframes flash-celebration {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.jackpot-card {
    animation: jackpot-glow 1.5s ease-in-out infinite;
    border: 2px solid #ffd700 !important;
}

.mega-jackpot-card {
    animation: mega-jackpot 1s ease-in-out infinite;
    border: 3px solid #ff00ff !important;
}

.grand-jackpot-card {
    animation: mega-jackpot 0.5s ease-in-out infinite, rainbow-border 2s linear infinite;
    border: 4px solid #ff00ff !important;
}

/* Heat meter styling */
.heat-meter {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #00ff88 0%, #ffff00 50%, #ff0000 100%);
    transition: width 0.3s ease;
}

.heat-container {
    background: #1a1a2e;
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid #333;
}

/* Slot machine styling */
.slot-container {
    display: inline-flex;
    gap: 4px;
    padding: 4px 8px;
    background: linear-gradient(180deg, #2a2a4e 0%, #1a1a2e 100%);
    border-radius: 8px;
    border: 2px solid #ffd700;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
}

.slot-reel {
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    background: #0a0a0f;
    border-radius: 4px;
    border: 1px solid #444;
}

/* Achievement badges */
.achievement-badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 9px;
    font-weight: bold;
    margin: 1px;
}

.achievement-legendary {
    background: linear-gradient(135deg, #ff8000, #ffd700);
    color: #000;
    animation: jackpot-glow 1s ease-in-out infinite;
}

.achievement-epic {
    background: linear-gradient(135deg, #a335ee, #cc77ff);
    color: #fff;
}

.achievement-rare {
    background: linear-gradient(135deg, #0070dd, #00aaff);
    color: #fff;
}

/* Leaderboard styling */
.leaderboard-item {
    background: linear-gradient(90deg, rgba(0,255,136,0.1) 0%, transparent 100%);
    border-left: 3px solid #00ff88;
    padding: 8px 12px;
    margin: 4px 0;
    transition: all 0.3s ease;
}

.leaderboard-item:hover {
    background: linear-gradient(90deg, rgba(0,255,136,0.2) 0%, transparent 100%);
    transform: translateX(5px);
}

.leaderboard-rank {
    font-size: 18px;
    font-weight: bold;
    color: #ffd700;
}

/* Alert flash */
.alert-flash {
    animation: flash-celebration 0.3s ease-in-out 5;
}

/* Neon text effects */
.neon-green {
    color: #00ff88;
    text-shadow: 0 0 5px #00ff88, 0 0 10px #00ff88;
}

.neon-red {
    color: #ff4444;
    text-shadow: 0 0 5px #ff4444, 0 0 10px #ff4444;
}

.neon-gold {
    color: #ffd700;
    text-shadow: 0 0 5px #ffd700, 0 0 10px #ffd700, 0 0 15px #ffd700;
}

.neon-purple {
    color: #ff00ff;
    text-shadow: 0 0 5px #ff00ff, 0 0 10px #ff00ff, 0 0 15px #ff00ff;
}

/* Card hover effects */
.hover-glow:hover {
    box-shadow: 0 0 15px rgba(0, 255, 136, 0.5);
    transition: box-shadow 0.3s ease;
}

/* Stats counter animation */
@keyframes count-up {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.stat-value {
    animation: count-up 0.3s ease-out;
}
"""


# ============================================================================
# DASH APP
# ============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
app.title = "🎰 Beyond Price and Time - JACKPOT EDITION"
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


def create_jackpot_display():
    """Create the main jackpot display section."""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                # Slot machine header
                html.Div([
                    html.H3("🎰 JACKPOT SCANNER 🎰", className="text-center neon-gold arcade-font mb-0",
                           style={'fontSize': '16px', 'letterSpacing': '2px'}),
                    html.P("Detecting multi-level alignment...", className="text-center text-muted",
                          style={'fontSize': '9px', 'marginBottom': '0'}),
                ]),
            ], className="mb-2"),
            
            # Current jackpots display
            html.Div(id='current-jackpots', className="text-center"),
            
        ], style={'padding': '10px'})
    ], style={
        'backgroundColor': 'rgba(26, 26, 46, 0.9)',
        'border': '2px solid #ffd700',
        'borderRadius': '10px',
        'boxShadow': '0 0 20px rgba(255, 215, 0, 0.3)',
    })


def create_heat_map_display():
    """Create a heat map showing all symbols' alignment status."""
    return dbc.Card([
        dbc.CardBody([
            html.H5("🔥 HEAT MAP", className="title-font text-warning mb-2", style={'fontSize': '12px'}),
            html.Div(id='heat-map-display'),
        ], style={'padding': '10px'})
    ], style={
        'backgroundColor': 'rgba(26, 26, 46, 0.9)',
        'border': '1px solid #ff8800',
        'borderRadius': '8px',
    })


def create_leaderboard():
    """Create the top signals leaderboard."""
    return dbc.Card([
        dbc.CardBody([
            html.H5("🏆 LEADERBOARD", className="title-font neon-gold mb-2", style={'fontSize': '12px'}),
            html.Div(id='leaderboard-display'),
        ], style={'padding': '10px', 'maxHeight': '300px', 'overflowY': 'auto'})
    ], style={
        'backgroundColor': 'rgba(26, 26, 46, 0.9)',
        'border': '1px solid #ffd700',
        'borderRadius': '8px',
    })


def create_achievement_panel():
    """Create the achievements panel."""
    return dbc.Card([
        dbc.CardBody([
            html.H5("🏅 ACHIEVEMENTS", className="title-font text-info mb-2", style={'fontSize': '12px'}),
            html.Div(id='achievements-display'),
        ], style={'padding': '10px'})
    ], style={
        'backgroundColor': 'rgba(26, 26, 46, 0.9)',
        'border': '1px solid #00ffff',
        'borderRadius': '8px',
    })


# Main layout
app.layout = dbc.Container([
    # Jackpot Toast Notification
    dbc.Toast(
        id="jackpot-toast",
        header="🎰 JACKPOT! 🎰",
        icon="warning",
        is_open=False,
        dismissable=True,
        duration=8000,
        style={
            "position": "fixed",
            "top": 66,
            "right": 10,
            "width": 400,
            "zIndex": 9999,
            "backgroundColor": "#1a1a2e",
            "border": "3px solid #ffd700",
            "boxShadow": "0 0 30px rgba(255, 215, 0, 0.5)",
        },
        header_style={"color": "#ffd700", "fontFamily": "Orbitron", "fontWeight": "bold", "fontSize": "16px"}
    ),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H1("🎰", style={'fontSize': '40px', 'marginRight': '10px'}),
                    html.Div([
                        html.H2("BEYOND PRICE AND TIME", className="neon-green mb-0 title-font",
                               style={'fontSize': '20px', 'fontWeight': '700', 'letterSpacing': '3px'}),
                        html.P("JACKPOT EDITION • MULTI-LEVEL ALIGNMENT DETECTOR", 
                              className="text-warning arcade-font",
                              style={'fontSize': '8px', 'letterSpacing': '1px', 'marginBottom': '0'}),
                    ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
                ], style={'display': 'flex', 'alignItems': 'center'})
            ])
        ], width=5),
        dbc.Col([
            html.Div(id='jackpot-counter', className="text-center")
        ], width=3),
        dbc.Col([
            html.Div(id='connection-status', className="text-end")
        ], width=2),
        dbc.Col([
            html.Div(id='stats-summary', className="text-end", style={'fontSize': '10px'})
        ], width=2)
    ], className="mb-2 mt-2", style={'alignItems': 'center'}),
    
    # Jackpot Display Row
    dbc.Row([
        dbc.Col([
            create_jackpot_display()
        ], width=6),
        dbc.Col([
            create_leaderboard()
        ], width=3),
        dbc.Col([
            create_achievement_panel()
        ], width=3),
    ], className="mb-2"),
    
    # Stats Bar
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div(id='stats-display', className="text-center")
                ], style={'padding': '6px'})
            ], style={'backgroundColor': 'rgba(26, 42, 58, 0.9)', 'border': '1px solid #00ff88'})
        ])
    ], className="mb-2"),
    
    # Filters
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("ALL", id="btn-all", color="secondary", outline=True, size="sm", 
                          className="title-font", style={'letterSpacing': '1px', 'fontSize': '9px', 'padding': '4px 8px'}),
                dbc.Button("🎰 JACKPOT", id="btn-jackpot", color="warning", outline=True, size="sm",
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
                                {'label': '📈 LONG', 'value': 'LONG'},
                                {'label': '📉 SHORT', 'value': 'SHORT'}],
                        value='ALL', clearable=False, style={'fontSize': '10px', 'minWidth': '80px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-merit',
                        options=[{'label': 'MERIT', 'value': 0},
                                {'label': '≥10', 'value': 10},
                                {'label': '≥25', 'value': 25},
                                {'label': '≥50', 'value': 50},
                                {'label': '≥100', 'value': 100}],
                        value=0, clearable=False, style={'fontSize': '10px', 'minWidth': '70px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-alignment',
                        options=[{'label': 'ALIGN', 'value': 0},
                                {'label': '100%', 'value': 100},
                                {'label': '≥75%', 'value': 75},
                                {'label': '≥50%', 'value': 50}],
                        value=0, clearable=False, style={'fontSize': '10px', 'minWidth': '70px'})
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
                        options=[{'label': '🔥 HEAT↓', 'value': 'heat'},
                                {'label': 'MERIT↓', 'value': 'merit'},
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
                    {'name': '🎰', 'id': 'Jackpot', 'sortable': False},
                    {'name': 'T', 'id': 'Type', 'sortable': True},
                    {'name': 'SYM', 'id': 'Symbol', 'sortable': True},
                    {'name': '🔥', 'id': 'Heat', 'sortable': True},
                    {'name': 'MERIT', 'id': 'Merit', 'sortable': True},
                    {'name': 'LVL', 'id': 'Levels', 'sortable': True},
                    {'name': 'ALIGN', 'id': 'Align', 'sortable': True},
                    {'name': 'BAND', 'id': 'Band', 'sortable': True},
                    {'name': 'STS', 'id': 'Stasis', 'sortable': True},
                    {'name': 'DIR', 'id': 'Dir', 'sortable': True},
                    {'name': 'STR', 'id': 'Str', 'sortable': True},
                    {'name': 'CURRENT', 'id': 'Current', 'sortable': True},
                    {'name': 'TP', 'id': 'TP', 'sortable': True},
                    {'name': 'SL', 'id': 'SL', 'sortable': True},
                    {'name': 'R:R', 'id': 'R:R', 'sortable': True},
                    {'name': 'DUR', 'id': 'Duration', 'sortable': True},
                    {'name': 'CHG', 'id': 'Chg', 'sortable': True},
                    {'name': '52W', 'id': '52W', 'sortable': True},
                    {'name': 'BITS', 'id': 'Recent', 'sortable': False},
                ],
                sort_action='native',
                sort_mode='multi',
                sort_by=[{'column_id': 'Merit', 'direction': 'desc'}],
                style_table={'height': '50vh', 'overflowY': 'auto'},
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
                    {'if': {'column_id': 'Jackpot'}, 'textAlign': 'center', 'fontSize': '14px', 'minWidth': '50px'},
                    {'if': {'column_id': 'Type'}, 'textAlign': 'center', 'fontWeight': '600', 'minWidth': '35px'},
                    {'if': {'column_id': 'Symbol'}, 'textAlign': 'left', 'fontWeight': '700', 'color': '#00ff88'},
                    {'if': {'column_id': 'Heat'}, 'textAlign': 'center', 'fontWeight': '700'},
                    {'if': {'column_id': 'Merit'}, 'textAlign': 'center', 'fontWeight': '700'},
                    {'if': {'column_id': 'Levels'}, 'textAlign': 'center', 'fontWeight': '600'},
                    {'if': {'column_id': 'Align'}, 'textAlign': 'center'},
                    {'if': {'column_id': 'Dir'}, 'textAlign': 'center'},
                    {'if': {'column_id': 'Str'}, 'textAlign': 'center'},
                    {'if': {'column_id': 'Recent'}, 'textAlign': 'left', 'minWidth': '90px'},
                ],
                style_header={
                    'backgroundColor': '#2a2a4e', 
                    'color': '#ffd700',
                    'fontWeight': '700', 
                    'fontSize': '9px',
                    'fontFamily': 'Orbitron, sans-serif',
                    'borderBottom': '2px solid #ffd700',
                    'textAlign': 'center',
                    'letterSpacing': '0.5px',
                },
                style_data_conditional=[
                    # Type coloring
                    {'if': {'filter_query': '{Type} = "ETF"', 'column_id': 'Type'}, 'color': '#00ffff'},
                    {'if': {'filter_query': '{Type} = "STK"', 'column_id': 'Type'}, 'color': '#ffaa00'},
                    # Jackpot rows
                    {'if': {'filter_query': '{Is_Jackpot} = true'}, 
                     'backgroundColor': 'rgba(255, 215, 0, 0.15)', 'border': '1px solid #ffd700'},
                    # Heat coloring
                    {'if': {'filter_query': '{Heat} >= 80', 'column_id': 'Heat'}, 'color': '#ff0000', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Heat} >= 60 && {Heat} < 80', 'column_id': 'Heat'}, 'color': '#ff8800'},
                    {'if': {'filter_query': '{Heat} >= 40 && {Heat} < 60', 'column_id': 'Heat'}, 'color': '#ffff00'},
                    {'if': {'filter_query': '{Heat} >= 20 && {Heat} < 40', 'column_id': 'Heat'}, 'color': '#88ff00'},
                    {'if': {'filter_query': '{Heat} < 20', 'column_id': 'Heat'}, 'color': '#00ff88'},
                    # Merit score coloring
                    {'if': {'filter_query': '{Merit_Val} >= 100', 'column_id': 'Merit'}, 
                     'color': '#ff00ff', 'fontWeight': 'bold', 'backgroundColor': '#3d2a4d'},
                    {'if': {'filter_query': '{Merit_Val} >= 50 && {Merit_Val} < 100', 'column_id': 'Merit'}, 
                     'color': '#ffff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Merit_Val} >= 25 && {Merit_Val} < 50', 'column_id': 'Merit'}, 
                     'color': '#00ffff', 'fontWeight': 'bold'},
                    # Level coloring
                    {'if': {'filter_query': '{Levels} >= 5', 'column_id': 'Levels'}, 'color': '#ff00ff', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Levels} >= 3 && {Levels} < 5', 'column_id': 'Levels'}, 'color': '#ffff00'},
                    # Alignment coloring
                    {'if': {'filter_query': '{Align_Val} = 100', 'column_id': 'Align'}, 'color': '#00ff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Align_Val} >= 75 && {Align_Val} < 100', 'column_id': 'Align'}, 'color': '#88ff00'},
                    # High merit row highlighting
                    {'if': {'filter_query': '{Merit_Val} >= 50'}, 'backgroundColor': '#2d3a4d'},
                    # Stasis coloring
                    {'if': {'filter_query': '{Stasis} >= 10'}, 'backgroundColor': '#2d4a2d'},
                    {'if': {'filter_query': '{Stasis} >= 7 && {Stasis} < 10'}, 'backgroundColor': '#2a3a2a'},
                    # Direction coloring
                    {'if': {'filter_query': '{Dir} = "LONG"', 'column_id': 'Dir'}, 'color': '#00ff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Dir} = "SHORT"', 'column_id': 'Dir'}, 'color': '#ff4444', 'fontWeight': 'bold'},
                    # Price columns
                    {'if': {'column_id': 'Current'}, 'color': '#00ffff', 'fontWeight': '600'},
                    {'if': {'column_id': 'TP'}, 'color': '#00ff00'},
                    {'if': {'column_id': 'SL'}, 'color': '#ff4444'},
                    # R:R coloring
                    {'if': {'filter_query': '{RR_Val} >= 2', 'column_id': 'R:R'}, 'color': '#00ff00', 'fontWeight': '600'},
                    # Change coloring
                    {'if': {'filter_query': '{Chg} contains "+"', 'column_id': 'Chg'}, 'color': '#00ff00'},
                    {'if': {'filter_query': '{Chg} contains "-"', 'column_id': 'Chg'}, 'color': '#ff4444'},
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
                html.Span("🎰 JACKPOT", style={'color': '#ffd700', 'fontSize': '9px', 'fontWeight': 'bold'}),
                html.Span(" = ALL LEVELS ALIGNED | ", style={'fontSize': '8px'}),
                html.Span("🔥 HEAT", style={'color': '#ff8800', 'fontSize': '9px'}),
                html.Span(" = Alignment Intensity | ", style={'fontSize': '8px'}),
                html.Span("📈 LONG", style={'color': '#00ff00', 'fontSize': '8px'}),
                html.Span("/", style={'fontSize': '8px'}),
                html.Span("📉 SHORT", style={'color': '#ff4444', 'fontSize': '8px'}),
                html.Span(" | ", style={'fontSize': '8px'}),
                html.Span("🏆 Hit 7-7-7 for MEGA JACKPOT!", style={'color': '#ff00ff', 'fontSize': '8px'}),
            ], className="text-center text-muted mt-1"),
            html.Hr(style={'borderColor': '#333', 'margin': '5px 0'}),
            html.P("© 2026 TRUTH COMMUNICATIONS LLC • JACKPOT EDITION",
                  className="text-muted text-center mb-0 title-font",
                  style={'fontSize': '8px', 'letterSpacing': '1px'}),
        ])
    ]),
    
    dcc.Store(id='view-mode', data='tradable'),
    dcc.Store(id='last-jackpot', data=None),
    dcc.Interval(id='refresh-interval', interval=config.update_interval_ms, n_intervals=0),
    
], fluid=True, className="p-2", style={'backgroundColor': 'transparent'})


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('btn-all', 'active'), Output('btn-jackpot', 'active'), Output('btn-tradable', 'active'), Output('view-mode', 'data')],
    [Input('btn-all', 'n_clicks'), Input('btn-jackpot', 'n_clicks'), Input('btn-tradable', 'n_clicks')],
    [State('view-mode', 'data')]
)
def toggle_view(n1, n2, n3, current):
    ctx = callback_context
    if not ctx.triggered:
        return False, False, True, 'tradable'
    btn = ctx.triggered[0]['prop_id'].split('.')[0]
    if btn == 'btn-all':
        return True, False, False, 'all'
    elif btn == 'btn-jackpot':
        return False, True, False, 'jackpot'
    return False, False, True, 'tradable'


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
    return html.Span(f"🟢 LIVE {status['message_count']:,}", 
                    className="text-success data-font", style={'fontSize': '11px'})


@app.callback(
    Output('jackpot-counter', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_jackpot_counter(n):
    if not manager.backfill_complete:
        return ""
    
    jackpots = manager.get_jackpots()
    
    # Count jackpots by tier
    grand = sum(1 for j in jackpots.values() if j.get('tier') == 'GRAND_JACKPOT')
    mega = sum(1 for j in jackpots.values() if j.get('tier') == 'MEGA_JACKPOT')
    super_jp = sum(1 for j in jackpots.values() if j.get('tier') == 'SUPER_JACKPOT')
    jackpot = sum(1 for j in jackpots.values() if j.get('tier') == 'JACKPOT')
    big_win = sum(1 for j in jackpots.values() if j.get('tier') == 'BIG_WIN')
    
    total_jackpots = grand + mega + super_jp + jackpot + big_win
    
    return html.Div([
        html.Span("🎰 ACTIVE JACKPOTS: ", className="title-font", style={'fontSize': '10px', 'color': '#ffd700'}),
        html.Span(f"{total_jackpots}", className="neon-gold", 
                 style={'fontSize': '18px', 'fontWeight': 'bold', 'fontFamily': 'Orbitron'}),
        html.Div([
            html.Span(f"💎{grand} ", style={'color': '#ff00ff', 'fontSize': '9px'}) if grand > 0 else None,
            html.Span(f"🎰{mega} ", style={'color': '#ffff00', 'fontSize': '9px'}) if mega > 0 else None,
            html.Span(f"💰{super_jp} ", style={'color': '#00ffff', 'fontSize': '9px'}) if super_jp > 0 else None,
            html.Span(f"🍀{jackpot} ", style={'color': '#00ff88', 'fontSize': '9px'}) if jackpot > 0 else None,
            html.Span(f"⭐{big_win}", style={'color': '#88ff88', 'fontSize': '9px'}) if big_win > 0 else None,
        ], style={'marginTop': '2px'})
    ], className="text-center")


@app.callback(
    Output('current-jackpots', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_current_jackpots(n):
    if not manager.backfill_complete:
        return html.Span("Scanning...", className="text-muted")
    
    jackpots = manager.get_jackpots()
    merit_scores = manager.get_merit_scores()
    
    # Get top 3 jackpots by heat level
    hot_jackpots = [(sym, j, merit_scores.get(sym, {})) 
                   for sym, j in jackpots.items() 
                   if j.get('is_jackpot') or j.get('heat_level', 0) >= 50]
    
    hot_jackpots.sort(key=lambda x: x[1].get('heat_level', 0), reverse=True)
    hot_jackpots = hot_jackpots[:3]
    
    if not hot_jackpots:
        return html.Div([
            html.Div([
                html.Span("⬜", className="slot-reel"),
                html.Span("⬜", className="slot-reel"),
                html.Span("⬜", className="slot-reel"),
            ], className="slot-container"),
            html.P("Scanning for alignment patterns...", className="text-muted mt-2", style={'fontSize': '10px'}),
        ])
    
    cards = []
    for symbol, jackpot, merit in hot_jackpots:
        tier = jackpot.get('tier')
        tier_name = jackpot.get('tier_name', 'NO SIGNAL')
        emoji = jackpot.get('emoji', '⬜')
        color = jackpot.get('color', '#333')
        heat = jackpot.get('heat_level', 0)
        slot_display = jackpot.get('slot_display', ['⬜', '⬜', '⬜'])
        achievements = jackpot.get('achievements', [])
        direction = merit.get('dominant_direction', '—')
        levels = merit.get('stasis_levels', 0)
        alignment = merit.get('direction_alignment', 0)
        
        # Card class based on tier
        card_class = ""
        if tier == 'GRAND_JACKPOT':
            card_class = "grand-jackpot-card"
        elif tier == 'MEGA_JACKPOT':
            card_class = "mega-jackpot-card"
        elif tier and 'JACKPOT' in tier:
            card_class = "jackpot-card"
        
        direction_icon = "📈" if direction == "LONG" else "📉" if direction == "SHORT" else "➖"
        direction_color = "#00ff00" if direction == "LONG" else "#ff4444" if direction == "SHORT" else "#888"
        
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    # Symbol and slot display
                    html.Div([
                        html.Span(symbol, style={'color': color, 'fontWeight': 'bold', 'fontSize': '14px'}),
                        html.Span(f" {emoji}", style={'fontSize': '16px'}),
                    ], className="mb-1"),
                    
                    # Slot machine reels
                    html.Div([
                        html.Span(s, className="slot-reel") for s in slot_display
                    ], className="slot-container mb-1"),
                    
                    # Tier name
                    html.Div(tier_name, style={'color': color, 'fontSize': '9px', 'fontWeight': 'bold'}),
                    
                    # Stats
                    html.Div([
                        html.Span(f"{direction_icon} {direction}", style={'color': direction_color, 'fontSize': '10px'}),
                        html.Span(f" | {levels}LVL | {alignment:.0f}%", style={'color': '#888', 'fontSize': '9px'}),
                    ], className="mt-1"),
                    
                    # Heat bar
                    html.Div([
                        html.Div(style={
                            'width': f'{heat}%',
                            'height': '4px',
                            'backgroundColor': get_heat_color(heat),
                            'borderRadius': '2px',
                            'transition': 'width 0.3s ease',
                        })
                    ], style={
                        'width': '100%',
                        'height': '4px',
                        'backgroundColor': '#1a1a2e',
                        'borderRadius': '2px',
                        'marginTop': '4px',
                    }),
                    
                    # Achievements
                    html.Div([
                        html.Span(
                            f"{ACHIEVEMENTS[a]['emoji']} {a.replace('_', ' ')}", 
                            className=f"achievement-badge achievement-{ACHIEVEMENTS[a]['rarity'].lower()}"
                        ) for a in achievements[:2]
                    ], className="mt-1") if achievements else None,
                    
                ], style={'padding': '8px', 'textAlign': 'center'})
            ], className=f"m-1 hover-glow {card_class}", style={
                'display': 'inline-block',
                'minWidth': '150px',
                'backgroundColor': 'rgba(26, 26, 46, 0.9)',
                'border': f'2px solid {color}',
                'borderRadius': '8px',
            })
        )
    
    return html.Div(cards, style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap'})


@app.callback(
    Output('leaderboard-display', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_leaderboard(n):
    if not manager.backfill_complete:
        return html.Span("Loading...", className="text-muted")
    
    merit_scores = manager.get_merit_scores()
    jackpots = manager.get_jackpots()
    
    # Sort by merit score
    sorted_symbols = sorted(merit_scores.items(), key=lambda x: x[1].get('weighted_score', 0), reverse=True)[:10]
    
    items = []
    for rank, (symbol, merit) in enumerate(sorted_symbols, 1):
        score = merit.get('weighted_score', 0)
        if score == 0:
            continue
            
        levels = merit.get('stasis_levels', 0)
        direction = merit.get('dominant_direction', '—')
        alignment = merit.get('direction_alignment', 0)
        jackpot = jackpots.get(symbol, {})
        emoji = jackpot.get('emoji', '')
        heat = jackpot.get('heat_level', 0)
        
        direction_icon = "📈" if direction == "LONG" else "📉" if direction == "SHORT" else ""
        direction_color = "#00ff00" if direction == "LONG" else "#ff4444" if direction == "SHORT" else "#888"
        
        rank_style = {
            1: {'color': '#ffd700', 'fontWeight': 'bold'},
            2: {'color': '#c0c0c0', 'fontWeight': 'bold'},
            3: {'color': '#cd7f32', 'fontWeight': 'bold'},
        }.get(rank, {'color': '#888'})
        
        items.append(
            html.Div([
                html.Span(f"#{rank}", className="leaderboard-rank me-2", style=rank_style),
                html.Span(f"{symbol}", style={'color': '#00ff88', 'fontWeight': 'bold', 'fontSize': '11px'}),
                html.Span(f" {emoji}", style={'fontSize': '12px'}),
                html.Span(f" {direction_icon}", style={'color': direction_color}),
                html.Div([
                    html.Span(f"Score: {score:.0f}", style={'color': '#ffd700', 'fontSize': '9px'}),
                    html.Span(f" | {levels}LVL", style={'color': '#888', 'fontSize': '9px'}),
                    html.Span(f" | {alignment:.0f}%", style={'color': '#00ff88' if alignment == 100 else '#888', 'fontSize': '9px'}),
                ], style={'marginTop': '2px'}),
            ], className="leaderboard-item", style={
                'borderLeftColor': get_heat_color(heat),
            })
        )
    
    if not items:
        return html.Span("No signals yet...", className="text-muted", style={'fontSize': '10px'})
    
    return html.Div(items)


@app.callback(
    Output('achievements-display', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_achievements(n):
    if not manager.backfill_complete:
        return html.Span("Loading...", className="text-muted")
    
    jackpots = manager.get_jackpots()
    
    # Collect all achievements
    all_achievements = []
    for symbol, jackpot in jackpots.items():
        for achievement in jackpot.get('achievements', []):
            all_achievements.append({
                'symbol': symbol,
                'achievement': achievement,
                'info': ACHIEVEMENTS.get(achievement, {}),
            })
    
    if not all_achievements:
        return html.Div([
            html.P("🏅 No achievements unlocked yet", className="text-muted text-center", style={'fontSize': '10px'}),
            html.P("Keep watching for perfect alignments!", className="text-muted text-center", style={'fontSize': '9px'}),
        ])
    
    # Group by rarity
    by_rarity = defaultdict(list)
    for a in all_achievements:
        rarity = a['info'].get('rarity', 'COMMON')
        by_rarity[rarity].append(a)
    
    sections = []
    for rarity in ['LEGENDARY', 'EPIC', 'RARE', 'UNCOMMON', 'COMMON']:
        if rarity in by_rarity:
            items = by_rarity[rarity]
            sections.append(
                html.Div([
                    html.Div([
                        html.Span(
                            f"{a['info'].get('emoji', '')} {a['symbol']}", 
                            className=f"achievement-badge achievement-{rarity.lower()} me-1"
                        ) for a in items[:5]
                    ])
                ], className="mb-1")
            )
    
    return html.Div(sections)


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
        html.Span(f"SIG:{tradable}", style={'color': '#00ff88', 'fontWeight': 'bold'}),
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
    jackpots = manager.get_jackpots()
    
    if not data:
        return html.Span("LOADING...", className="text-muted title-font")
    
    tradable = [d for d in data if d['is_tradable']]
    long_count = sum(1 for d in tradable if d['direction'] == 'LONG')
    short_count = sum(1 for d in tradable if d['direction'] == 'SHORT')
    max_stasis = max([d['stasis'] for d in data]) if data else 0
    
    top_merit = max(merit_scores.values(), key=lambda x: x['weighted_score'], default={'weighted_score': 0})
    top_merit_symbol = [k for k, v in merit_scores.items() if v['weighted_score'] == top_merit['weighted_score']]
    top_merit_symbol = top_merit_symbol[0] if top_merit_symbol else "—"
    
    multi_level = sum(1 for m in merit_scores.values() if m['stasis_levels'] >= 2)
    perfect_align = sum(1 for m in merit_scores.values() if m['direction_alignment'] == 100 and m['stasis_levels'] >= 2)
    
    # Get hottest symbol
    hottest = max(jackpots.items(), key=lambda x: x[1].get('heat_level', 0), default=(None, {'heat_level': 0}))
    hottest_symbol = hottest[0] if hottest[0] else "—"
    hottest_heat = hottest[1].get('heat_level', 0)
    
    return html.Div([
        html.Span("🎯 SIGNALS: ", className="title-font", style={'fontSize': '10px'}),
        html.Span(f"{len(tradable)}", className="data-font text-success stat-value", style={'fontSize': '11px', 'fontWeight': '600'}),
        html.Span("  🏆 TOP: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{top_merit_symbol}({top_merit['weighted_score']:.0f})", 
                 className="data-font text-warning stat-value", style={'fontSize': '11px', 'fontWeight': '600'}),
        html.Span("  🔥 HOT: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{hottest_symbol}({hottest_heat})", 
                 className="data-font stat-value", style={'fontSize': '11px', 'color': get_heat_color(hottest_heat)}),
        html.Span("  ✅ ALIGNED: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{perfect_align}", className="data-font text-info stat-value", style={'fontSize': '11px'}),
        html.Span("  📈 L: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{long_count}", className="data-font text-success stat-value", style={'fontSize': '11px'}),
        html.Span("  📉 S: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{short_count}", className="data-font text-danger stat-value", style={'fontSize': '11px'}),
        html.Span("  ⚡ MAX: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{max_stasis}", className="data-font text-warning stat-value", style={'fontSize': '11px'}),
    ])


@app.callback(
    [Output('jackpot-toast', 'is_open'),
     Output('jackpot-toast', 'children'),
     Output('last-jackpot', 'data')],
    [Input('refresh-interval', 'n_intervals')],
    [State('last-jackpot', 'data')]
)
def show_jackpot_notification(n, last_jackpot):
    if not manager.backfill_complete:
        return False, "", last_jackpot
    
    recent_jackpots = manager.get_recent_jackpots()
    
    if not recent_jackpots:
        return False, "", last_jackpot
    
    # Get the most recent jackpot
    latest = recent_jackpots[-1]
    
    # Check if this is a new jackpot
    if last_jackpot and latest['symbol'] == last_jackpot.get('symbol') and latest['tier'] == last_jackpot.get('tier'):
        return False, "", last_jackpot
    
    tier_info = JACKPOT_TIERS.get(latest['tier'], {})
    
    content = html.Div([
        html.Div([
            html.Span(tier_info.get('emoji', '🎰'), style={'fontSize': '30px'}),
        ], className="text-center mb-2"),
        html.Div([
            html.Span(latest['symbol'], style={'color': '#00ff88', 'fontWeight': 'bold', 'fontSize': '20px'}),
            html.Span(f" hit ", style={'color': 'white'}),
            html.Span(latest['tier'].replace('_', ' '), 
                     style={'color': tier_info.get('color', '#ffd700'), 'fontWeight': 'bold', 'fontSize': '16px'}),
        ], className="text-center"),
        html.Div([
            html.Span(f"{latest['levels']} levels aligned ", style={'color': '#ffff00', 'fontSize': '12px'}),
            html.Span(f"{'📈 LONG' if latest['direction'] == 'LONG' else '📉 SHORT'}", 
                     style={'color': '#00ff00' if latest['direction'] == 'LONG' else '#ff4444', 'fontSize': '12px'}),
        ], className="text-center mt-2"),
        html.Div([
            html.Span(f"Multiplier: {tier_info.get('multiplier', '?x')}", 
                     style={'color': '#ff00ff', 'fontSize': '14px', 'fontWeight': 'bold'}),
        ], className="text-center mt-2"),
    ], className="data-font")
    
    return True, content, {'symbol': latest['symbol'], 'tier': latest['tier']}


@app.callback(
    Output('main-table', 'data'),
    [Input('refresh-interval', 'n_intervals'),
     Input('view-mode', 'data'),
     Input('filter-type', 'value'),
     Input('filter-symbol', 'value'),
     Input('filter-threshold', 'value'),
     Input('filter-stasis', 'value'),
     Input('filter-direction', 'value'),
     Input('filter-merit', 'value'),
     Input('filter-alignment', 'value'),
     Input('filter-rows', 'value'),
     Input('filter-sort', 'value')]
)
def update_table(n, view_mode, type_filter, sym, thresh, stasis, direction, merit, alignment, rows, sort):
    df = get_table_data()
    if df.empty:
        return []
    
    if view_mode == 'tradable':
        df = df[df['Is_Tradable'] == True]
    elif view_mode == 'jackpot':
        df = df[df['Is_Jackpot'] == True]
    
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
    if merit is not None and merit > 0:
        df = df[df['Merit_Val'] >= merit]
    if alignment is not None and alignment > 0:
        df = df[df['Align_Val'] >= alignment]
    
    if sort == 'heat':
        df = df.sort_values(['Heat', 'Merit_Val'], ascending=[False, False])
    elif sort == 'merit':
        df = df.sort_values(['Merit_Val', 'Stasis'], ascending=[False, False])
    elif sort == 'stasis':
        df = df.sort_values(['Stasis', 'Merit_Val'], ascending=[False, False])
    elif sort == 'rr':
        df = df.sort_values(['RR_Val', 'Merit_Val'], ascending=[False, False])
    
    df = df.head(rows)
    
    drop_cols = ['Band_Val', 'Current_Val', 'Anchor', 'RR_Val', 'Dur_Val', 'Chg_Val', '52W_Val', 
                 'Is_Tradable', 'Merit_Val', 'Bits', 'Is_ETF', 'Is_Jackpot', 'Align_Val', 'Jackpot_Tier',
                 '→TP', '→SL']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    return df.to_dict('records')


# ============================================================================
# INITIALIZATION
# ============================================================================

_initialized = False
_init_lock = threading.Lock()


def initialize_app():
    """
    Initialize the application with parallel data fetching where possible.
    This runs in a background thread to allow the server to start quickly.
    """
    global _initialized
    
    with _init_lock:
        if _initialized:
            return
        
        print("=" * 70)
        print("  🎰 BEYOND PRICE AND TIME - JACKPOT EDITION 🎰")
        print("  Finding the 7-7-7 alignment across all price levels")
        print("  © 2026 Truth Communications LLC. All Rights Reserved.")
        print("=" * 70)
        
        print(f"\n📊 Total Symbols: {len(config.symbols)}")
        print(f"   ETFs: {len(config.etf_symbols)}")
        print(f"   Stocks: {len(config.symbols) - len(config.etf_symbols)}")
        
        print(f"\n📏 Threshold Levels ({len(config.thresholds)}):")
        for t in config.thresholds:
            print(f"   • {t*100:.4f}%")
        
        print(f"\n🎰 Jackpot Tiers:")
        for tier_name, tier_info in JACKPOT_TIERS.items():
            print(f"   {tier_info['emoji']} {tier_name}: {tier_info['min_levels']}+ levels @ {tier_info['min_alignment']}% alignment = {tier_info['multiplier']}")
        
        # ================================================================
        # PARALLEL DATA FETCHING
        # Fetch 52-week data and volume data simultaneously
        # ================================================================
        print("\n" + "=" * 70)
        print("📡 FETCHING MARKET DATA (PARALLEL)")
        print("=" * 70)
        
        week52_result = {}
        volume_result = {}
        fetch_errors = []
        
        def fetch_52week_thread():
            nonlocal week52_result
            try:
                print("   🔄 Starting 52-week data fetch...")
                week52_result = fetch_52_week_data()
                print(f"   ✅ 52-week data complete: {len(week52_result)} symbols")
            except Exception as e:
                fetch_errors.append(f"52-week fetch error: {e}")
                print(f"   ❌ 52-week data error: {e}")
        
        def fetch_volume_thread():
            nonlocal volume_result
            try:
                print("   🔄 Starting volume data fetch...")
                volume_result = fetch_volume_data()
                print(f"   ✅ Volume data complete: {len(volume_result)} symbols")
            except Exception as e:
                fetch_errors.append(f"Volume fetch error: {e}")
                print(f"   ❌ Volume data error: {e}")
        
        # Start parallel fetches
        t1 = threading.Thread(target=fetch_52week_thread, daemon=True)
        t2 = threading.Thread(target=fetch_volume_thread, daemon=True)
        
        t1.start()
        t2.start()
        
        # Wait for both to complete
        t1.join()
        t2.join()
        
        # Store results in config
        config.week52_data = week52_result
        config.volumes = volume_result
        
        if fetch_errors:
            print(f"\n⚠️ Fetch warnings: {len(fetch_errors)}")
            for err in fetch_errors:
                print(f"   - {err}")
        
        # ================================================================
        # BACKFILL HISTORICAL DATA
        # ================================================================
        print("\n" + "=" * 70)
        print("📜 BACKFILLING HISTORICAL DATA")
        print("=" * 70)
        
        manager.backfill()
        
        # ================================================================
        # START REAL-TIME SERVICES
        # ================================================================
        print("\n" + "=" * 70)
        print("🚀 STARTING REAL-TIME SERVICES")
        print("=" * 70)
        
        # Load saved alerts
        alert_stats = alert_manager.get_stats()
        print(f"🔔 Loaded Alerts: {alert_stats['active']} active, {alert_stats['triggered']} triggered")
        
        # Start price feed
        price_feed.start()
        
        # Start bitstream manager (includes alert checking)
        manager.start()
        
        print("\n" + "=" * 70)
        print("✅ INITIALIZATION COMPLETE")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"   • Symbols monitored: {len(config.symbols)}")
        print(f"   • Threshold levels: {len(config.thresholds)}")
        print(f"   • Total bitstreams: {len(config.symbols) * len(config.thresholds)}")
        print(f"   • 52-week data: {len(config.week52_data)} symbols")
        print(f"   • Volume data: {len(config.volumes)} symbols")
        print(f"   • Active alerts: {alert_stats['active']}")
        
        print("\n🎰 JACKPOT EDITION FEATURES:")
        print("   • Multi-level alignment detection (like 7-7-7 slots)")
        print("   • Heat map showing signal intensity")
        print("   • Leaderboard ranking top opportunities")
        print("   • Achievement badges for special patterns")
        print("   • Jackpot notifications with celebrations")
        print("   • Price alerts with real-time monitoring")
        print("=" * 70 + "\n")
        
        _initialized = True


# Start initialization in background thread
_init_thread = threading.Thread(target=initialize_app, daemon=True)
_init_thread.start()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import os
    
    # Wait for initialization to complete
    _init_thread.join()
    
    print("\n✅ Server starting: http://127.0.0.1:8050")
    print("🎰 Ready to find jackpots!\n")
    
    # Auto-open browser after short delay
    threading.Thread(
        target=lambda: (time.sleep(2), webbrowser.open('http://127.0.0.1:8050')), 
        daemon=True
    ).start()
    
    # Get port from environment (for cloud deployment) or default to 8050
    port = int(os.environ.get('PORT', 8050))
    
    # Run the Dash server
    app.run(debug=False, host='0.0.0.0', port=port)

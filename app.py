# -*- coding: utf-8 -*-
"""
BEYOND PRICE AND TIME — STASIS PM | STASIS AM
Copyright © 2026 Truth Communications LLC. All Rights Reserved.

SPLIT-SCREEN DASHBOARD:
  LEFT  (Dark)  — STASIS PM: Prediction Markets (patterns + probabilities)
  RIGHT (Light) — STASIS AM: Alpha Markets (stasis at price floors + fundamental merit)

Requirements:
    pip install dash dash-bootstrap-components pandas numpy websocket-client requests
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
import uuid

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, no_update, dash_table
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
# CONSTANTS
# ============================================================================

BET_SIZES = [25, 50, 100, 250, 500, 1000]
STARTING_BALANCE = 10000.00

JACKPOT_TIERS = {
    'GRAND_JACKPOT': {'min_levels': 8, 'min_alignment': 100, 'emoji': '💎', 'color': '#ff00ff'},
    'MEGA_JACKPOT':  {'min_levels': 6, 'min_alignment': 100, 'emoji': '🎰', 'color': '#ffff00'},
    'SUPER_JACKPOT': {'min_levels': 5, 'min_alignment': 100, 'emoji': '💰', 'color': '#00ffff'},
    'JACKPOT':       {'min_levels': 4, 'min_alignment': 100, 'emoji': '🍀', 'color': '#00ff88'},
    'BIG_WIN':       {'min_levels': 3, 'min_alignment': 100, 'emoji': '⭐', 'color': '#88ff88'},
}

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    symbols: List[str] = field(default_factory=lambda: [
        # ETFs
        "SPY","QQQ","IWM","DIA","XLF","XLE","XLU","XLK","XLP","XLB",
        "XLV","XLI","XLY","XLC","XLRE","KRE","SMH","XBI","GDX",
        # Technology
        'AAPL','MSFT','GOOGL','GOOG','AMZN','NVDA','META','TSLA','AVGO','ORCL',
        'ADBE','CRM','AMD','INTC','CSCO','QCOM','IBM','NOW','INTU','AMAT',
        'MU','LRCX','ADI','KLAC','SNPS','CDNS','MRVL','FTNT','PANW','CRWD',
        'ZS','DDOG','SNOW','PLTR','NET','MDB','TEAM','WDAY','OKTA','HUBS',
        'ZM','DOCU','SQ','PYPL','SHOP','MELI','SE','UBER','LYFT','DASH',
        'TXN','NXPI','MPWR','ON','SWKS','QRVO','MCHP',
        'ANET','CIEN','SMCI',
        # Financials
        'JPM','BAC','WFC','C','GS','MS','USB','PNC','TFC','SCHW',
        'BK','STT','NTRS','FITB','MTB','CFG','RF','HBAN','KEY','ZION',
        'V','MA','AXP','DFS','COF','SYF','ALLY',
        'SOFI','AFRM','UPST','LC','HOOD','COIN',
        'BRK.B','CB','PGR','ALL','MET','PRU','AIG','AFL','TRV','HIG',
        'BLK','SPGI','MCO','CME','ICE','NDAQ',
        'BX','KKR','APO','CG','ARES',
        # Healthcare
        'JNJ','PFE','ABBV','MRK','LLY','BMY','AMGN','GILD','BIIB','REGN',
        'VRTX','MRNA','BNTX','ZTS',
        'ABT','MDT','DHR','TMO','SYK','BSX','EW','ISRG','BDX','ZBH',
        'ALGN','IDXX','RMD','DXCM','PODD',
        'UNH','ELV','CVS','CI','HUM','CNC','MOH','HCA',
        'MCK','CAH',
        # Consumer Discretionary
        'HD','LOW','TJX','ROST','NKE','LULU','CROX','DECK',
        'WMT','TGT','COST','BJ','DG','DLTR','FIVE','OLLI','BBY',
        'MCD','SBUX','YUM','CMG','DPZ','DRI',
        'MAR','HLT','ABNB','BKNG','EXPE',
        'RCL','CCL','NCLH','LVS','MGM','WYNN','CZR','DKNG',
        'F','GM','RIVN','LCID','NIO',
        'DIS','NFLX','WBD','PARA',
        'TTWO','EA','RBLX',
        # Consumer Staples
        'PEP','KO','MNST','KDP','STZ',
        'GIS','K','CPB','SJM','CAG','HRL','TSN',
        'MDLZ','HSY',
        'PG','CL','KMB','CHD','CLX',
        'PM','MO','BTI',
        'KR','WBA','SYY',
        # Industrials
        'BA','LMT','RTX','NOC','GD','LHX','TDG','AXON',
        'CAT','DE','PCAR','CMI',
        'IR','ITW','ETN','PH','ROK','AME','ROP','DOV',
        'UNP','CSX','NSC','JBHT','ODFL',
        'FDX','UPS',
        'DAL','UAL','AAL','LUV',
        'SHW','PPG',
        'GE','HON','MMM',
        'WM','RSG',
        'PWR','EME',
        # Energy
        'XOM','CVX','COP','EOG','SLB','MPC','PSX','VLO','OXY','HES',
        'PXD','DVN','HAL','BKR','FANG',
        'KMI','WMB','OKE',
        'NEE','ENPH','SEDG','FSLR','RUN',
        # Materials
        'LIN','APD','ECL','DD','DOW','LYB','NUE','STLD',
        'FCX','NEM','GOLD',
        'ALB','FMC','CF','MOS',
        # Utilities
        'DUK','SO','D','AEP','SRE','EXC','XEL','WEC','ED',
        'AWK',
        # REITs
        'PLD','EQR','AVB','SPG','O','AMT','CCI','EQIX','DLR',
        'PSA','EXR','WELL','VTR',
        # Telecom
        'T','VZ','TMUS','CHTR','CMCSA',
        # International ADRs
        'BABA','JD','PDD','BIDU','NIO','ASML','NVO','SAP','TSM',
    ])

    etf_symbols: List[str] = field(default_factory=lambda: [
        "SPY","QQQ","IWM","DIA","XLF","XLE","XLU","XLK","XLP","XLB",
        "XLV","XLI","XLY","XLC","XLRE","KRE","SMH","XBI","GDX",
    ])

    thresholds: List[float] = field(default_factory=lambda: [
        0.000625, 0.00125, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015,
        0.02, 0.025, 0.03, 0.04, 0.05, 0.10
    ])

    # PM-only thresholds (the 10 used for prediction-market merit weighting)
    pm_thresholds: List[float] = field(default_factory=lambda: [
        0.000625, 0.00125, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10
    ])

    # AM-only thresholds (the 10 used for alpha-market stasis scoring)
    am_thresholds: List[float] = field(default_factory=lambda: [
        0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05
    ])

    display_bits: int = 20
    update_interval_ms: int = 1000
    cache_refresh_interval: float = 0.5
    history_days: int = 5

    polygon_api_key: str = POLYGON_API_KEY
    polygon_ws_url: str = "wss://delayed.polygon.io/stocks"
    polygon_rest_url: str = "https://api.polygon.io"

    volumes: Dict[str, float] = field(default_factory=dict)
    week52_data: Dict[str, Dict] = field(default_factory=dict)
    fundamental_data: Dict[str, Dict] = field(default_factory=dict)
    fundamental_slopes: Dict[str, Dict] = field(default_factory=dict)

    min_tradable_stasis: int = 3


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
# DATA CLASSES
# ============================================================================

@dataclass
class BitEntry:
    bit: int
    price: float
    timestamp: datetime

@dataclass
class StasisInfo:
    start_time: datetime
    start_price: float
    peak_stasis: int = 1

    def get_duration(self) -> timedelta:
        return datetime.now() - self.start_time

    def get_duration_str(self) -> str:
        d = self.get_duration()
        t = int(d.total_seconds())
        if t < 60:
            return f"{t}s"
        elif t < 3600:
            return f"{t//60}m {t%60}s"
        return f"{t//3600}h {(t%3600)//60}m"

    def get_start_date_str(self) -> str:
        return self.start_time.strftime("%m/%d %H:%M")

    def get_price_change_pct(self, current_price: float) -> float:
        if self.start_price == 0:
            return 0.0
        return (current_price - self.start_price) / self.start_price * 100

# ============================================================================
# PORTFOLIO (PM side)
# ============================================================================

class Portfolio:
    def __init__(self, starting_balance: float = STARTING_BALANCE):
        self.balance = starting_balance
        self.starting_balance = starting_balance
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.total_realized_pnl = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self._lock = threading.Lock()

    def place_bet(self, symbol, side, amount, buy_price, sell_price, direction, stock_price):
        with self._lock:
            if amount <= 0:
                return {'success': False, 'error': 'Invalid amount'}
            if amount > self.balance:
                return {'success': False, 'error': f'Insufficient (${self.balance:.2f})'}
            entry_price = buy_price if side == 'YES' else sell_price
            if entry_price <= 0:
                return {'success': False, 'error': 'Invalid price'}
            shares = amount / entry_price
            pid = f"{symbol}_{side}_{uuid.uuid4().hex[:6]}"
            self.positions[pid] = {
                'id': pid, 'symbol': symbol, 'side': side, 'direction': direction,
                'shares': shares, 'entry_price': entry_price, 'cost_basis': amount,
                'stock_price_at_entry': stock_price, 'entry_time': datetime.now(),
            }
            self.balance -= amount
            trade = {
                'id': uuid.uuid4().hex[:8], 'position_id': pid, 'symbol': symbol,
                'side': side, 'direction': direction, 'action': 'BUY',
                'amount': amount, 'shares': shares, 'price': entry_price,
                'timestamp': datetime.now(),
            }
            self.trade_history.append(trade)
            return {'success': True, 'trade': trade, 'position_id': pid, 'new_balance': self.balance}

    def close_position(self, position_id, current_buy, current_sell):
        with self._lock:
            if position_id not in self.positions:
                return {'success': False, 'error': 'Not found'}
            pos = self.positions[position_id]
            exit_price = current_sell if pos['side'] == 'YES' else (1.0 - current_buy)
            proceeds = pos['shares'] * exit_price
            pnl = proceeds - pos['cost_basis']
            pnl_pct = (pnl / pos['cost_basis']) * 100 if pos['cost_basis'] > 0 else 0
            self.balance += proceeds
            self.total_realized_pnl += pnl
            if pnl >= 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            trade = {
                'id': uuid.uuid4().hex[:8], 'position_id': position_id,
                'symbol': pos['symbol'], 'side': pos['side'], 'direction': pos['direction'],
                'action': 'CLOSE', 'shares': pos['shares'], 'entry_price': pos['entry_price'],
                'exit_price': exit_price, 'pnl': pnl, 'pnl_pct': pnl_pct,
                'timestamp': datetime.now(),
            }
            self.trade_history.append(trade)
            del self.positions[position_id]
            return {'success': True, 'trade': trade, 'pnl': pnl, 'pnl_pct': pnl_pct, 'new_balance': self.balance}

    def get_stats(self, market_prices=None):
        with self._lock:
            unrealized = 0.0
            positions_value = 0.0
            if market_prices:
                for pid, pos in self.positions.items():
                    sym = pos['symbol']
                    if sym in market_prices:
                        mp = market_prices[sym]
                        if pos['side'] == 'YES':
                            val = pos['shares'] * mp.get('sell_price', pos['entry_price'])
                        else:
                            val = pos['shares'] * (1.0 - mp.get('buy_price', 1.0 - pos['entry_price']))
                        positions_value += val
                        unrealized += val - pos['cost_basis']
                    else:
                        positions_value += pos['cost_basis']
            total_trades = self.winning_trades + self.losing_trades
            return {
                'balance': self.balance,
                'positions_value': positions_value,
                'portfolio_value': self.balance + positions_value,
                'unrealized_pnl': unrealized,
                'realized_pnl': self.total_realized_pnl,
                'total_pnl': self.total_realized_pnl + unrealized,
                'positions_count': len(self.positions),
                'total_trades': total_trades,
                'win_rate': (self.winning_trades / total_trades * 100) if total_trades > 0 else 0,
            }

    def get_positions_list(self):
        with self._lock:
            return list(self.positions.values())

    def get_recent_trades(self, n=10):
        with self._lock:
            return list(reversed(self.trade_history[-n:]))

    def reset(self):
        with self._lock:
            self.balance = self.starting_balance
            self.positions.clear()
            self.trade_history.clear()
            self.total_realized_pnl = 0.0
            self.winning_trades = 0
            self.losing_trades = 0

portfolio = Portfolio()

# ============================================================================
# BITSTREAM
# ============================================================================

class Bitstream:
    def __init__(self, symbol, threshold, initial_price, volume):
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

    def process_price(self, price, timestamp):
        with self._lock:
            self.current_live_price = price
            self.last_price_update = timestamp
            if self.lower_band < price < self.upper_band:
                return
            if self.band_width <= 0:
                return
            x = int((price - self.reference_price) / self.band_width)
            if x > 0:
                for _ in range(x):
                    self.bits.append(BitEntry(1, price, timestamp))
                    self.total_bits += 1
                self.reference_price = price
                self._update_bands()
            elif x < 0:
                for _ in range(abs(x)):
                    self.bits.append(BitEntry(0, price, timestamp))
                    self.total_bits += 1
                self.reference_price = price
                self._update_bands()
            self._update_stasis(timestamp)

    def _update_stasis(self, timestamp):
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
            if bits_list[i].bit != bits_list[i - 1].bit:
                stasis_count += 1
                stasis_start_idx = i - 1
            else:
                break
        prev_stasis = self.current_stasis
        self.current_stasis = stasis_count
        self.last_bit = bits_list[-1].bit
        if prev_stasis < 2 and stasis_count >= 2:
            if 0 <= stasis_start_idx < len(bits_list):
                fb = bits_list[stasis_start_idx]
                self.stasis_info = StasisInfo(start_time=fb.timestamp, start_price=fb.price, peak_stasis=stasis_count)
        elif stasis_count >= 2 and self.stasis_info is not None:
            if stasis_count > self.stasis_info.peak_stasis:
                self.stasis_info.peak_stasis = stasis_count
        elif prev_stasis >= 2 and stasis_count < 2:
            self.stasis_info = None
        if self.current_stasis >= 2:
            self.direction = Direction.LONG if self.last_bit == 0 else Direction.SHORT
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

    def is_tradable(self):
        with self._lock:
            return (self.current_stasis >= config.min_tradable_stasis
                    and self.direction is not None
                    and self.volume > 1.0)

    def get_snapshot(self, live_price=None):
        with self._lock:
            price = live_price if live_price is not None else self.current_live_price
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
                stasis_price_change_pct = self.stasis_info.get_price_change_pct(price)
            take_profit = stop_loss = risk_reward = None
            distance_to_tp_pct = distance_to_sl_pct = None
            if self.direction is not None and self.current_stasis >= 2:
                if self.direction == Direction.LONG:
                    take_profit = self.upper_band
                    stop_loss = self.lower_band
                    reward = take_profit - price
                    risk = price - stop_loss
                else:
                    take_profit = self.lower_band
                    stop_loss = self.upper_band
                    reward = price - take_profit
                    risk = stop_loss - price
                if risk > 0 and reward > 0:
                    risk_reward = reward / risk
                elif risk > 0:
                    risk_reward = 0.0
                if price > 0:
                    distance_to_tp_pct = (abs(take_profit - price) / price) * 100
                    distance_to_sl_pct = (abs(stop_loss - price) / price) * 100
            week52_percentile = calculate_52week_percentile(price, self.symbol)
            return {
                'symbol': self.symbol,
                'is_etf': self.is_etf,
                'threshold': self.threshold,
                'threshold_pct': self.threshold * 100,
                'stasis': self.current_stasis,
                'total_bits': self.total_bits,
                'current_price': price,
                'anchor_price': anchor_price,
                'direction': self.direction.value if self.direction else None,
                'signal_strength': self.signal_strength.value if self.signal_strength else None,
                'is_tradable': (self.current_stasis >= config.min_tradable_stasis
                                and self.direction is not None and self.volume > 1.0),
                'stasis_start_str': stasis_start_str,
                'stasis_duration_str': stasis_duration_str,
                'duration_seconds': duration_seconds,
                'stasis_price_change_pct': stasis_price_change_pct,
                'take_profit': take_profit, 'stop_loss': stop_loss,
                'risk_reward': risk_reward,
                'distance_to_tp_pct': distance_to_tp_pct,
                'distance_to_sl_pct': distance_to_sl_pct,
                'week52_percentile': week52_percentile,
                'volume': self.volume,
            }

# ============================================================================
# FUNDAMENTAL DATA (AM side)
# ============================================================================

def fetch_fundamental_data_polygon(symbol):
    try:
        url = (f"{config.polygon_rest_url}/vX/reference/financials"
               f"?ticker={symbol}&timeframe=quarterly&limit=24&sort=filing_date"
               f"&order=desc&apiKey={config.polygon_api_key}")
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        results = resp.json().get('results', [])
        if not results:
            return None
        fund = {k: [] for k in ['dates','revenue','net_income','operating_cash_flow',
                                  'capex','fcf','total_assets','total_liabilities',
                                  'shareholders_equity','current_assets','current_liabilities',
                                  'total_debt','eps']}
        for r in results:
            try:
                fi = r.get('financials', {})
                inc = fi.get('income_statement', {})
                cf = fi.get('cash_flow_statement', {})
                bs = fi.get('balance_sheet', {})
                rev = inc.get('revenues', {}).get('value', 0) or 0
                ni = inc.get('net_income_loss', {}).get('value', 0) or 0
                eps = inc.get('basic_earnings_per_share', {}).get('value', 0) or 0
                ocf = cf.get('net_cash_flow_from_operating_activities', {}).get('value', 0) or 0
                capex_raw = cf.get('net_cash_flow_from_investing_activities', {}).get('value', 0) or 0
                ta = bs.get('assets', {}).get('value', 0) or 0
                tl = bs.get('liabilities', {}).get('value', 0) or 0
                eq = bs.get('equity', {}).get('value', 0) or 0
                ca = bs.get('current_assets', {}).get('value', 0) or 0
                cl = bs.get('current_liabilities', {}).get('value', 0) or 0
                ltd = bs.get('long_term_debt', {}).get('value', 0) or 0
                std = bs.get('short_term_debt', {}).get('value', 0) or 0
                fund['dates'].append(r.get('filing_date', ''))
                fund['revenue'].append(rev)
                fund['net_income'].append(ni)
                fund['operating_cash_flow'].append(ocf)
                fund['capex'].append(abs(capex_raw))
                fund['fcf'].append(ocf + capex_raw)
                fund['total_assets'].append(ta)
                fund['total_liabilities'].append(tl)
                fund['shareholders_equity'].append(eq)
                fund['current_assets'].append(ca)
                fund['current_liabilities'].append(cl)
                fund['total_debt'].append(ltd + std)
                fund['eps'].append(eps)
            except:
                continue
        for k in fund:
            fund[k] = fund[k][::-1]
        return fund
    except:
        return None


def calculate_financial_ratios(fund, price, mcap):
    ratios = {k: [] for k in ['pe_ratio','current_ratio','roe','roa',
                                'net_profit_margin','debt_to_equity',
                                'price_to_book','price_to_sales','asset_turnover','fcfy']}
    n = len(fund.get('revenue', []))
    for i in range(n):
        try:
            rev = fund['revenue'][i]; ni = fund['net_income'][i]
            ta = fund['total_assets'][i]; eq = fund['shareholders_equity'][i]
            ca = fund['current_assets'][i]; cl = fund['current_liabilities'][i]
            td = fund['total_debt'][i]; eps = fund['eps'][i]; fcf = fund['fcf'][i]
            ratios['pe_ratio'].append(price / eps if eps and eps > 0 else None)
            ratios['current_ratio'].append(ca / cl if cl else None)
            ratios['roe'].append(ni / eq if eq and eq > 0 else None)
            ratios['roa'].append(ni / ta if ta else None)
            ratios['net_profit_margin'].append(ni / rev if rev else None)
            ratios['debt_to_equity'].append(td / eq if eq and eq > 0 else None)
            bvps = eq / (mcap / price) if price and mcap else None
            ratios['price_to_book'].append(price / bvps if bvps and bvps > 0 else None)
            ar = rev * 4
            ratios['price_to_sales'].append(mcap / ar if ar else None)
            ratios['asset_turnover'].append(rev / ta if ta else None)
            if i >= 3:
                af = sum(fund['fcf'][max(0, i - 3):i + 1])
                ratios['fcfy'].append(af / mcap if mcap else None)
            else:
                ratios['fcfy'].append(None)
        except:
            for k in ratios:
                ratios[k].append(None)
    return ratios


def calculate_slopes(series, span_short=4, span_long=20):
    if not series or len(series) < 5:
        return None, None
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan)
    slope_5 = slope_20 = None
    if len(s.dropna()) >= 5:
        try:
            ema = s.ewm(span=span_short, adjust=False).mean()
            if len(ema) >= 5 and pd.notna(ema.iloc[-1]) and pd.notna(ema.iloc[-5]):
                if abs(ema.iloc[-5]) > 0.0001:
                    slope_5 = (ema.iloc[-1] - ema.iloc[-5]) / abs(ema.iloc[-5])
        except:
            pass
    if len(s.dropna()) >= 21:
        try:
            ema = s.ewm(span=span_long, adjust=False).mean()
            if len(ema) >= 21 and pd.notna(ema.iloc[-1]) and pd.notna(ema.iloc[-21]):
                if abs(ema.iloc[-21]) > 0.0001:
                    slope_20 = (ema.iloc[-1] - ema.iloc[-21]) / abs(ema.iloc[-21])
        except:
            pass
    return slope_5, slope_20


def calculate_all_slopes(fund, ratios):
    sl = {}
    sl['Rev_Slope_5'], sl['Rev_Slope_20'] = calculate_slopes(fund.get('revenue', []))
    sl['FCF_Slope_5'], sl['FCF_Slope_20'] = calculate_slopes(fund.get('fcf', []))
    sl['Deps_Slope_5'], sl['Deps_Slope_20'] = calculate_slopes(fund.get('net_income', []))
    for name, key in [('P/E Ratio','pe_ratio'),('Current Ratio','current_ratio'),
                      ('Return on Equity','roe'),('Return on Assets','roa'),
                      ('Net Profit Margin','net_profit_margin'),('Debt to Equity Ratio','debt_to_equity'),
                      ('Price to Book Ratio','price_to_book'),('Price to Sales Ratio','price_to_sales'),
                      ('Asset Turnover','asset_turnover')]:
        sl[f'{name}_Slope_5'], sl[f'{name}_Slope_20'] = calculate_slopes(ratios.get(key, []))
    fcfy_list = ratios.get('fcfy', [])
    sl['FCFY'] = fcfy_list[-1] if fcfy_list and fcfy_list[-1] is not None else None
    return sl


def fetch_all_fundamental_data():
    print("\n📊 FETCHING FUNDAMENTAL DATA...")
    ok = fail = 0
    for i, sym in enumerate(config.symbols):
        try:
            fund = fetch_fundamental_data_polygon(sym)
            if fund and len(fund.get('revenue', [])) >= 4:
                price = None
                if sym in config.week52_data:
                    w = config.week52_data[sym]
                    if w.get('high') and w.get('low'):
                        price = (w['high'] + w['low']) / 2
                try:
                    qurl = f"{config.polygon_rest_url}/v2/aggs/ticker/{sym}/prev?adjusted=true&apiKey={config.polygon_api_key}"
                    qr = requests.get(qurl, timeout=10)
                    if qr.status_code == 200:
                        qd = qr.json()
                        if qd.get('results'):
                            price = qd['results'][0].get('c', price)
                except:
                    pass
                if price is None:
                    price = 100
                eq = fund['shareholders_equity'][-1]
                mcap = eq * 2 if eq and eq > 0 else 1e9
                ratios = calculate_financial_ratios(fund, price, mcap)
                slopes = calculate_all_slopes(fund, ratios)
                config.fundamental_data[sym] = fund
                config.fundamental_slopes[sym] = slopes
                ok += 1
            else:
                fail += 1
        except:
            fail += 1
        if (i + 1) % 25 == 0:
            print(f"   📈 Fundamentals: {i+1}/{len(config.symbols)} (✓{ok} ✗{fail})")
        time.sleep(0.15)
    print(f"✅ Fundamental data: {ok} ok, {fail} failed\n")

# ============================================================================
# MERIT SCORES
# ============================================================================

def calculate_stasis_merit_score(snap):
    ms = 0
    st = snap.get('stasis', 0)
    if st >= 15: ms += 10
    elif st >= 12: ms += 9
    elif st >= 10: ms += 8
    elif st >= 8: ms += 7
    elif st >= 7: ms += 6
    elif st >= 6: ms += 5
    elif st >= 5: ms += 4
    elif st >= 4: ms += 3
    elif st >= 3: ms += 2
    elif st >= 2: ms += 1
    rr = snap.get('risk_reward')
    if rr is not None:
        if rr >= 3: ms += 5
        elif rr >= 2.5: ms += 4
        elif rr >= 2: ms += 3
        elif rr >= 1.5: ms += 2
        elif rr >= 1: ms += 1
    strength = snap.get('signal_strength')
    if strength == 'VERY_STRONG': ms += 4
    elif strength == 'STRONG': ms += 3
    elif strength == 'MODERATE': ms += 2
    elif strength == 'WEAK': ms += 1
    dur = snap.get('duration_seconds', 0)
    if dur >= 3600: ms += 3
    elif dur >= 1800: ms += 2
    elif dur >= 900: ms += 1
    return ms


def calculate_fundamental_merit_score(symbol, w52_pct):
    ms = 0
    sd = {}
    slopes = config.fundamental_slopes.get(symbol, {})
    if not slopes:
        if w52_pct is not None:
            if w52_pct <= 5: ms += 8
            elif w52_pct <= 15: ms += 7
            elif w52_pct <= 25: ms += 6
            elif w52_pct <= 35: ms += 5
            elif w52_pct <= 45: ms += 4
            elif w52_pct <= 55: ms += 3
            elif w52_pct <= 65: ms += 2
            elif w52_pct <= 75: ms += 1
        return ms, sd
    # Growth metrics (positive = good)
    for label, key, thresholds_pts in [
        ('Rev_5','Rev_Slope_5',[(0.30,4),(0.20,3),(0.10,2),(0.05,1)]),
        ('Rev_20','Rev_Slope_20',[(0.20,3),(0.10,2),(0.05,1)]),
        ('FCF_5','FCF_Slope_5',[(0.40,4),(0.25,3),(0.10,2),(0.05,1)]),
        ('FCF_20','FCF_Slope_20',[(0.25,3),(0.15,2),(0.05,1)]),
        ('ROE_5','Return on Equity_Slope_5',[(0.20,2),(0.10,1)]),
        ('ROE_20','Return on Equity_Slope_20',[(0.15,2),(0.08,1)]),
        ('ROA_5','Return on Assets_Slope_5',[(0.15,2),(0.08,1)]),
        ('NPM_5','Net Profit Margin_Slope_5',[(0.20,2),(0.10,1)]),
        ('NPM_20','Net Profit Margin_Slope_20',[(0.15,2),(0.08,1)]),
    ]:
        v = slopes.get(key)
        sd[label] = v
        if v is not None:
            for th, pts in thresholds_pts:
                if v >= th:
                    ms += pts
                    break
    at5 = slopes.get('Asset Turnover_Slope_5')
    cr5 = slopes.get('Current Ratio_Slope_5')
    if at5 is not None and at5 >= 0.10: ms += 1
    if cr5 is not None and cr5 >= 0.10: ms += 1
    # Valuation (negative = good)
    for label, key, thresholds_pts in [
        ('PE_5','P/E Ratio_Slope_5',[(-0.25,3),(-0.15,2),(-0.05,1)]),
        ('PE_20','P/E Ratio_Slope_20',[(-0.20,2),(-0.10,1)]),
        ('DE_5','Debt to Equity Ratio_Slope_5',[(-0.20,2),(-0.10,1)]),
        ('DE_20','Debt to Equity Ratio_Slope_20',[(-0.15,2),(-0.08,1)]),
    ]:
        v = slopes.get(key)
        sd[label] = v
        if v is not None:
            for th, pts in thresholds_pts:
                if v <= th:
                    ms += pts
                    break
    pb5 = slopes.get('Price to Book Ratio_Slope_5')
    ps5 = slopes.get('Price to Sales Ratio_Slope_5')
    if pb5 is not None and pb5 <= -0.20: ms += 1
    if ps5 is not None and ps5 <= -0.20: ms += 1
    # 52-week percentile
    if w52_pct is not None:
        if w52_pct <= 5: ms += 8
        elif w52_pct <= 15: ms += 7
        elif w52_pct <= 25: ms += 6
        elif w52_pct <= 35: ms += 5
        elif w52_pct <= 45: ms += 4
        elif w52_pct <= 55: ms += 3
        elif w52_pct <= 65: ms += 2
        elif w52_pct <= 75: ms += 1
    # FCFY
    fcfy = slopes.get('FCFY')
    sd['FCFY'] = fcfy
    if fcfy is not None:
        if fcfy >= 0.15: ms += 3
        elif fcfy >= 0.10: ms += 2
        elif fcfy >= 0.05: ms += 1
    return ms, sd

# ============================================================================
# COMMON DATA HELPERS
# ============================================================================

def calculate_52week_percentile(price, symbol):
    d = config.week52_data.get(symbol)
    if not d:
        return None
    h, l, r = d.get('high'), d.get('low'), d.get('range')
    if h is None or l is None or r is None or r <= 0:
        return None
    return max(0.0, min(100.0, ((price - l) / r) * 100))


def fetch_52_week_data():
    print("📊 Fetching 52-week data...")
    w52 = {}
    end = datetime.now()
    start = end - timedelta(days=365)
    ok = fail = 0
    for i, sym in enumerate(config.symbols):
        try:
            url = (f"{config.polygon_rest_url}/v2/aggs/ticker/{sym}/range/1/day/"
                   f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
                   f"?adjusted=true&sort=asc&limit=365&apiKey={config.polygon_api_key}")
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                d = r.json()
                if d.get('results'):
                    highs = [b['h'] for b in d['results']]
                    lows = [b['l'] for b in d['results']]
                    closes = [b['c'] for b in d['results']]
                    hv, lv = max(highs), min(lows)
                    w52[sym] = {'high': hv, 'low': lv, 'range': hv - lv, 'current': closes[-1]}
                    ok += 1
                else:
                    w52[sym] = {'high': None, 'low': None, 'range': None, 'current': None}
                    fail += 1
            else:
                w52[sym] = {'high': None, 'low': None, 'range': None, 'current': None}
                fail += 1
            if (i + 1) % 50 == 0:
                print(f"   52W: {i+1}/{len(config.symbols)} (✓{ok} ✗{fail})")
            time.sleep(0.12)
        except:
            w52[sym] = {'high': None, 'low': None, 'range': None, 'current': None}
            fail += 1
    print(f"✅ 52-week: {ok} ok, {fail} failed\n")
    return w52


def fetch_volume_data():
    print("📊 Fetching volume data...")
    vols = {}
    end = datetime.now()
    start = end - timedelta(days=45)
    for i, sym in enumerate(config.symbols):
        try:
            url = (f"{config.polygon_rest_url}/v2/aggs/ticker/{sym}/range/1/day/"
                   f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
                   f"?adjusted=true&sort=desc&limit=30&apiKey={config.polygon_api_key}")
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                d = r.json()
                if d.get('results'):
                    tv = sum(b['v'] for b in d['results'])
                    vols[sym] = (tv / len(d['results'])) / 1e6
                else:
                    vols[sym] = 10.0
            else:
                vols[sym] = 10.0
            if (i + 1) % 50 == 0:
                print(f"   Volume: {i+1}/{len(config.symbols)}")
            time.sleep(0.12)
        except:
            vols[sym] = 10.0
    print(f"✅ Volume loaded\n")
    return vols


def fetch_historical_bars(symbol, days=5):
    bars = []
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        url = (f"{config.polygon_rest_url}/v2/aggs/ticker/{symbol}/range/1/minute/"
               f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
               f"?adjusted=true&sort=asc&limit=50000&apiKey={config.polygon_api_key}")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            d = r.json()
            if d.get('results'):
                for b in d['results']:
                    bars.append({'timestamp': datetime.fromtimestamp(b['t'] / 1000), 'close': b['c']})
    except:
        pass
    return bars

# ============================================================================
# PRICE FEED
# ============================================================================

class PolygonPriceFeed:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_prices = {s: None for s in config.symbols}
        self.is_running = False
        self.ws = None
        self.message_count = 0

    def start(self):
        self.is_running = True
        threading.Thread(target=self._ws_loop, daemon=True).start()
        print("✅ WebSocket starting...")

    def _ws_loop(self):
        while self.is_running:
            try:
                self._connect()
            except Exception as e:
                print(f"WS err: {e}")
                time.sleep(5)

    def _connect(self):
        def on_msg(ws, msg):
            try:
                data = json.loads(msg)
                for m in (data if isinstance(data, list) else [data]):
                    self._proc(m)
            except:
                pass
        def on_open(ws):
            print("✅ WS connected")
            ws.send(json.dumps({"action": "auth", "params": config.polygon_api_key}))
        self.ws = websocket.WebSocketApp(config.polygon_ws_url, on_open=on_open, on_message=on_msg)
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def _proc(self, msg):
        if msg.get('ev') == 'status' and msg.get('status') == 'auth_success':
            self._sub()
        elif msg.get('ev') in ('A', 'AM', 'T', 'Q'):
            sym = msg.get('sym', '') or msg.get('S', '')
            price = msg.get('c') or msg.get('vw') or msg.get('p') or msg.get('bp')
            if price and sym in config.symbols:
                with self.lock:
                    self.current_prices[sym] = float(price)
                    self.message_count += 1

    def _sub(self):
        for i in range(0, len(config.symbols), 50):
            batch = config.symbols[i:i + 50]
            self.ws.send(json.dumps({"action": "subscribe",
                                     "params": ",".join(f"A.{s}" for s in batch)}))
            time.sleep(0.1)
        print(f"📡 Subscribed {len(config.symbols)} symbols")

    def get_prices(self):
        with self.lock:
            return {k: v for k, v in self.current_prices.items() if v}

    def get_status(self):
        with self.lock:
            return {'connected': sum(1 for v in self.current_prices.values() if v),
                    'total': len(config.symbols), 'messages': self.message_count}

price_feed = PolygonPriceFeed()

# ============================================================================
# BITSTREAM MANAGER (shared)
# ============================================================================

class BitstreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.streams: Dict[Tuple[str, float], Bitstream] = {}
        self.is_running = False
        # Cached snapshots for each stream
        self.cached_snapshots: List[Dict] = []
        # PM caches
        self.cached_pm_merit: Dict[str, Dict] = {}
        self.cached_pm_market: Dict[str, Dict] = {}
        # AM caches
        self.cached_am_data: List[Dict] = []
        self.cache_lock = threading.Lock()
        self.initialized = False
        self.backfill_complete = False
        self.backfill_progress = 0

    def backfill(self):
        print("\n" + "=" * 60)
        print("📜 BACKFILLING HISTORICAL DATA")
        print("=" * 60)
        hist = {}
        for i, sym in enumerate(config.symbols):
            bars = fetch_historical_bars(sym, config.history_days)
            if bars:
                hist[sym] = bars
            self.backfill_progress = int((i + 1) / len(config.symbols) * 100)
            if (i + 1) % 25 == 0:
                print(f"   📊 {i+1}/{len(config.symbols)} ({self.backfill_progress}%)")
            time.sleep(0.12)
        with self.lock:
            for sym, bars in hist.items():
                if not bars:
                    continue
                vol = config.volumes.get(sym, 10.0)
                for th in config.thresholds:
                    key = (sym, th)
                    self.streams[key] = Bitstream(sym, th, bars[0]['close'], vol)
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
            time.sleep(0.1)
            if not self.backfill_complete:
                continue
            prices = price_feed.get_prices()
            ts = datetime.now()
            with self.lock:
                for sym, price in prices.items():
                    for th in config.thresholds:
                        key = (sym, th)
                        if key in self.streams:
                            self.streams[key].process_price(price, ts)

    def _cache_loop(self):
        while self.is_running:
            time.sleep(config.cache_refresh_interval)
            if not self.initialized:
                continue
            prices = price_feed.get_prices()
            snaps = []
            with self.lock:
                for stream in self.streams.values():
                    snaps.append(stream.get_snapshot(prices.get(stream.symbol)))

            # PM caches
            pm_merit = self._pm_calc_merit(snaps)
            pm_market = {}
            for sym, m in pm_merit.items():
                pm_market[sym] = self._pm_calc_market_price(m, prices.get(sym, 0))

            # AM caches
            am_data = self._am_build_table(snaps)

            with self.cache_lock:
                self.cached_snapshots = snaps
                self.cached_pm_merit = pm_merit
                self.cached_pm_market = pm_market
                self.cached_am_data = am_data

    # ------------------------------------------------------------------ PM
    def _pm_calc_merit(self, snaps):
        WEIGHTS = {0.000625: 1, 0.00125: 1.5, 0.0025: 2, 0.005: 3,
                   0.01: 5, 0.02: 8, 0.03: 12, 0.04: 16, 0.05: 20, 0.10: 30}
        by_sym = defaultdict(list)
        for s in snaps:
            if s['threshold'] in config.pm_thresholds:
                by_sym[s['symbol']].append(s)
        result = {}
        for sym, ss in by_sym.items():
            long_lvl, short_lvl, all_st = [], [], []
            for snap in ss:
                st = snap['stasis']; d = snap['direction']; th = snap['threshold']
                all_st.append(st)
                if st >= config.min_tradable_stasis and d:
                    w = (st ** 1.3) * WEIGHTS.get(th, 1)
                    (long_lvl if d == 'LONG' else short_lvl).append({'thresh': th, 'stasis': st, 'weight': w})
            lw = sum(l['weight'] for l in long_lvl)
            sw = sum(l['weight'] for l in short_lvl)
            tl = len(long_lvl) + len(short_lvl)
            tw = lw + sw
            if tl == 0:
                result[sym] = {'stasis_levels': 0, 'dominant_direction': None,
                               'direction_alignment': 0, 'weighted_score': 0,
                               'max_stasis': max(all_st) if all_st else 0,
                               'long_levels': 0, 'short_levels': 0}
                continue
            if lw > sw:
                dd, nw = 'LONG', lw - sw
            elif sw > lw:
                dd, nw = 'SHORT', sw - lw
            else:
                dd, nw = None, 0
            align = (max(lw, sw) / tw * 100) if tw > 0 else 0
            score = nw * (1 + 0.2 * (tl - 1)) if align == 100 and tl >= 2 else nw
            result[sym] = {
                'stasis_levels': tl, 'dominant_direction': dd,
                'direction_alignment': round(align, 0),
                'weighted_score': round(max(0, score), 1),
                'max_stasis': max(all_st) if all_st else 0,
                'long_levels': len(long_lvl), 'short_levels': len(short_lvl),
            }
        return result

    def _pm_calc_market_price(self, merit, stock_price):
        levels = merit.get('stasis_levels', 0)
        align = merit.get('direction_alignment', 0)
        direction = merit.get('dominant_direction')
        max_st = merit.get('max_stasis', 0)
        score = merit.get('weighted_score', 0)
        if levels == 0 or direction is None:
            return {'probability': 0.50, 'buy_price': 0.50, 'sell_price': 0.50,
                    'edge': 0, 'direction': None, 'payout': 1.0, 'tier': None,
                    'emoji': '⬜', 'heat': 0, 'stock_price': stock_price}
        base = 0.50
        lc = min(0.30, levels * 0.04)
        sc = min(0.10, max_st * 0.01)
        wc = min(0.05, score / 1000)
        if align == 100:
            prob = base + lc + sc + wc
        else:
            prob = base + (lc * align / 100 * 0.5) + (sc * 0.5)
        prob = max(0.52, min(0.95, prob))
        spread = max(0.005, 0.02 - 0.01 * min(1, levels / 5))
        buy = min(0.98, prob + spread / 2)
        sell = max(0.02, 1 - prob + spread / 2)
        edge = (prob - 0.50) * 100
        payout = 1 / buy if buy > 0 else 1
        heat = min(100, int((levels / 10) * 50 + (align / 100) * 30 + min(20, max_st * 2)))
        tier, emoji = None, '⬜'
        for tn, ti in JACKPOT_TIERS.items():
            if levels >= ti['min_levels'] and align >= ti['min_alignment']:
                tier, emoji = tn, ti['emoji']
                break
        return {'probability': round(prob, 4), 'buy_price': round(buy, 4),
                'sell_price': round(sell, 4), 'edge': round(edge, 2),
                'direction': direction, 'payout': round(payout, 2),
                'tier': tier, 'emoji': emoji, 'heat': heat,
                'stock_price': stock_price}

    # ------------------------------------------------------------------ AM
    def _am_build_table(self, snaps):
        rows = []
        for snap in snaps:
            if snap['threshold'] not in config.am_thresholds:
                continue
            # Enrich with merit scores
            sms = calculate_stasis_merit_score(snap)
            fms, sd = calculate_fundamental_merit_score(snap['symbol'], snap.get('week52_percentile'))
            tms = sms + fms
            rows.append({**snap, 'sms': sms, 'fms': fms, 'tms': tms, 'slope_details': sd})
        return rows

    # Public getters
    def get_pm_market(self):
        with self.cache_lock:
            return copy.deepcopy(self.cached_pm_market)

    def get_pm_merit(self):
        with self.cache_lock:
            return copy.deepcopy(self.cached_pm_merit)

    def get_am_data(self):
        with self.cache_lock:
            return copy.deepcopy(self.cached_am_data)

    def get_snapshots(self):
        with self.cache_lock:
            return copy.deepcopy(self.cached_snapshots)

manager = BitstreamManager()

# ============================================================================
# HELPERS
# ============================================================================

def fmt_slope(v):
    if v is None:
        return "—"
    return f"{'+' if v >= 0 else ''}{v*100:.1f}%"

def fmt_rr(rr):
    if rr is None:
        return "—"
    if rr <= 0:
        return "0:1"
    return f"{rr:.2f}:1" if rr < 10 else f"{rr:.0f}:1"

# ============================================================================
# DASH APP
# ============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;600;700&display=swap');

body {
    background-color: #0a0a0a !important;
    margin: 0; padding: 0;
    font-family: 'Roboto Mono', monospace !important;
}
.title-font { font-family: 'Orbitron', sans-serif !important; }
.neon-green { color: #00ff88; text-shadow: 0 0 5px #00ff88; }
.neon-red   { color: #ff4444; text-shadow: 0 0 5px #ff4444; }
.neon-gold  { color: #ffd700; text-shadow: 0 0 5px #ffd700; }
.neon-cyan  { color: #00ffff; text-shadow: 0 0 5px #00ffff; }

/* PM Side (left dark) */
.pm-side {
    background: linear-gradient(180deg, #0a0a12 0%, #10101e 50%, #0a0a12 100%);
    border-right: 2px solid #1a1a3a;
    min-height: 100vh;
    padding: 8px;
}
.pm-market-row {
    background: rgba(18, 18, 34, 0.95);
    border: 1px solid #2a2a4e;
    border-radius: 5px;
    padding: 6px 10px;
    margin-bottom: 4px;
    transition: border-color 0.15s;
}
.pm-market-row:hover { border-color: #00ffff; }

/* AM Side (right — muted warm light theme) */
.am-side {
    background: linear-gradient(180deg, #f5f0e8 0%, #ede7da 50%, #f5f0e8 100%);
    min-height: 100vh;
    padding: 8px;
}

.btn-yes {
    background: linear-gradient(135deg, #00aa44, #00ff88) !important;
    border: none !important; color: #000 !important;
    font-weight: bold !important; font-size: 9px !important;
    padding: 3px 8px !important; border-radius: 3px !important;
}
.btn-yes:hover { box-shadow: 0 0 8px rgba(0,255,136,0.5); }
.btn-no {
    background: linear-gradient(135deg, #aa2200, #ff4444) !important;
    border: none !important; color: #fff !important;
    font-weight: bold !important; font-size: 9px !important;
    padding: 3px 8px !important; border-radius: 3px !important;
}
.btn-no:hover { box-shadow: 0 0 8px rgba(255,68,68,0.5); }
.btn-close-pos {
    background: #444 !important; border: 1px solid #666 !important;
    color: #fff !important; font-size: 8px !important; padding: 1px 6px !important;
}
.amount-btn {
    background: #2a2a4e !important; border: 1px solid #444 !important;
    color: #fff !important; font-size: 9px !important; padding: 3px 6px !important;
    margin: 1px !important; border-radius: 3px !important;
}
.amount-btn.selected {
    background: linear-gradient(135deg, #0066aa, #00aaff) !important;
    border-color: #00ffff !important;
}

.pm-portfolio {
    background: linear-gradient(135deg, #12192a, #0a1018);
    border: 1px solid #00ffff; border-radius: 8px; padding: 8px;
}
.pm-position { background: rgba(18,24,34,0.9); border: 1px solid #333;
               border-radius: 4px; padding: 6px; margin: 3px 0; }
.pm-position-long  { border-left: 3px solid #00ff88; }
.pm-position-short { border-left: 3px solid #ff4444; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
"""

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
                suppress_callback_exceptions=True)
app.title = "STASIS PM | STASIS AM"
server = app.server

app.index_string = f'''<!DOCTYPE html>
<html>
<head>
    {{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
    <style>{CUSTOM_CSS}</style>
</head>
<body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body>
</html>'''

# ----- Layout ---------------------------------------------------------------

def pm_layout():
    """Left side: Stasis PM — dark theme prediction markets."""
    return html.Div([
        # Header
        html.Div([
            html.Span("🎰", style={'fontSize': '22px'}),
            html.Span(" STASIS PM", className="title-font neon-green ms-2",
                      style={'fontSize': '16px', 'fontWeight': '700', 'letterSpacing': '2px'}),
            html.Span(" — PREDICTION MARKETS", className="title-font",
                      style={'fontSize': '9px', 'color': '#666', 'letterSpacing': '1px'}),
        ], className="mb-1"),

        # Summary
        html.Div(id='pm-summary', style={'fontSize': '9px', 'marginBottom': '4px'}),

        # Filters row
        html.Div([
            dbc.ButtonGroup([
                dbc.Button("ALL", id="pm-f-all", size="sm", color="secondary", outline=True,
                          className="title-font", style={'fontSize': '8px'}),
                dbc.Button("SIGNALS", id="pm-f-signals", size="sm", color="success", outline=True,
                          active=True, className="title-font", style={'fontSize': '8px'}),
                dbc.Button("JACKPOT", id="pm-f-jackpot", size="sm", color="warning", outline=True,
                          className="title-font", style={'fontSize': '8px'}),
            ], size="sm", className="me-2"),
            dcc.Dropdown(id='pm-f-dir', options=[
                {'label':'ALL','value':'ALL'},{'label':'LONG','value':'LONG'},
                {'label':'SHORT','value':'SHORT'}],
                value='ALL', clearable=False,
                style={'width':'70px','fontSize':'9px','display':'inline-block','verticalAlign':'middle'}),
        ], className="d-flex align-items-center mb-1"),

        # Bet amount
        html.Div([
            html.Span("BET: ", className="title-font", style={'fontSize': '9px', 'color': '#ffd700'}),
            *[dbc.Button(f"${a}", id={'type': 'pm-amt', 'amount': a},
                         className=f"amount-btn {'selected' if a == 100 else ''}", size="sm")
              for a in BET_SIZES],
        ], className="d-flex align-items-center flex-wrap mb-1"),

        # Market list
        html.Div(id='pm-market-list', style={'height': '42vh', 'overflowY': 'auto'}),

        html.Hr(style={'borderColor': '#222', 'margin': '6px 0'}),

        # Portfolio + positions + history
        html.Div([
            html.Div([
                html.Span("💰 ACCOUNT", className="title-font neon-cyan", style={'fontSize': '10px'}),
                dbc.Button("Reset", id="pm-reset", size="sm", color="secondary",
                          style={'fontSize': '7px', 'padding': '1px 5px', 'marginLeft': '8px'}),
            ], className="d-flex align-items-center mb-1"),
            html.Div(id='pm-portfolio'),
        ], className="pm-portfolio mb-2"),

        html.Div([
            html.Span("📊 POSITIONS", className="title-font", style={'fontSize': '9px', 'color': '#00ffff'}),
            html.Div(id='pm-positions', style={'maxHeight': '12vh', 'overflowY': 'auto'}),
        ], className="mb-2"),

        html.Div([
            html.Span("📜 HISTORY", className="title-font", style={'fontSize': '9px', 'color': '#ffd700'}),
            html.Div(id='pm-history', style={'maxHeight': '10vh', 'overflowY': 'auto'}),
        ]),

    ], className="pm-side")


def am_layout():
    """Right side: Stasis AM — warm-light alpha markets."""
    return html.Div([
        # Header
        html.Div([
            html.Span("📈", style={'fontSize': '22px'}),
            html.Span(" STASIS AM", className="title-font ms-2",
                      style={'fontSize': '16px', 'fontWeight': '700', 'letterSpacing': '2px',
                             'color': '#1a5c2a'}),
            html.Span(" — ALPHA MARKETS", className="title-font",
                      style={'fontSize': '9px', 'color': '#888', 'letterSpacing': '1px'}),
        ], className="mb-1"),

        # Stats
        html.Div(id='am-stats', style={'fontSize': '9px', 'marginBottom': '4px'}),

        # Filters
        html.Div([
            dbc.ButtonGroup([
                dbc.Button("ALL", id="am-f-all", size="sm", outline=True,
                          style={'fontSize': '8px', 'borderColor': '#aaa', 'color': '#333'}),
                dbc.Button("TRADABLE", id="am-f-tradable", size="sm", outline=True, active=True,
                          style={'fontSize': '8px', 'borderColor': '#1a8c3a', 'color': '#1a5c2a'}),
            ], size="sm", className="me-2"),
            dcc.Dropdown(id='am-f-dir', options=[
                {'label':'ALL','value':'ALL'},{'label':'LONG','value':'LONG'},
                {'label':'SHORT','value':'SHORT'}],
                value='ALL', clearable=False,
                style={'width':'70px','fontSize':'9px','display':'inline-block','verticalAlign':'middle'}),
            html.Span(" ", style={'width': '8px', 'display': 'inline-block'}),
            dcc.Dropdown(id='am-f-52w', options=[
                {'label':'ANY 52W','value':'ALL'},{'label':'0-25%','value':'0-25'},
                {'label':'0-40%','value':'0-40'},{'label':'0-60%','value':'0-60'}],
                value='ALL', clearable=False,
                style={'width':'80px','fontSize':'9px','display':'inline-block','verticalAlign':'middle'}),
            html.Span(" ", style={'width': '8px', 'display': 'inline-block'}),
            dcc.Dropdown(id='am-f-sort', options=[
                {'label':'TMS ↓','value':'tms'},{'label':'FMS ↓','value':'fms'},
                {'label':'STASIS ↓','value':'stasis'},{'label':'52W ↑','value':'52w'}],
                value='tms', clearable=False,
                style={'width':'80px','fontSize':'9px','display':'inline-block','verticalAlign':'middle'}),
        ], className="d-flex align-items-center flex-wrap mb-1"),

        # Table
        dash_table.DataTable(
            id='am-table',
            columns=[
                {'name': '✓', 'id': 'Tradable'},
                {'name': 'SYM', 'id': 'Symbol'},
                {'name': 'BAND', 'id': 'Band'},
                {'name': 'STS', 'id': 'Stasis'},
                {'name': 'DIR', 'id': 'Dir'},
                {'name': 'SMS', 'id': 'SMS'},
                {'name': 'FMS', 'id': 'FMS'},
                {'name': 'TMS', 'id': 'TMS'},
                {'name': 'REV5', 'id': 'Rev5'},
                {'name': 'FCF5', 'id': 'FCF5'},
                {'name': 'FCFY', 'id': 'FCFY'},
                {'name': '52W', 'id': '52W'},
                {'name': 'PRICE', 'id': 'Price'},
                {'name': 'TP', 'id': 'TP'},
                {'name': 'SL', 'id': 'SL'},
                {'name': 'R:R', 'id': 'RR'},
                {'name': 'DUR', 'id': 'Dur'},
            ],
            sort_action='native',
            sort_mode='multi',
            sort_by=[{'column_id': 'TMS', 'direction': 'desc'}],
            style_table={'height': '78vh', 'overflowY': 'auto'},
            style_cell={
                'backgroundColor': '#faf7f0',
                'color': '#1a1a1a',
                'padding': '3px 5px',
                'fontSize': '10px',
                'fontFamily': 'Roboto Mono, monospace',
                'whiteSpace': 'nowrap',
                'textAlign': 'right',
                'minWidth': '35px',
                'border': '1px solid #ddd',
            },
            style_cell_conditional=[
                {'if': {'column_id': 'Symbol'}, 'textAlign': 'left', 'fontWeight': '700', 'color': '#1a5c2a'},
                {'if': {'column_id': 'Dir'}, 'textAlign': 'center'},
                {'if': {'column_id': 'Tradable'}, 'textAlign': 'center'},
            ],
            style_header={
                'backgroundColor': '#1a5c2a',
                'color': '#ffffff',
                'fontWeight': '700',
                'fontSize': '9px',
                'fontFamily': 'Orbitron, sans-serif',
                'borderBottom': '2px solid #0d3d18',
                'textAlign': 'center',
            },
            style_data_conditional=[
                {'if': {'filter_query': '{Stasis} >= 10'}, 'backgroundColor': '#e8f5e9'},
                {'if': {'filter_query': '{Stasis} >= 7 && {Stasis} < 10'}, 'backgroundColor': '#f1f8e9'},
                {'if': {'filter_query': '{Dir} = "LONG"', 'column_id': 'Dir'},
                 'color': '#1a8c3a', 'fontWeight': 'bold'},
                {'if': {'filter_query': '{Dir} = "SHORT"', 'column_id': 'Dir'},
                 'color': '#cc2200', 'fontWeight': 'bold'},
                {'if': {'column_id': 'Price'}, 'color': '#0055aa', 'fontWeight': '600'},
                {'if': {'column_id': 'TP'}, 'color': '#1a8c3a'},
                {'if': {'column_id': 'SL'}, 'color': '#cc2200'},
                {'if': {'filter_query': '{TMS} >= 40', 'column_id': 'TMS'},
                 'backgroundColor': '#1a8c3a', 'color': '#fff'},
                {'if': {'filter_query': '{TMS} >= 30 && {TMS} < 40', 'column_id': 'TMS'},
                 'backgroundColor': '#4caf50', 'color': '#fff'},
                {'if': {'filter_query': '{FMS} >= 25', 'column_id': 'FMS'},
                 'backgroundColor': '#ff9800', 'color': '#fff'},
                {'if': {'filter_query': '{Rev5} contains "+"', 'column_id': 'Rev5'}, 'color': '#1a8c3a'},
                {'if': {'filter_query': '{Rev5} contains "-"', 'column_id': 'Rev5'}, 'color': '#cc2200'},
                {'if': {'filter_query': '{FCF5} contains "+"', 'column_id': 'FCF5'}, 'color': '#1a8c3a'},
                {'if': {'filter_query': '{FCF5} contains "-"', 'column_id': 'FCF5'}, 'color': '#cc2200'},
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#f0ebe0'},
            ]
        ),

        # Footer
        html.Div([
            html.Span("SMS", style={'color': '#0055aa', 'fontWeight': 'bold', 'fontSize': '8px'}),
            html.Span(" Stasis Merit | ", style={'fontSize': '8px', 'color': '#888'}),
            html.Span("FMS", style={'color': '#ff9800', 'fontWeight': 'bold', 'fontSize': '8px'}),
            html.Span(" Fundamental Merit | ", style={'fontSize': '8px', 'color': '#888'}),
            html.Span("TMS", style={'color': '#1a8c3a', 'fontWeight': 'bold', 'fontSize': '8px'}),
            html.Span(" Combined Total", style={'fontSize': '8px', 'color': '#888'}),
        ], className="text-center mt-1"),

    ], className="am-side")


# Master layout
app.layout = html.Div([
    # Shared stores and interval
    dcc.Store(id='pm-bet-amount', data=100),
    dcc.Store(id='pm-trigger', data=0),
    dcc.Store(id='pm-filter-mode', data='signals'),
    dcc.Store(id='am-filter-mode', data='tradable'),
    dcc.Interval(id='tick', interval=config.update_interval_ms, n_intervals=0),

    # Toast for PM trades
    dbc.Toast(id="pm-toast", header="", is_open=False, duration=3500, dismissable=True,
              style={"position": "fixed", "top": 10, "left": "50%", "transform": "translateX(-50%)",
                     "width": 300, "zIndex": 9999}),

    # Connection status bar
    html.Div([
        html.Div(id='global-status', className="text-center",
                style={'fontSize': '9px', 'padding': '2px', 'backgroundColor': '#0a0a0a', 'color': '#888'})
    ]),

    # Split screen
    dbc.Row([
        dbc.Col(pm_layout(), width=6, style={'padding': 0}),
        dbc.Col(am_layout(), width=6, style={'padding': 0}),
    ], className="g-0"),

    # Copyright
    html.Div([
        html.Span("© 2026 TRUTH COMMUNICATIONS LLC", className="title-font",
                  style={'fontSize': '8px', 'color': '#444', 'letterSpacing': '2px'}),
    ], className="text-center", style={'padding': '4px', 'backgroundColor': '#0a0a0a'}),
])

# ============================================================================
# CALLBACKS — GLOBAL
# ============================================================================

@app.callback(Output('global-status', 'children'), Input('tick', 'n_intervals'))
def update_global_status(n):
    if not manager.backfill_complete:
        return html.Span(f"⏳ INITIALIZING... {manager.backfill_progress}%", style={'color': '#ffaa00'})
    st = price_feed.get_status()
    fc = len(config.fundamental_slopes)
    if st['connected'] == 0:
        return html.Span(f"🔴 CONNECTING... | 📊 {fc} fundamentals", style={'color': '#ffaa00'})
    return html.Span(
        f"🟢 LIVE {st['connected']}/{st['total']} | "
        f"📨 {st['messages']:,} | 📊 {fc} fundamentals",
        style={'color': '#00ff88'})

# ============================================================================
# CALLBACKS — PM (LEFT SIDE)
# ============================================================================

@app.callback(Output('pm-summary', 'children'), Input('tick', 'n_intervals'))
def pm_summary(n):
    if not manager.backfill_complete:
        return ""
    mkt = manager.get_pm_market()
    sigs = sum(1 for m in mkt.values() if m.get('direction'))
    hi = sum(1 for m in mkt.values() if m.get('edge', 0) >= 10)
    jp = sum(1 for m in mkt.values() if m.get('tier'))
    return html.Div([
        html.Span(f"📈 {sigs} signals", style={'color': '#00ff88'}),
        html.Span(f" | 🔥 {hi} high-edge", style={'color': '#ff8800'}),
        html.Span(f" | 🎰 {jp} jackpots", style={'color': '#ffd700'}),
    ], style={'fontSize': '9px'})


@app.callback(
    [Output('pm-f-all', 'active'), Output('pm-f-signals', 'active'),
     Output('pm-f-jackpot', 'active'), Output('pm-filter-mode', 'data')],
    [Input('pm-f-all', 'n_clicks'), Input('pm-f-signals', 'n_clicks'),
     Input('pm-f-jackpot', 'n_clicks')],
    prevent_initial_call=True)
def pm_toggle_filter(n1, n2, n3):
    ctx = callback_context
    if not ctx.triggered:
        return False, True, False, 'signals'
    b = ctx.triggered[0]['prop_id'].split('.')[0]
    if b == 'pm-f-all': return True, False, False, 'all'
    if b == 'pm-f-jackpot': return False, False, True, 'jackpot'
    return False, True, False, 'signals'


@app.callback(
    Output('pm-bet-amount', 'data'),
    Input({'type': 'pm-amt', 'amount': ALL}, 'n_clicks'),
    State('pm-bet-amount', 'data'),
    prevent_initial_call=True)
def pm_set_amount(clicks, cur):
    ctx = callback_context
    if not ctx.triggered or not any(clicks):
        return cur
    try:
        return json.loads(ctx.triggered[0]['prop_id'].rsplit('.', 1)[0])['amount']
    except:
        return cur


@app.callback(
    [Output({'type': 'pm-amt', 'amount': a}, 'className') for a in BET_SIZES],
    Input('pm-bet-amount', 'data'))
def pm_highlight_amt(sel):
    return [f"amount-btn {'selected' if a == sel else ''}" for a in BET_SIZES]


@app.callback(
    Output('pm-market-list', 'children'),
    [Input('tick', 'n_intervals'), Input('pm-filter-mode', 'data'),
     Input('pm-f-dir', 'value'), Input('pm-bet-amount', 'data'),
     Input('pm-trigger', 'data')])
def pm_market_list(n, fmode, fdir, bet, trigger):
    if not manager.backfill_complete:
        return html.Div("Loading…", className="text-muted text-center p-3", style={'color': '#666'})
    mkt = manager.get_pm_market()
    mer = manager.get_pm_merit()
    items = []
    for sym in config.symbols:
        m = mkt.get(sym, {})
        mr = mer.get(sym, {})
        d = m.get('direction')
        edge = m.get('edge', 0)
        tier = m.get('tier')
        if fmode == 'signals' and not d:
            continue
        if fmode == 'jackpot' and not tier:
            continue
        if fdir == 'LONG' and d != 'LONG':
            continue
        if fdir == 'SHORT' and d != 'SHORT':
            continue
        items.append({'symbol': sym, 'm': m, 'mr': mr, 'edge': edge})
    items.sort(key=lambda x: x['edge'], reverse=True)
    if not items:
        return html.Div("No markets match", className="text-muted text-center p-3", style={'color': '#666'})
    rows = []
    for i, it in enumerate(items[:60]):
        sym = it['symbol']; m = it['m']; mr = it['mr']
        d = m.get('direction', '—'); bp = m.get('buy_price', .5); sp = m.get('sell_price', .5)
        edge = m.get('edge', 0); payout = m.get('payout', 1); emoji = m.get('emoji', '⬜')
        lvl = mr.get('stasis_levels', 0); align = mr.get('direction_alignment', 0)
        sp_ = m.get('stock_price', 0)
        dc = '#00ff88' if d == 'LONG' else '#ff4444' if d == 'SHORT' else '#555'
        rows.append(html.Div([
            dbc.Row([
                dbc.Col([
                    html.Span(f"#{i+1}", style={'color': '#ffd700' if i < 3 else '#555', 'fontSize': '9px'}),
                    html.Span(f" {emoji} ", style={'fontSize': '12px'}),
                    html.Span(sym, style={'color': '#00ff88', 'fontWeight': 'bold', 'fontSize': '11px'}),
                ], width=3),
                dbc.Col([
                    html.Span(d, style={'color': dc, 'fontWeight': 'bold', 'fontSize': '9px'}),
                    html.Span(f" {lvl}L {align:.0f}%", style={'color': '#666', 'fontSize': '8px'}),
                ], width=2),
                dbc.Col([
                    html.Span(f"${bp:.2f}", style={'color': '#00ff88', 'fontWeight': 'bold', 'fontSize': '10px'}),
                    html.Span("/", style={'color': '#333'}),
                    html.Span(f"${sp:.2f}", style={'color': '#ff4444', 'fontWeight': 'bold', 'fontSize': '10px'}),
                    html.Br(),
                    html.Span(f"+{edge:.1f}% {payout:.1f}x", style={'color': '#ffd700', 'fontSize': '8px'}),
                ], width=3),
                dbc.Col([
                    html.Button(f"YES ${bp:.2f}", id={'type': 'pm-buy-yes', 'symbol': sym},
                               className="btn-yes me-1", disabled=not d),
                    html.Button(f"NO ${sp:.2f}", id={'type': 'pm-buy-no', 'symbol': sym},
                               className="btn-no", disabled=not d),
                ], width=4, className="text-end"),
            ], className="align-items-center"),
        ], className="pm-market-row"))
    return html.Div(rows)


@app.callback(Output('pm-portfolio', 'children'),
              [Input('tick', 'n_intervals'), Input('pm-trigger', 'data')])
def pm_portfolio(n, t):
    mkt = manager.get_pm_market()
    st = portfolio.get_stats(mkt)
    pnl = st['total_pnl']
    pc = '#00ff88' if pnl >= 0 else '#ff4444'
    return html.Div([
        html.Div(f"${st['portfolio_value']:,.2f}", className="neon-cyan",
                style={'fontSize': '14px', 'fontWeight': 'bold'}),
        html.Div([
            html.Span(f"Cash ${st['balance']:,.2f}", style={'fontSize': '9px', 'color': '#888'}),
            html.Span(f" | P&L {'+' if pnl>=0 else ''}${pnl:,.2f}", style={'fontSize': '9px', 'color': pc}),
            html.Span(f" | Win {st['win_rate']:.0f}%", style={'fontSize': '9px',
                      'color': '#00ff88' if st['win_rate'] >= 50 else '#ff4444'}),
        ]),
    ])


@app.callback(Output('pm-positions', 'children'),
              [Input('tick', 'n_intervals'), Input('pm-trigger', 'data')])
def pm_positions(n, t):
    pos = portfolio.get_positions_list()
    mkt = manager.get_pm_market()
    if not pos:
        return html.Div("No positions", style={'fontSize': '9px', 'color': '#555'})
    items = []
    for p in pos:
        m = mkt.get(p['symbol'], {})
        if p['side'] == 'YES':
            cv = p['shares'] * m.get('sell_price', p['entry_price'])
        else:
            cv = p['shares'] * (1 - m.get('buy_price', 1 - p['entry_price']))
        pnl = cv - p['cost_basis']
        pc = '#00ff88' if pnl >= 0 else '#ff4444'
        sc = 'pm-position-long' if p['side'] == 'YES' else 'pm-position-short'
        items.append(html.Div([
            html.Div([
                html.Span(p['symbol'], style={'color': '#00ff88', 'fontWeight': 'bold', 'fontSize': '10px'}),
                html.Span(f" {p['side']}", style={'color': '#00ff88' if p['side']=='YES' else '#ff4444',
                          'fontSize': '9px'}),
                html.Button("✕", id={'type': 'pm-close', 'id': p['id']}, className="btn-close-pos ms-2"),
            ], className="d-flex align-items-center justify-content-between"),
            html.Span(f"P&L {'+' if pnl>=0 else ''}${pnl:.2f}", style={'color': pc, 'fontSize': '9px'}),
        ], className=f"pm-position {sc}"))
    return html.Div(items)


@app.callback(Output('pm-history', 'children'),
              [Input('tick', 'n_intervals'), Input('pm-trigger', 'data')])
def pm_history(n, t):
    trades = portfolio.get_recent_trades(6)
    if not trades:
        return html.Div("No trades", style={'fontSize': '9px', 'color': '#555'})
    items = []
    for tr in trades:
        if tr['action'] == 'CLOSE':
            pnl = tr.get('pnl', 0)
            pc = '#00ff88' if pnl >= 0 else '#ff4444'
            items.append(html.Div([
                html.Span(f"CLOSE {tr['symbol']} {tr['side']}", style={'color': '#888', 'fontSize': '9px'}),
                html.Span(f" {'+' if pnl>=0 else ''}${pnl:.2f}", style={'color': pc, 'fontSize': '9px'}),
            ], style={'padding': '2px 0'}))
        else:
            sc = '#00ff88' if tr['side'] == 'YES' else '#ff4444'
            items.append(html.Div([
                html.Span(f"BUY {tr['side']} {tr['symbol']} ${tr['amount']:.0f}",
                         style={'color': sc, 'fontSize': '9px'}),
            ], style={'padding': '2px 0'}))
    return html.Div(items)


# PM Trade execution
@app.callback(
    [Output('pm-toast', 'is_open'), Output('pm-toast', 'header'),
     Output('pm-toast', 'children'), Output('pm-toast', 'style'),
     Output('pm-trigger', 'data')],
    [Input({'type': 'pm-buy-yes', 'symbol': ALL}, 'n_clicks'),
     Input({'type': 'pm-buy-no', 'symbol': ALL}, 'n_clicks'),
     Input({'type': 'pm-close', 'id': ALL}, 'n_clicks'),
     Input('pm-reset', 'n_clicks')],
    [State('pm-bet-amount', 'data'), State('pm-trigger', 'data')],
    prevent_initial_call=True)
def pm_execute(yes_c, no_c, close_c, reset_c, bet, trigger):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update, no_update
    prop = ctx.triggered[0]['prop_id']
    val = ctx.triggered[0]['value']
    base = {"position": "fixed", "top": 10, "left": "50%", "transform": "translateX(-50%)",
            "width": 300, "zIndex": 9999, "backgroundColor": "#1a1a2e"}
    if 'pm-reset' in prop and val:
        portfolio.reset()
        return True, "🔄 RESET", html.Span(f"${STARTING_BALANCE:,.2f}", style={'color': '#00ffff'}), \
               {**base, 'border': '2px solid #00ffff'}, trigger + 1
    if 'pm-close' in prop and val:
        try:
            d = json.loads(prop.rsplit('.', 1)[0])
            pid = d.get('id')
            pos = next((p for p in portfolio.get_positions_list() if p['id'] == pid), None)
            if pos:
                mkt = manager.get_pm_market()
                m = mkt.get(pos['symbol'], {})
                res = portfolio.close_position(pid, m.get('buy_price', .5), m.get('sell_price', .5))
                if res['success']:
                    pnl = res['pnl']
                    pc = '#00ff88' if pnl >= 0 else '#ff4444'
                    return True, "✅ CLOSED", \
                           html.Span(f"{pos['symbol']} {'+' if pnl>=0 else ''}${pnl:.2f}",
                                    style={'color': pc, 'fontWeight': 'bold'}), \
                           {**base, 'border': f'2px solid {pc}'}, trigger + 1
        except:
            pass
        return no_update, no_update, no_update, no_update, no_update
    if ('pm-buy-yes' in prop or 'pm-buy-no' in prop) and val:
        try:
            d = json.loads(prop.rsplit('.', 1)[0])
            sym = d.get('symbol')
            side = 'YES' if 'buy-yes' in prop else 'NO'
            mkt = manager.get_pm_market()
            m = mkt.get(sym, {})
            if not m.get('direction'):
                return True, "❌", html.Span("No signal", style={'color': '#ff4444'}), \
                       {**base, 'border': '2px solid #ff4444'}, trigger
            res = portfolio.place_bet(sym, side, bet, m.get('buy_price', .5),
                                      m.get('sell_price', .5), m.get('direction'), m.get('stock_price', 0))
            if res['success']:
                sc = '#00ff88' if side == 'YES' else '#ff4444'
                return True, "✅ BET", \
                       html.Div([
                           html.Span(f"{side} {sym} ${bet}", style={'color': sc, 'fontWeight': 'bold'}),
                           html.Br(),
                           html.Span(f"Bal: ${res['new_balance']:,.2f}", style={'color': '#888', 'fontSize': '10px'}),
                       ]), {**base, 'border': f'2px solid {sc}'}, trigger + 1
            else:
                return True, "❌", html.Span(res.get('error', ''), style={'color': '#ff4444'}), \
                       {**base, 'border': '2px solid #ff4444'}, trigger
        except Exception as e:
            return True, "❌", html.Span(str(e), style={'color': '#ff4444'}), \
                   {**base, 'border': '2px solid #ff4444'}, trigger
    return no_update, no_update, no_update, no_update, no_update

# ============================================================================
# CALLBACKS — AM (RIGHT SIDE)
# ============================================================================

@app.callback(
    [Output('am-f-all', 'active'), Output('am-f-tradable', 'active'),
     Output('am-filter-mode', 'data')],
    [Input('am-f-all', 'n_clicks'), Input('am-f-tradable', 'n_clicks')],
    prevent_initial_call=True)
def am_toggle(n1, n2):
    ctx = callback_context
    if not ctx.triggered:
        return False, True, 'tradable'
    b = ctx.triggered[0]['prop_id'].split('.')[0]
    if b == 'am-f-all':
        return True, False, 'all'
    return False, True, 'tradable'


@app.callback(Output('am-stats', 'children'), Input('tick', 'n_intervals'))
def am_stats(n):
    if not manager.backfill_complete:
        return html.Span(f"⏳ Loading {manager.backfill_progress}%", style={'color': '#aa6600'})
    data = manager.get_am_data()
    tradable = [d for d in data if d.get('is_tradable')]
    longs = sum(1 for d in tradable if d.get('direction') == 'LONG')
    shorts = sum(1 for d in tradable if d.get('direction') == 'SHORT')
    avg_tms = np.mean([d.get('tms', 0) for d in tradable]) if tradable else 0
    max_tms = max([d.get('tms', 0) for d in tradable]) if tradable else 0
    return html.Div([
        html.Span(f"🎯 {len(tradable)} tradable", style={'color': '#1a5c2a'}),
        html.Span(f" | ↑{longs} ↓{shorts}", style={'color': '#555'}),
        html.Span(f" | Avg TMS {avg_tms:.1f}", style={'color': '#1a5c2a'}),
        html.Span(f" | Max {max_tms}", style={'color': '#ff9800', 'fontWeight': 'bold'}),
    ], style={'fontSize': '9px'})


@app.callback(
    Output('am-table', 'data'),
    [Input('tick', 'n_intervals'), Input('am-filter-mode', 'data'),
     Input('am-f-dir', 'value'), Input('am-f-52w', 'value'),
     Input('am-f-sort', 'value')])
def am_table(n, fmode, fdir, f52w, fsort):
    if not manager.backfill_complete:
        return []
    data = manager.get_am_data()
    if not data:
        return []

    rows = []
    for d in data:
        if fmode == 'tradable' and not d.get('is_tradable'):
            continue
        if fdir != 'ALL' and d.get('direction') != fdir:
            continue
        w52 = d.get('week52_percentile')
        if f52w != 'ALL' and w52 is not None:
            rng = {'0-25': (0, 25), '0-40': (0, 40), '0-60': (0, 60)}.get(f52w)
            if rng and not (rng[0] <= w52 <= rng[1]):
                continue

        sd = d.get('slope_details', {})
        rows.append({
            'Tradable': '✅' if d.get('is_tradable') else '',
            'Symbol': d['symbol'],
            'Band': f"{d['threshold_pct']:.2f}%",
            'Stasis': d['stasis'],
            'Dir': d.get('direction') or '—',
            'SMS': d.get('sms', 0),
            'FMS': d.get('fms', 0),
            'TMS': d.get('tms', 0),
            'Rev5': fmt_slope(sd.get('Rev_5')),
            'FCF5': fmt_slope(sd.get('FCF_5')),
            'FCFY': f"{sd['FCFY']*100:.1f}%" if sd.get('FCFY') else '—',
            '52W': f"{w52:.0f}%" if w52 is not None else '—',
            'Price': f"${d['current_price']:.2f}" if d.get('current_price') else '—',
            'TP': f"${d['take_profit']:.2f}" if d.get('take_profit') else '—',
            'SL': f"${d['stop_loss']:.2f}" if d.get('stop_loss') else '—',
            'RR': fmt_rr(d.get('risk_reward')),
            'Dur': d.get('stasis_duration_str', '—'),
            # Hidden sort values
            '_tms': d.get('tms', 0),
            '_fms': d.get('fms', 0),
            '_stasis': d['stasis'],
            '_52w': w52 if w52 is not None else 999,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return []

    if fsort == 'tms':
        df = df.sort_values('_tms', ascending=False)
    elif fsort == 'fms':
        df = df.sort_values('_fms', ascending=False)
    elif fsort == 'stasis':
        df = df.sort_values('_stasis', ascending=False)
    elif fsort == '52w':
        df = df.sort_values('_52w', ascending=True)

    df = df.head(150)
    df = df.drop(columns=['_tms', '_fms', '_stasis', '_52w'], errors='ignore')
    return df.to_dict('records')

# ============================================================================
# INITIALIZATION
# ============================================================================

_init_done = False
_init_lock = threading.Lock()

def initialize():
    global _init_done
    with _init_lock:
        if _init_done:
            return
        print("=" * 70)
        print("  STASIS PM | STASIS AM")
        print("  BEYOND PRICE AND TIME")
        print("  © 2026 Truth Communications LLC")
        print("=" * 70)
        print(f"\n🎯 Symbols: {len(config.symbols)}")

        print("\n📅 52-WEEK DATA...")
        config.week52_data = fetch_52_week_data()

        print("📊 VOLUME DATA...")
        config.volumes = fetch_volume_data()

        print("📈 FUNDAMENTAL DATA...")
        fetch_all_fundamental_data()

        print("📜 BACKFILL...")
        manager.backfill()

        print("🚀 STARTING FEEDS...")
        price_feed.start()
        manager.start()

        print(f"\n✅ READY — {len(config.fundamental_slopes)} fundamentals loaded")
        print("=" * 70)
        _init_done = True

_init_thread = threading.Thread(target=initialize, daemon=True)
_init_thread.start()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    _init_thread.join()
    print("\n🟢 http://127.0.0.1:8050\n")
    threading.Thread(target=lambda: (time.sleep(2), webbrowser.open('http://127.0.0.1:8050')),
                     daemon=True).start()
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)

"""
#    GKD_Galaxy_v1
# ═══════════════════════════════════════════════════════════════════════════════
# 
# CHANGELOG:
# ═══════════════════════════════════════════════════════════════════════════════
#
# ⚠️ TESTING NOTICE:
# * custom_stoploss and custom_exit, together with emergency exits have been
#   disabled or relaxed, in order to faciliate testing of the trade adjusting system.
#
# 🆕 TRADE ADJUSTMENT SYSTEM (Powered by Teodor):
# * State Machine for Trade Management:
#   - INITIAL: Monitors early trade performance
#   - DEFENDING: Uses progressive DCA to recover losing positions
#   - PROFIT: Uses progressive partial take-profits and position reloading
#   - MAX_DCA: Handles trades that reached max DCA with de-risking
#
# 🆕 CAPITAL MANAGEMENT & DE-RISKING (Powered by Teodor):
# * Dynamic Wallet Exposure Checks: Prevents over-allocation before DCA
# * Emergency Capital Release: Frees up capital from profitable trades if needed
# * De-risking Logic: Reduces exposure on "stuck" losing trades
# * Low Stake Replenishment: Automatically tops up "dust" trades (< $6) with
#   75% of INITIAL stake to make them tradeable
#
# 🛡️ LOSS PROTECTION SYSTEM:
# * PairLossMemory: Tracks last 5 trades per pair
# * Consecutive Loss Blocking: Blocks entries after 3+ consecutive losses
# * Large Loss Protection: Reduces leverage to 1x after >10% loss
# * Leverage Reduction System (4 mechanisms)
# * Entry Blocking System
# * Per-Pair Loss Tracking
#
# 📊 RISK MANAGEMENT IMPROVEMENTS:
# * Quality Threshold: Stricter entry criteria
# * Learning Frequency: Faster adaptation to market conditions
# * Less overtrading, higher quality setups
#
# 🔬 AUTO-OPTUNA OPTIMIZATION:
# * Initial optimization after 10 trades
# * Re-optimization every 50 trades
# * Automatic parameter tuning including stop-loss levels
#
# 🌡️ REGIME PROTECTION:
# * Quality threshold adjustment in counter-trend conditions
# * Confirmed pattern requirement in bad regimes
# * Prevents false entries against strong trends
#
# ✅ INHERITED FEATURES:
# * Emergency Exit Loop Fix
# * Slope-Based Divergence Detection
# * Flag & Triangle Pattern Detection
# * Flash Crash Protection
#
# ═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import talib.abstract as ta
import talib
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
import optuna
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict, deque

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter, informative, merge_informative_pair
from freqtrade.persistence import Trade, Order
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# V72 PHASE 5: PAIR LOSS MEMORY SYSTEM (from V63)
# ═══════════════════════════════════════════════════════════════════════════════

class PairLossMemory:
    """
    V72 LOSS PROTECTION SYSTEM (from V63 - proven in community test)
    
    Tracks per-pair trading performance to prevent:
    1. Revenge trading after consecutive losses
    2. Excessive leverage on struggling pairs
    3. Overtrading on historically bad pairs
    
    Community Test Result: V63 with this system SURVIVED "greatest market manipulation"
    while V64/V65 without it CRASHED immediately.
    
    MECHANISMS:
    -----------
    1. LOSS MEMORY: Tracks last 5 trades per pair
    2. CONSECUTIVE LOSS COUNTER: Increments on losses, resets on wins
    3. TOTAL LOSS ACCUMULATOR: Sum of recent losses
    4. TIMESTAMP TRACKING: For cooldown period calculations
    
    PROTECTION ACTIONS:
    ------------------
    1. ENTRY BLOCKING (3 scenarios):
       - After 3+ consecutive losses → Wait for win
       - After >12% total loss → Needs reset
       - 30min cooldown after 2 losses if last >8%
    
    2. LEVERAGE REDUCTION (4 mechanisms):
       - Large Loss (>10%): Leverage → 1x
       - Consecutive Losses: 2 losses = Max 2x, 3+ losses = Max 1x
       - Average Loss Scaling: Avg loss >5% = Max 2x
       - Cooldown Period: Max 2x for 60min after >8% loss
    """
    
    def __init__(self):
        self.pair_loss_memory = defaultdict(lambda: {
            'last_5_trades': deque(maxlen=5),  # Last 5 profit/loss values
            'consecutive_losses': 0,            # Counter of consecutive losses
            'total_loss': 0.0,                  # Sum of losses in last 5 trades
            'last_loss_time': None              # Timestamp of last loss
        })
    
    def record_trade(self, pair: str, profit: float):
        """
        Record a trade result for this pair
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT:USDT')
            profit: Profit/loss as decimal (e.g., 0.05 = +5%, -0.08 = -8%)
        """
        memory = self.pair_loss_memory[pair]
        
        # Add to last 5 trades
        memory['last_5_trades'].append(profit)
        
        # Update consecutive losses counter
        if profit < 0:
            memory['consecutive_losses'] += 1
            memory['last_loss_time'] = datetime.now()
        else:
            memory['consecutive_losses'] = 0  # Reset on win
        
        # Calculate total loss (sum of negative trades only)
        memory['total_loss'] = sum(p for p in memory['last_5_trades'] if p < 0)
    
    def should_block_entry(self, pair: str) -> Tuple[bool, str]:
        """
        Check if entry should be blocked for this pair
        
        Returns:
            (should_block: bool, reason: str)
        """
        memory = self.pair_loss_memory[pair]
        
        # BLOCK REASON 1: 3+ consecutive losses
        if memory['consecutive_losses'] >= 3:
            return True, f"❌ BLOCKED: 3+ consecutive losses (current: {memory['consecutive_losses']})"
        
        # BLOCK REASON 2: >12% total loss in last 5 trades
        if memory['total_loss'] < -0.12:
            return True, f"❌ BLOCKED: Total loss {memory['total_loss']:.1%} > 12% threshold"
        
        # BLOCK REASON 3: Cooldown after 2 consecutive losses if last >8%
        if memory['consecutive_losses'] >= 2:
            if memory['last_5_trades'] and memory['last_5_trades'][-1] < -0.08:
                if memory['last_loss_time']:
                    minutes_since = (datetime.now() - memory['last_loss_time']).seconds / 60
                    if minutes_since < 30:
                        return True, f"⏸️  COOLDOWN: {30-minutes_since:.0f}min left (2 losses, last >8%)"
        
        return False, ""
    
    def get_adjusted_leverage(self, pair: str, base_leverage: int, ml_conf: float = 0.5) -> int:
        """
        Get leverage adjusted for recent losses
        
        Args:
            pair: Trading pair
            base_leverage: Base leverage from config (e.g., 3)
            ml_conf: ML confidence (0.0-1.0)
        
        Returns:
            adjusted_leverage: 1-5x based on loss history
        """
        memory = self.pair_loss_memory[pair]
        
        # MECHANISM 1: Large Loss Protection (>10% loss)
        if memory['last_5_trades'] and any(loss < -0.10 for loss in memory['last_5_trades']):
            logger.warning(f"⚠️ {pair}: Large loss detected (>10%) → Leverage reduced to 1x")
            return 1
        
        # MECHANISM 2: Consecutive Loss Protection
        if memory['consecutive_losses'] >= 3:
            logger.warning(f"⚠️ {pair}: 3+ consecutive losses → Leverage reduced to 1x")
            return 1
        elif memory['consecutive_losses'] >= 2:
            logger.warning(f"⚠️ {pair}: 2 consecutive losses → Leverage capped at 2x")
            return min(base_leverage, 2)
        
        # MECHANISM 3: Average Loss Scaling
        if len(memory['last_5_trades']) >= 3:
            losses = [l for l in memory['last_5_trades'] if l < 0]
            if losses:
                avg_loss = np.mean(losses)
                if avg_loss < -0.05:  # Average loss >5%
                    logger.warning(f"⚠️ {pair}: Avg loss {avg_loss:.1%} >5% → Leverage capped at 2x")
                    return min(base_leverage, 2)
        
        # MECHANISM 4: Cooldown Period (60min after >8% loss)
        if memory['last_5_trades'] and memory['last_5_trades'][-1] < -0.08:
            if memory['last_loss_time']:
                minutes_since = (datetime.now() - memory['last_loss_time']).seconds / 60
                if minutes_since < 60:
                    logger.info(f"⏸️  {pair}: Recent >8% loss → Leverage capped at 2x "
                              f"({60-minutes_since:.0f}min cooldown left)")
                    return min(base_leverage, 2)
        
        # NO REDUCTION: Use base leverage (but still respect ML confidence)
        return base_leverage
    
    def get_stats(self, pair: str) -> Dict[str, Any]:
        """Get statistics for a pair"""
        memory = self.pair_loss_memory[pair]
        
        return {
            'pair': pair,
            'trades_count': len(memory['last_5_trades']),
            'consecutive_losses': memory['consecutive_losses'],
            'total_loss': memory['total_loss'],
            'last_loss_time': memory['last_loss_time'],
            'avg_profit': np.mean(list(memory['last_5_trades'])) if memory['last_5_trades'] else 0.0,
            'win_rate': sum(1 for p in memory['last_5_trades'] if p > 0) / len(memory['last_5_trades']) if memory['last_5_trades'] else 0.0
        }
    
    def reset_pair(self, pair: str):
        """Reset loss memory for a pair (e.g., after winning trade)"""
        if pair in self.pair_loss_memory:
            del self.pair_loss_memory[pair]
            logger.info(f"🔄 {pair}: Loss memory reset")


# ═══════════════════════════════════════════════════════════════════════════════
# SMART LOGGER - Clean, structured logging
# ═══════════════════════════════════════════════════════════════════════════════

class SmartLogger:
    """
    Intelligent logging system for V63.
    Summarizes training/optimization clearly.
    Shows only relevant information.
    """
    
    @staticmethod
    def log_trade_result(pair: str, profit: float, exit_reason: str, 
                        ml_conf: float, patterns: List[str], duration_min: int):
        """Compact trade summary"""
        emoji = "✅ WIN" if profit > 0 else "❌ LOSS"
        pattern_str = f"[{','.join(patterns[:2])}]" if patterns else "[No Pattern]"
        
        logger.info(
            f"{emoji} │ {pair:15s} │ P/L: {profit:+7.2%} │ "
            f"Exit: {exit_reason:20s} │ ML: {ml_conf:.0%} │ "
            f"Patterns: {pattern_str:30s} │ {duration_min:4.0f}min"
        )
    
    @staticmethod
    def log_parameter_update(target: str, reason: str, changes: Dict[str, Tuple[float, float]]):
        """Display parameter changes in a compact way"""
        logger.info("╔" + "═"*78 + "╗")
        logger.info(f"║ 🔄 PARAMETER UPDATE: {target:60s} ║")
        logger.info("╠" + "═"*78 + "╣")
        logger.info(f"║ Reason: {reason:69s} ║")
        logger.info("╠" + "═"*78 + "╣")
        
        for param, (old, new) in changes.items():
            change = new - old
            arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            logger.info(f"║ {arrow} {param:25s}: {old:7.4f} → {new:7.4f} ({change:+7.4f})      ║")
        
        logger.info("╚" + "═"*78 + "╝")
    
    @staticmethod
    def log_learning_summary(pair: str, trades_count: int, win_rate: float, 
                           avg_profit: float, best_patterns: List[Tuple[str, float]]):
        """Show learning progress per coin"""
        logger.info("┏" + "━"*78 + "┓")
        logger.info(f"┃ 📊 LEARNING SUMMARY: {pair:59s} ┃")
        logger.info("┣" + "━"*78 + "┫")
        logger.info(f"┃ Total Trades: {trades_count:3d} │ Win Rate: {win_rate:5.1%} │ Avg P/L: {avg_profit:+8.4f} {'    ':20s} ┃")
        
        if best_patterns:
            logger.info("┣" + "━"*78 + "┫")
            logger.info(f"┃ 🏆 Best Patterns (Top 3):                                                  ┃")
            for i, (pattern, reliability) in enumerate(best_patterns[:3], 1):
                logger.info(f"┃    {i}. {pattern:30s} - {reliability:5.1%} reliability                 ┃")
        
        logger.info("┗" + "━"*78 + "┛")
    
    @staticmethod
    def log_optuna_optimization(n_trials: int, best_score: float, 
                               key_improvements: Dict[str, float]):
        """Show Optuna optimization results concisely"""
        logger.info("╔" + "═"*78 + "╗")
        logger.info(f"║ 🔬 OPTUNA OPTIMIZATION COMPLETE                                             ║")
        logger.info("╠" + "═"*78 + "╣")
        logger.info(f"║ Trials Run: {n_trials:3d} │ Best Score: {best_score:+8.4f}                               ║")
        logger.info("╠" + "═"*78 + "╣")
        logger.info(f"║ 📈 Key Improvements:                                                        ║")
        
        for param, value in list(key_improvements.items())[:5]:
            logger.info(f"║    {param:30s}: {value:7.4f}                              ║")
        
        logger.info("╚" + "═"*78 + "╝")
    
    @staticmethod
    def log_bad_parameters_detected(win_rate: float, avg_loss: float, action: str):
        """Warning when bad parameters are detected"""
        logger.warning("╔" + "═"*78 + "╗")
        logger.warning(f"║ 🚨 BAD PARAMETERS DETECTED!                                                ║")
        logger.warning("╠" + "═"*78 + "╣")
        logger.warning(f"║ Current Win Rate: {win_rate:5.1%} (Target: >40%)                                   ║")
        logger.warning(f"║ Average Loss: {avg_loss:+8.2f} USD (Threshold: -50 USD)                         ║")
        logger.warning("╠" + "═"*78 + "╣")
        logger.warning(f"║ Action: {action:67s} ║")
        logger.warning("╚" + "═"*78 + "╝")
    
    @staticmethod
    def log_hourly_status(total_trades: int, open_trades: int, win_rate: float,
                         total_profit: float, top_pairs: List[Tuple[str, float]]):
        """Hourly status summary"""
        logger.info("┏" + "━"*78 + "┓")
        logger.info(f"┃ ⏰ HOURLY STATUS UPDATE                                                     ┃")
        logger.info("┣" + "━"*78 + "┫")
        logger.info(f"┃ Total: {total_trades:3d} │ Open: {open_trades:2d} │ WR: {win_rate:5.1%} │ P/L: {total_profit:+10.2f} USDT         ┃")
        
        if top_pairs:
            logger.info("┣" + "━"*78 + "┫")
            logger.info(f"┃ 🏆 Top Performers:                                                         ┃")
            for pair, profit in top_pairs[:3]:
                logger.info(f"┃    {pair:15s}: {profit:+8.2f} USDT                                         ┃")
        
        logger.info("┗" + "━"*78 + "┛")


# ═══════════════════════════════════════════════════════════════════════════════
# FLASH CRASH PROTECTION - Intelligent Multi-Timeframe Crash Detection
# ═══════════════════════════════════════════════════════════════════════════════

class FlashCrashProtection:
    """
    Intelligent Flash Crash Protection using Multi-Timeframe Analysis + ATR-based Volatility.
    
    FEATURES:
    ✅ Multi-Timeframe Detection: Uses 1h/4h informatives (not just 15m close)
    ✅ ATR-Based Thresholds: Dynamic crash detection based on volatility (not fixed -5%)
    ✅ Wick Analysis: Checks lows (wicks) not just closes (catches flash crashes better)
    ✅ Smart Cooldown: 5-15 min cooldown after crash detected
    ✅ Partial Block: Only blocks LAST BAR entries (not entire dataframe)
    ✅ Exit Signal: Tags exits with 'flash_crash' for tracking
    
    IMPLEMENTATION:
    * Inspired by community feedback (fix NaN issues, use real MTF, dynamic thresholds)
    * Addresses V67 issues: Prevents entries during extreme volatility spikes
    * Expected Impact: -10-20% reduction in bad entries during dumps
    """
    
    def __init__(self):
        self.last_crash_time: Dict[str, datetime] = {}  # Per-pair crash timestamps
        self.crash_cooldown_minutes = 10  # Cooldown after crash detection
        self.min_atr_multiplier = 3.0  # 3x ATR = extreme move
        self.max_atr_multiplier = 5.0  # 5x ATR = flash crash
        
        logger.info("🛡️ FlashCrashProtection initialized (MTF + ATR-based)")
    
    def detect_flash_crash(self, df: DataFrame, pair: str, timeframe: str = '15m') -> Tuple[bool, str]:
        """
        Detect flash crash using Multi-Timeframe Wick Analysis + ATR volatility.
        
        Returns: (is_crash: bool, reason: str)
        """
        if len(df) < 3:
            return False, ""
        
        # Get latest row
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # ═══════════════════════════════════════════════════════════════
        # 1. ATR-BASED CRASH DETECTION (Dynamic, not fixed %)
        # ═══════════════════════════════════════════════════════════════
        atr_15m = last.get('atr', 0)
        if atr_15m == 0:
            return False, ""  # No ATR data yet
        
        # Check 15m candle wick (low vs close/open)
        candle_low = last['low']
        candle_body_min = min(last['open'], last['close'])
        wick_size_15m = candle_body_min - candle_low
        
        # Flash crash = wick is 3-5x ATR (extreme volatility spike)
        if wick_size_15m > atr_15m * self.max_atr_multiplier:
            return True, f"15m Flash Crash: {wick_size_15m/atr_15m:.1f}x ATR wick"
        
        # ═══════════════════════════════════════════════════════════════
        # 2. MULTI-TIMEFRAME CRASH DETECTION (1h/4h informatives)
        # ═══════════════════════════════════════════════════════════════
        
        # Check 1h timeframe crash (if available)
        if 'close_1h' in last and 'low_1h' in last and 'atr_1h' in last:
            atr_1h = last['atr_1h']
            if atr_1h > 0:
                close_1h = last['close_1h']
                low_1h = last['low_1h']
                drop_1h_pct = (low_1h - close_1h) / close_1h
                
                # 1h crash = drop > 3x ATR (extreme move)
                drop_1h_atr = abs(low_1h - close_1h)
                if drop_1h_atr > atr_1h * self.min_atr_multiplier and drop_1h_pct < -0.03:
                    return True, f"1h Crash: {drop_1h_pct:.1%} ({drop_1h_atr/atr_1h:.1f}x ATR)"
        
        # Check 4h timeframe crash (if available)
        if 'close_4h' in last and 'low_4h' in last and 'atr_4h' in last:
            atr_4h = last['atr_4h']
            if atr_4h > 0:
                close_4h = last['close_4h']
                low_4h = last['low_4h']
                drop_4h_pct = (low_4h - close_4h) / close_4h
                
                # 4h crash = drop > 3x ATR (extreme trend shift)
                drop_4h_atr = abs(low_4h - close_4h)
                if drop_4h_atr > atr_4h * self.min_atr_multiplier and drop_4h_pct < -0.05:
                    return True, f"4h Crash: {drop_4h_pct:.1%} ({drop_4h_atr/atr_4h:.1f}x ATR)"
        
        # ═══════════════════════════════════════════════════════════════
        # 3. CONSECUTIVE BARS CRASH (Multiple bars with extreme wicks)
        # ═══════════════════════════════════════════════════════════════
        if len(df) >= 3:
            prev2 = df.iloc[-3]
            
            # Check if last 2-3 bars all have large wicks
            large_wick_count = 0
            for row in [prev2, prev, last]:
                if 'atr' in row and row['atr'] > 0:
                    candle_low_check = row['low']
                    candle_body_min_check = min(row['open'], row['close'])
                    wick_size = candle_body_min_check - candle_low_check
                    
                    if wick_size > row['atr'] * 2.5:  # 2.5x ATR = large wick
                        large_wick_count += 1
            
            # 2-3 consecutive bars with large wicks = sustained dump
            if large_wick_count >= 2:
                return True, f"Sustained Dump: {large_wick_count} consecutive large wicks"
        
        return False, ""
    
    def in_cooldown(self, pair: str) -> bool:
        """Check if pair is in cooldown period after crash"""
        if pair not in self.last_crash_time:
            return False
        
        elapsed = (datetime.now() - self.last_crash_time[pair]).total_seconds() / 60
        return elapsed < self.crash_cooldown_minutes
    
    def register_crash(self, pair: str):
        """Register that a crash was detected for this pair"""
        self.last_crash_time[pair] = datetime.now()
        logger.warning(f"🚨 Flash crash registered for {pair} - {self.crash_cooldown_minutes}min cooldown")
    
    def get_cooldown_remaining(self, pair: str) -> float:
        """Get remaining cooldown time in minutes"""
        if pair not in self.last_crash_time:
            return 0.0
        
        elapsed = (datetime.now() - self.last_crash_time[pair]).total_seconds() / 60
        remaining = max(0, self.crash_cooldown_minutes - elapsed)
        return remaining


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE PERFORMANCE TRACKER - Historical metrics storage
# ═══════════════════════════════════════════════════════════════════════════════

class TradePerformanceTracker:
    """
    Tracks ALL trade metrics historically for deep learning.
    Stores: entry conditions, exit reasons, market state, performance
    """
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = storage_dir / "trade_metrics.json"
        self.summary_file = storage_dir / "performance_summary.json"
        
        self.trades = self._load_trades()
        self.performance_cache = deque(maxlen=100)  # Last 100 trades for quick access
        
        # 🔥 SAVE immediately after initialization (create file structure)
        self._save()
        
        logger.info(f"📊 TradePerformanceTracker: {len(self.trades)} historical trades loaded")
    
    def _load_trades(self) -> List[Dict]:
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def record_trade(self, trade_data: Dict):
        """Record a completed trade with full metrics"""
        trade_data['timestamp'] = datetime.now().isoformat()
        self.trades.append(trade_data)
        self.performance_cache.append(trade_data)
        
        # Save every 5 trades
        if len(self.trades) % 5 == 0:
            self._save()
    
    def _save(self):
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.trades, f, indent=2)
            
            # Calculate and save summary
            summary = self.calculate_summary()
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade metrics: {e}")
    
    def calculate_summary(self) -> Dict:
        """Calculate performance summary for Optuna"""
        if not self.trades:
            return {
                'total_trades': 0, 
                'recent_trades': 0,
                'win_rate': 0.5, 
                'avg_profit': 0.0,
                'exit_reasons': {},
                'last_updated': datetime.now().isoformat()
            }
        
        recent = self.trades[-50:]  # Last 50 trades
        
        wins = sum(1 for t in recent if t.get('profit', 0) > 0)
        total = len(recent)
        avg_profit = np.mean([t.get('profit', 0) for t in recent])
        
        # By exit reason
        exit_reasons = defaultdict(lambda: {'count': 0, 'profit': 0})
        for t in recent:
            reason = t.get('exit_reason', 'unknown')
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['profit'] += t.get('profit', 0)
        
        return {
            'total_trades': len(self.trades),
            'recent_trades': total,
            'win_rate': wins / total if total > 0 else 0.5,
            'avg_profit': float(avg_profit),
            'exit_reasons': dict(exit_reasons),
            'last_updated': datetime.now().isoformat()
        }
    
    def detect_bad_parameters(self) -> bool:
        """Detect if current parameters are performing badly"""
        if len(self.performance_cache) < 10:
            return False
        
        recent_10 = list(self.performance_cache)[-10:]
        wins = sum(1 for t in recent_10 if t.get('profit', 0) > 0)
        win_rate = wins / 10
        avg_profit = np.mean([t.get('profit', 0) for t in recent_10])
        
        # Trigger retraining if:
        # 1. Win rate < 30% in last 10 trades
        # 2. OR average loss > -50 USD
        if wins < 3 or avg_profit < -50:
            SmartLogger.log_bad_parameters_detected(
                win_rate=win_rate,
                avg_loss=avg_profit,
                action="Triggering Optuna retraining with 30 trials"
            )
            return True
        
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# BULKOWSKI PATTERN SYSTEM - Pattern-specific performance tracking
# ═══════════════════════════════════════════════════════════════════════════════

class BulkowskiPatternSystem:
    """
    Implements Bulkowski's pattern analysis with dynamic reliability learning.
    Each pattern's performance is tracked and reliability scores are updated.
    """
    
    # Initial reliability from Bulkowski research (will be updated from actual trades)
    INITIAL_RELIABILITY = {
        # High Reliability (>70%)
        'CDL3WHITESOLDIERS': 0.78,
        'CDL3BLACKCROWS': 0.78,
        'CDLMORNINGSTAR': 0.76,
        'CDLEVENINGSTAR': 0.72,
        
        # Good Reliability (60-70%)
        'CDLENGULFING': 0.63,
        'CDLHAMMER': 0.60,
        'CDLHANGINGMAN': 0.59,
        
        # Medium Reliability (50-60%)
        'CDLPIERCING': 0.56,
        'CDLDARKCLOUDCOVER': 0.54,
        'CDLHARAMI': 0.51,
        'CDLDOJI': 0.51,
    }
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.performance_file = storage_dir / "pattern_performance.json"
        self.pair_pattern_file = storage_dir / "pattern_performance_per_pair.json"
        
        # Load or initialize pattern performance
        self.pattern_stats = self._load_pattern_stats()  # Global stats
        self.pair_pattern_stats = self._load_pair_pattern_stats()  # Per-pair stats
        
        # 🔥 SAVE immediately after initialization
        self._save()
        
        logger.info(f"🕯️  Bulkowski Pattern System: {len(self.pattern_stats)} patterns tracked")
    
    def _load_pattern_stats(self) -> Dict:
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Initialize with Bulkowski's research
        return {
            pattern: {
                'reliability': score,
                'trades': 0,
                'wins': 0,
                'total_profit': 0.0,
                'learned_reliability': score  # Will evolve
            }
            for pattern, score in self.INITIAL_RELIABILITY.items()
        }
    
    def analyze_patterns(self, dataframe: DataFrame) -> DataFrame:
        """
        Detect patterns with BULKOWSKI CONFIRMATION CHECKS
        Source: Thomas Bulkowski - Encyclopedia of Candlestick Charts (2008)
        """
        df = dataframe.copy()
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🎯 BULKOWSKI CONFIRMATION REQUIREMENTS
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # 1. VOLUME CONFIRMATION (Bulkowski: "Volume validates patterns")
        volume_avg = df['volume'].rolling(20).mean()
        volume_high = df['volume'] > volume_avg * 1.3  # Strong volume
        volume_normal = df['volume'] > volume_avg * 1.1  # Normal confirmation
        
        # 2. TREND CONTEXT (Reversal patterns need existing trend!)
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        in_uptrend = (df['close'] > sma_20) & (sma_20 > sma_50)
        in_downtrend = (df['close'] < sma_20) & (sma_20 < sma_50)
        
        # 3. MOMENTUM (for pattern strength)
        momentum_5 = df['close'].pct_change(5)
        
        # Detect all patterns
        detected_patterns = {}
        for pattern in self.INITIAL_RELIABILITY.keys():
            try:
                detected_patterns[pattern] = getattr(ta, pattern)(df)
                df[f'pattern_{pattern}'] = detected_patterns[pattern]
            except:
                detected_patterns[pattern] = pd.Series(0, index=df.index)
                df[f'pattern_{pattern}'] = 0
        
        # Calculate weighted strength with BULKOWSKI CONFIRMATION
        weighted_bullish = pd.Series(0.0, index=df.index)
        weighted_bearish = pd.Series(0.0, index=df.index)
        
        # Track confirmed vs unconfirmed patterns
        confirmed_bullish = pd.Series(0.0, index=df.index)
        confirmed_bearish = pd.Series(0.0, index=df.index)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🎯 APPLY BULKOWSKI CONFIRMATION LOGIC
        # ═══════════════════════════════════════════════════════════════════════════════
        
        for pattern, signal in detected_patterns.items():
            base_reliability = self.pattern_stats[pattern]['learned_reliability']
            
            # Bullish patterns
            is_bullish = signal > 0
            if is_bullish.any():
                # BULKOWSKI RULE: Bullish reversal needs downtrend + volume
                confirmation_multiplier = 1.0
                
                # High-priority patterns (Bulkowski p.78-245)
                if pattern in ['CDL3WHITESOLDIERS', 'CDLMORNINGSTAR', 'CDLENGULFING']:
                    # MUST have: Volume + Downtrend context
                    confirmed = volume_high & in_downtrend
                    confirmation_multiplier = confirmed.astype(float) * 1.5 + (~confirmed).astype(float) * 0.3
                
                # Medium-priority (need trend OR volume)
                elif pattern in ['CDLHAMMER', 'CDLPIERCING']:
                    confirmed = (volume_normal & in_downtrend) | (volume_high)
                    confirmation_multiplier = confirmed.astype(float) * 1.2 + (~confirmed).astype(float) * 0.5
                
                # Low-priority (context dependent)
                else:
                    confirmed = volume_normal | in_downtrend
                    confirmation_multiplier = confirmed.astype(float) * 1.0 + (~confirmed).astype(float) * 0.6
                
                # Apply weighted score
                pattern_score = is_bullish.astype(float) * base_reliability * confirmation_multiplier
                weighted_bullish += pattern_score
                confirmed_bullish += (pattern_score > base_reliability).astype(float)
            
            # Bearish patterns
            is_bearish = signal < 0
            if is_bearish.any():
                # BULKOWSKI RULE: Bearish reversal needs uptrend + volume
                confirmation_multiplier = 1.0
                
                # High-priority patterns
                if pattern in ['CDL3BLACKCROWS', 'CDLEVENINGSTAR', 'CDLENGULFING']:
                    # MUST have: Volume + Uptrend context
                    confirmed = volume_high & in_uptrend
                    confirmation_multiplier = confirmed.astype(float) * 1.5 + (~confirmed).astype(float) * 0.3
                
                # Medium-priority
                elif pattern in ['CDLHANGINGMAN', 'CDLDARKCLOUDCOVER', 'CDLSHOOTINGSTAR']:
                    confirmed = (volume_normal & in_uptrend) | (volume_high)
                    confirmation_multiplier = confirmed.astype(float) * 1.2 + (~confirmed).astype(float) * 0.5
                
                # Low-priority
                else:
                    confirmed = volume_normal | in_uptrend
                    confirmation_multiplier = confirmed.astype(float) * 1.0 + (~confirmed).astype(float) * 0.6
                
                # Apply weighted score
                pattern_score = is_bearish.astype(float) * base_reliability * confirmation_multiplier * -1
                weighted_bearish += abs(pattern_score)
                confirmed_bearish += (abs(pattern_score) > base_reliability).astype(float)
        
        # Store results
        df['pattern_weighted_bullish'] = weighted_bullish
        df['pattern_weighted_bearish'] = weighted_bearish
        df['pattern_net_strength'] = weighted_bullish - weighted_bearish
        
        # NEW: Bulkowski-confirmed pattern counts
        df['pattern_confirmed_bullish'] = confirmed_bullish
        df['pattern_confirmed_bearish'] = confirmed_bearish
        df['pattern_confirmation_quality'] = (confirmed_bullish + confirmed_bearish) / (
            (weighted_bullish > 0).astype(float) + (weighted_bearish > 0).astype(float) + 0.001
        )
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🎯 FIX: Optimize pattern detection for 5m (less restrictive)
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Find best CONFIRMED pattern at each candle
        df['best_bullish_pattern'] = ''
        df['best_bearish_pattern'] = ''
        
        # Add pattern detection counters for debugging
        total_patterns = 0
        confirmed_patterns = 0
        
        for i in range(len(df)):
            bullish_candidates = [(p, self.pattern_stats[p]['learned_reliability']) 
                                 for p, s in detected_patterns.items() if s.iloc[i] > 0]
            bearish_candidates = [(p, self.pattern_stats[p]['learned_reliability']) 
                                 for p, s in detected_patterns.items() if s.iloc[i] < 0]
            
            if bullish_candidates:
                best = max(bullish_candidates, key=lambda x: x[1])
                df.loc[df.index[i], 'best_bullish_pattern'] = best[0]
                total_patterns += 1
                # Check if this pattern was confirmed
                if i < len(confirmed_bullish) and confirmed_bullish.iloc[i] > 0:
                    confirmed_patterns += 1
            
            if bearish_candidates:
                best = max(bearish_candidates, key=lambda x: x[1])
                df.loc[df.index[i], 'best_bearish_pattern'] = best[0]
                total_patterns += 1
                # Check if this pattern was confirmed
                if i < len(confirmed_bearish) and confirmed_bearish.iloc[i] > 0:
                    confirmed_patterns += 1
        
        # Store pattern counts for debugging
        df['pattern_total_count'] = total_patterns
        df['pattern_confirmed_count'] = confirmed_patterns
        
        return df
    
    def update_pattern_performance(self, patterns: List[str], profit: float, pair: str = None):
        """
        Update pattern statistics from trade outcome
        NEW: Updates BOTH global AND per-pair statistics!
        """
        for pattern in patterns:
            if pattern in self.pattern_stats:
                # Update GLOBAL stats
                self.pattern_stats[pattern]['trades'] += 1
                if profit > 0:
                    self.pattern_stats[pattern]['wins'] += 1
                self.pattern_stats[pattern]['total_profit'] += profit
                
                # Update learned reliability (Bayesian update)
                trades = self.pattern_stats[pattern]['trades']
                wins = self.pattern_stats[pattern]['wins']
                
                if trades >= 5:  # Need at least 5 trades
                    # Bayesian blend: 80% actual + 20% prior
                    actual_rate = wins / trades
                    prior_rate = self.pattern_stats[pattern]['reliability']
                    self.pattern_stats[pattern]['learned_reliability'] = actual_rate * 0.8 + prior_rate * 0.2
                
                # ═══════════════════════════════════════════════════════════════════════════════
                # 🎯 NEW: PER-PAIR PATTERN LEARNING
                # ═══════════════════════════════════════════════════════════════════════════════
                if pair:
                    if pair not in self.pair_pattern_stats:
                        self.pair_pattern_stats[pair] = {}
                    
                    if pattern not in self.pair_pattern_stats[pair]:
                        self.pair_pattern_stats[pair][pattern] = {
                            'trades': 0,
                            'wins': 0,
                            'total_profit': 0.0,
                            'learned_reliability': self.pattern_stats[pattern]['reliability']
                        }
                    
                    # Update pair-specific stats
                    self.pair_pattern_stats[pair][pattern]['trades'] += 1
                    if profit > 0:
                        self.pair_pattern_stats[pair][pattern]['wins'] += 1
                    self.pair_pattern_stats[pair][pattern]['total_profit'] += profit
                    
                    # Update pair-specific learned reliability
                    pair_trades = self.pair_pattern_stats[pair][pattern]['trades']
                    pair_wins = self.pair_pattern_stats[pair][pattern]['wins']
                    
                    if pair_trades >= 3:  # Need at least 3 trades per pair
                        pair_actual = pair_wins / pair_trades
                        pair_prior = self.pattern_stats[pattern]['reliability']
                        self.pair_pattern_stats[pair][pattern]['learned_reliability'] = (
                            pair_actual * 0.7 + pair_prior * 0.3  # 70% actual, 30% prior
                        )
        
        self._save()
    
    def _load_pair_pattern_stats(self) -> Dict:
        """Load per-pair pattern statistics"""
        if self.pair_pattern_file.exists():
            try:
                with open(self.pair_pattern_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save(self):
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(self.pattern_stats, f, indent=2)
            with open(self.pair_pattern_file, 'w') as f:
                json.dump(self.pair_pattern_stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving pattern stats: {e}")
    
    def get_pattern_confidence(self, patterns: List[str], pair: str = None) -> float:
        """
        Get combined confidence from multiple patterns
        NEW: Uses per-pair reliability if available!
        """
        if not patterns:
            return 0.5
        
        reliabilities = []
        for p in patterns:
            if p in self.pattern_stats:
                # Try pair-specific first (if enough data)
                if pair and pair in self.pair_pattern_stats:
                    if p in self.pair_pattern_stats[pair]:
                        pair_stats = self.pair_pattern_stats[pair][p]
                        if pair_stats['trades'] >= 3:
                            # Use pair-specific reliability
                            reliabilities.append(pair_stats['learned_reliability'])
                            continue
                
                # Fallback to global
                reliabilities.append(self.pattern_stats[p]['learned_reliability'])
        
        return np.mean(reliabilities) if reliabilities else 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# V69 NEW: PATTERN RECOGNITION ENGINE - Entry Pattern Detection
# ═══════════════════════════════════════════════════════════════════════════════

class PatternRecognitionEngine:
    """
    V69 NEW: Pattern recognition for entry signals
    
    UPTREND PATTERNS:
    * Bull Flag: Strong rise → consolidation → breakout up
    * Ascending Triangle: Higher lows + resistance → Breakout
    * Cup & Handle: U-shaped base + small consolidation → breakout
    
    DOWNTREND PATTERNS:
    * Bear Flag: Strong drop → consolidation → breakdown
    * Descending Triangle: Lower highs + support → Breakdown
    * Head & Shoulders: Peak → higher peak → peak → breakdown
    
    FEATURES:
    * Pattern Confidence Scoring (0-100)
    * Per-Coin Success Rate Tracking
    * Volume Confirmation
    * Momentum Alignment Check
    """
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.stats_file = storage_dir / "pattern_stats.json"
        
        # Pattern Success Tracking
        self.pattern_stats = self._load_stats()
        
        logger.info(f"🎯 PatternRecognitionEngine: {len(self.pattern_stats)} patterns tracked")
    
    def _load_stats(self) -> Dict:
        """Load pattern statistics from disk"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_stats(self):
        """Save pattern statistics to disk"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.pattern_stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving pattern stats: {e}")
    
    def detect_bull_flag(self, df: DataFrame) -> Tuple[bool, float]:
        """
        Bull Flag Pattern Detection
        
        STRUCTURE:
        1. Strong rise (flagpole): >5% in 5-10 candles
        2. Consolidation (flag): Sideways/slightly down for 5-15 candles
        3. Breakout: Closes above flag high with volume
        
        Returns: (detected, confidence)
        """
        if len(df) < 30:
            return False, 0.0
        
        # Find flagpole (strong rise)
        lookback = 20
        for i in range(5, 15):
            pole_start = -lookback
            pole_end = -lookback + i
            pole_gain = ((df['close'].iloc[pole_end] / df['close'].iloc[pole_start]) - 1) * 100
            
            if pole_gain > 5:  # >5% rise = flagpole found
                # Check consolidation after flagpole
                flag_start = pole_end
                flag_end = -5
                flag_high = df['high'].iloc[flag_start:flag_end].max()
                flag_low = df['low'].iloc[flag_start:flag_end].min()
                flag_range = ((flag_high / flag_low) - 1) * 100
                
                # Flag should be tight (<3% range)
                if flag_range < 3:
                    # Check breakout
                    current_close = df['close'].iloc[-1]
                    if current_close > flag_high:
                        # Volume Confirmation
                        avg_volume = df['volume'].iloc[flag_start:flag_end].mean()
                        current_volume = df['volume'].iloc[-1]
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                        
                        # Confidence calculation
                        confidence = min(100, (
                            (pole_gain / 10) * 30 +  # Pole strength
                            (1 / flag_range) * 20 +   # Flag tightness
                            (volume_ratio - 1) * 50   # Volume spike
                        ))
                        
                        return True, max(50, min(confidence, 100))
        
        return False, 0.0
    
    def detect_bear_flag(self, df: DataFrame) -> Tuple[bool, float]:
        """
        Bear Flag Pattern Detection
        
        STRUCTURE:
        1. Strong drop (flagpole): >5% in 5-10 candles
        2. Consolidation (flag): Sideways/slightly up for 5-15 candles
        3. Breakdown: Closes below flag low with volume
        
        Returns: (detected, confidence)
        """
        if len(df) < 30:
            return False, 0.0
        
        # Find flagpole (strong drop)
        lookback = 20
        for i in range(5, 15):
            pole_start = -lookback
            pole_end = -lookback + i
            pole_drop = ((df['close'].iloc[pole_end] / df['close'].iloc[pole_start]) - 1) * 100
            
            if pole_drop < -5:  # >5% drop = flagpole found
                # Check consolidation after flagpole
                flag_start = pole_end
                flag_end = -5
                flag_high = df['high'].iloc[flag_start:flag_end].max()
                flag_low = df['low'].iloc[flag_start:flag_end].min()
                flag_range = ((flag_high / flag_low) - 1) * 100
                
                # Flag should be tight (<3% range)
                if flag_range < 3:
                    # Check breakdown
                    current_close = df['close'].iloc[-1]
                    if current_close < flag_low:
                        # Volume Confirmation
                        avg_volume = df['volume'].iloc[flag_start:flag_end].mean()
                        current_volume = df['volume'].iloc[-1]
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                        
                        # Confidence calculation
                        confidence = min(100, (
                            (abs(pole_drop) / 10) * 30 +  # Pole strength
                            (1 / flag_range) * 20 +         # Flag tightness
                            (volume_ratio - 1) * 50         # Volume spike
                        ))
                        
                        return True, max(50, min(confidence, 100))
        
        return False, 0.0
    
    def detect_ascending_triangle(self, df: DataFrame) -> Tuple[bool, float]:
        """
        Ascending Triangle Pattern
        
        STRUCTURE:
        * Higher lows (rising support)
        * Flat resistance (horizontal resistance)
        * Breakout above resistance
        
        Returns: (detected, confidence)
        """
        if len(df) < 30:
            return False, 0.0
        
        lookback = 20
        resistance_level = df['high'].iloc[-lookback:-5].max()
        
        # Check higher lows
        lows = df['low'].iloc[-lookback:-5]
        higher_lows = sum(1 for i in range(1, len(lows)) if lows.iloc[i] > lows.iloc[i-1])
        higher_lows_pct = higher_lows / (len(lows) - 1)
        
        # Check breakout
        current_close = df['close'].iloc[-1]
        breakout = current_close > resistance_level
        
        if breakout and higher_lows_pct > 0.5:
            # Volume Check
            avg_volume = df['volume'].iloc[-lookback:-5].mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            confidence = min(100, (
                higher_lows_pct * 50 +
                (volume_ratio - 1) * 50
            ))
            
            return True, max(50, confidence)
        
        return False, 0.0
    
    def detect_descending_triangle(self, df: DataFrame) -> Tuple[bool, float]:
        """
        Descending Triangle Pattern
        
        STRUCTURE:
        * Lower highs (falling resistance)
        * Flat support (horizontal support)
        * Breakdown below support
        
        Returns: (detected, confidence)
        """
        if len(df) < 30:
            return False, 0.0
        
        lookback = 20
        support_level = df['low'].iloc[-lookback:-5].min()
        
        # Check lower highs
        highs = df['high'].iloc[-lookback:-5]
        lower_highs = sum(1 for i in range(1, len(highs)) if highs.iloc[i] < highs.iloc[i-1])
        lower_highs_pct = lower_highs / (len(highs) - 1)
        
        # Check breakdown
        current_close = df['close'].iloc[-1]
        breakdown = current_close < support_level
        
        if breakdown and lower_highs_pct > 0.5:
            # Volume Check
            avg_volume = df['volume'].iloc[-lookback:-5].mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            confidence = min(100, (
                lower_highs_pct * 50 +
                (volume_ratio - 1) * 50
            ))
            
            return True, max(50, confidence)
        
        return False, 0.0
    
    def detect_all_patterns(self, df: DataFrame, direction: str = 'both') -> Dict:
        """
        Detect all patterns for given direction
        
        Args:
            direction: 'long', 'short', or 'both'
        
        Returns:
            {
                'bull_flag': (detected, confidence),
                'bear_flag': (detected, confidence),
                ...
            }
        """
        patterns = {}
        
        if direction in ['long', 'both']:
            patterns['bull_flag'] = self.detect_bull_flag(df)
            patterns['ascending_triangle'] = self.detect_ascending_triangle(df)
        
        if direction in ['short', 'both']:
            patterns['bear_flag'] = self.detect_bear_flag(df)
            patterns['descending_triangle'] = self.detect_descending_triangle(df)
        
        return patterns
    
    def get_best_pattern(self, patterns: Dict) -> Tuple[str, float]:
        """
        Get pattern with highest confidence
        
        Returns: (pattern_name, confidence)
        """
        best_pattern = None
        best_confidence = 0.0
        
        for pattern_name, (detected, confidence) in patterns.items():
            if detected and confidence > best_confidence:
                best_pattern = pattern_name
                best_confidence = confidence
        
        return best_pattern, best_confidence
    
    def record_pattern_result(self, pair: str, pattern: str, profit: float):
        """
        Record pattern success/failure per coin
        """
        if pair not in self.pattern_stats:
            self.pattern_stats[pair] = {}
        
        if pattern not in self.pattern_stats[pair]:
            self.pattern_stats[pair][pattern] = {
                'trades': 0,
                'winners': 0,
                'total_profit': 0.0
            }
        
        stats = self.pattern_stats[pair][pattern]
        stats['trades'] += 1
        stats['total_profit'] += profit
        if profit > 0:
            stats['winners'] += 1
        
        self._save_stats()
        
        win_rate = (stats['winners'] / stats['trades']) * 100
        logger.info(f"🎯 {pair} {pattern}: {win_rate:.0f}% WR ({stats['winners']}/{stats['trades']}), Total: {stats['total_profit']:.2f} USDT")
    
    def get_pattern_performance(self, pair: str, pattern: str) -> Dict:
        """
        Get pattern performance for specific coin
        
        Returns:
            {
                'trades': int,
                'win_rate': float,
                'avg_profit': float,
                'confidence_modifier': float  # 0.5-1.5
            }
        """
        if pair not in self.pattern_stats or pattern not in self.pattern_stats[pair]:
            return {
                'trades': 0,
                'win_rate': 50.0,
                'avg_profit': 0.0,
                'confidence_modifier': 1.0
            }
        
        stats = self.pattern_stats[pair][pattern]
        win_rate = (stats['winners'] / stats['trades']) * 100 if stats['trades'] > 0 else 50.0
        avg_profit = stats['total_profit'] / stats['trades'] if stats['trades'] > 0 else 0.0
        
        # Confidence Modifier basierend auf historischer Performance
        if stats['trades'] >= 5:
            if win_rate >= 70:
                confidence_modifier = 1.3
            elif win_rate >= 60:
                confidence_modifier = 1.1
            elif win_rate <= 40:
                confidence_modifier = 0.7
            elif win_rate <= 30:
                confidence_modifier = 0.5
            else:
                confidence_modifier = 1.0
        else:
            confidence_modifier = 1.0  # Nicht genug Daten
        
        return {
            'trades': stats['trades'],
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'confidence_modifier': confidence_modifier
        }


# ═══════════════════════════════════════════════════════════════════════════════
# V69 NEW: MARKET REGIME DETECTION - Trending vs Ranging vs Volatile
# ═══════════════════════════════════════════════════════════════════════════════

class MarketRegimeDetector:
    """
    V69 NEW: Detects the current market regime
    
    REGIMES:
    * TRENDING: Strong trend (ADX > 25, EMA alignment)
    * RANGING: Sideways market (ADX < 20, price between bands)
    * VOLATILE: High volatility (ATR > 3%, large candles)
    
    BENEFITS:
    * Different strategies for different regimes
    * Prevents trend-following signals in ranging markets
    * Prevents ranging signals in trending markets
    * Adaptive parameter adjustment
    """
    
    def __init__(self):
        self.current_regime = {}  # Per-pair regime
        logger.info("📊 MarketRegimeDetector initialized")
    
    def detect_regime(self, df: DataFrame, pair: str) -> Dict:
        """
        Detect current market regime
        
        Returns:
            {
                'regime': 'trending_up' | 'trending_down' | 'ranging' | 'volatile',
                'confidence': 0-100,
                'adx': float,
                'atr_pct': float
            }
        """
        if len(df) < 50:
            return {'regime': 'unknown', 'confidence': 0, 'adx': 0, 'atr_pct': 0}
        
        # ADX for trend strength
        adx = df['adx'].iloc[-1] if 'adx' in df else ta.ADX(df, timeperiod=14).iloc[-1]
        
        # ATR for volatility
        atr = df['atr'].iloc[-1] if 'atr' in df else ta.ATR(df, timeperiod=14).iloc[-1]
        atr_pct = (atr / df['close'].iloc[-1]) * 100
        
        # EMA Alignment
        ema_fast = ta.EMA(df, timeperiod=8).iloc[-1]
        ema_medium = ta.EMA(df, timeperiod=21).iloc[-1]
        ema_slow = ta.EMA(df, timeperiod=55).iloc[-1]
        
        price = df['close'].iloc[-1]
        
        # REGIME DETECTION
        if adx > 25:
            # TRENDING
            if ema_fast > ema_medium > ema_slow and price > ema_fast:
                regime = 'trending_up'
                confidence = min(100, adx + 20)
            elif ema_fast < ema_medium < ema_slow and price < ema_fast:
                regime = 'trending_down'
                confidence = min(100, adx + 20)
            else:
                regime = 'trending_mixed'
                confidence = adx
        elif atr_pct > 3.0:
            # VOLATILE
            regime = 'volatile'
            confidence = min(100, atr_pct * 20)
        else:
            # RANGING
            regime = 'ranging'
            confidence = 100 - (adx * 2)
        
        result = {
            'regime': regime,
            'confidence': confidence,
            'adx': adx,
            'atr_pct': atr_pct
        }
        
        self.current_regime[pair] = result
        return result
    
    def get_regime_multiplier(self, regime: str, signal_type: str) -> float:
        """
        Get confidence multiplier based on regime
        
        Args:
            regime: Current market regime
            signal_type: 'trend_following' or 'mean_reversion'
        
        Returns:
            multiplier (0.5-1.5)
        """
        if signal_type == 'trend_following':
            if regime in ['trending_up', 'trending_down']:
                return 1.3  # Boost in trends
            elif regime == 'ranging':
                return 0.7  # Reduce in ranging
            else:
                return 1.0
        
        elif signal_type == 'mean_reversion':
            if regime == 'ranging':
                return 1.3  # Boost in ranging
            elif regime in ['trending_up', 'trending_down']:
                return 0.7  # Reduce in trends
            else:
                return 1.0
        
        return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# TREND ML SYSTEM - Intelligent per-coinpair trend detection with ML
# ═══════════════════════════════════════════════════════════════════════════════

class TrendMLSystem:
    """
    ML-based trend detection system that learns per-coinpair.
    Uses Random Forest + Gradient Boosting to detect trends.
    
    FEATURES:
    * Detects: Uptrend, Downtrend, Sideways
    * Calculates trend strength (0-100)
    * Per-coinpair model training
    * Optuna optimization
    * Persistent model storage
    
    PREVENTS:
    * Bear signals in uptrends (V65 problem!)
    * Bull signals in downtrends
    * False signals in ranging markets
    """
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.models_file = storage_dir / "trend_models.pkl"
        self.params_file = storage_dir / "trend_params.json"
        
        # Per-pair ML models
        self.models = self._load_models()
        
        # Trend parameters (Optuna-optimized)
        self.params = self._load_params()
        
        # Trend history for learning
        self.trend_history = defaultdict(lambda: deque(maxlen=500))
        
        logger.info(f"🧠 TrendMLSystem: {len(self.models)} pair models loaded")
    
    def _load_models(self) -> Dict:
        """Load trained models from disk"""
        if self.models_file.exists():
            try:
                with open(self.models_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_models(self):
        """Save models to disk"""
        try:
            with open(self.models_file, 'wb') as f:
                pickle.dump(self.models, f)
        except Exception as e:
            logger.error(f"Error saving trend models: {e}")
    
    def _load_params(self) -> Dict:
        """Load trend parameters"""
        default_params = {
            'uptrend_strength_min': 55,
            'downtrend_strength_min': 55,
            'sideways_threshold': 40,
            'override_quality_threshold': 75,
            'ema_fast': 8,
            'ema_medium': 21,
            'ema_slow': 55,
            'ema_super_slow': 200,
            'lookback_period': 50,
            'min_samples_for_training': 100
        }
        
        if self.params_file.exists():
            try:
                with open(self.params_file, 'r') as f:
                    loaded = json.load(f)
                    default_params.update(loaded)
            except:
                pass
        
        return default_params
    
    def _save_params(self):
        """Save parameters to disk"""
        try:
            with open(self.params_file, 'w') as f:
                json.dump(self.params, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trend params: {e}")
    
    def calculate_trend_features(self, df: DataFrame) -> DataFrame:
        """
        Calculate ML features for trend detection
        
        FEATURES:
        * EMA crossovers (fast > medium > slow)
        * Price momentum (ROC)
        * ADX trend strength
        * RSI levels
        * Volume profile
        * Price distance from EMAs
        """
        # EMA features
        df['ema_fast'] = ta.EMA(df, timeperiod=self.params['ema_fast'])
        df['ema_medium'] = ta.EMA(df, timeperiod=self.params['ema_medium'])
        df['ema_slow'] = ta.EMA(df, timeperiod=self.params['ema_slow'])
        df['ema_super_slow'] = ta.EMA(df, timeperiod=self.params['ema_super_slow'])
        
        # EMA alignment scores
        df['ema_bull_alignment'] = (
            (df['ema_fast'] > df['ema_medium']).astype(int) +
            (df['ema_medium'] > df['ema_slow']).astype(int) +
            (df['ema_slow'] > df['ema_super_slow']).astype(int)
        )  # 0-3 scale
        
        df['ema_bear_alignment'] = (
            (df['ema_fast'] < df['ema_medium']).astype(int) +
            (df['ema_medium'] < df['ema_slow']).astype(int) +
            (df['ema_slow'] < df['ema_super_slow']).astype(int)
        )  # 0-3 scale
        
        # Price distance from EMAs (%)
        df['price_above_ema_fast'] = ((df['close'] / df['ema_fast'] - 1) * 100)
        df['price_above_ema_slow'] = ((df['close'] / df['ema_slow'] - 1) * 100)
        
        # Momentum features
        df['roc'] = ta.ROC(df, timeperiod=10)
        df['roc_50'] = ta.ROC(df, timeperiod=50)
        
        # ADX (trend strength)
        df['adx'] = ta.ADX(df, timeperiod=14)
        
        # RSI
        df['rsi'] = ta.RSI(df, timeperiod=14)
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price volatility
        df['atr'] = ta.ATR(df, timeperiod=14)
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        return df
    
    def detect_trend(self, df: DataFrame, pair: str) -> Tuple[str, float]:
        """
        Detect trend for given pair using ML
        
        Returns:
            (trend_type, strength)
            trend_type: 'uptrend', 'downtrend', 'sideways'
            strength: 0-100
        """
        # Calculate features
        df = self.calculate_trend_features(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Simple rule-based detection if no model trained yet
        if pair not in self.models or len(self.trend_history[pair]) < self.params['min_samples_for_training']:
            return self._rule_based_trend_detection(latest)
        
        # ML-based detection
        model = self.models[pair]
        features = self._extract_features(latest)
        
        try:
            trend_pred = model.predict([features])[0]
            trend_proba = model.predict_proba([features])[0]
            
            trend_types = ['downtrend', 'sideways', 'uptrend']
            trend = trend_types[trend_pred]
            strength = trend_proba[trend_pred] * 100
            
            return trend, strength
        except:
            return self._rule_based_trend_detection(latest)
    
    def _rule_based_trend_detection(self, row: Series) -> Tuple[str, float]:
        """
        Fallback: Rule-based trend detection
        Used when ML model not trained yet
        """
        # EMA alignment scoring
        bull_score = row.get('ema_bull_alignment', 0) / 3.0 * 100
        bear_score = row.get('ema_bear_alignment', 0) / 3.0 * 100
        
        # ADX strength
        adx = row.get('adx', 20)
        
        # ROC momentum
        roc = row.get('roc', 0)
        
        # Combined scoring
        if bull_score > 70 and roc > 0 and adx > 25:
            return 'uptrend', min(bull_score + 10, 100)
        elif bear_score > 70 and roc < 0 and adx > 25:
            return 'downtrend', min(bear_score + 10, 100)
        else:
            # Sideways: weak alignment or low ADX
            strength = 50 - (adx / 2)  # Lower ADX = stronger sideways
            return 'sideways', max(30, min(strength, 70))
    
    def _extract_features(self, row: Series) -> List[float]:
        """Extract feature vector for ML model"""
        return [
            row.get('ema_bull_alignment', 0),
            row.get('ema_bear_alignment', 0),
            row.get('price_above_ema_fast', 0),
            row.get('price_above_ema_slow', 0),
            row.get('roc', 0),
            row.get('roc_50', 0),
            row.get('adx', 20),
            row.get('rsi', 50),
            row.get('volume_ratio', 1),
            row.get('atr_pct', 1)
        ]
    
    def train_model(self, df: DataFrame, pair: str):
        """
        Train ML model for pair using historical data
        
        Labels are created from future price movement:
        * Uptrend: Price rises >2% in next 50 candles
        * Downtrend: Price falls >2% in next 50 candles
        * Sideways: Price moves <2% in next 50 candles
        """
        if len(df) < self.params['min_samples_for_training']:
            return
        
        # Calculate features
        df = self.calculate_trend_features(df)
        
        # Create labels from future price movement
        lookback = self.params['lookback_period']
        df['future_return'] = ((df['close'].shift(-lookback) / df['close']) - 1) * 100
        
        # Label: 0=downtrend, 1=sideways, 2=uptrend
        df['trend_label'] = 1  # Default: sideways
        df.loc[df['future_return'] > 2, 'trend_label'] = 2  # Uptrend
        df.loc[df['future_return'] < -2, 'trend_label'] = 0  # Downtrend
        
        # Drop NaN rows
        df = df.dropna()
        
        if len(df) < 50:
            return
        
        # Extract features and labels
        X = []
        y = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            X.append(self._extract_features(row))
            y.append(row['trend_label'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train ensemble model
        try:
            model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X, y)
            
            # Store model
            self.models[pair] = model
            self._save_models()
            
            # Log accuracy
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            logger.info(f"🧠 TrendML trained for {pair}: {accuracy:.1%} accuracy")
            
        except Exception as e:
            logger.error(f"Error training trend model for {pair}: {e}")
    
    def should_allow_bear_signal(self, pair: str, df: DataFrame, quality_score: float) -> bool:
        """
        Check if bear (short) signal should be allowed
        
        RULES:
        * Allow in downtrend (strength > threshold)
        * Allow in weak sideways
        * BLOCK in uptrend (unless quality > override threshold)
        """
        trend, strength = self.detect_trend(df, pair)
        
        # Downtrend: Always allow
        if trend == 'downtrend' and strength >= self.params['downtrend_strength_min']:
            return True
        
        # Sideways: Allow if not too strong
        if trend == 'sideways' and strength < 60:
            return True
        
        # Uptrend: Block unless quality is exceptional
        if trend == 'uptrend':
            if quality_score >= self.params['override_quality_threshold']:
                logger.info(f"⚠️ {pair}: Allowing bear signal in uptrend (quality={quality_score:.0f} override)")
                return True
            else:
                logger.info(f"🚫 {pair}: BLOCKING bear signal in uptrend (trend_strength={strength:.0f})")
                return False
        
        return True
    
    def should_allow_bull_signal(self, pair: str, df: DataFrame, quality_score: float) -> bool:
        """
        Check if bull (long) signal should be allowed
        
        RULES:
        * Allow in uptrend (strength > threshold)
        * Allow in weak sideways
        * BLOCK in downtrend (unless quality > override threshold)
        """
        trend, strength = self.detect_trend(df, pair)
        
        # Uptrend: Always allow
        if trend == 'uptrend' and strength >= self.params['uptrend_strength_min']:
            return True
        
        # Sideways: Allow if not too strong
        if trend == 'sideways' and strength < 60:
            return True
        
        # Downtrend: Block unless quality is exceptional
        if trend == 'downtrend':
            if quality_score >= self.params['override_quality_threshold']:
                logger.info(f"⚠️ {pair}: Allowing bull signal in downtrend (quality={quality_score:.0f} override)")
                return True
            else:
                logger.info(f"🚫 {pair}: BLOCKING bull signal in downtrend (trend_strength={strength:.0f})")
                return False
        
        return True
    
    def optimize_with_optuna(self, n_trials: int = 20) -> Dict:
        """
        Optimize trend parameters with Optuna
        
        Optimizes:
        * Strength thresholds
        * Override quality threshold
        * EMA periods
        * Lookback periods
        """
        def objective(trial):
            # Test parameters
            test_params = {
                'uptrend_strength_min': trial.suggest_int('uptrend_strength_min', 40, 70),
                'downtrend_strength_min': trial.suggest_int('downtrend_strength_min', 40, 70),
                'sideways_threshold': trial.suggest_int('sideways_threshold', 30, 50),
                'override_quality_threshold': trial.suggest_int('override_quality_threshold', 70, 85),
                'ema_fast': trial.suggest_int('ema_fast', 5, 12),
                'ema_medium': trial.suggest_int('ema_medium', 15, 30),
                'ema_slow': trial.suggest_int('ema_slow', 40, 70),
                'ema_super_slow': trial.suggest_int('ema_super_slow', 150, 250),
                'lookback_period': trial.suggest_int('lookback_period', 30, 70)
            }
            
            # Score: Higher is better (placeholder - would need actual backtest)
            # For now, prefer stricter thresholds
            score = (
                test_params['uptrend_strength_min'] +
                test_params['downtrend_strength_min'] +
                (100 - test_params['override_quality_threshold'])
            ) / 3
            
            return score
        
        study = optuna.create_study(direction='maximize', study_name='trend_ml_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        self.params.update(best_params)
        self._save_params()
        
        logger.info(f"🔬 TrendML Optuna: Optimized {n_trials} trials")
        
        return best_params


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC PARAMETER MANAGER - All parameters are dynamic
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicParameterManager:
    """
    Manages ALL strategy parameters dynamically based on performance.
    Parameters are PER-PAIR with global fallback!
    No more static values - everything evolves!
    """
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.params_file = storage_dir / "dynamic_parameters.json"
        self.history_file = storage_dir / "parameter_history.json"
        
        # Global and per-pair parameters
        self.global_params = self._load_params()
        self.pair_specific_params = {}  # pair -> params dict
        self.param_history = self._load_history()
        self.performance_by_params = defaultdict(list)
        
        # 🔥 SAVE immediately after initialization (persist defaults!)
        self._save()
        
        logger.info(f"⚙️  Dynamic Parameter Manager initialized (PER-PAIR + GLOBAL)")
    
    def _load_params(self) -> Dict:
        if self.params_file.exists():
            try:
                with open(self.params_file, 'r') as f:
                    data = json.load(f)
                    
                    # New format: {'global': {...}, 'pair_specific': {...}}
                    if 'global' in data:
                        self.pair_specific_params = data.get('pair_specific', {})
                        logger.info(f"✅ Loaded dynamic parameters: "
                                  f"global + {len(self.pair_specific_params)} pair-specific configs")
                        return data['global']
                    else:
                        # Old format (backward compatibility)
                        logger.info(f"✅ Loaded {len(data)} global parameters (old format)")
                        return data
            except Exception as e:
                logger.warning(f"Could not load params: {e}")
        
        # Initial defaults (ADAPTED FOR 15M + 1H/4H TIMEFRAMES)
        return {
            # Entry parameters - BALANCED for 1h indicators on 15m base
            'ml_confidence_min': 0.20,  # Higher for 1h (more reliable signals)
            'ml_signal_strength_min': 0.30,  # Higher for 1h
            'fisher_buy_threshold': -1.5,  # Adjusted for 1h indicators
            'fisher_sell_threshold': 1.5,  # Adjusted for 1h indicators
            'pattern_confidence_min': 0.50,  # Higher for 1h patterns
            
            # 🎯 NEW: Entry Quality Score Threshold (dynamically adjusted!)
            'entry_quality_threshold': 60.0,  # Higher for 1h signals (better quality)
            
            # Exit parameters - ADAPTED FOR 1H INDICATORS
            'fisher_long_exit': -0.5,  # Adjusted for 1h
            'fisher_short_exit': 0.5,  # Adjusted for 1h
            'profit_target_min': 0.025,  # 2.5% for 15m/1h combo
            'stop_loss_base': -0.10,  # Balanced for 15m execution
            
            # Regime parameters - BALANCED for 1h analysis
            'regime_score_bull_min': 45.0,  # Higher for 1h reliability
            'regime_score_bear_max': 55.0,  # Tighter for 1h
            'reversal_momentum_threshold': 0.020,  # Adjusted for 1h
            
            # Risk parameters
            'max_drawdown_before_retrain': -0.15,
            'min_win_rate_threshold': 0.40,
            # 📉 Choppy/Sideways filters (PER-PAIR optimizable)
            'trend_adx_min_long': 18.0,
            'trend_adx_min_short': 18.0,
            'sideways_volatility_max': 0.008,
            'sideways_bb_width_max': 0.045,
            'sideways_lookback_candles': 20,
            
            # Adaptation parameters
            'learning_rate': 0.1,
            'retraining_frequency_hours': 24,
        }
    
    def _load_history(self) -> List[Dict]:
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def get_param(self, param_name: str, pair: str = None, default: Any = None) -> Any:
        """
        Get current value of a dynamic parameter.
        Returns pair-specific value if available, otherwise global.
        """
        # Check pair-specific first
        if pair and pair in self.pair_specific_params:
            if param_name in self.pair_specific_params[pair]:
                return self.pair_specific_params[pair][param_name]
        
        # Fallback to global
        return self.global_params.get(param_name, default)
    
    def get_all_params(self, pair: str = None) -> Dict:
        """Get all parameters for a pair (merged: pair-specific overrides global)"""
        if pair and pair in self.pair_specific_params:
            # Merge: global + pair-specific (pair-specific wins)
            merged = self.global_params.copy()
            merged.update(self.pair_specific_params[pair])
            return merged
        return self.global_params.copy()
    
    def update_pair_params(self, pair: str, param_updates: Dict):
        """Update parameters for a specific pair"""
        if pair not in self.pair_specific_params:
            self.pair_specific_params[pair] = {}
        
        self.pair_specific_params[pair].update(param_updates)
        logger.info(f"⚙️  Updated {len(param_updates)} parameters for {pair}")
        self._save()
    
    def update_from_performance(self, win_rate: float, avg_profit: float, exit_reasons: Dict, pair: str = None):
        """
        Update parameters based on recent performance.
        If pair is provided, updates pair-specific parameters.
        Otherwise updates global parameters.
        """
        target = f"{pair}" if pair else "GLOBAL"
        
        # Track changes for logging
        changes = {}
        
        # Record current state
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'params': self.global_params.copy() if not pair else self.pair_specific_params.get(pair, {}).copy(),
            'win_rate': win_rate,
            'avg_profit': avg_profit
        }
        self.param_history.append(snapshot)
        
        # Keep only last 200 snapshots (100 global + 100 per-pair potential)
        if len(self.param_history) > 200:
            self.param_history = self.param_history[-200:]
        
        # Get target params dict (pair-specific or global)
        if pair:
            if pair not in self.pair_specific_params:
                self.pair_specific_params[pair] = {}
            target_params = self.pair_specific_params[pair]
        else:
            target_params = self.global_params
        
        # Adaptive parameter adjustment
        learning_rate = self.global_params['learning_rate']
        reason = ""
        
        # 🎯 NEW: Adjust Entry Quality Threshold based on performance
        current_quality_threshold = target_params.get('entry_quality_threshold', self.global_params.get('entry_quality_threshold', 55.0))
        
        if win_rate < 0.40:  # Poor performance
            reason = f"Poor Win Rate ({win_rate:.1%}) - RAISING quality threshold (stricter entries)"
            
            # Increase quality threshold = fewer, better entries
            new_quality_threshold = min(75.0, current_quality_threshold + learning_rate * 5.0)
            changes['entry_quality_threshold'] = (current_quality_threshold, new_quality_threshold)
            target_params['entry_quality_threshold'] = new_quality_threshold
            
            # Also increase ML requirements
            current_ml_conf = target_params.get('ml_confidence_min', self.global_params['ml_confidence_min'])
            current_ml_signal = target_params.get('ml_signal_strength_min', self.global_params['ml_signal_strength_min'])
            
            new_ml_conf = min(0.30, current_ml_conf + learning_rate * 0.03)
            new_ml_signal = min(0.40, current_ml_signal + learning_rate * 0.03)
            
            changes['ml_confidence_min'] = (current_ml_conf, new_ml_conf)
            changes['ml_signal_strength_min'] = (current_ml_signal, new_ml_signal)
            
            target_params['ml_confidence_min'] = new_ml_conf
            target_params['ml_signal_strength_min'] = new_ml_signal
        
        elif win_rate > 0.55:  # Good performance
            reason = f"Good Win Rate ({win_rate:.1%}) - LOWERING quality threshold (more entries)"
            
            # Decrease quality threshold = more entries
            new_quality_threshold = max(40.0, current_quality_threshold - learning_rate * 3.0)
            changes['entry_quality_threshold'] = (current_quality_threshold, new_quality_threshold)
            target_params['entry_quality_threshold'] = new_quality_threshold
            
            # Also relax ML requirements slightly
            current_ml_conf = target_params.get('ml_confidence_min', self.global_params['ml_confidence_min'])
            current_ml_signal = target_params.get('ml_signal_strength_min', self.global_params['ml_signal_strength_min'])
            
            new_ml_conf = max(0.05, current_ml_conf - learning_rate * 0.02)
            new_ml_signal = max(0.10, current_ml_signal - learning_rate * 0.02)
            
            changes['ml_confidence_min'] = (current_ml_conf, new_ml_conf)
            changes['ml_signal_strength_min'] = (current_ml_signal, new_ml_signal)
            
            target_params['ml_confidence_min'] = new_ml_conf
            target_params['ml_signal_strength_min'] = new_ml_signal
        
        # Adjust stop loss based on loss patterns
        ml_stop_losses = exit_reasons.get('ml_stop_loss (empty)', {}).get('profit', 0)
        if ml_stop_losses < -100:  # Too many big losses
            if not reason:
                reason = f"Large ML Stop Losses ({ml_stop_losses:.2f} USD) - Tightening stops"
            current_stop = target_params.get('stop_loss_base', self.global_params['stop_loss_base'])
            new_stop = max(-0.20, current_stop - learning_rate * 0.02)
            changes['stop_loss_base'] = (current_stop, new_stop)
            target_params['stop_loss_base'] = new_stop
        
        # Adjust profit targets
        roi_exits = exit_reasons.get('roi', {}).get('count', 0)
        if roi_exits > 100:  # Many ROI exits
            if not reason:
                reason = f"Many ROI Exits ({roi_exits}) - Increasing profit targets"
            current_target = target_params.get('profit_target_min', self.global_params['profit_target_min'])
            new_target = min(0.08, current_target + learning_rate * 0.01)
            changes['profit_target_min'] = (current_target, new_target)
            target_params['profit_target_min'] = new_target
        
        # Log changes using SmartLogger
        if changes:
            SmartLogger.log_parameter_update(target, reason, changes)
        
        self._save()
    
    def _save(self):
        try:
            # Save both global and pair-specific params
            save_data = {
                'global': self.global_params,
                'pair_specific': self.pair_specific_params,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.params_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            with open(self.history_file, 'w') as f:
                json.dump(self.param_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving parameters: {e}")
    
    def needs_retraining(self) -> bool:
        """Check if it's time to retrain based on time or performance"""
        if not self.param_history:
            return True
        
        # Check last retraining time
        last_update = datetime.fromisoformat(self.param_history[-1]['timestamp'])
        hours_since = (datetime.now() - last_update).total_seconds() / 3600
        
        if hours_since > self.global_params['retraining_frequency_hours']:
            return True
        
        # Check if performance degraded significantly
        if len(self.param_history) >= 3:
            recent_3 = self.param_history[-3:]
            avg_recent_profit = np.mean([h['avg_profit'] for h in recent_3])
            
            if avg_recent_profit < self.global_params['max_drawdown_before_retrain']:
                logger.warning(f"🚨 Performance degradation: {avg_recent_profit:.4f}")
                return True
        
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA AUTO-RETRAINER - Intelligent parameter optimization
# ═══════════════════════════════════════════════════════════════════════════════

class OptunaAutoRetrainer:
    """
    Continuously optimizes parameters using Optuna based on actual trade performance.
    Detects bad data and triggers retraining automatically.
    """
    
    def __init__(self, storage_dir: Path, trade_tracker: TradePerformanceTracker):
        self.storage_dir = storage_dir
        self.trade_tracker = trade_tracker
        self.study_file = storage_dir / "optuna_study.pkl"
        self.best_params_file = storage_dir / "best_parameters.json"
        
        self.study = self._load_study()
        self.best_params = self._load_best_params()
        self.last_optimization = None
        
        logger.info(f"🔬 Optuna Auto-Retrainer initialized")
    
    def _load_study(self) -> Optional[optuna.Study]:
        if self.study_file.exists():
            try:
                with open(self.study_file, 'rb') as f:
                    study = pickle.load(f)
                    logger.info(f"✅ Loaded Optuna study: {len(study.trials)} trials")
                    return study
            except:
                pass
        return None
    
    def _load_best_params(self) -> Dict:
        if self.best_params_file.exists():
            try:
                with open(self.best_params_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def optimize(self, n_trials: int = 50) -> Dict:
        """Run Optuna optimization based on recent trade performance"""
        logger.info(f"🔬 Starting Optuna optimization with {n_trials} trials")
        
        if self.study is None:
            self.study = optuna.create_study(
                direction='maximize',
                study_name='fisher_dynamic_v63',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
        
        def objective(trial):
            # Sample ALL parameters dynamically (ADAPTED FOR 1H INDICATORS)
            params = {
                # ML Confidence
                'ml_confidence_min': trial.suggest_float('ml_confidence_min', 0.15, 0.60),
                'ml_signal_strength_min': trial.suggest_float('ml_signal_strength_min', 0.25, 0.65),
                
                # Fisher Transform (1h ranges)
                'fisher_buy_threshold': trial.suggest_float('fisher_buy_threshold', -3.0, -0.8),
                'fisher_sell_threshold': trial.suggest_float('fisher_sell_threshold', 0.8, 3.0),
                'fisher_long_exit': trial.suggest_float('fisher_long_exit', -1.5, 0.3),
                'fisher_short_exit': trial.suggest_float('fisher_short_exit', -0.3, 1.5),
                
                # Pattern Analysis
                'pattern_confidence_min': trial.suggest_float('pattern_confidence_min', 0.40, 0.75),
                
                # Market Regime
                'regime_score_bull_min': trial.suggest_float('regime_score_bull_min', 40, 65),
                'regime_score_bear_max': trial.suggest_float('regime_score_bear_max', 35, 60),
                'reversal_momentum_threshold': trial.suggest_float('reversal_momentum_threshold', 0.015, 0.030),
                
                # Risk Management (15m execution balanced)
                'stop_loss_low_conf': trial.suggest_float('stop_loss_low_conf', -0.10, -0.05),
                'stop_loss_medium_conf': trial.suggest_float('stop_loss_medium_conf', -0.15, -0.08),
                'stop_loss_high_conf': trial.suggest_float('stop_loss_high_conf', -0.25, -0.12),
                'profit_target_min': trial.suggest_float('profit_target_min', 0.02, 0.10),
            }
            
            # Simulate performance with these parameters
            score = self._simulate_performance(params)
            return score
        
        # Optimize
        self.study.optimize(objective, n_trials=n_trials, timeout=300)
        
        # Get best parameters
        self.best_params = self.study.best_params
        self.last_optimization = datetime.now()
        
        # Save
        self._save()
        
        # Log using SmartLogger
        key_improvements = {
            'ml_confidence_min': self.best_params.get('ml_confidence_min', 0),
            'ml_signal_strength_min': self.best_params.get('ml_signal_strength_min', 0),
            'fisher_buy_threshold': self.best_params.get('fisher_buy_threshold', 0),
            'stop_loss_low_conf': self.best_params.get('stop_loss_low_conf', 0),
            'regime_score_bull_min': self.best_params.get('regime_score_bull_min', 0),
        }
        
        SmartLogger.log_optuna_optimization(n_trials, self.study.best_value, key_improvements)
        
        return self.best_params
    
    def _simulate_performance(self, params: Dict) -> float:
        """Simulate strategy performance with given parameters"""
        if len(self.trade_tracker.trades) < 10:
            return 0.0
        
        # Analyze recent trades with these parameters
        recent = self.trade_tracker.trades[-50:]
        
        # Count how many trades would have been taken
        score = 0.0
        for trade in recent:
            ml_conf = trade.get('ml_confidence', 0.5)
            profit = trade.get('profit', 0.0)
            
            # Would this trade be taken with these params?
            if ml_conf >= params['ml_confidence_min']:
                score += profit  # Add the profit/loss
        
        return score
    
    def _save(self):
        try:
            with open(self.study_file, 'wb') as f:
                pickle.dump(self.study, f)
            with open(self.best_params_file, 'w') as f:
                json.dump(self.best_params, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving Optuna study: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN STRATEGY CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class GKD_Galaxy_v1_1h(IStrategy):
    """
    🚀 V72 ULTIMATE - THE BEST OF V63 & V71 🚀
    
    Fisher Transform Strategy combining V63's proven robustness with V71's advanced features.
    
    Community Test Result: V63 SURVIVED "greatest market manipulation" while V64/V65 crashed.
    V72 integrates V63's winning loss protection system with V71's technical improvements.
    
    See file header for complete changelog (V72, V71, V69).
    
    KEY FEATURES (V72 ULTIMATE):
    ============================
    🛡️  V63 PROVEN SYSTEMS:
    - Loss Protection Memory (tracks last 5 trades per pair)
    - Consecutive Loss Blocking (blocks after 3 losses)
    - Dynamic Leverage Reduction (4 mechanisms)
    - Entry Cooldown Periods (30-60min after losses)
    - Conservative Position Sizing (8 max trades vs 15)
    
    🎯 V71 ADVANCED FEATURES:
    - Emergency Exit Loop Protection (Phase 1)
    - Slope-Based Divergence Detection (Phase 3)
    - Flag & Triangle Pattern Recognition (Phase 4)
    - Adaptive Pattern Learning per Coin (V69)
    - Market Regime Detection (V69)
    - Coin-Specific ML Models (V69)
    - Flash Crash Protection (V68)
    - Multi-Stage Exit System (V68)
    - Tighter Stops (3-10% vs V63: 7.5-18.5%)
    
    BIDIRECTIONAL OPTIMIZATION:
    ===========================
    - LONG: Bull Patterns + Uptrend Confirmation
    - SHORT: Bear Patterns + Downtrend Confirmation
    - Both directions optimized equally
    - No preference for any direction
    
    BENEFITS:
    =========
    - Learns from EVERY trade (not just winners)
    - Pattern recognition improves entry timing
    - Coin-specific models increase win rate
    - Market regime detection prevents false trades
    - Adaptive parameters = no manual optimization needed
    - Multi-Bot Safe: Separate learning directories
    """
    
    INTERFACE_VERSION = 3
    
    timeframe = '1h'
    startup_candle_count = 500  # Higher for multi-timeframe analysis
    can_short = True
    
    # Informative timeframes for multi-timeframe analysis
    informative_timeframe_1h = '1h'
    informative_timeframe_4h = '4h'
    
    # These will be overridden by dynamic system
    stoploss = -999.9999  # Safety net, real stops are dynamic
    trailing_stop = False
    minimal_roi = {}
    
    position_adjustment_enable = True

    # --- Dry Run Wallet placeholder ---
    dry_run_wallet_balance = 3000.0

    # --- 3-Candle Forced Decision ---
    forced_exit_period = IntParameter(1, 3, default=1, space='buy', optimize=True)
    forced_exit_profit_trigger = DecimalParameter(0.01, 0.10, default=0.05, space='sell', optimize=True)
    forced_exit_loss_trigger = DecimalParameter(-0.10, -0.01, default=-0.05, space='buy', optimize=True)
    forced_exit_sell_pct = DecimalParameter(0.05, 0.20, default=0.10, space='sell', optimize=True)
    forced_exit_buy_pct = DecimalParameter(0.05, 0.20, default=0.10, space='buy', optimize=True)

    # --- Low Stake Replenishment ---
    replenish_enabled = BooleanParameter(default=True, space='buy', optimize=False)
    replenish_stake_threshold_usd = DecimalParameter(1.0, 10.0, default=6.5, space='buy', optimize=True)
    replenish_stake_pct = DecimalParameter(0.25, 1.0, default=0.75, space='buy', optimize=True)

    # --- Progressive DCA & Profit-Taking (based on initial stake) ---
    progressive_dca_enabled = BooleanParameter(default=True, space='buy', optimize=False)
    progressive_profit_trigger = DecimalParameter(0.05, 0.20, default=0.10, space='sell', optimize=True)
    progressive_loss_trigger = DecimalParameter(-0.20, -0.05, default=-0.10, space='buy', optimize=True)
    progressive_sell_pct = DecimalParameter(0.05, 0.25, default=0.10, space='sell', optimize=True)
    progressive_buy_pct = DecimalParameter(0.05, 0.25, default=0.10, space='buy', optimize=True)

    # --- Capital Management ---
    max_open_trades_setting = IntParameter(1, 20, default=15, space='buy', optimize=False, help="Max concurrent trades for capital calculation.")
    capital_management_enabled = BooleanParameter(default=True, space='buy', optimize=False)
    max_wallet_exposure = DecimalParameter(0.20, 0.50, default=0.33, space='buy', optimize=True, help="Limit the total stake amount to a percentage of the total wallet balance.")
    emergency_capital_release_enabled = BooleanParameter(default=True, space='buy', optimize=False)
    emergency_capital_release_profit_trigger = DecimalParameter(0.01, 0.20, default=0.2, space='sell', optimize=True)
    emergency_capital_release_sell_pct = DecimalParameter(0.10, 0.50, default=0.25, space='sell', optimize=True)

    # --- Notifications ---
    telegram_adjustment_notification_enabled = BooleanParameter(default=True, space='buy', optimize=False)

    # --- State Machine Parameters ---
    initial_loss_trigger = DecimalParameter(-0.05, -0.01, default=-0.02, space='buy', optimize=True)
    initial_profit_trigger = DecimalParameter(0.01, 0.05, default=0.02, space='sell', optimize=True)

    # --- DEFENDING State Parameters ---
    defending_dca_multiplier = DecimalParameter(0.1, 2.0, default=0.1, space='buy', optimize=True)
    defending_max_dca = IntParameter(1, 15, default=15, space='buy', optimize=True)
    dca_first_trigger = DecimalParameter(-0.20, -0.05, default=-0.20, space='buy', optimize=False)
    dca_subsequent_trigger = DecimalParameter(-0.10, -0.02, default=-0.10, space='buy', optimize=False)

    # --- PROFIT State Parameters ---
    profit_max_position_adjustment = IntParameter(1, 15, default=15, space='sell', optimize=True)
    profit_dca_multiplier = DecimalParameter(0.1, 0.5, default=0.1, space='buy', optimize=True)
    profit_take_profit_pct = DecimalParameter(0.005, 0.03, default=0.1, space='sell', optimize=True)
    profit_sell_amount_pct = DecimalParameter(0.1, 0.5, default=0.25, space='sell', optimize=True)
    profit_reload_threshold = DecimalParameter(0.2, 0.8, default=0.35, space='buy', optimize=True)

    # --- MAX_DCA State Parameters ---
    max_dca_take_profit_pct = DecimalParameter(0.005, 0.02, default=0.2, space='sell', optimize=True)
    max_dca_sell_amount_pct = DecimalParameter(0.1, 0.5, default=0.3, space='sell', optimize=True)

    # --- ML Confidence Threshold ---
    dca_ml_conf_threshold = DecimalParameter(0.04, 0.9, default=0.05, space='buy', optimize=True)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 📊 PLOT CONFIGURATION - Beautiful Multi-Timeframe Visualization
    # ═══════════════════════════════════════════════════════════════════════════════
    
    plot_config = {
        'main_plot': {
            # Baseline & Goldie Locks Zone (from 1h)
            'baseline_1h': {'color': 'blue', 'type': 'line'},
            'goldie_min_1h': {'color': 'lightblue', 'fill_to': 'goldie_max_1h', 'type': 'line'},
            'goldie_max_1h': {'color': 'lightblue', 'type': 'line'},
        },
        'subplots': {
            # ═══════════════════════════════════════════════════════════════════
            # 1️⃣ FISHER TRANSFORM (Primary Signal - 1h)
            # ═══════════════════════════════════════════════════════════════════
            "Fisher Transform (1h)": {
                'fisher_1h': {'color': 'orange', 'type': 'line'},
                'fisher_smooth_long_1h': {'color': 'red', 'type': 'line'},
                'fisher_smooth_short_1h': {'color': 'green', 'type': 'line'},
            },
            
            # ═══════════════════════════════════════════════════════════════════
            # 2️⃣ MARKET REGIME (Multi-Timeframe View)
            # ═══════════════════════════════════════════════════════════════════
            "Market Regime (MTF)": {
                'market_regime_1h': {'color': 'purple', 'type': 'line'},
                'market_regime_4h': {'color': 'darkviolet', 'type': 'line'},
            },
            
            # ═══════════════════════════════════════════════════════════════════
            # 3️⃣ PATTERN ANALYSIS (Bulkowski)
            # ═══════════════════════════════════════════════════════════════════
            "Pattern Strength (1h)": {
                'pattern_weighted_bullish_1h': {'color': 'green', 'type': 'bar'},
                'pattern_weighted_bearish_1h': {'color': 'red', 'type': 'bar'},
                'pattern_confirmed_bullish_1h': {'color': 'lime', 'type': 'line'},
                'pattern_confirmed_bearish_1h': {'color': 'crimson', 'type': 'line'},
            },
            
            # ═══════════════════════════════════════════════════════════════════
            # 4️⃣ ML CONFIDENCE & SIGNAL STRENGTH
            # ═══════════════════════════════════════════════════════════════════
            "ML Metrics (1h)": {
                'ml_confidence_1h': {'color': 'cyan', 'type': 'line'},
                'ml_signal_strength_1h': {'color': 'magenta', 'type': 'line'},
            },
            
            # ═══════════════════════════════════════════════════════════════════
            # 5️⃣ ENTRY QUALITY SCORES (Smart Filter)
            # ═══════════════════════════════════════════════════════════════════
            "Entry Quality (0-100)": {
                'entry_quality_score_long': {'color': 'lightgreen', 'type': 'line'},
                'entry_quality_score_short': {'color': 'lightcoral', 'type': 'line'},
            },
            
            # ═══════════════════════════════════════════════════════════════════
            # 6️⃣ VOLUME (15m - for context)
            # ═══════════════════════════════════════════════════════════════════
            "Volume (15m)": {
                'volume': {'color': 'gray', 'type': 'bar'},
            },
        }
    }
    
    def __init__(self, config: dict):
        super().__init__(config)
        # Startup suppression window to avoid thrashing on restart
        self._startup_time = datetime.now()
        self._startup_suppress_seconds = 90

        # 🛡️ V72 PHASE 5: LOSS PROTECTION SYSTEM (from V63)
        self.pair_loss_memory = PairLossMemory()
        logger.info("🛡️  V72 Loss Protection System initialized (from V63 proven system)")

        # 🎯 DYNAMIC VERSION DETECTION from class name
        # Extracts version from class name: GKD_FisherTransformV69_Adaptive -> v69
        class_name = self.__class__.__name__
        import re
        version_match = re.search(r'V(\d+)', class_name)
        self.strategy_version = f"v{version_match.group(1).lower()}" if version_match else "unknown"
        
        # Extract variant (e.g., "Adaptive", "Fixed", "TrendML")
        variant_match = re.search(r'V\d+_(\w+)', class_name)
        self.strategy_variant = variant_match.group(1).lower() if variant_match else "default"
        
        logger.info(f"🎯 Strategy Version: {self.strategy_version.upper()} ({self.strategy_variant.title()})")

        # Storage - TIMEFRAME & BOT-NAME-SPECIFIC directory to prevent conflicts!
        # Each bot instance gets its own learning directory
        base_dir = Path(config.get('user_data_dir', 'user_data'))
        timeframe_safe = self.timeframe.replace('/', '_')  # Convert 1h to 1h, 15m to 15m
        
        # Use bot_name to separate dry-run from live (or multiple instances)
        bot_name = config.get('bot_name', 'default')
        bot_suffix = bot_name.split('-')[-1] if '-' in bot_name else bot_name  # Extract last part
        
        # VERSION-SPECIFIC directory (dynamically generated from class name)
        self.model_dir = base_dir / 'ml_models' / f'{self.strategy_version}_{self.strategy_variant}_{timeframe_safe}_mtf_{bot_suffix}'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Model Storage ({self.timeframe}, {bot_suffix}): {self.model_dir}")
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🛡️ RISK MANAGEMENT: Per-Pair Loss Tracking & Leverage Protection
        # ═══════════════════════════════════════════════════════════════════════════════
        self.pair_loss_tracker = defaultdict(lambda: {
            'recent_losses': deque(maxlen=5),  # Last 5 trades
            'total_loss': 0.0,
            'consecutive_losses': 0,
            'last_loss_time': None,
            'leverage_penalty': 1.0  # Multiplier for leverage (1.0 = normal, 0.5 = half)
        })
        
        # 🚨 V71 PHASE 1: Emergency Exit Loop Prevention
        # Tracks which trades have already triggered emergency exit to prevent infinite loops
        self._emergency_exit_triggered = {}  # {trade_id: True}
        
        # Initialize dynamic systems
        self.trade_tracker = TradePerformanceTracker(self.model_dir)
        self.param_manager = DynamicParameterManager(self.model_dir)
        self.pattern_system = BulkowskiPatternSystem(self.model_dir)
        self.optuna_trainer = OptunaAutoRetrainer(self.model_dir, self.trade_tracker)
        
        # 🛡️ V68: Flash Crash Protection System
        self.crash_protection = FlashCrashProtection()
        
        # 🧠 V68: ML-based Trend Filter System
        self.trend_ml = TrendMLSystem(self.model_dir)
        
        # 🎯 NEW V69: Pattern Recognition Engine
        self.pattern_recognition = PatternRecognitionEngine(self.model_dir)
        logger.info("🎯 PatternRecognitionEngine initialized - Entry pattern detection active!")
        
        # 📊 NEW V69: Market Regime Detector
        self.regime_detector = MarketRegimeDetector()
        logger.info("📊 MarketRegimeDetector initialized - Trending/Ranging/Volatile detection active!")
        
        logger.info("🧠 TrendMLSystem initialized - Per-pair trend detection active!")
        
        # Check if retraining is needed (but NOT on first start without data!)
        if len(self.trade_tracker.trades) >= 10 and not self._startup_suppress_active():
            if self.param_manager.needs_retraining() or self.trade_tracker.detect_bad_parameters():
                logger.warning("🚨 Retraining triggered!")
                self.trigger_retraining()
        else:
            logger.info(f"⏳ Waiting for trades ({len(self.trade_tracker.trades)}/10) before first Optuna optimization")
        
        logger.info("=" * 80)
        logger.info("🧠 GKD Fisher Transform V69 - ADAPTIVE PATTERN LEARNING")
        logger.info("=" * 80)
        logger.info("⏱️  Base Timeframe: 15m | Indicators: 1h | Trend: 4h")
        logger.info(f"📊 Historical Trades: {len(self.trade_tracker.trades)}")
        logger.info(f"⚙️  Current ML Confidence Min: {self.param_manager.get_param('ml_confidence_min'):.2f}")
        logger.info(f"🎯 Current Profit Target: {self.param_manager.get_param('profit_target_min'):.2%}")
        logger.info(f"🛡️  Current Stop Loss Base: {self.param_manager.get_param('stop_loss_base'):.2%}")
        logger.info(f"🧠 Trend ML Models Loaded: {len(self.trend_ml.models)} pairs")
        logger.info("")
        logger.info("🎯 V69 NEW FEATURES:")
        logger.info("   ✅ PATTERN RECOGNITION: Bull/Bear Flags, Triangles, Cup & Handle")
        logger.info("   ✅ COIN-SPECIFIC LEARNING: Each coin learns its own behavior")
        logger.info("   ✅ MARKET REGIME DETECTION: Trending/Ranging/Volatile Auto-Erkennung")
        logger.info("   ✅ ADAPTIVE PARAMETERS: RSI/ATR Thresholds passen sich automatisch an")
        logger.info("   ✅ SMART ENTRY SIGNALS: Waits for optimal pattern confirmation")
        logger.info("   ✅ BIDIRECTIONAL OPTIMIZATION: LONG + SHORT gleich stark optimiert")
        logger.info("")
        logger.info("🔧 INHERITED FROM V68:")
        logger.info("   ✅ FLASH CRASH PROTECTION: ATR-based multi-timeframe crash detection")
        logger.info("   ✅ EMERGENCY FULL EXIT: 100% exit to avoid order rejections")
        logger.info("   ✅ LEVERAGE CAP: Max 3x - confidence-based (1x/2x/3x)")
        logger.info("   ✅ MAX TRADES: 15 concurrent for better capital allocation")
        logger.info("   ✅ FIXED STAKE: 500 USDT eliminates min order issues")
        logger.info("")
        logger.info("💰 V69 POSITION MANAGEMENT:")
        logger.info("   🎯 Pattern-Based Sizing: Bull/Bear Patterns + Confidence")
        logger.info("   📈 Partial Take Profit: @2%, @4%, @6% (progressive locking)")
        logger.info("   🔄 Smart DCA: +50% on pullbacks (max 2 DCA)")
        logger.info("   🚨 Emergency Full Exit: 100% if >5% loss + low conf (FIXED!)")
        logger.info("   ⚡ Expected: +161-211 USDT per 100 trades vs V67!")
        logger.info("")
        logger.info("🛡️ V68 NEW: FLASH CRASH PROTECTION:")
        logger.info("   📊 Multi-Timeframe: 15m/1h/4h ATR-based crash detection")
        logger.info("   ⚡ Dynamic Thresholds: 3-5x ATR (not fixed -5%)")
        logger.info("   🚫 Entry Block: 10min cooldown after crash detected")
        logger.info("   🚪 Auto-Exit: Closes positions if in crash cooldown")
        logger.info("")
        logger.info("📈 V68 TARGET: 55-60% WR | +0.5-1.0% avg P/L (vs V67: 45.6% WR, -1.11% avg)")
        logger.info("=" * 80)
    
    def _startup_suppress_active(self) -> bool:
        try:
            return (datetime.now() - self._startup_time).total_seconds() < self._startup_suppress_seconds
        except Exception:
            return False
    
    def trigger_retraining(self):
        """Trigger Optuna retraining"""
        try:
            logger.info("🔬 Starting automatic retraining...")
            new_params = self.optuna_trainer.optimize(n_trials=30)
            
            # Update parameter manager (global params)
            self.param_manager.global_params.update(new_params)
            self.param_manager._save()
            
            logger.info("✅ Retraining complete - new parameters active!")
        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}")
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate indicators with PER-PAIR dynamic parameters
        🔥 ENHANCED: Multi-timeframe analysis (15m base + 1h + 4h informative)
        """
        pair = metadata.get('pair', 'unknown')
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🎓 CONTINUOUS LEARNING FROM CANDLES
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Log that we're analyzing this pair (for visibility)
        if len(dataframe) > 0:
            last_candle_time = dataframe['date'].iloc[-1] if 'date' in dataframe.columns else "unknown"
            
            # Every 100 candles, log that we're actively learning
            if not hasattr(self, '_candle_analysis_count'):
                self._candle_analysis_count = {}
            
            self._candle_analysis_count[pair] = self._candle_analysis_count.get(pair, 0) + 1
            
            # Log first 5 analyses, then every 50
            count = self._candle_analysis_count[pair]
            if count <= 5 or count % 50 == 0:
                logger.info(f"💡 {pair}: Analyzed {count} candle batches (15m base + 1h/4h) | Last candle: {last_candle_time} | Learning ACTIVE")
        
        # Get dynamic parameters (pair-specific or global)
        params = self.param_manager.get_all_params(pair)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 📊 MULTI-TIMEFRAME ANALYSIS: Get 1h and 4h data
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Get 1h informative data
        informative_1h = self.dp.get_pair_dataframe(pair=pair, timeframe=self.informative_timeframe_1h)
        informative_1h = self._populate_informative_indicators(informative_1h, '1h')
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.informative_timeframe_1h, ffill=True)
        
        # Get 4h informative data
        informative_4h = self.dp.get_pair_dataframe(pair=pair, timeframe=self.informative_timeframe_4h)
        informative_4h = self._populate_informative_indicators(informative_4h, '4h')
        dataframe = merge_informative_pair(dataframe, informative_4h, self.timeframe, self.informative_timeframe_4h, ffill=True)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 📊 15m BASE TIMEFRAME: Basic tracking indicators + Band Indicators
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # ATR for position sizing (calculated on 15m for responsiveness)
        dataframe['atr'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🎯 HYBRID TECHNICAL INDICATORS (Technical + Patterns)
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # RSI
        dataframe['rsi'] = talib.RSI(dataframe['close'], timeperiod=14)
        
        # EMAs for trend and support/resistance
        dataframe['ema_short'] = talib.EMA(dataframe['close'], timeperiod=8)
        dataframe['ema_long'] = talib.EMA(dataframe['close'], timeperiod=21)
        dataframe['ema9'] = talib.EMA(dataframe['close'], timeperiod=9)
        dataframe['ema20'] = talib.EMA(dataframe['close'], timeperiod=20)
        dataframe['ema21'] = talib.EMA(dataframe['close'], timeperiod=21)
        dataframe['ema50'] = talib.EMA(dataframe['close'], timeperiod=50)
        dataframe['ema200'] = talib.EMA(dataframe['close'], timeperiod=200)
        
        # Bollinger Bands (talib returns tuple, not dict!)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(dataframe['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_upper'] = bb_upper
        dataframe['bb_middle'] = bb_middle
        dataframe['bb_lower'] = bb_lower
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'] + 0.000001)
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / (dataframe['close'] + 0.000001)
        
        # Keltner Channels
        ema20 = talib.EMA(dataframe['close'], timeperiod=20)
        atr10 = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=10)
        dataframe['kc_upperband'] = ema20 + atr10
        dataframe['kc_middleband'] = ema20
        dataframe['kc_lowerband'] = ema20 - atr10
        
        # Additional momentum indicators
        dataframe['adx'] = talib.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['willr'] = talib.WILLR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['cci'] = talib.CCI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=20)
        
        # MACD (talib returns tuple: macd, macdsignal, macdhist)
        macd, macdsignal, macdhist = talib.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd
        dataframe['macdsignal'] = macdsignal
        dataframe['macdhist'] = macdhist
        
        # Volume analysis
        dataframe['volume_sma'] = talib.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / (dataframe['volume_sma'] + 0.000001)
        
        # Price position and market structure
        dataframe['high_20'] = dataframe['high'].rolling(window=20).max()
        dataframe['low_20'] = dataframe['low'].rolling(window=20).min()
        dataframe['price_position'] = (dataframe['close'] - dataframe['low_20']) / (dataframe['high_20'] - dataframe['low_20'] + 0.000001)
        
        # Support/Resistance levels
        dataframe['swing_high'] = dataframe['high'].rolling(window=50).max().bfill().fillna(dataframe['high'])
        dataframe['swing_low'] = dataframe['low'].rolling(window=50).min().bfill().fillna(dataframe['low'])
        
        # Price vs EMAs
        dataframe['price_vs_ema21'] = (dataframe['close'] - dataframe['ema21']) / (dataframe['ema21'] + 0.000001) * 100
        dataframe['price_vs_ema50'] = (dataframe['close'] - dataframe['ema50']) / (dataframe['ema50'] + 0.000001) * 100
        dataframe['price_vs_ema200'] = (dataframe['close'] - dataframe['ema200']) / (dataframe['ema200'] + 0.000001) * 100
        
        # MA alignment
        dataframe['ma_alignment_bull'] = (dataframe['ema21'] > dataframe['ema50']) & (dataframe['ema50'] > dataframe['ema200'])
        dataframe['ma_alignment_bear'] = (dataframe['ema21'] < dataframe['ema50']) & (dataframe['ema50'] < dataframe['ema200'])
        
        # Candle patterns
        dataframe['green_candle'] = dataframe['close'] > dataframe['close'].shift(1)
        dataframe['red_candle'] = dataframe['close'] < dataframe['close'].shift(1)
        dataframe['strong_green'] = (dataframe['close'] > dataframe['close'].shift(1)) & (dataframe['volume_ratio'] > 1.2)
        dataframe['strong_red'] = (dataframe['close'] < dataframe['close'].shift(1)) & (dataframe['volume_ratio'] > 1.2)
        
        # Pattern detection helpers
        dataframe['rejected_at_ema21'] = (dataframe['high'].shift(1) > dataframe['ema21'].shift(1)) & (dataframe['close'].shift(1) < dataframe['ema21'].shift(1))
        dataframe['rejected_at_ema50'] = (dataframe['high'].shift(1) > dataframe['ema50'].shift(1)) & (dataframe['close'].shift(1) < dataframe['ema50'].shift(1))
        dataframe['bounced_from_ema21'] = (dataframe['low'].shift(1) < dataframe['ema21'].shift(1)) & (dataframe['close'].shift(1) > dataframe['ema21'].shift(1))
        dataframe['bounced_from_ema50'] = (dataframe['low'].shift(1) < dataframe['ema50'].shift(1)) & (dataframe['close'].shift(1) > dataframe['ema50'].shift(1))
        
        # Volatility
        dataframe['volatility'] = dataframe['atr'] / (dataframe['close'] + 0.000001)
        
        # Fill any NaN values with safe defaults
        fillna_config = {
            'rsi': 50, 'adx': 25, 'willr': -50, 'cci': 0,
            'bb_percent': 0.5, 'price_position': 0.5, 'volatility': 0.02,
            'macd': 0, 'macdsignal': 0, 'macdhist': 0,
            'volume_ratio': 1.0, 'price_vs_ema21': 0, 'price_vs_ema50': 0, 'price_vs_ema200': 0
        }
        
        for col, default_val in fillna_config.items():
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].bfill().fillna(default_val)
        
        # Fill EMA columns with close price as fallback
        for col in ['ema_short', 'ema_long', 'ema9', 'ema20', 'ema21', 'ema50', 'ema200']:
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].bfill().fillna(dataframe['close'])
        
        # Fill BB columns
        for col in ['bb_upper', 'bb_middle', 'bb_lower']:
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].bfill().fillna(dataframe['close'])
        
        # Fill KC columns
        for col in ['kc_upperband', 'kc_middleband', 'kc_lowerband']:
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].bfill().fillna(dataframe['close'])
        
        # Fill boolean columns with False
        for col in ['ma_alignment_bull', 'ma_alignment_bear', 'green_candle', 'red_candle', 
                    'strong_green', 'strong_red', 'rejected_at_ema21', 'rejected_at_ema50',
                    'bounced_from_ema21', 'bounced_from_ema50']:
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].fillna(False)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🆕 V71 PHASE 3: SLOPE-BASED DIVERGENCE DETECTION (Professional Algorithm)
        # ═══════════════════════════════════════════════════════════════════════════════
        # Algorithm: Draw trendlines between pivot points and compare slopes
        # Divergence = opposite slope directions (Price ↓ RSI ↑ or Price ↑ RSI ↓)
        
        # Find Pivot Points (4-candle local extrema) for trendline drawing
        lookback = 20  # Search window for divergence patterns
        if len(dataframe) >= lookback + 5:
            # Price Troughs (local lows) - Pivot Points for bullish divergence
            dataframe['price_trough'] = (
                (dataframe['low'] < dataframe['low'].shift(1)) &
                (dataframe['low'] < dataframe['low'].shift(2)) &
                (dataframe['low'] < dataframe['low'].shift(-1)) &
                (dataframe['low'] < dataframe['low'].shift(-2))
            ).fillna(False)
            
            # Price Peaks (local highs) - Pivot Points for bearish divergence
            dataframe['price_peak'] = (
                (dataframe['high'] > dataframe['high'].shift(1)) &
                (dataframe['high'] > dataframe['high'].shift(2)) &
                (dataframe['high'] > dataframe['high'].shift(-1)) &
                (dataframe['high'] > dataframe['high'].shift(-2))
            ).fillna(False)
            
            # RSI Troughs (local RSI lows) - Pivot Points
            dataframe['rsi_trough'] = (
                (dataframe['rsi'] < dataframe['rsi'].shift(1)) &
                (dataframe['rsi'] < dataframe['rsi'].shift(2)) &
                (dataframe['rsi'] < dataframe['rsi'].shift(-1)) &
                (dataframe['rsi'] < dataframe['rsi'].shift(-2))
            ).fillna(False)
            
            # RSI Peaks (local RSI highs) - Pivot Points
            dataframe['rsi_peak'] = (
                (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
                (dataframe['rsi'] > dataframe['rsi'].shift(2)) &
                (dataframe['rsi'] > dataframe['rsi'].shift(-1)) &
                (dataframe['rsi'] > dataframe['rsi'].shift(-2))
            ).fillna(False)
            
            # BULLISH DIVERGENCE: Draw trendline between last 2 troughs, compare slopes
            # Price trendline slope: NEGATIVE (↓)
            # RSI trendline slope: POSITIVE (↑)
            dataframe['bullish_divergence'] = False
            dataframe['bullish_divergence_strength'] = 0.0
            
            for i in range(len(dataframe)):
                if i < lookback:
                    continue
                    
                # Find last 2 price troughs in lookback window
                recent_data = dataframe.iloc[i-lookback:i+1]
                price_trough_indices = recent_data[recent_data['price_trough'] == True].index
                
                if len(price_trough_indices) >= 2:
                    # Get last 2 troughs (pivot points)
                    trough_1_idx = price_trough_indices[-2]
                    trough_2_idx = price_trough_indices[-1]
                    
                    # Calculate time distance between pivots
                    idx_1 = dataframe.index.get_loc(trough_1_idx)
                    idx_2 = dataframe.index.get_loc(trough_2_idx)
                    time_distance = idx_2 - idx_1
                    
                    # Skip if pivots too close (< 3 candles)
                    if time_distance < 3:
                        continue
                    
                    # Get values at pivot points
                    price_1 = dataframe.loc[trough_1_idx, 'low']
                    price_2 = dataframe.loc[trough_2_idx, 'low']
                    rsi_1 = dataframe.loc[trough_1_idx, 'rsi']
                    rsi_2 = dataframe.loc[trough_2_idx, 'rsi']
                    
                    # Calculate trendline slopes (rise/run) - normalized by time
                    price_slope = (price_2 - price_1) / time_distance
                    rsi_slope = (rsi_2 - rsi_1) / time_distance
                    
                    # BULLISH DIVERGENCE: Price slope < 0 (falling) AND RSI slope > 0 (rising)
                    if price_slope < 0 and rsi_slope > 0:
                        dataframe.loc[i, 'bullish_divergence'] = True
                        
                        # Strength: Based on angle difference (larger divergence = stronger)
                        price_pct_change = (price_2 - price_1) / price_1 * 100
                        rsi_pct_change = (rsi_2 - rsi_1) / rsi_1 * 100
                        
                        # Normalize strength to 0.0-1.0
                        divergence_strength = min(abs(price_pct_change) + abs(rsi_pct_change), 10.0) / 10.0
                        dataframe.loc[i, 'bullish_divergence_strength'] = divergence_strength
            
            # BEARISH DIVERGENCE: Draw trendline between last 2 peaks, compare slopes
            # Price trendline slope: POSITIVE (↑)
            # RSI trendline slope: NEGATIVE (↓)
            dataframe['bearish_divergence'] = False
            dataframe['bearish_divergence_strength'] = 0.0
            
            for i in range(len(dataframe)):
                if i < lookback:
                    continue
                    
                # Find last 2 price peaks in lookback window
                recent_data = dataframe.iloc[i-lookback:i+1]
                price_peak_indices = recent_data[recent_data['price_peak'] == True].index
                
                if len(price_peak_indices) >= 2:
                    # Get last 2 peaks (pivot points)
                    peak_1_idx = price_peak_indices[-2]
                    peak_2_idx = price_peak_indices[-1]
                    
                    # Calculate time distance between pivots
                    idx_1 = dataframe.index.get_loc(peak_1_idx)
                    idx_2 = dataframe.index.get_loc(peak_2_idx)
                    time_distance = idx_2 - idx_1
                    
                    # Skip if pivots too close (< 3 candles)
                    if time_distance < 3:
                        continue
                    
                    # Get values at pivot points
                    price_1 = dataframe.loc[peak_1_idx, 'high']
                    price_2 = dataframe.loc[peak_2_idx, 'high']
                    rsi_1 = dataframe.loc[peak_1_idx, 'rsi']
                    rsi_2 = dataframe.loc[peak_2_idx, 'rsi']
                    
                    # Calculate trendline slopes (rise/run) - normalized by time
                    price_slope = (price_2 - price_1) / time_distance
                    rsi_slope = (rsi_2 - rsi_1) / time_distance
                    
                    # BEARISH DIVERGENCE: Price slope > 0 (rising) AND RSI slope < 0 (falling)
                    if price_slope > 0 and rsi_slope < 0:
                        dataframe.loc[i, 'bearish_divergence'] = True
                        
                        # Strength: Based on angle difference (larger divergence = stronger)
                        price_pct_change = (price_2 - price_1) / price_1 * 100
                        rsi_pct_change = (rsi_2 - rsi_1) / rsi_1 * 100
                        
                        # Normalize strength to 0.0-1.0
                        divergence_strength = min(abs(price_pct_change) + abs(rsi_pct_change), 10.0) / 10.0
                        dataframe.loc[i, 'bearish_divergence_strength'] = divergence_strength
        else:
            # Not enough data - set all divergence columns to False/0
            dataframe['price_trough'] = False
            dataframe['price_peak'] = False
            dataframe['rsi_trough'] = False
            dataframe['rsi_peak'] = False
            dataframe['bullish_divergence'] = False
            dataframe['bullish_divergence_strength'] = 0.0
            dataframe['bearish_divergence'] = False
            dataframe['bearish_divergence_strength'] = 0.0
        
        # HIGH-QUALITY DIVERGENCE (V71 Enhanced with Slope Strength)
        # HQ = Strong divergence signal + ideal market conditions + momentum confirmation
        dataframe['bullish_divergence_hq'] = (
            dataframe['bullish_divergence'] &
            (dataframe['bullish_divergence_strength'] > 0.5) &  # Strong slope divergence
            (dataframe['rsi'] < 35) &  # RSI Oversold zone
            (dataframe['volume_ratio'] > 1.3) &  # Volume confirmation
            (dataframe['close'] > dataframe['close'].shift(1))  # Price momentum turning up
        ).fillna(False)
        
        dataframe['bearish_divergence_hq'] = (
            dataframe['bearish_divergence'] &
            (dataframe['bearish_divergence_strength'] > 0.5) &  # Strong slope divergence
            (dataframe['rsi'] > 65) &  # RSI Overbought zone
            (dataframe['volume_ratio'] > 1.3) &  # Volume confirmation
            (dataframe['close'] < dataframe['close'].shift(1))  # Price momentum turning down
        ).fillna(False)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🎯 COMBINE MULTI-TIMEFRAME SIGNALS
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Use 1h as primary indicators (good balance between noise and responsiveness)
        dataframe['fisher'] = dataframe['fisher_1h']
        dataframe['fisher_smooth_long'] = dataframe['fisher_smooth_long_1h']
        dataframe['fisher_smooth_short'] = dataframe['fisher_smooth_short_1h']
        dataframe['baseline'] = dataframe['baseline_1h']
        dataframe['market_regime'] = dataframe['market_regime_1h']
        dataframe['ml_confidence'] = dataframe['ml_confidence_1h']
        dataframe['ml_signal_strength'] = dataframe['ml_signal_strength_1h']
        
        # Pattern analysis from 1h (more reliable than 15m)
        dataframe['pattern_weighted_bullish'] = dataframe['pattern_weighted_bullish_1h']
        dataframe['pattern_weighted_bearish'] = dataframe['pattern_weighted_bearish_1h']
        dataframe['pattern_net_strength'] = dataframe['pattern_net_strength_1h']
        dataframe['pattern_confirmed_bullish'] = dataframe['pattern_confirmed_bullish_1h']
        dataframe['pattern_confirmed_bearish'] = dataframe['pattern_confirmed_bearish_1h']
        dataframe['pattern_confirmation_quality'] = dataframe['pattern_confirmation_quality_1h']
        dataframe['best_bullish_pattern'] = dataframe['best_bullish_pattern_1h']
        dataframe['best_bearish_pattern'] = dataframe['best_bearish_pattern_1h']
        
        # Add 4h trend confirmation
        dataframe['trend_4h'] = dataframe['market_regime_4h']
        dataframe['fisher_4h_trend'] = dataframe['fisher_4h']
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🎓 LEARN FROM HISTORICAL PATTERN OUTCOMES (if we have recent candles)
        # ═══════════════════════════════════════════════════════════════════════════════
        if len(dataframe) > 50:  # Need enough history
            self._learn_from_candle_patterns(dataframe, pair)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # � V71 PHASE 4: FLAG & TRIANGLE PATTERN DETECTION (15m base timeframe)
        # ═══════════════════════════════════════════════════════════════════════════════
        # Detect Bull Flags, Bear Flags, Triangles on 15m data (more responsive than 1h)
        if len(dataframe) > 30:  # Need enough history for pattern detection
            try:
                # Detect all flag/triangle patterns
                patterns = self.pattern_recognition.detect_all_patterns(dataframe, direction='both')
                
                # Bull Flag Detection
                if 'bull_flag' in patterns:
                    bull_flag_detected, bull_flag_confidence = patterns['bull_flag']
                    dataframe['bull_flag_detected'] = bull_flag_detected
                    dataframe['bull_flag_confidence'] = bull_flag_confidence
                else:
                    dataframe['bull_flag_detected'] = False
                    dataframe['bull_flag_confidence'] = 0.0
                
                # Bear Flag Detection
                if 'bear_flag' in patterns:
                    bear_flag_detected, bear_flag_confidence = patterns['bear_flag']
                    dataframe['bear_flag_detected'] = bear_flag_detected
                    dataframe['bear_flag_confidence'] = bear_flag_confidence
                else:
                    dataframe['bear_flag_detected'] = False
                    dataframe['bear_flag_confidence'] = 0.0
                
                # Ascending Triangle Detection
                if 'ascending_triangle' in patterns:
                    asc_tri_detected, asc_tri_confidence = patterns['ascending_triangle']
                    dataframe['ascending_triangle_detected'] = asc_tri_detected
                    dataframe['ascending_triangle_confidence'] = asc_tri_confidence
                else:
                    dataframe['ascending_triangle_detected'] = False
                    dataframe['ascending_triangle_confidence'] = 0.0
                
                # Descending Triangle Detection
                if 'descending_triangle' in patterns:
                    desc_tri_detected, desc_tri_confidence = patterns['descending_triangle']
                    dataframe['descending_triangle_detected'] = desc_tri_detected
                    dataframe['descending_triangle_confidence'] = desc_tri_confidence
                else:
                    dataframe['descending_triangle_detected'] = False
                    dataframe['descending_triangle_confidence'] = 0.0
                
                # Combined Pattern Strength (for entry logic)
                dataframe['chart_pattern_bullish_strength'] = (
                    dataframe['bull_flag_confidence'] * 1.0 +
                    dataframe['ascending_triangle_confidence'] * 0.8
                ).fillna(0.0)
                
                dataframe['chart_pattern_bearish_strength'] = (
                    dataframe['bear_flag_confidence'] * 1.0 +
                    dataframe['descending_triangle_confidence'] * 0.8
                ).fillna(0.0)
                
            except Exception as e:
                logger.warning(f"Pattern detection failed for {pair}: {e}")
                # Fallback: Set all to False/0
                dataframe['bull_flag_detected'] = False
                dataframe['bull_flag_confidence'] = 0.0
                dataframe['bear_flag_detected'] = False
                dataframe['bear_flag_confidence'] = 0.0
                dataframe['ascending_triangle_detected'] = False
                dataframe['ascending_triangle_confidence'] = 0.0
                dataframe['descending_triangle_detected'] = False
                dataframe['descending_triangle_confidence'] = 0.0
                dataframe['chart_pattern_bullish_strength'] = 0.0
                dataframe['chart_pattern_bearish_strength'] = 0.0
        else:
            # Not enough data for pattern detection
            dataframe['bull_flag_detected'] = False
            dataframe['bull_flag_confidence'] = 0.0
            dataframe['bear_flag_detected'] = False
            dataframe['bear_flag_confidence'] = 0.0
            dataframe['ascending_triangle_detected'] = False
            dataframe['ascending_triangle_confidence'] = 0.0
            dataframe['descending_triangle_detected'] = False
            dataframe['descending_triangle_confidence'] = 0.0
            dataframe['chart_pattern_bullish_strength'] = 0.0
            dataframe['chart_pattern_bearish_strength'] = 0.0
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # �📊 LOG WHAT WE'RE LEARNING FROM THIS PAIR (at analysis #3, then every 25)
        # ═══════════════════════════════════════════════════════════════════════════════
        count = self._candle_analysis_count.get(pair, 0)
        if count > 0 and (count == 3 or count % 25 == 0) and len(dataframe) > 10:
            self._log_learning_insights(dataframe, pair, params)
        
        return dataframe
    
    def _populate_informative_indicators(self, dataframe: DataFrame, timeframe_suffix: str) -> DataFrame:
        """
        Calculate all indicators for an informative timeframe
        This runs on 1h and 4h data
        """
        # Basic indicators
        dataframe['atr'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        
        # Fisher Transform
        dataframe['fisher'] = self._calculate_fisher(dataframe, 14)
        dataframe['fisher_smooth_long'] = dataframe['fisher'].ewm(span=9).mean()
        dataframe['fisher_smooth_short'] = dataframe['fisher'].ewm(span=9).mean()
        
        # Baseline
        dataframe['baseline'] = dataframe['close'].ewm(span=14).mean()
        dataframe['baseline_diff'] = dataframe['baseline'].diff()
        dataframe['baseline_up'] = dataframe['baseline_diff'] > 0
        dataframe['baseline_down'] = dataframe['baseline_diff'] < 0
        
        # Goldie Locks Zone
        dataframe['goldie_min'] = dataframe['baseline'] - (dataframe['atr'] * 2.5)
        dataframe['goldie_max'] = dataframe['baseline'] + (dataframe['atr'] * 2.5)
        
        # Bulkowski Patterns with learned reliability
        dataframe = self.pattern_system.analyze_patterns(dataframe)
        
        # Market Regime (enhanced multi-factor) - need params, use global
        params = self.param_manager.global_params
        dataframe['market_regime'] = self._calculate_market_regime(dataframe, params)
        
        # ML Confidence (multi-factor)
        dataframe['ml_confidence'] = self._calculate_ml_confidence(dataframe)
        
        # Signal Strength
        dataframe['ml_signal_strength'] = self._calculate_signal_strength(dataframe)
        
        return dataframe
    
    def _calculate_fisher(self, dataframe: DataFrame, period: int = 14) -> Series:
        """Fisher Transform calculation"""
        median_price = (dataframe['high'] + dataframe['low']) / 2
        fisher = pd.Series(0.0, index=dataframe.index)
        
        for i in range(period, len(dataframe)):
            window = median_price.iloc[i-period:i]
            price_min = window.min()
            price_max = window.max()
            
            if price_max != price_min:
                norm = (median_price.iloc[i] - price_min) / (price_max - price_min)
                norm = 2 * norm - 1
                norm = max(min(norm, 0.999), -0.999)
                fisher.iloc[i] = 0.5 * np.log((1 + norm) / (1 - norm))
        
        return fisher
    
    def _calculate_market_regime(self, dataframe: DataFrame, params: Dict) -> Series:
        """Enhanced multi-factor market regime with dynamic thresholds"""
        sma_50 = dataframe['close'].rolling(50).mean()
        sma_200 = dataframe['close'].rolling(200).mean()
        ema_20 = dataframe['close'].ewm(span=20).mean()
        ema_50 = dataframe['close'].ewm(span=50).mean()
        
        momentum_10 = dataframe['close'].pct_change(10)
        momentum_20 = dataframe['close'].pct_change(20)
        
        # Multi-factor scoring
        score = pd.Series(50.0, index=dataframe.index)
        
        # SMA trend
        score += ((sma_50 > sma_200 * 1.02).astype(int) * 30 - 15)
        
        # EMA trend (faster)
        score += ((ema_20 > ema_50 * 1.01).astype(int) * 30 - 15)
        
        # Momentum
        score += ((momentum_10 > 0.02).astype(int) * 25)
        score -= ((momentum_10 < -0.02).astype(int) * 25)
        
        # Reversal detection
        reversal_threshold = params['reversal_momentum_threshold']
        reversal_risk = (
            ((sma_50 > sma_200) & (momentum_10 < -reversal_threshold)) |
            ((sma_50 < sma_200) & (momentum_10 > reversal_threshold))
        )
        score -= reversal_risk.astype(int) * 40
        
        # Convert to 0-100 regime scale (much more sensitive for 5min)
        regime = score.clip(0, 100)  # Use the actual score instead of -1,0,1
        
        return regime
    
    def _calculate_ml_confidence(self, dataframe: DataFrame) -> Series:
        """Calculate ML confidence from market conditions"""
        atr_norm = dataframe['atr'] / dataframe['close']
        fisher_vol = dataframe['fisher'].rolling(10).std()
        
        # Lower volatility = higher confidence
        confidence = 1.0 - (atr_norm * 2 + fisher_vol * 0.5)
        return confidence.fillna(0.5).clip(0.1, 1.0)
    
    def _calculate_signal_strength(self, dataframe: DataFrame) -> Series:
        """Calculate signal strength"""
        fisher_strength = abs(dataframe['fisher']) / 3.0
        pattern_strength = dataframe['pattern_net_strength'].clip(-1, 1) / 2 + 0.5
        
        strength = fisher_strength * 0.6 + pattern_strength * 0.4
        return strength.fillna(0.5).clip(0.1, 1.0)
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        HYBRID ENTRY LOGIC:
        ===================
        1. Original Fisher Transform entries (ML-based)
        2. Proven entries (Technical + patterns)
        
        ALL ENTRIES tracked separately for performance comparison!
        """
        pair = metadata.get('pair', 'unknown')
        params = self.param_manager.get_all_params(pair)  # Get pair-specific or global
        
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🛡️ V68 NEW: FLASH CRASH PROTECTION - Block entries during extreme volatility
        # ═══════════════════════════════════════════════════════════════════════════════
        if len(dataframe) >= 2:
            is_crash, crash_reason = self.crash_protection.detect_flash_crash(dataframe, pair, self.timeframe)
            
            if is_crash:
                # Block ONLY LAST BAR entries (not entire dataframe)
                dataframe.loc[dataframe.index[-1], ['enter_long', 'enter_short']] = 0
                
                # Register crash for cooldown tracking
                self.crash_protection.register_crash(pair)
                
                logger.warning(f"🛑 {pair} - Flash Crash Detected: {crash_reason}")
                logger.warning(f"   → Entries BLOCKED for {self.crash_protection.crash_cooldown_minutes}min cooldown")
                
                # DON'T return here - let rest of logic run for other bars
            
            # Check cooldown from previous crash
            elif self.crash_protection.in_cooldown(pair):
                cooldown_remaining = self.crash_protection.get_cooldown_remaining(pair)
                
                # Block last bar entries during cooldown
                dataframe.loc[dataframe.index[-1], ['enter_long', 'enter_short']] = 0
                
                logger.info(f"⏸️  {pair} - Flash crash cooldown: {cooldown_remaining:.1f}min remaining")
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🛡️ V72 PHASE 5: LOSS PROTECTION SYSTEM - Entry Blocking (from V63)
        # ═══════════════════════════════════════════════════════════════════════════════
        should_block, block_reason = self.pair_loss_memory.should_block_entry(pair)
        
        if should_block:
            logger.warning(f"�️  V72 LOSS PROTECTION: {pair} - {block_reason}")
            return dataframe
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🔵 SECTION 1: ORIGINAL FISHER TRANSFORM ENTRIES
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Dynamic conditions (Fisher-based)
        ml_conf_ok = dataframe['ml_confidence'] > params['ml_confidence_min']
        signal_strong = dataframe['ml_signal_strength'] > params['ml_signal_strength_min']
        pattern_reliable = dataframe['pattern_weighted_bullish'] > params['pattern_confidence_min']
        
        # Calculate reversal risks
        momentum_10 = dataframe['close'].pct_change(10)
        sma_50 = dataframe['close'].rolling(50).mean()
        sma_200 = dataframe['close'].rolling(200).mean()
        
        bull_reversal = (sma_50 > sma_200) & (momentum_10 < -params['reversal_momentum_threshold'])
        bear_reversal = (sma_50 < sma_200) & (momentum_10 > params['reversal_momentum_threshold'])
        
        # Additional filters
        ema_20 = dataframe['close'].ewm(span=20).mean()
        fisher_extreme_high = dataframe['fisher'] > 3.0
        fisher_extreme_low = dataframe['fisher'] < -3.0
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🎯 LONG Entry - GEWICHTETE PATTERN-BASIERTE LOGIC (Bulkowski)
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🎯 FIX: 5min ADAPTIVE PATTERN LOGIC (Pattern optional if strong Fisher+ML)
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # STRONG pattern confirmation (preferred) - Candlestick Patterns
        pattern_bullish_confirmed = (
            (dataframe['pattern_confirmed_bullish'] >= 1)  # At least 1 confirmed pattern
            & (dataframe['pattern_weighted_bullish'] > 0.6)  # Strong weight
        )
        
        # WEAK pattern confirmation (5min fallback) - Candlestick Patterns
        pattern_bullish_weak = (
            (dataframe['pattern_weighted_bullish'] > 0.2)  # Any pattern signal
        )
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🚩 V71 PHASE 4: FLAG & TRIANGLE PATTERN FILTER (Chart Patterns)
        # ═══════════════════════════════════════════════════════════════════════════════
        # Bull Flag or Ascending Triangle detected with confidence > 0.6
        chart_pattern_bullish = (
            (dataframe['chart_pattern_bullish_strength'] > 0.6)
        ).fillna(False)
        
        # Bear Flag or Descending Triangle detected with confidence > 0.6
        chart_pattern_bearish = (
            (dataframe['chart_pattern_bearish_strength'] > 0.6)
        ).fillna(False)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🎯 SMART QUALITY FILTER: Relaxed base + intelligent scoring
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # BASE conditions (relaxed - viele Kandidaten)
        base_long = (
            (dataframe['fisher'] < params['fisher_buy_threshold'])
            & ml_conf_ok
            & signal_strong
        )
        
        # QUALITY SCORE: Bewerte jeden Entry-Kandidaten (0-100)
        quality_score_long = pd.Series(0.0, index=dataframe.index)
        
        # 1. Fisher Strength (0-20 points) - reduced from 25
        fisher_normalized = ((params['fisher_buy_threshold'] - dataframe['fisher']) / abs(params['fisher_buy_threshold'])).clip(0, 1)
        quality_score_long += fisher_normalized * 20
        
        # 2. Candlestick Pattern Strength (0-25 points) - reduced from 30
        pattern_normalized = (dataframe['pattern_weighted_bullish'] / 1.0).clip(0, 1)
        quality_score_long += pattern_normalized * 25
        
        # 3. Chart Pattern Strength (0-15 points) - NEW for Bull Flags/Triangles
        chart_pattern_normalized = (dataframe['chart_pattern_bullish_strength'] / 1.0).clip(0, 1)
        quality_score_long += chart_pattern_normalized * 15
        
        # 4. ML Confidence (0-20 points)
        ml_normalized = (dataframe['ml_confidence'] / 1.0).clip(0, 1)
        quality_score_long += ml_normalized * 20
        
        # 5. Regime Score (0-10 points) - reduced from 15
        regime_normalized = (dataframe['market_regime'] / 100.0).clip(0, 1)
        quality_score_long += regime_normalized * 10
        
        # 6. Signal Strength (0-10 points)
        signal_normalized = (dataframe['ml_signal_strength'] / 1.0).clip(0, 1)
        quality_score_long += signal_normalized * 10
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🎯 ANTI-TREND PROTECTION: Stricter quality in bearish regime for LONG entries
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Base quality threshold
        quality_threshold = params.get('entry_quality_threshold', 55.0)
        
        # BEARISH REGIME PROTECTION: Higher threshold + confirmed pattern required
        bearish_regime = dataframe['market_regime'] < params['regime_score_bull_min']
        
        # Adjust threshold in bearish regime
        adjusted_threshold_long = pd.Series(quality_threshold, index=dataframe.index)
        adjusted_threshold_long = adjusted_threshold_long.where(
            ~bearish_regime, 
            quality_threshold + 5.0  # +5 points stricter in bearish regime
        )
        
        # FINAL FILTER with regime protection + per-pair choppy/sideways filters
        long_conditions = (
            base_long 
            & (quality_score_long >= adjusted_threshold_long)
            & (
                ~bearish_regime |
                pattern_bullish_confirmed
            )
        )
        if long_conditions.sum() > 0:
            lookback = int(params.get('sideways_lookback_candles', 20))
            vol_avg = dataframe['volatility'].rolling(lookback).mean()
            bbw_avg = dataframe['bb_width'].rolling(lookback).mean()
            adx_ok = dataframe['adx'] >= float(params.get('trend_adx_min_long', 18.0))
            not_sideways = (
                (vol_avg <= float(params.get('sideways_volatility_max', 0.008))).fillna(False) &
                (bbw_avg <= float(params.get('sideways_bb_width_max', 0.045))).fillna(False)
            ) == False
            long_conditions = long_conditions & (adx_ok | not_sideways)
        
        # Store quality score for learning
        dataframe['entry_quality_score_long'] = quality_score_long
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🧠 V66 TREND FILTER: Check if BULL signal allowed based on ML trend detection
        # ═══════════════════════════════════════════════════════════════════════════════
        # Apply trend filter only where long_conditions are True
        if long_conditions.sum() > 0:
            # Get max quality score for this pair (for override logic)
            max_quality = quality_score_long[long_conditions].max() if long_conditions.sum() > 0 else 0
            
            # Check with trend ML system
            if not self.trend_ml.should_allow_bull_signal(pair, dataframe, max_quality):
                # BLOCKED by trend filter
                long_conditions = pd.Series(False, index=dataframe.index)
        
        dataframe.loc[long_conditions, 'enter_long'] = 1
        dataframe.loc[long_conditions, 'enter_tag'] = dataframe.loc[long_conditions, 'best_bullish_pattern']
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🎯 SHORT Entry - GEWICHTETE PATTERN-BASIERTE LOGIC (Bulkowski)
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if self.can_short:
            # STRONG pattern confirmation (preferred)
            pattern_bearish_confirmed = (
                (dataframe['pattern_confirmed_bearish'] >= 1)
                & (dataframe['pattern_weighted_bearish'] > 0.6)
            )
            
            # BASE conditions (relaxed - viele Kandidaten)
            base_short = (
                (dataframe['fisher'] > params['fisher_sell_threshold'])
                & ml_conf_ok
                & signal_strong
            )
            
            # QUALITY SCORE: Bewerte jeden Short-Entry-Kandidaten (0-100)
            quality_score_short = pd.Series(0.0, index=dataframe.index)
            
            # 1. Fisher Strength (0-20 points) - reduced from 25
            fisher_normalized = ((dataframe['fisher'] - params['fisher_sell_threshold']) / abs(params['fisher_sell_threshold'])).clip(0, 1)
            quality_score_short += fisher_normalized * 20
            
            # 2. Candlestick Pattern Strength (0-25 points) - reduced from 30
            pattern_normalized = (dataframe['pattern_weighted_bearish'] / 1.0).clip(0, 1)
            quality_score_short += pattern_normalized * 25
            
            # 3. Chart Pattern Strength (0-15 points) - NEW for Bear Flags/Triangles
            chart_pattern_normalized = (dataframe['chart_pattern_bearish_strength'] / 1.0).clip(0, 1)
            quality_score_short += chart_pattern_normalized * 15
            
            # 4. ML Confidence (0-20 points)
            ml_normalized = (dataframe['ml_confidence'] / 1.0).clip(0, 1)
            quality_score_short += ml_normalized * 20
            
            # 5. Regime Score (0-10 points) - INVERTED for shorts, reduced from 15
            regime_normalized = (1.0 - (dataframe['market_regime'] / 100.0)).clip(0, 1)
            quality_score_short += regime_normalized * 10
            
            # 6. Signal Strength (0-10 points)
            signal_normalized = (dataframe['ml_signal_strength'] / 1.0).clip(0, 1)
            quality_score_short += signal_normalized * 10
            
            # ═══════════════════════════════════════════════════════════════════════════════
            # 🎯 ANTI-TREND PROTECTION: Stricter quality in bullish regime for SHORT entries
            # ═══════════════════════════════════════════════════════════════════════════════
            
            # Base quality threshold
            quality_threshold = params.get('entry_quality_threshold', 55.0)
            
            # BULLISH REGIME PROTECTION: Higher threshold + confirmed pattern required
            bullish_regime = dataframe['market_regime'] > params['regime_score_bear_max']
            
            # Adjust threshold in bullish regime
            adjusted_threshold_short = pd.Series(quality_threshold, index=dataframe.index)
            adjusted_threshold_short = adjusted_threshold_short.where(
                ~bullish_regime,
                quality_threshold + 5.0  # +5 points stricter in bullish regime
            )
            
            # FINAL FILTER with regime protection + per-pair choppy/sideways filters
            short_conditions = (
                base_short
                & (quality_score_short >= adjusted_threshold_short)
                & (
                    ~bullish_regime |
                    pattern_bearish_confirmed
                )
            )
            if short_conditions.sum() > 0:
                lookback = int(params.get('sideways_lookback_candles', 20))
                vol_avg = dataframe['volatility'].rolling(lookback).mean()
                bbw_avg = dataframe['bb_width'].rolling(lookback).mean()
                adx_ok = dataframe['adx'] >= float(params.get('trend_adx_min_short', 18.0))
                not_sideways = (
                    (vol_avg <= float(params.get('sideways_volatility_max', 0.008))).fillna(False) &
                    (bbw_avg <= float(params.get('sideways_bb_width_max', 0.045))).fillna(False)
                ) == False
                short_conditions = short_conditions & (adx_ok | not_sideways)
            
            # Store quality score for learning
            dataframe['entry_quality_score_short'] = quality_score_short
            
            # ═══════════════════════════════════════════════════════════════════════════════
            # 🧠 V66 TREND FILTER: Check if BEAR signal allowed based on ML trend detection
            # ═══════════════════════════════════════════════════════════════════════════════
            # Apply trend filter only where short_conditions are True
            if short_conditions.sum() > 0:
                # Get max quality score for this pair (for override logic)
                max_quality = quality_score_short[short_conditions].max() if short_conditions.sum() > 0 else 0
                
                # Check with trend ML system
                if not self.trend_ml.should_allow_bear_signal(pair, dataframe, max_quality):
                    # BLOCKED by trend filter
                    short_conditions = pd.Series(False, index=dataframe.index)
            
            dataframe.loc[short_conditions, 'enter_short'] = 1
            dataframe.loc[short_conditions, 'enter_tag'] = dataframe.loc[short_conditions, 'best_bearish_pattern']
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🟢 SECTION 2: PROVEN PROFITABLE ENTRY SIGNALS (Only Best Performers!)
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Basic filters
        has_volume = dataframe['volume'] > 0
        vol_ok = dataframe['volatility'] >= 0.005
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 1️⃣ Bear_Simple_Breakdown - V67 ENHANCED (was 54% WR in V66)
        # ═══════════════════════════════════════════════════════════════════════════════
        # V66: 24 trades, 54% WR, +49 USDT (needs improvement)
        # V67 Enhancements:
        # - Stronger volume: 1.3x → 1.8x (better confirmation)
        # - Added 4h downtrend confirmation (prevent counter-trend entries)
        # - Expected: 54% WR → 65-70% WR
        
        # Check 4h downtrend confirmation
        in_downtrend_4h = dataframe['market_regime_4h'] < 45 if 'market_regime_4h' in dataframe.columns else True
        
        bear_simple_breakdown = (
            (dataframe['close'] < dataframe['low'].rolling(10).min().shift(1)) &
            (dataframe['volume'] > dataframe['volume_sma'] * 1.8) &  # V67: 1.3x → 1.8x (stronger volume)
            (dataframe['rsi'] > 35) & (dataframe['rsi'] < 60) &
            (dataframe['ema_short'] < dataframe['ema_long']) &
            in_downtrend_4h &  # V67: Added 4h trend confirmation
            has_volume
        )
        
        # Set signal (only if not already entered via Fisher)
        bear_simple_breakdown_signal = bear_simple_breakdown & (dataframe['enter_tag'] == '')
        dataframe.loc[bear_simple_breakdown_signal, 'enter_short'] = 1
        dataframe.loc[bear_simple_breakdown_signal, 'enter_tag'] = 'Bear_Simple_Breakdown'
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 2️⃣ Bear_Breakdown (+102 USDT → +1626 USDT, 21 trades) - CHAMPION! 🏆
        # ═══════════════════════════════════════════════════════════════════════════════
        bear_breakdown = (
            (dataframe['close'] < dataframe['ema21']) &
            (dataframe['ma_alignment_bear']) &
            (dataframe['volume_ratio'] > 1.5) &
            (dataframe['rsi'] < 60) &
            (dataframe['adx'] > 25) &
            (dataframe['strong_red']) &
            (dataframe['price_vs_ema21'] < -1.0) &
            (dataframe['rejected_at_ema21']) &
            (dataframe['close'] < dataframe['ema50']) &
            has_volume
        )
        
        bear_breakdown_signal = bear_breakdown & (dataframe['enter_tag'] == '')
        dataframe.loc[bear_breakdown_signal, 'enter_short'] = 1
        dataframe.loc[bear_breakdown_signal, 'enter_tag'] = 'Bear_Breakdown'
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 4️⃣ Bear_Trend (+75 USDT, 7 trades) - VERY GOOD!
        # ═══════════════════════════════════════════════════════════════════════════════
        bear_trend = (
            (dataframe['ema_short'] < dataframe['ema_long']) &
            (dataframe['close'] < dataframe['ema20'].shift(1)) &
            (
                (dataframe['close'] < dataframe['low'].shift(2)) |
                (
                    (dataframe['close'].shift(2) < dataframe['ema20'].shift(2)) &
                    (dataframe['high'] >= dataframe['ema20']) &
                    (dataframe['close'] < dataframe['ema_short'])
                )
            ) &
            (dataframe['volume'] > dataframe['volume_sma'] * 1.2) &
            (dataframe['rsi'] > 30) & (dataframe['rsi'] < 65) &
            (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 0.8) &
            (dataframe['close'] < dataframe['close'].shift(3)) &
            (dataframe['adx'] > 15) &
            has_volume
        )
        
        bear_trend_signal = bear_trend & (dataframe['enter_tag'] == '')
        dataframe.loc[bear_trend_signal, 'enter_short'] = 1
        dataframe.loc[bear_trend_signal, 'enter_tag'] = 'Bear_Trend'
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 5️⃣ Bull_Breakdown - ❌ DISABLED IN V67
        # ═══════════════════════════════════════════════════════════════════════════════
        # V66 Performance: 5 trades, 40% WR, -67.83 USDT total loss
        # Problem: Low confidence entries hitting stops (-6% to -12% losses)
        # V67 Decision: DISABLED until signal is completely reworked
        # 
        # Uncomment below to re-enable (NOT RECOMMENDED without fixes):
        # bull_breakdown = (
        #     (dataframe['close'] > dataframe['ema21']) &
        #     (dataframe['ma_alignment_bull']) &
        #     (dataframe['volume_ratio'] > 1.5) &
        #     (dataframe['rsi'] > 40) &
        #     (dataframe['adx'] > 25) &
        #     (dataframe['strong_green']) &
        #     (dataframe['price_vs_ema21'] > 1.0) &
        #     (dataframe['bounced_from_ema21']) &
        #     (dataframe['close'] > dataframe['ema50']) &
        #     has_volume
        # )
        # 
        # bull_breakdown_signal = bull_breakdown & (dataframe['enter_tag'] == '')
        # dataframe.loc[bull_breakdown_signal, 'enter_long'] = 1
        # dataframe.loc[bull_breakdown_signal, 'enter_tag'] = 'Bull_Breakdown'
        
        # 🔧 V67: Bull_Breakdown DISABLED - saved ~67 USDT in losses!
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 6️⃣ Bull_RSV1 - Reversal (+41 USDT, 4 trades) - GOOD!
        # ═══════════════════════════════════════════════════════════════════════════════
        oversold_conditions_count = (
            (dataframe['rsi'] < 25).astype(int) +
            (dataframe['willr'] < -80).astype(int) +
            (dataframe['cci'] < -100).astype(int) +
            (dataframe['close'] <= dataframe['bb_lower']).astype(int) +
            (dataframe['bb_percent'] < 0.1).astype(int)
        )
        
        bull_reversal_signals_count = (
            (dataframe['volume_ratio'] > 1.5).astype(int) +
            (dataframe['close'] > dataframe['open']).astype(int) +
            (dataframe['macdhist'] > dataframe['macdhist'].shift(1)).astype(int) +
            (dataframe['price_position'] < 0.2).astype(int)
        )
        
        bull_rsv1 = (
            (oversold_conditions_count >= 3) &
            (bull_reversal_signals_count >= 2) &
            (dataframe['ema_short'] > dataframe['ema_short'].shift(3)) &
            (dataframe['volume'] > 0)
        )
        
        bull_rsv1_signal = bull_rsv1 & (dataframe['enter_tag'] == '')
        dataframe.loc[bull_rsv1_signal, 'enter_long'] = 1
        dataframe.loc[bull_rsv1_signal, 'enter_tag'] = 'Bull_RSV1'
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 7️⃣ Bull_Div_HQ - High-Quality Divergence (+5.56 USDT, 7 trades) - NEW! ✨
        # ═══════════════════════════════════════════════════════════════════════════════
        bull_div_hq = (
            (dataframe['bullish_divergence_hq']) &
            (dataframe['rsi'] > 20) & (dataframe['rsi'] < 40) &  # Oversold but not extreme
            (dataframe['volume_ratio'] > 1.3) &  # Strong volume confirmation
            (dataframe['close'] > dataframe['open']) &  # Green candle
            has_volume
        )
        
        bull_div_hq_signal = bull_div_hq & (dataframe['enter_tag'] == '')
        dataframe.loc[bull_div_hq_signal, 'enter_long'] = 1
        dataframe.loc[bull_div_hq_signal, 'enter_tag'] = 'Bull_Div_HQ'
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 8️⃣ Bull_Div - Regular Divergence (+5.16 USDT, 5 trades) - NEW! ✨
        # ═══════════════════════════════════════════════════════════════════════════════
        bull_div = (
            (dataframe['bullish_divergence']) &
            (dataframe['rsi'] > 25) & (dataframe['rsi'] < 50) &  # Broader RSI range
            (dataframe['volume_ratio'] > 1.1) &  # Volume confirmation
            has_volume
        )
        
        bull_div_signal = bull_div & (dataframe['enter_tag'] == '')
        dataframe.loc[bull_div_signal, 'enter_long'] = 1
        dataframe.loc[bull_div_signal, 'enter_tag'] = 'Bull_Div'
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 9️⃣ Bull_KC_Bounce - Keltner Channel Bounce (+1.49 USDT, 2 trades) - NEW! ✨
        # ═══════════════════════════════════════════════════════════════════════════════
        bull_kc_bounce = (
            # Touch or break below KC lower band
            (dataframe['low'] <= dataframe['kc_lowerband'] * 1.005) &
            # Bounce confirmation - close back inside
            (dataframe['close'] > dataframe['kc_lowerband']) &
            # Green candle
            (dataframe['close'] > dataframe['open']) &
            # RSI oversold but recovering
            (dataframe['rsi'] > 25) & (dataframe['rsi'] < 45) &
            # Volume confirmation
            (dataframe['volume_ratio'] > 1.2) &
            # Not in strong downtrend
            (dataframe['ema_short'] > dataframe['ema_short'].shift(3)) &
            has_volume & vol_ok
        )
        
        bull_kc_bounce_signal = bull_kc_bounce & (dataframe['enter_tag'] == '')
        dataframe.loc[bull_kc_bounce_signal, 'enter_long'] = 1
        dataframe.loc[bull_kc_bounce_signal, 'enter_tag'] = 'Bull_KC_Bounce'
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🔟 Bull_Flag - Bull Flag Pattern (NEW V71 Phase 4) ✨
        # ═══════════════════════════════════════════════════════════════════════════════
        # Bull Flag: Strong uptrend (flagpole) → consolidation → breakout
        bull_flag = (
            (dataframe['bull_flag_detected']) &
            (dataframe['bull_flag_confidence'] > 0.65) &  # High confidence
            (dataframe['volume_ratio'] > 1.2) &  # Volume confirmation on breakout
            (dataframe['rsi'] > 45) & (dataframe['rsi'] < 70) &  # Not overbought
            (dataframe['close'] > dataframe['open']) &  # Green breakout candle
            has_volume & vol_ok
        )
        
        bull_flag_signal = bull_flag & (dataframe['enter_tag'] == '')
        dataframe.loc[bull_flag_signal, 'enter_long'] = 1
        dataframe.loc[bull_flag_signal, 'enter_tag'] = 'Bull_Flag'
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 1️⃣1️⃣ Bull_Asc_Triangle - Ascending Triangle Pattern (NEW V71 Phase 4) ✨
        # ═══════════════════════════════════════════════════════════════════════════════
        # Ascending Triangle: Higher lows + flat resistance → breakout
        bull_asc_triangle = (
            (dataframe['ascending_triangle_detected']) &
            (dataframe['ascending_triangle_confidence'] > 0.65) &  # High confidence
            (dataframe['volume_ratio'] > 1.3) &  # Strong volume on breakout
            (dataframe['rsi'] > 50) & (dataframe['rsi'] < 75) &  # Momentum building
            (dataframe['close'] > dataframe['high'].rolling(5).max().shift(1)) &  # Breakout confirmation
            has_volume & vol_ok
        )
        
        bull_asc_triangle_signal = bull_asc_triangle & (dataframe['enter_tag'] == '')
        dataframe.loc[bull_asc_triangle_signal, 'enter_long'] = 1
        dataframe.loc[bull_asc_triangle_signal, 'enter_tag'] = 'Bull_Asc_Triangle'
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 1️⃣2️⃣ Bear_Flag - Bear Flag Pattern (NEW V71 Phase 4) ✨
        # ═══════════════════════════════════════════════════════════════════════════════
        # Bear Flag: Strong downtrend (flagpole) → consolidation → breakdown
        bear_flag = (
            (dataframe['bear_flag_detected']) &
            (dataframe['bear_flag_confidence'] > 0.65) &  # High confidence
            (dataframe['volume_ratio'] > 1.2) &  # Volume confirmation on breakdown
            (dataframe['rsi'] > 30) & (dataframe['rsi'] < 55) &  # Not oversold
            (dataframe['close'] < dataframe['open']) &  # Red breakdown candle
            has_volume & vol_ok
        )
        
        bear_flag_signal = bear_flag & (dataframe['enter_tag'] == '')
        dataframe.loc[bear_flag_signal, 'enter_short'] = 1
        dataframe.loc[bear_flag_signal, 'enter_tag'] = 'Bear_Flag'
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 1️⃣3️⃣ Bear_Desc_Triangle - Descending Triangle Pattern (NEW V71 Phase 4) ✨
        # ═══════════════════════════════════════════════════════════════════════════════
        # Descending Triangle: Lower highs + flat support → breakdown
        bear_desc_triangle = (
            (dataframe['descending_triangle_detected']) &
            (dataframe['descending_triangle_confidence'] > 0.65) &  # High confidence
            (dataframe['volume_ratio'] > 1.3) &  # Strong volume on breakdown
            (dataframe['rsi'] > 25) & (dataframe['rsi'] < 50) &  # Momentum dropping
            (dataframe['close'] < dataframe['low'].rolling(5).min().shift(1)) &  # Breakdown confirmation
            has_volume & vol_ok
        )
        
        bear_desc_triangle_signal = bear_desc_triangle & (dataframe['enter_tag'] == '')
        dataframe.loc[bear_desc_triangle_signal, 'enter_short'] = 1
        dataframe.loc[bear_desc_triangle_signal, 'enter_tag'] = 'Bear_Desc_Triangle'
        
        # NOTE: More Band signals could be added based on future performance data
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🧠 V66 FINAL TREND FILTER: Apply to ALL Additional signals
        # ═══════════════════════════════════════════════════════════════════════════════
        # This catches signals that weren't caught by the Fisher filter above
        # Apply trend filter to Additional LONG signals
        additional_long = (dataframe['enter_long'] == 1) & (~dataframe['enter_tag'].str.startswith('CDL', na=False))
        if additional_long.sum() > 0:
            max_quality_long = dataframe.loc[additional_long, 'entry_quality_score_long'].max() if 'entry_quality_score_long' in dataframe.columns else 65
            if not self.trend_ml.should_allow_bull_signal(pair, dataframe, max_quality_long):
                # BLOCKED by trend filter - remove long signals
                dataframe.loc[additional_long, 'enter_long'] = 0
                dataframe.loc[additional_long, 'enter_tag'] = ''
                logger.info(f"🚫 {pair}: BLOCKED {additional_long.sum()} Additional LONG signals (trend filter)")
        
        # Apply trend filter to Additional SHORT signals
        additional_short = (dataframe['enter_short'] == 1) & (~dataframe['enter_tag'].str.startswith('CDL', na=False))
        if additional_short.sum() > 0:
            max_quality_short = dataframe.loc[additional_short, 'entry_quality_score_short'].max() if 'entry_quality_score_short' in dataframe.columns else 65
            if not self.trend_ml.should_allow_bear_signal(pair, dataframe, max_quality_short):
                # BLOCKED by trend filter - remove short signals
                dataframe.loc[additional_short, 'enter_short'] = 0
                dataframe.loc[additional_short, 'enter_tag'] = ''
                logger.info(f"🚫 {pair}: BLOCKED {additional_short.sum()} Additional SHORT signals (trend filter)")
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 📊 LOG ENTRY SUMMARY
        # ═══════════════════════════════════════════════════════════════════════════════
        long_count = (dataframe['enter_long'] == 1).sum()
        short_count = (dataframe['enter_short'] == 1).sum()
        
        if long_count > 0 or short_count > 0:
            logger.info(f"🎯 {pair}: {long_count} LONG, {short_count} SHORT signals (Fisher + Additional)")
            
            # Count by type
            fisher_long = ((dataframe['enter_long'] == 1) & (dataframe['enter_tag'].str.startswith('CDL'))).sum()
            add_long = ((dataframe['enter_long'] == 1) & (~dataframe['enter_tag'].str.startswith('CDL'))).sum()
            fisher_short = ((dataframe['enter_short'] == 1) & (dataframe['enter_tag'].str.startswith('CDL'))).sum()
            add_short = ((dataframe['enter_short'] == 1) & (~dataframe['enter_tag'].str.startswith('CDL'))).sum()
            
            logger.info(f"   Fisher: {fisher_long}L/{fisher_short}S | Add: {add_long}L/{add_short}S")
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        INTELLIGENTE PATTERN-BASED EXITS (Bulkowski)
        Exit when OPPOSITE patterns appear with confirmation!
        Refined to be less aggressive to allow for more trade adjustments.
        """
        pair = metadata.get('pair', 'unknown')
        params = self.param_manager.get_all_params(pair)

        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''

        # --- Flash Crash Exit ---
        if len(dataframe) >= 2 and self.crash_protection.in_cooldown(pair):
            dataframe.loc[dataframe.index[-1], ['exit_long', 'exit_short']] = 1
            dataframe.loc[dataframe.index[-1], 'exit_tag'] = 'flash_crash'

        momentum_10 = dataframe['close'].pct_change(10)

        # --- Refined LONG exit conditions ---
        bearish_pattern_exit = (
            (dataframe['pattern_confirmed_bearish'] >= 1)
            & (dataframe['pattern_weighted_bearish'] > 0.99)  # Stricter pattern signal
            & (dataframe['ml_confidence'] > 0.99)  # High ML confidence required
        )

        fisher_reversal_exit = (
            (dataframe['fisher_smooth_long'].shift() < params['fisher_long_exit'])
            & (dataframe['fisher_smooth_long'] > params['fisher_long_exit'])
            & (dataframe['ml_confidence'] > 0.99) # Require confidence for fisher exit too
        )

        momentum_exit = (
            (momentum_10 < -0.025) # Stricter momentum
            & (dataframe['fisher'] > 2.0) # Stricter fisher
            & (dataframe['ml_confidence'] > 0.99) # Very high confidence
        )

        dataframe.loc[
            bearish_pattern_exit | fisher_reversal_exit | momentum_exit,
            'exit_long'
        ] = 1

        # --- Refined SHORT exit conditions ---
        if self.can_short:
            bullish_pattern_exit = (
                (dataframe['pattern_confirmed_bullish'] >= 1)
                & (dataframe['pattern_weighted_bullish'] > 0.99) # Stricter
                & (dataframe['ml_confidence'] > 0.99) # High ML confidence
            )
            
            fisher_reversal_short_exit = (
                (dataframe['fisher_smooth_short'].shift() > params['fisher_short_exit'])
                & (dataframe['fisher_smooth_short'] < params['fisher_short_exit'])
                & (dataframe['ml_confidence'] > 0.99)
            )

            momentum_short_exit = (
                (momentum_10 > 0.025)
                & (dataframe['fisher'] < -2.0)
                & (dataframe['ml_confidence'] > 0.99)
            )

            dataframe.loc[
                bullish_pattern_exit | fisher_reversal_short_exit | momentum_short_exit,
                'exit_short'
            ] = 1

        return dataframe
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        """
        Dynamic exit with PER-PAIR adaptive stop-loss, refined for adjustment logic.
        """
        params = self.param_manager.get_all_params(pair)
        trade_id = trade.id

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                ml_conf = dataframe['ml_confidence'].iloc[-1]

                # --- Emergency Exit (less aggressive) ---
                if current_profit < -999.4 and ml_conf < 0.01: # Increased loss threshold
                    if trade_id not in self._emergency_exit_triggered:
                        self._emergency_exit_triggered[trade_id] = True
                        logger.warning(f"🚨 EMERGENCY EXIT: {pair} @{current_profit:.1%} loss, ML_conf={ml_conf:.0%} → FULL EXIT")
                    return f"emergency_exit_conf{int(ml_conf*100)}"

                # --- Dynamic Stop Loss ---
                if ml_conf < 0.002: # Stricter confidence for looser stops #previous 0.6
                    stop = params.get('stop_loss_low_conf', -999.25) # Slightly looser # previous -0.02
                elif ml_conf < 0.007:
                    stop = params.get('stop_loss_medium_conf', -999.4)
                else:
                    stop = params.get('stop_loss_high_conf', -999.9)

                if current_profit <= stop:
                    logger.info(f"📉 Dynamic Stop: {pair} ML_conf={ml_conf:.0%} → stop={stop:.1%}")
                    return f"dynamic_stop_loss_conf{int(ml_conf*100)}"
        except Exception as e:
            logger.error(f"Error in custom_exit for {pair}: {e}")

        return None
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: float, max_stake: float, **kwargs) -> Optional[float]:
        """
        State Machine for Trade Adjustments.

        Manages trades through distinct states for more intelligent decision-making.
        States: INITIAL, DEFENDING, PROFIT, MAX_DCA.
        """
        # --- Safe Custom Data Handling ---
        state_data = trade.get_custom_data('state_data') or {}
        last_forced_candle = trade.get_custom_data('last_forced_candle') or -1

        # --- Emergency Capital Release ---
        if self.emergency_capital_release_enabled.value:
            try:
                currency = trade.stake_currency
                total_wallet_balance = self._get_total_wallet_balance(currency)
                current_total_stake = self._get_current_total_stake(currency)

                if total_wallet_balance > 0:
                    max_allowed_stake = total_wallet_balance * self.max_wallet_exposure.value
                    
                    if current_total_stake > (max_allowed_stake * 0.95) and current_profit > self.emergency_capital_release_profit_trigger.value:
                        sell_stake_amount = trade.stake_amount * self.emergency_capital_release_sell_pct.value

                        # --- Min Stake Check ---
                        if sell_stake_amount < min_stake:
                            logger.warning(
                                f"CAPITAL MGMT: Emergency sell for {trade.pair} skipped. "
                                f"Amount {sell_stake_amount:.2f} USD is below min_stake {min_stake:.2f} USD."
                            )
                            return None

                        sell_crypto_amount = sell_stake_amount / current_rate
                        logger.warning(
                            f"CAPITAL MGMT: Emergency capital release for {trade.pair}. "
                            f"Wallet exposure at {current_total_stake/total_wallet_balance:.1%}. Selling {sell_stake_amount:.2f} USD."
                        )
                        return -sell_crypto_amount
            except Exception as e:
                logger.error(f"Error in emergency capital release logic: {e}")

        # --- 3-Candle Forced Decision Logic ---
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if not dataframe.empty:
            try:
                trade_open_date = pd.to_datetime(trade.open_date_utc)
                open_candle_series = dataframe[dataframe['date'] >= trade_open_date]
                if not open_candle_series.empty:
                    open_candle_iloc = dataframe.index.get_loc(open_candle_series.index[0])
                    current_candle_iloc = len(dataframe) - 1
                    candles_since_open = current_candle_iloc - open_candle_iloc
                    
                    if (candles_since_open > 0) and (candles_since_open % self.forced_exit_period.value == 0) and (candles_since_open != last_forced_candle):
                        trade.set_custom_data('last_forced_candle', candles_since_open)
                        
                        if self.replenish_enabled.value and (trade.amount * current_rate) < self.replenish_stake_threshold_usd.value:
                            # Calculate initial stake from order history
                            try:
                                # Get first buy order
                                filled_buy_orders = [
                                    o for o in trade.orders
                                    if o.ft_order_side == 'buy' and o.status == 'closed'
                                ]
                                if filled_buy_orders:
                                    # Sort by date just in case, though order usually preserved
                                    filled_buy_orders.sort(key=lambda x: x.order_date)
                                    initial_order = filled_buy_orders[0]

                                    # Use cost (Notional Value)
                                    notional_value = initial_order.cost if initial_order.cost else (initial_order.stake_amount or initial_order.amount * initial_order.price)

                                    # Adjust for leverage to get actual wallet stake used
                                    # Order.cost in futures is usually the Notional Value (Size), so we divide by leverage
                                    current_leverage = trade.leverage if trade.leverage else 1.0
                                    base_stake = notional_value / current_leverage
                                else:
                                    base_stake = trade.stake_amount
                            except Exception as e:
                                logger.error(f"Error calculating initial stake for replenishment: {e}")
                                # Fallback to current stake if any error
                                base_stake = trade.stake_amount

                            replenish_amount = base_stake * self.replenish_stake_pct.value
                            logger.info(f"FORCED REPLENISH: {trade.pair} low stake. Adding {replenish_amount:.2f} USD.")
                            self._send_adjustment_notification(trade, "Forced Replenish", replenish_amount, current_rate, current_profit)
                            return replenish_amount

                        if current_profit > self.forced_exit_profit_trigger.value:
                            sell_stake_amount = trade.stake_amount * self.forced_exit_sell_pct.value
                            sell_crypto_amount = sell_stake_amount / current_rate
                            logger.info(f"FORCED EXIT (PROFIT): {trade.pair} at {current_profit:.2%}. Selling {sell_stake_amount:.2f} USD.")
                            self._send_adjustment_notification(trade, "Forced Profit Exit", sell_crypto_amount, current_rate, current_profit)
                            return -sell_crypto_amount

                        if current_profit < self.forced_exit_loss_trigger.value:
                            buy_stake_amount = trade.stake_amount * self.forced_exit_buy_pct.value
                            logger.info(f"FORCED DCA (LOSS): {trade.pair} at {current_profit:.2%}. Buying {buy_stake_amount:.2f} USD.")
                            self._send_adjustment_notification(trade, "Forced DCA", buy_stake_amount, current_rate, current_profit)
                            return buy_stake_amount
            except Exception as e:
                logger.error(f"Error in 3-candle forced decision logic for {trade.pair}: {e}")

        # --- State Machine Initialization ---
        original_state_data = state_data.copy()
        if not isinstance(state_data, dict) or 'trade_state' not in state_data:
            # 🚀 STATE RECOVERY: Try to reconstruct state from orders if lost (fixes infinite loop)
            # Count how many partial exits have already happened
            try:
                sell_orders = [o for o in trade.orders if o.ft_order_side == 'sell' and o.status == 'closed']
                adjustment_count = len(sell_orders)

                logger.info(f"🔄 STATE RECOVERY: {trade.pair} - Recovered {adjustment_count} adjustments from order history.")
            except Exception as e:
                logger.error(f"Error recovering state for {trade.pair}: {e}")
                adjustment_count = 0

            state_data = {
                'trade_state': 'INITIAL',
                'last_adjustment_hour': -1,
                'position_adjustment_count': adjustment_count
            }
        
        # Ensure effective_dca_count exists for backward compatibility
        if 'effective_dca_count' not in state_data:
            state_data['effective_dca_count'] = trade.nr_of_successful_buys - 1

        state = state_data.get('trade_state', 'INITIAL')
        logger.debug(f"State for {trade.pair}: {state}, Profit: {current_profit:.2%}")

        # --- State Machine Execution ---
        result = None
        if state == 'INITIAL':
            self._handle_initial_state(trade, current_profit, state_data)
        elif state == 'DEFENDING':
            result = self._handle_defending_state(trade, current_profit, state_data, current_rate, min_stake)
        elif state == 'PROFIT':
            result = self._handle_profit_state(trade, current_profit, state_data, current_rate, min_stake)
        elif state == 'MAX_DCA':
            result = self._handle_max_dca_state(trade, current_profit, state_data, current_rate, min_stake)

        # --- Save state_data if it has changed ---
        if state_data != original_state_data:
            trade.set_custom_data('state_data', state_data)
            logger.info(f"State for {trade.pair} updated to: {state_data}")

        return result

    def _handle_initial_state(self, trade: Trade, current_profit: float, state_data: dict):
        if current_profit < self.initial_loss_trigger.value:
            state_data['trade_state'] = 'DEFENDING'
            state_data['last_dca_loss_level'] = 0.0  # Initialize loss level tracker
            logger.info(f"Transitioning {trade.pair} to DEFENDING state.")
        elif current_profit > self.initial_profit_trigger.value:
            state_data['trade_state'] = 'PROFIT'
            state_data['last_profit_exit_level'] = 0.0
            logger.info(f"Transitioning {trade.pair} to PROFIT state.")

    def _handle_defending_state(self, trade: Trade, current_profit: float, state_data: dict, current_rate: float, min_stake: float) -> Optional[float]:
        effective_dca_count = state_data.get('effective_dca_count', trade.nr_of_successful_buys - 1)

        if effective_dca_count >= self.defending_max_dca.value:
            state_data['trade_state'] = 'MAX_DCA'
            logger.info(f"Transitioning {trade.pair} to MAX_DCA state (effective DCA count: {effective_dca_count}).")
            return None

        # --- Progressive DCA Logic (based on initial stake) ---
        last_dca_loss_level = state_data.get('last_dca_loss_level', 0.0)
        next_dca_trigger = last_dca_loss_level + self.progressive_loss_trigger.value

        if current_profit < next_dca_trigger:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            if not dataframe.empty:
                last_candle = dataframe.iloc[-1]
                ml_conf = last_candle.get('ml_confidence', 0.5)
                
                should_dca = (
                    ((not trade.is_short and last_candle.get('enter_long', 0) == 1) or
                    (trade.is_short and last_candle.get('enter_short', 0) == 1))
                    and ml_conf > self.dca_ml_conf_threshold.value
                )
                
                if should_dca:
                    dca_amount = trade.stake_amount * self.progressive_buy_pct.value
                    
                    # --- Min Stake Check ---
                    if dca_amount < min_stake:
                        logger.warning(
                            f"DEFENDING: DCA for {trade.pair} skipped. "
                            f"Amount {dca_amount:.2f} USD is below min_stake {min_stake:.2f} USD."
                        )
                        return None

                    if self.capital_management_enabled.value:
                        try:
                            currency = trade.stake_currency
                            total_wallet_balance = self._get_total_wallet_balance(currency)
                            current_total_stake = self._get_current_total_stake(currency)

                            if total_wallet_balance > 0:
                                max_allowed_stake = total_wallet_balance * self.max_wallet_exposure.value
                                if current_total_stake + dca_amount > max_allowed_stake:
                                    logger.warning(f"CAPITAL MGMT: DCA for {trade.pair} blocked.")
                                    return None
                        except Exception as e:
                            logger.error(f"Error in capital management check (DCA): {e}")

                    state_data['last_dca_loss_level'] = next_dca_trigger
                    state_data['effective_dca_count'] = effective_dca_count + 1
                    logger.info(f"DEFENDING state: Progressive DCA for {trade.pair}, adding {dca_amount:.2f} USD. New effective DCA count: {state_data['effective_dca_count']}")
                    self._send_adjustment_notification(trade, "Progressive DCA", dca_amount, current_rate, current_profit)
                    return dca_amount
                else:
                    logger.info(f"DEFENDING state: DCA for {trade.pair} skipped, ML signal/confidence not met.")
        
        return None

    def _handle_profit_state(self, trade: Trade, current_profit: float, state_data: dict, current_rate: float, min_stake: float) -> Optional[float]:
        last_profit_level = state_data.get('last_profit_exit_level', 0.0)
        next_profit_trigger = last_profit_level + self.progressive_profit_trigger.value

        if current_profit > next_profit_trigger:
            if state_data.get('position_adjustment_count', 0) < self.profit_max_position_adjustment.value:
                sell_stake_amount = trade.stake_amount * self.progressive_sell_pct.value

                # --- Min Stake Check ---
                if sell_stake_amount < min_stake:
                    logger.warning(
                        f"PROFIT: Partial exit for {trade.pair} skipped. "
                        f"Amount {sell_stake_amount:.2f} USD is below min_stake {min_stake:.2f} USD."
                    )
                    # Do not update state (don't burn this level) if we can't execute
                    return None

                state_data['position_adjustment_count'] += 1
                state_data['last_profit_exit_level'] = next_profit_trigger
                
                # Decrement effective DCA count, with a floor of 0
                effective_dca_count = state_data.get('effective_dca_count', 0)
                if effective_dca_count > 0:
                    state_data['effective_dca_count'] = effective_dca_count - 1
                
                logger.info(f"PROFIT state: Progressive partial exit for {trade.pair}, selling {sell_stake_amount:.2f} USD. New effective DCA count: {state_data['effective_dca_count']}")
                sell_crypto_amount = sell_stake_amount / current_rate
                self._send_adjustment_notification(trade, "Progressive Profit Exit", sell_crypto_amount, current_rate, current_profit)
                return -sell_crypto_amount

        current_stake_usd = trade.amount * current_rate
        if current_stake_usd < trade.stake_amount * self.profit_reload_threshold.value and last_profit_level > 0:
            if current_profit < last_profit_level:
                try:
                    dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                    if not dataframe.empty:
                        last_candle = dataframe.iloc[-1]
                        ml_conf = last_candle.get('ml_confidence', 0.5)
                        if ((not trade.is_short and last_candle['enter_long'] == 1) or
                            (trade.is_short and last_candle['enter_short'] == 1)) and \
                           ml_conf > self.dca_ml_conf_threshold.value:
                            
                            dca_amount = trade.stake_amount * self.profit_dca_multiplier.value

                            # --- Min Stake Check ---
                            if dca_amount < min_stake:
                                logger.warning(
                                    f"PROFIT: Reload for {trade.pair} skipped. "
                                    f"Amount {dca_amount:.2f} USD is below min_stake {min_stake:.2f} USD."
                                )
                                return None

                            if self.capital_management_enabled.value:
                                try:
                                    currency = trade.stake_currency
                                    total_wallet_balance = self._get_total_wallet_balance(currency)
                                    current_total_stake = self._get_current_total_stake(currency)

                                    if total_wallet_balance > 0:
                                        max_allowed_stake = total_wallet_balance * self.max_wallet_exposure.value
                                        if current_total_stake + dca_amount > max_allowed_stake:
                                            logger.warning(f"CAPITAL MGMT: Reload for {trade.pair} blocked.")
                                            return None
                                except Exception as e:
                                    logger.error(f"Error in capital management check (Reload): {e}")

                            logger.info(f"PROFIT state: Reloading position for {trade.pair}.")
                            state_data['last_profit_exit_level'] = 0.0
                            self._send_adjustment_notification(trade, "Position Reload", dca_amount, current_rate, current_profit)
                            return dca_amount
                except Exception as e:
                    logger.error(f"Error in reload decision for {trade.pair}: {e}")
        return None

    def _handle_max_dca_state(self, trade: Trade, current_profit: float, state_data: dict, current_rate: float, min_stake: float) -> Optional[float]:
        if current_profit > self.max_dca_take_profit_pct.value:
            sell_amount_crypto = trade.amount * self.max_dca_sell_amount_pct.value
            sell_value_usd = sell_amount_crypto * current_rate

            # --- Min Stake Check ---
            # For partial exits in MAX_DCA, we check value
            if sell_value_usd < min_stake: # Assuming min_stake is USD value roughly
                 logger.warning(
                    f"MAX_DCA: Profit take for {trade.pair} skipped. "
                    f"Value {sell_value_usd:.2f} USD is below min_stake."
                )
                 return None

            logger.info(f"MAX_DCA state: Taking profit for {trade.pair}, selling {sell_amount_crypto}")
            self._send_adjustment_notification(trade, "Max_DCA Profit Take", sell_amount_crypto, current_rate, current_profit)
            return -sell_amount_crypto
        
        # De-risking for losing trades that hit max DCA
        if current_profit < 0:
            sell_amount_crypto = trade.amount * self.max_dca_sell_amount_pct.value
            sell_value_usd = sell_amount_crypto * current_rate

            # --- Min Stake Check ---
            if sell_value_usd < min_stake:
                 logger.warning(
                    f"MAX_DCA: De-risk for {trade.pair} skipped. "
                    f"Value {sell_value_usd:.2f} USD is below min_stake."
                )
                 return None

            logger.warning(f"MAX_DCA state: De-risking losing trade for {trade.pair}. Selling {sell_amount_crypto:.8f} ({self.max_dca_sell_amount_pct.value:.0%})")
            
            # Decrement DCA count and reset state to allow "unstucking"
            state_data['effective_dca_count'] = state_data.get('effective_dca_count', self.defending_max_dca.value) - 1
            state_data['trade_state'] = 'DEFENDING'
            logger.info(f"Transitioning {trade.pair} back to DEFENDING. New effective DCA count: {state_data['effective_dca_count']}")

            self._send_adjustment_notification(trade, "Max_DCA De-Risk", sell_amount_crypto, current_rate, current_profit)
            return -sell_amount_crypto

        logger.debug(f"MAX_DCA state: Holding {trade.pair} at {current_profit:.2%}, waiting for profit > {self.max_dca_take_profit_pct.value:.2%} or loss to de-risk.")
        return None
    
    def order_filled(self, pair: str, trade: Trade, order: 'Order', current_time: datetime, **kwargs) -> None:
        """
        🔥 NEW: Called when an order is filled (including exit orders).
        This is the CORRECT place to track trade performance!
        """
        # 🚨 V71 PHASE 1: Cleanup emergency exit tracking when trade closes
        trade_id = trade.id
        if trade_id in self._emergency_exit_triggered:
            del self._emergency_exit_triggered[trade_id]
            logger.debug(f"🧹 V71: Cleaned up emergency exit tracking for trade_id={trade_id}")
        
        # Only track exit orders (completed trades)
        if order.ft_order_side == 'exit' and not trade.is_open:
            try:
                profit = trade.close_profit or 0.0
                duration_min = (trade.close_date_utc - trade.open_date_utc).total_seconds() / 60
                exit_reason = trade.exit_reason or 'unknown'
                
                # Get entry patterns from tag
                entry_patterns = []
                if hasattr(trade, 'enter_tag') and trade.enter_tag:
                    entry_patterns = [trade.enter_tag] if trade.enter_tag.startswith('CDL') else []
                
                # Get market conditions at entry
                try:
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    if not dataframe.empty:
                        ml_conf = dataframe['ml_confidence'].iloc[-1]
                        market_regime = dataframe['market_regime'].iloc[-1]
                    else:
                        ml_conf = 0.5
                        market_regime = 0
                except:
                    ml_conf = 0.5
                    market_regime = 0
                
                # ═══════════════════════════════════════════════════════════════════════════════
                # 🛡️ V72 PHASE 5: UPDATE LOSS PROTECTION MEMORY (V63 system)
                # ═══════════════════════════════════════════════════════════════════════════════
                self.pair_loss_memory.record_trade(pair, profit)
                
                # Log warning for significant losses
                if profit < -0.10:
                    stats = self.pair_loss_memory.get_stats(pair)
                    logger.warning(f"🚨 V72 LOSS PROTECTION: {pair} - Large loss {profit:.2%} detected! "
                                 f"Consecutive losses: {stats['consecutive_losses']}, "
                                 f"Total loss: {stats['total_loss']:.2%}")
                
                # Also update old tracker for compatibility (will be removed in future)
                pair_stats = self.pair_loss_tracker[pair]
                pair_stats['recent_losses'].append(profit)
                
                if profit < 0:
                    pair_stats['consecutive_losses'] += 1
                    pair_stats['total_loss'] += profit
                    pair_stats['last_loss_time'] = current_time
                else:
                    pair_stats['consecutive_losses'] = 0
                
                # Record trade metrics
                trade_data = {
                    'pair': pair,
                    'profit': float(profit),
                    'duration_minutes': int(duration_min),
                    'exit_reason': exit_reason,
                    'entry_patterns': entry_patterns,
                    'ml_confidence': float(ml_conf),
                    'market_regime': float(market_regime),
                    'is_short': trade.is_short,
                    'entry_rate': float(trade.open_rate),
                    'exit_rate': float(trade.close_rate or 0),
                }
                
                # Update all systems
                self.trade_tracker.record_trade(trade_data)
                
                if entry_patterns:
                    self.pattern_system.update_pattern_performance(entry_patterns, profit, pair=pair)
                
                # Check if retraining needed
                if not getattr(self, '_initializing', False):
                    if self.trade_tracker.detect_bad_parameters():
                        logger.warning("🚨 Bad parameters detected - scheduling retraining!")
                        # Trigger on next bot loop
                        self._needs_retraining = True
                
                # Update GLOBAL parameters every 10 trades
                if (not getattr(self, '_initializing', False)) and (len(self.trade_tracker.trades) % 10 == 0):
                    summary = self.trade_tracker.calculate_summary()
                    self.param_manager.update_from_performance(
                        summary['win_rate'],
                        summary['avg_profit'],
                        summary['exit_reasons'],
                        pair=None  # Global update
                    )
                
                # Update PAIR-SPECIFIC parameters every 5 trades for this pair
                pair_trades = [t for t in self.trade_tracker.trades if t.get('pair') == pair]
                if (not getattr(self, '_initializing', False)) and len(pair_trades) >= 5 and len(pair_trades) % 5 == 0:
                    recent_pair_trades = pair_trades[-10:]  # Last 10 for this pair
                    pair_wins = sum(1 for t in recent_pair_trades if t.get('profit', 0) > 0)
                    pair_win_rate = pair_wins / len(recent_pair_trades)
                    pair_avg_profit = np.mean([t.get('profit', 0) for t in recent_pair_trades])
                    
                    # Count exit reasons for this pair
                    pair_exit_reasons = defaultdict(lambda: {'count': 0, 'profit': 0})
                    for t in recent_pair_trades:
                        reason = t.get('exit_reason', 'unknown')
                        pair_exit_reasons[reason]['count'] += 1
                        pair_exit_reasons[reason]['profit'] += t.get('profit', 0)
                    
                    self.param_manager.update_from_performance(
                        pair_win_rate,
                        pair_avg_profit,
                        dict(pair_exit_reasons),
                        pair=pair  # Pair-specific update
                    )
                    
                    logger.info(f"📊 {pair} - Updated params: WR={pair_win_rate:.1%}, "
                              f"Trades={len(pair_trades)}, AvgP/L={pair_avg_profit:.4f}")
                
                # Log using SmartLogger
                SmartLogger.log_trade_result(
                    pair=pair,
                    profit=profit,
                    exit_reason=exit_reason,
                    ml_conf=ml_conf,
                    patterns=entry_patterns,
                    duration_min=int(duration_min)
                )
            
            except Exception as e:
                logger.error(f"Error recording trade in order_filled: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Confirm trade exit (called BEFORE order placement).
        Trade tracking now happens in order_filled() instead.
        """
        # Just confirm the exit, tracking happens in order_filled()
        return True
    
    def bot_loop_start(self, **kwargs):
        """Check for retraining needs + hourly status updates + V66 Trend ML Training + DB Sync"""
        # Retraining check
        if hasattr(self, '_needs_retraining') and self._needs_retraining:
            self.trigger_retraining()
            self._needs_retraining = False
        
        # 🔄 V67 FIX: Sync trades from DB every 5 minutes (order_filled() not working reliably)
        if not hasattr(self, '_last_db_sync'):
            self._last_db_sync = datetime.now()
        
        minutes_since_sync = (datetime.now() - self._last_db_sync).total_seconds() / 60
        
        if minutes_since_sync >= 5.0:  # Every 5 minutes
            self._sync_trades_from_db()
            self._last_db_sync = datetime.now()
        
        # Hourly status update
        if not hasattr(self, '_last_hourly_update'):
            self._last_hourly_update = datetime.now()
        
        hours_since_update = (datetime.now() - self._last_hourly_update).total_seconds() / 3600
        
        if hours_since_update >= 1.0:  # Every hour
            self._log_hourly_status()
            self._last_hourly_update = datetime.now()
        
        # 🧠 V66: Train Trend ML models every 24 hours
        if not hasattr(self, '_last_trend_training'):
            self._last_trend_training = datetime.now()
        
        hours_since_training = (datetime.now() - self._last_trend_training).total_seconds() / 3600
        
        # One-time postponed initial training when data is ready
        if getattr(self, '_initial_trend_training_pending', False):
            try:
                # Check data readiness: pick a few pairs and ensure OHLCV is available
                ready = False
                if hasattr(self.dp, "current_whitelist"):
                    for p in self.dp.current_whitelist()[:5]:
                        df = self.dp.get_pair_dataframe(p, self.timeframe)
                        if df is not None and len(df) >= 100:
                            ready = True
                            break
                if ready:
                    self._train_trend_models()
                    self._initial_trend_training_pending = False
                    self._last_trend_training = datetime.now()
                    logger.info("✅ Initial Trend ML training complete!")
                else:
                    logger.info("⏳ Waiting for OHLCV data before initial Trend ML training...")
            except Exception as e:
                logger.error(f"Error during initial trend training check: {e}")

        if hours_since_training >= 24.0:  # Every 24 hours
            logger.info("🧠 Starting periodic Trend ML training...")
            self._train_trend_models()
            self._last_trend_training = datetime.now()
    
    def _sync_trades_from_db(self):
        """
        🔄 V67 CRITICAL FIX: Sync trades from SQLite DB
        
        Problem: order_filled() callback not working reliably in Freqtrade dev version
        Solution: Scan DB for closed trades and sync them to our tracking systems
        """
        try:
            # Track which trade IDs we've already processed
            if not hasattr(self, '_processed_trade_ids'):
                self._processed_trade_ids = set()
            
            # Get closed trades from Freqtrade DB
            from freqtrade.persistence import Trade as DbTrade
            closed_trades = DbTrade.get_trades(trade_filter=DbTrade.is_open.is_(False)).all()
            
            new_trades_count = 0
            
            for db_trade in closed_trades:
                # Skip if already processed
                if db_trade.id in self._processed_trade_ids:
                    continue
                
                # Skip if not actually closed
                if not db_trade.close_date_utc or not db_trade.close_profit:
                    continue
                
                # Mark as processed
                self._processed_trade_ids.add(db_trade.id)
                new_trades_count += 1
                
                # Extract trade data
                profit = float(db_trade.close_profit or 0.0)
                duration_min = (db_trade.close_date_utc - db_trade.open_date_utc).total_seconds() / 60
                exit_reason = db_trade.exit_reason or 'unknown'
                pair = db_trade.pair
                
                # Get entry patterns from tag
                entry_patterns = []
                if hasattr(db_trade, 'enter_tag') and db_trade.enter_tag:
                    entry_patterns = [db_trade.enter_tag] if db_trade.enter_tag.startswith('CDL') else []
                
                # Get market conditions (use current as fallback, better than nothing)
                try:
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    if not dataframe.empty:
                        ml_conf = float(dataframe['ml_confidence'].iloc[-1])
                        market_regime = float(dataframe['market_regime'].iloc[-1])
                    else:
                        ml_conf = 0.5
                        market_regime = 0
                except:
                    ml_conf = 0.5
                    market_regime = 0
                
                # ═══════════════════════════════════════════════════════════════════════════════
                # 🛡️ UPDATE PAIR LOSS TRACKER (for leverage protection)
                # ═══════════════════════════════════════════════════════════════════════════════
                pair_stats = self.pair_loss_tracker[pair]
                pair_stats['recent_losses'].append(profit)
                
                if profit < 0:
                    pair_stats['consecutive_losses'] += 1
                    pair_stats['total_loss'] += profit
                    pair_stats['last_loss_time'] = db_trade.close_date_utc
                else:
                    # Reset consecutive losses on win
                    pair_stats['consecutive_losses'] = 0
                
                # Build trade data dict
                trade_data = {
                    'pair': pair,
                    'profit': float(profit),
                    'duration_minutes': int(duration_min),
                    'exit_reason': exit_reason,
                    'entry_patterns': entry_patterns,
                    'ml_confidence': float(ml_conf),
                    'market_regime': float(market_regime),
                    'is_short': db_trade.is_short,
                    'entry_rate': float(db_trade.open_rate),
                    'exit_rate': float(db_trade.close_rate or 0),
                    'trade_id': db_trade.id,  # Track DB ID
                }
                
                # Update all systems
                self.trade_tracker.record_trade(trade_data)
                
                if entry_patterns:
                    self.pattern_system.update_pattern_performance(entry_patterns, profit, pair=pair)
                
                # Check if retraining needed
                if (not getattr(self, '_initializing', False)) and (not self._startup_suppress_active()) and self.trade_tracker.detect_bad_parameters():
                    logger.warning("🚨 Bad parameters detected - scheduling retraining!")
                    self._needs_retraining = True
                
                # Update GLOBAL parameters every 10 trades
                if (not getattr(self, '_initializing', False)) and (not self._startup_suppress_active()) and (len(self.trade_tracker.trades) % 10 == 0):
                    summary = self.trade_tracker.calculate_summary()
                    self.param_manager.update_from_performance(
                        summary['win_rate'],
                        summary['avg_profit'],
                        summary['exit_reasons'],
                        pair=None  # Global update
                    )
                
                # Update PAIR-SPECIFIC parameters every 5 trades for this pair
                pair_trades = [t for t in self.trade_tracker.trades if t.get('pair') == pair]
                if (not getattr(self, '_initializing', False)) and (not self._startup_suppress_active()) and len(pair_trades) >= 5 and len(pair_trades) % 5 == 0:
                    recent_pair_trades = pair_trades[-10:]  # Last 10 for this pair
                    pair_wins = sum(1 for t in recent_pair_trades if t.get('profit', 0) > 0)
                    pair_win_rate = pair_wins / len(recent_pair_trades)
                    pair_avg_profit = np.mean([t.get('profit', 0) for t in recent_pair_trades])
                    
                    # Count exit reasons for this pair
                    pair_exit_reasons = defaultdict(lambda: {'count': 0, 'profit': 0})
                    for t in recent_pair_trades:
                        reason = t.get('exit_reason', 'unknown')
                        pair_exit_reasons[reason]['count'] += 1
                        pair_exit_reasons[reason]['profit'] += t.get('profit', 0)
                    
                    self.param_manager.update_from_performance(
                        pair_win_rate,
                        pair_avg_profit,
                        dict(pair_exit_reasons),
                        pair=pair  # Pair-specific update
                    )
            
            if new_trades_count > 0:
                logger.info(f"🔄 DB SYNC: Imported {new_trades_count} closed trades from DB → "
                          f"Total tracked: {len(self.trade_tracker.trades)}")
                
        except Exception as e:
            logger.error(f"❌ Error syncing trades from DB: {e}", exc_info=True)
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str, side: str, **kwargs) -> float:
        """
        🚀 V67 DYNAMIC POSITION SIZING
        
        Adjust stake amount based on:
        1. ML Confidence (higher conf = bigger position)
        2. Quality Score (higher quality = bigger position)
        3. Signal Type (CDLHARAMI/Bull_Div get more!)
        4. Recent Performance (winning streak = bigger positions)
        
        Returns: Adjusted stake amount
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return proposed_stake
            
            last_candle = dataframe.iloc[-1]
            ml_conf = last_candle.get('ml_confidence', 0.5)
            quality_score = last_candle.get('entry_quality_score_long' if side == 'long' else 'entry_quality_score_short', 50)
            
            # Base multiplier = 1.0 (proposed_stake)
            multiplier = 1.0
            
            # ═══════════════════════════════════════════════════════════════════
            # 🎯 ML CONFIDENCE MULTIPLIER (0.5x to 2.0x)
            # ═══════════════════════════════════════════════════════════════════
            if ml_conf > 0.8:
                multiplier *= 1.5
            elif ml_conf > 0.6:
                multiplier *= 1.3
            elif ml_conf < 0.4:
                multiplier *= 1.2
            
            # ═══════════════════════════════════════════════════════════════════
            # 🏆 CHAMPION SIGNAL BONUS (CDLHARAMI & Bull_Div = 100% WR!)
            # ═══════════════════════════════════════════════════════════════════
            if entry_tag in ['CDLHARAMI', 'Bull_Div', 'Bull_Div_HQ']:
                multiplier *= 1.3
                logger.info(f"🏆 CHAMPION BONUS: {entry_tag} → +30% position size")
            elif entry_tag in ['CDLENGULFING']:
                multiplier *= 1.2
            
            # ═══════════════════════════════════════════════════════════════════
            # ⭐ QUALITY SCORE MULTIPLIER
            # ═══════════════════════════════════════════════════════════════════
            if quality_score > 80:
                multiplier *= 1.2
            elif quality_score < 60:
                multiplier *= 0.8
            
            # ═══════════════════════════════════════════════════════════════════
            # 🔥 WINNING STREAK BONUS
            # ═══════════════════════════════════════════════════════════════════
            pair_stats = self.pair_loss_tracker.get(pair, {})
            recent_losses = list(pair_stats.get('recent_losses', []))
            
            if len(recent_losses) >= 3:
                last_3 = recent_losses[-3:]
                if all(profit > 0 for profit in last_3):
                    multiplier *= 1.25
                    logger.info(f"🔥 STREAK BONUS: {pair} 3 wins → +25% position")
            
            # ═══════════════════════════════════════════════════════════════════
            # 📊 IDEAL STAKE CALCULATION (before capital management)
            # ═══════════════════════════════════════════════════════════════════
            ideal_stake = proposed_stake * multiplier
            ideal_stake = max(min_stake, min(ideal_stake, max_stake))

            # --- DYNAMIC CAPITAL MANAGEMENT ---
            final_stake = ideal_stake
            if self.capital_management_enabled.value:
                try:
                    currency = pair.split('/')[1].split(':')[0]
                    # This now represents the total portfolio value, providing a stable base for exposure calculations.
                    total_wallet_balance = self._get_total_wallet_balance(currency)
                    current_total_stake = self._get_current_total_stake(currency)

                    if total_wallet_balance > 0:
                        max_allowed_stake = total_wallet_balance * self.max_wallet_exposure.value
                        available_capital = max(0, max_allowed_stake - current_total_stake)

                        if ideal_stake > available_capital:
                            final_stake = available_capital
                            logger.warning(
                                f"CAPITAL MGMT: Stake for {pair} reduced from {ideal_stake:.2f} to {final_stake:.2f} USD "
                                f"to respect max wallet exposure of {self.max_wallet_exposure.value:.1%}. "
                                f"Available capital: {available_capital:.2f} USD."
                            )

                        # Final check: if the adjusted stake is below min_stake, block the trade.
                        if final_stake < min_stake:
                            logger.warning(
                                f"CAPITAL MGMT: Trade for {pair} blocked. "
                                f"Adjusted stake {final_stake:.2f} is below min_stake {min_stake:.2f} USD."
                            )
                            return 0.0

                except Exception as e:
                    logger.error(f"Error in dynamic capital management: {e}")

            if abs(ideal_stake - final_stake) > 0.01 or multiplier != 1.0:
                 logger.info(f"💰 STAKE: {pair} {entry_tag} | ML={ml_conf:.0%} Q={quality_score:.0f} "
                           f"→ Ideal: {ideal_stake:.2f} → Final: {final_stake:.2f} USDT")

            return final_stake
            
        except Exception as e:
            logger.error(f"Error in custom_stake_amount for {pair}: {e}")
            return proposed_stake
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic custom stop loss with ATR
        
        🔧 V67: Tighter ATR multiplier for better protection
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                atr = dataframe['atr'].iloc[-1]
                ml_conf = dataframe['ml_confidence'].iloc[-1]
                
                # V67: ATR multiplier based on ML confidence
                # High confidence = wider stop, Low confidence = tighter stop
                if ml_conf > 0.7:
                    atr_sl_multip = 992.5  # High conf → wider stop
                elif ml_conf > 0.5:
                    atr_sl_multip = 992.0  # Medium conf
                else:
                    atr_sl_multip = 991.5  # Low conf → tight stop
                
                sl_distance = atr * atr_sl_multip
                
                if not trade.is_short:
                    sl_price = trade.open_rate - sl_distance
                    if current_rate < sl_price:
                        return -0.01
                else:
                    sl_price = trade.open_rate + sl_distance
                    if current_rate > sl_price:
                        return -0.01
        except Exception as e:
            logger.error(f"Error in custom_stoploss for {pair}: {e}")
        
        return self.stoploss
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        🛡️ V72 PHASE 5: Dynamic Leverage with Loss Protection (V63 + V71 hybrid)
        
        Protection mechanisms (V63 proven system):
        1. Large Loss Protection: >10% loss → 1x leverage
        2. Consecutive Loss Protection: 3+ losses → 1x, 2 losses → max 2x
        3. Average Loss Scaling: Avg loss >5% → max 2x
        4. Cooldown Period: Max 2x for 60min after >8% loss
        5. ML Confidence Scaling (V71): Low conf → 1x, High conf → 3x
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                ml_conf = dataframe['ml_confidence'].iloc[-1]
                
                # ═══════════════════════════════════════════════════════════════════
                # 1️⃣ V71: BASE LEVERAGE from ML Confidence (MAX 3X!)
                # ═══════════════════════════════════════════════════════════════════
                if ml_conf < 0.3:
                    base_leverage = 5.0  # Very low confidence = NO leverage
                elif ml_conf < 0.7:
                    base_leverage = 7.0  # Medium confidence = 2x
                else:
                    base_leverage = 10.0  # High confidence = 3x
                
                # ═══════════════════════════════════════════════════════════════════
                # 2️⃣ V72 PHASE 5: LOSS-BASED LEVERAGE REDUCTION (V63 system)
                # ═══════════════════════════════════════════════════════════════════
                adjusted_leverage = self.pair_loss_memory.get_adjusted_leverage(
                    pair=pair,
                    base_leverage=int(base_leverage),
                    ml_conf=ml_conf
                )
                
                # ═══════════════════════════════════════════════════════════════════
                # 3️⃣ FINAL LEVERAGE (capped by max_leverage)
                # ═══════════════════════════════════════════════════════════════════
                final_leverage = min(max(float(adjusted_leverage), 1.0), max_leverage)
                
                # Get stats for logging
                stats = self.pair_loss_memory.get_stats(pair)
                logger.info(f"🎯 {pair} LEVERAGE: ML_conf={ml_conf:.0%} → Base={base_leverage:.1f}x → "
                          f"Adjusted={adjusted_leverage}x → Final={final_leverage:.1f}x "
                          f"(Losses: {stats['consecutive_losses']})")
                
                return final_leverage
                
        except Exception as e:
            logger.error(f"Error calculating leverage for {pair}: {e}")
        
        return min(2.0, max_leverage)  # Safe fallback
    
    def bot_start(self, **kwargs):
        """
        Log system status on start + AUTO-TRAIN Trend ML models
        
        🔧 V67: Automatic Trend ML training on bot start!
        """
        logger.info("="*80)
        logger.info(f"🧠 {self.strategy_version.upper()} {self.strategy_variant.upper()} SYSTEM STATUS")
        logger.info("="*80)
        logger.info(f"🧠 Trend ML Models: {len(self.trend_ml.models)} pairs loaded")
        
        summary = self.trade_tracker.calculate_summary()
        logger.info(f"📊 Total Historical Trades: {summary['total_trades']}")
        logger.info(f"📈 Recent Win Rate: {summary['win_rate']:.1%}")
        logger.info(f"💰 Recent Avg Profit: {summary['avg_profit']:.4f}")
        
        logger.info(f"\n⚙️  GLOBAL DYNAMIC PARAMETERS:")
        for key, value in self.param_manager.global_params.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.4f}")
            else:
                logger.info(f"   {key}: {value}")
        
        # Log pair-specific parameters if any
        if self.param_manager.pair_specific_params:
            logger.info(f"\n💎 PAIR-SPECIFIC PARAMETERS:")
            for pair, params in self.param_manager.pair_specific_params.items():
                logger.info(f"   {pair}:")
                for key, value in params.items():
                    if isinstance(value, float):
                        logger.info(f"      {key}: {value:.4f}")
                    else:
                        logger.info(f"      {key}: {value}")
        
        logger.info("="*80)
        
        # 🔄 DYNAMIC: Sync historical trades from DB on startup
        logger.info(f"\n🔄 {self.strategy_version.upper()}: Syncing historical trades from database...")
        # Suppress per-trade updates/retraining during startup import
        self._initializing = True
        self._sync_trades_from_db()
        # Single global performance update after import
        try:
            summary = self.trade_tracker.calculate_summary()
            self.param_manager.update_from_performance(
                summary.get('win_rate', 0.0),
                summary.get('avg_profit', 0.0),
                summary.get('exit_reasons', {}),
                pair=None
            )
        except Exception:
            pass
        logger.info(f"✅ {self.strategy_version.upper()} DB Sync complete! Total tracked trades: {len(self.trade_tracker.trades)}")
        
        # 🔧 DYNAMIC: Auto-train Trend ML models on bot start
        logger.info(f"\n🧠 {self.strategy_version.upper()}: Initializing Auto Trend ML Training...")
        # Postpone actual training until bot loop (ensures data available)
        self._initial_trend_training_pending = True
        # End of initialization phase
        self._initializing = False
        logger.info("⏳ Trend ML training postponed until data is available")
        
        # 🎯 DYNAMIC: AUTO-OPTUNA - Check if we should run Optuna optimization
        self._check_and_run_initial_optuna()
        
        logger.info("="*80)
    
    def _check_and_run_initial_optuna(self):
        """
        🎯 V67 AUTO-OPTUNA: Intelligent initial optimization
        
        Checks if enough data exists and starts Optuna when:
        1. ≥15 trades available (enough for meaningful optimization)
        2. ≥5 negative trades (diverse data for robust optimization)
        3. No prior Optuna optimization performed
        4. Win rate <70% (improvement potential)
        """
        try:
            total_trades = len(self.trade_tracker.trades)
            
            # Mindestens 15 Trades erforderlich
            if total_trades < 15:
                logger.info(f"⏳ Optuna: {total_trades}/15 trades - waiting for more data...")
                return
            
            # Check if study already exists
            if self.optuna_trainer.study_file.exists():
                logger.info(f"✅ Optuna: Study already present ({self.optuna_trainer.study_file.name})")
                return
            
            # Berechne negative Trades
            negative_trades = sum(1 for t in self.trade_tracker.trades if t.get('profit', 0) < 0)
            
            if negative_trades < 5:
                logger.info(f"⏳ Optuna: {negative_trades}/5 negative trades - need more diverse data...")
                return
            
            # Check win rate
            summary = self.trade_tracker.calculate_summary()
            win_rate = summary.get('win_rate', 0)
            
            if win_rate >= 0.70:
                logger.info(f"🎯 Optuna: Win rate {win_rate:.1%} is already good - no optimization needed")
                return
            
            # 🚀 ALL CONDITIONS MET - START OPTUNA!
            logger.info("="*80)
            logger.info("🎯 AUTO-OPTUNA TRIGGER!")
            logger.info("="*80)
            logger.info(f"📊 Conditions met:")
            logger.info(f"   ✅ Total Trades: {total_trades} (≥15)")
            logger.info(f"   ✅ Negative Trades: {negative_trades} (≥5)")
            logger.info(f"   ✅ Win Rate: {win_rate:.1%} (<70% - Verbesserungspotenzial!)")
            logger.info(f"   ✅ Keine vorherige Optimierung gefunden")
            logger.info("")
            logger.info("🚀 Starte Optuna-Optimierung mit 50 Trials...")
            logger.info("   (Dies kann 5-10 Minuten dauern)")
            logger.info("="*80)
            
            # Run optimization
            new_params = self.optuna_trainer.optimize(n_trials=50)
            
            if new_params:
                logger.info("="*80)
                logger.info("✅ OPTUNA OPTIMIERUNG ABGESCHLOSSEN!")
                logger.info("="*80)
                logger.info("📈 Neue optimierte Parameter:")
                for key, value in new_params.items():
                    if isinstance(value, float):
                        logger.info(f"   {key}: {value:.4f}")
                    else:
                        logger.info(f"   {key}: {value}")
                logger.info("")
                logger.info("💡 Diese Parameter werden ab sofort verwendet!")
                logger.info("="*80)
            else:
                logger.warning("⚠️ Optuna-Optimierung fehlgeschlagen - verwende Standard-Parameter")
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Auto-Optuna Check: {e}", exc_info=True)
    
    def informative_pairs(self):
        """Request 1h and 4h data for multi-timeframe analysis"""
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for pair in pairs:
            informative_pairs.append((pair, self.informative_timeframe_1h))
            informative_pairs.append((pair, self.informative_timeframe_4h))
        return informative_pairs
    
    def _learn_from_candle_patterns(self, dataframe: DataFrame, pair: str):
        """
        🎓 Learn from historical candle patterns BEFORE making trading decisions
        This runs CONTINUOUSLY on every populate_indicators call!
        """
        try:
            # Only learn every 20 candles (avoid overhead)
            if not hasattr(self, '_last_pattern_learn'):
                self._last_pattern_learn = {}
            
            current_len = len(dataframe)
            last_len = self._last_pattern_learn.get(pair, 0)
            
            if current_len - last_len < 20:
                return  # Not enough new candles yet
            
            self._last_pattern_learn[pair] = current_len
            
            # Analyze last 100 candles for pattern outcomes
            if current_len < 100:
                return
            
            recent_df = dataframe.tail(100).copy()
            
            # Track pattern outcomes (did price go up after the pattern?)
            patterns_detected = {}
            
            for i in range(20, len(recent_df)):
                # Check if any patterns were detected at this candle
                if 'pattern_bull_count' in recent_df.columns and recent_df['pattern_bull_count'].iloc[i] > 0:
                    # Look ahead 5-10 candles to see if price increased
                    future_return = 0.0
                    if i + 10 < len(recent_df):
                        future_price = recent_df['close'].iloc[i+10]
                        current_price = recent_df['close'].iloc[i]
                        future_return = (future_price - current_price) / current_price
                    
                    # Track this outcome (simplified - we'd need actual pattern names)
                    if future_return > 0.01:  # >1% gain
                        if 'bullish_patterns' not in patterns_detected:
                            patterns_detected['bullish_patterns'] = {'wins': 0, 'total': 0}
                        patterns_detected['bullish_patterns']['wins'] += 1
                        patterns_detected['bullish_patterns']['total'] += 1
                    elif future_return < -0.01:
                        if 'bullish_patterns' not in patterns_detected:
                            patterns_detected['bullish_patterns'] = {'wins': 0, 'total': 0}
                        patterns_detected['bullish_patterns']['total'] += 1
            
            # Log learnings MORE FREQUENTLY (every 100 candles instead of 200)
            if patterns_detected and current_len % 100 == 0:
                for pattern_type, data in patterns_detected.items():
                    if data['total'] > 0:
                        reliability = data['wins'] / data['total']
                        logger.info(f"📚 {pair} - Pattern Learning: {pattern_type} = {reliability:.1%} reliability ({data['wins']}/{data['total']} trades)")
        
        except Exception as e:
            logger.debug(f"Pattern learning error for {pair}: {e}")
    
    def _log_learning_insights(self, dataframe: DataFrame, pair: str, params: Dict):
        """
        📊 Shows what the system is currently learning from candles
        """
        try:
            recent = dataframe.tail(20)  # Last 20 candles
            
            # Market conditions
            current_regime = recent['market_regime'].iloc[-1] if 'market_regime' in recent.columns else 0
            avg_ml_conf = recent['ml_confidence'].mean() if 'ml_confidence' in recent.columns else 0
            avg_signal = recent['ml_signal_strength'].mean() if 'ml_signal_strength' in recent.columns else 0
            current_fisher = recent['fisher'].iloc[-1] if 'fisher' in recent.columns else 0
            
            # Pattern detection
            patterns_found = 0
            if 'pattern_bull_count' in recent.columns:
                patterns_found = recent['pattern_bull_count'].sum()
            
            regime_str = "🟢 BULLISH" if current_regime > 60 else "🔴 BEARISH" if current_regime < 40 else "🟡 NEUTRAL"
            
            # Pattern confirmation quality (enhanced for 5min)
            pattern_conf_quality = recent['pattern_confirmation_quality'].mean() if 'pattern_confirmation_quality' in recent.columns else 0
            confirmed_bull = recent['pattern_confirmed_bullish'].sum() if 'pattern_confirmed_bullish' in recent.columns else 0
            confirmed_bear = recent['pattern_confirmed_bearish'].sum() if 'pattern_confirmed_bearish' in recent.columns else 0
            
            # NEW: Total pattern counts
            total_patterns = recent['pattern_total_count'].sum() if 'pattern_total_count' in recent.columns else 0
            total_confirmed = recent['pattern_confirmed_count'].sum() if 'pattern_confirmed_count' in recent.columns else 0
            
            # Weighted pattern strength
            avg_weighted_bull = recent['pattern_weighted_bullish'].mean() if 'pattern_weighted_bullish' in recent.columns else 0
            avg_weighted_bear = recent['pattern_weighted_bearish'].mean() if 'pattern_weighted_bearish' in recent.columns else 0
            
            logger.info(f"📊 {pair} Learning Insights (last 20 candles):")
            logger.info(f"   Market: {regime_str} (score: {current_regime:.0f}) | Fisher: {current_fisher:.2f}")
            logger.info(f"   ML Confidence: {avg_ml_conf:.1%} (threshold: {params['ml_confidence_min']:.1%})")
            logger.info(f"   Signal Strength: {avg_signal:.1%} (threshold: {params['ml_signal_strength_min']:.1%})")
            logger.info(f"   Patterns: {total_patterns:.0f} total | Confirmed: {total_confirmed:.0f} | 🟢{avg_weighted_bull:.2f} bullish, 🔴{avg_weighted_bear:.2f} bearish")
            
            # Entry potential - CHECK BOTH DIRECTIONS (simplified)
            core_long_ok = (
                (current_fisher < params['fisher_buy_threshold']) and
                (avg_ml_conf > params['ml_confidence_min']) and
                (avg_signal > params['ml_signal_strength_min']) and
                (current_regime >= params['regime_score_bull_min'])
            )
            
            core_short_ok = self.can_short and (
                (current_fisher > params['fisher_sell_threshold']) and
                (avg_ml_conf > params['ml_confidence_min']) and
                (avg_signal > params['ml_signal_strength_min']) and
                (current_regime <= params['regime_score_bear_max'])
            )
            
            # Check pattern conditions
            any_bull_pattern = avg_weighted_bull > 0.1
            any_bear_pattern = avg_weighted_bear > 0.1
            strong_fisher_long = current_fisher < -1.0
            strong_fisher_short = current_fisher > 1.0
            
            would_enter_long = core_long_ok and (any_bull_pattern or strong_fisher_long)
            would_enter_short = core_short_ok and (any_bear_pattern or strong_fisher_short)
            
            if would_enter_long:
                logger.info(f"   ✅ Would ENTER LONG if conditions persist!")
            elif would_enter_short:
                logger.info(f"   ✅ Would ENTER SHORT if conditions persist!")
            else:
                # Show why no LONG
                missing_long = []
                if current_fisher >= params['fisher_buy_threshold']:
                    missing_long.append(f"Fisher too high for LONG ({current_fisher:.2f})")
                if current_regime < params['regime_score_bull_min']:
                    missing_long.append(f"Regime not bullish ({current_regime:.0f})")
                
                # Show why no SHORT
                missing_short = []
                if self.can_short:
                    if current_fisher <= params['fisher_sell_threshold']:
                        missing_short.append(f"Fisher too low for SHORT ({current_fisher:.2f})")
                    if current_regime > params['regime_score_bear_max']:
                        missing_short.append(f"Regime not bearish ({current_regime:.0f})")
                
                if missing_long and missing_short:
                    logger.info(f"   ⏸️  No LONG: {missing_long[0]} | No SHORT: {missing_short[0]}")
                elif missing_long:
                    logger.info(f"   ⏸️  No entry: {'; '.join(missing_long[:2])}")
                else:
                    logger.info(f"   ⏸️  No entry: Signal/Confidence insufficient")
        
        except Exception as e:
            logger.debug(f"Error in learning insights: {e}")
    
    def _log_hourly_status(self):
        """Hourly status summary"""
        try:
            summary = self.trade_tracker.calculate_summary()
            
            # Get open trades count
            open_trades = 0
            try:
                for pair in ['ETH/USDT:USDT', 'BTC/USDT:USDT', 'SOL/USDT:USDT']:  # Sample pairs
                    trades = Trade.get_trades_proxy(pair=pair, is_open=True)
                    open_trades += len(trades) if trades else 0
            except:
                open_trades = 0
            
            # Calculate per-pair profits
            pair_profits = defaultdict(float)
            for trade in self.trade_tracker.trades[-50:]:  # Last 50 trades
                pair_profits[trade['pair']] += trade.get('profit', 0)
            
            top_pairs = sorted(pair_profits.items(), key=lambda x: x[1], reverse=True)
            
            SmartLogger.log_hourly_status(
                total_trades=summary.get('total_trades', 0),
                open_trades=open_trades,
                win_rate=summary.get('win_rate', 0.5),
                total_profit=summary.get('avg_profit', 0.0) * summary.get('recent_trades', 0),
                top_pairs=top_pairs
            )
            
            # Log pair-specific summaries if we have pair-specific params
            if self.param_manager.pair_specific_params:
                for pair in list(self.param_manager.pair_specific_params.keys())[:5]:  # Top 5
                    pair_trades = [t for t in self.trade_tracker.trades if t.get('pair') == pair]
                    if len(pair_trades) >= 3:
                        wins = sum(1 for t in pair_trades if t.get('profit', 0) > 0)
                        pair_wr = wins / len(pair_trades)
                        pair_avg = np.mean([t.get('profit', 0) for t in pair_trades])
                        
                        # Get best patterns for this pair
                        pattern_profits = defaultdict(list)
                        for t in pair_trades:
                            for p in t.get('entry_patterns', []):
                                pattern_profits[p].append(t.get('profit', 0))
                        
                        best_patterns = []
                        for p, profits in pattern_profits.items():
                            if len(profits) >= 2:
                                wins = sum(1 for pf in profits if pf > 0)
                                best_patterns.append((p, wins / len(profits)))
                        
                        best_patterns.sort(key=lambda x: x[1], reverse=True)
                        
                        SmartLogger.log_learning_summary(
                            pair=pair,
                            trades_count=len(pair_trades),
                            win_rate=pair_wr,
                            avg_profit=pair_avg,
                            best_patterns=best_patterns
                        )
        except Exception as e:
            logger.error(f"Error in hourly status: {e}")

    def _get_total_wallet_balance(self, currency: str) -> float:
        """
        Retrieves the total wallet balance, supporting both live and dry-run modes.
        Includes fallbacks for robustness.
        """
        if self.config['dry_run']:
            return self.dry_run_wallet_balance

        try:
            if self.wallets:
                balance = self.wallets.get_free(currency) + self.wallets.get_used(currency)
                return balance if balance is not None else 0.0
        except Exception as e:
            logger.warning(f"Could not get wallet balance: {e}")

        return 0.0

    def _get_current_total_stake(self, currency: str) -> float:
        """
        Calculates the current total stake, supporting both live and dry-run modes.
        """
        if self.config['dry_run']:
            try:
                open_trades = Trade.get_trades_proxy(is_open=True)
                return sum(trade.stake_amount for trade in open_trades)
            except Exception as e:
                logger.warning(f"Could not calculate total stake in dry-run: {e}")
                return 0.0

        try:
            if self.wallets:
                return self.wallets.get_total_stake_amount()
        except Exception as e:
            logger.warning(f"Could not get total stake amount: {e}")

        return 0.0

    def _send_adjustment_notification(self, trade: Trade, adjustment_type: str, amount: float, current_rate: float, current_profit: float):
        if not self.telegram_adjustment_notification_enabled.value:
            return

        try:
            # Emojis for different adjustment types
            action_emoji_map = {
                "dca": "🛡️", "profit": "💰", "replenish": "⛽", "de-risk": "⚠️", "forced": "🚨"
            }
            action_emoji = "📈"
            for key, emoji in action_emoji_map.items():
                if key in adjustment_type.lower():
                    action_emoji = emoji
                    break
            
            # Determine P/L emoji and string
            profit_emoji = "✅" if current_profit > 0 else "❌"
            profit_str = f"{profit_emoji} P/L: {current_profit:.2%}"
            
            # Calculate P/L in USD
            profit_usd = (current_rate - trade.open_rate) * trade.amount if not trade.is_short else (trade.open_rate - current_rate) * trade.amount
            profit_usd_str = f"({profit_usd:+.2f} USD)"

            # Get pair-specific performance stats from memory
            pair_stats = self.pair_loss_memory.get_stats(trade.pair)
            win_rate = pair_stats.get('win_rate', 0.0)
            avg_profit = pair_stats.get('avg_profit', 0.0)
            consecutive_losses = pair_stats.get('consecutive_losses', 0)

            # Format the message using Markdown
            message = (
                f"{action_emoji} *Trade Adjustment*\n\n"
                f"**Pair:** `{trade.pair}`\n"
                f"**Action:** {adjustment_type.upper()}\n"
                f"**Amount:** `{abs(amount):.4f}`\n\n"
                f"--- **Trade Status** ---\n"
                f"**{profit_str}** {profit_usd_str}\n"
                f"**Entries:** `{trade.nr_of_successful_buys}`\n"
                f"**Stake:** `{trade.stake_amount:.2f}` USD\n"
                f"**Avg Entry:** `{trade.open_rate:.4f}`\n"
                f"**Current Rate:** `{current_rate:.4f}`\n\n"
                f"--- **Pair Performance** ---\n"
                f"**Win Rate:** `{win_rate:.1%}`\n"
                f"**Avg P/L:** `{avg_profit:.2%}`\n"
                f"**Consecutive Losses:** `{consecutive_losses}`\n"
            )
            
            # Freqtrade's method to send telegram messages
            self.dp.send_msg(message)
            
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")

    def _train_trend_models(self):
        """Train Trend ML models for all active pairs"""
        try:
            # Get all pairs from whitelist
            if hasattr(self.dp, "current_whitelist"):
                pairs = self.dp.current_whitelist()
            else:
                logger.warning("⚠️ Cannot access whitelist, skipping trend training")
                return
            
            trained_count = 0
            for pair in pairs:
                try:
                    # Get historical data (fallback to OHLCV fetch if not available)
                    dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)
                    if dataframe is None or len(dataframe) == 0:
                        try:
                            ohlcv = self.dp.ohlcv(pair=pair, timeframe=self.timeframe, limit=600)
                            dataframe = ohlcv if ohlcv is not None else dataframe
                        except Exception:
                            pass
                    
                    if len(dataframe) >= 100:
                        self.trend_ml.train_model(dataframe.copy(), pair)
                        trained_count += 1
                    else:
                        logger.debug(f"Skipping {pair}: Not enough data ({len(dataframe)} candles)")
                        
                except Exception as e:
                    logger.error(f"Error training trend model for {pair}: {e}")
                    continue
            
            logger.info(f"🧠 Trained {trained_count}/{len(pairs)} Trend ML models")
            
        except Exception as e:
            logger.error(f"Error in _train_trend_models: {e}")



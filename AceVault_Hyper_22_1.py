"""
AceVault_Hyper01 (Freqtrade Strategy)
-------------------------------------
Obiettivo: replicare la logica del wallet Hyperliquid descritta:
- 3 posizioni LONG (BTC, ETH, HYPE) che consumano ~50% del margin utilizzato (circa 20% del capitale, dato margin totale ~40%).
- Molte posizioni SHORT su altcoin che consumano il restante ~50% del margin utilizzato (~20% del capitale).
- Nessun ROI/TP/SL/trailing classico: gestione tramite entrate/uscite parziali.
- Entrate parziali piccole e regolari quando la posizione è in perdita; uscite parziali piccole e regolari quando è in profitto.
- Futures cross margin, leverage 5x.

Note operative:
- La strategia non usa indicatori per generare segnali: tiene posizioni "sempre attive" gestendo l'esposizione con i callback.
- Per attivare gli SHORT è necessario usare Freqtrade in modalità futures con un exchange che supporta derivati.
- Configurare il pairlist affinché includa le 3 core coin (BTC, ETH, HYPE) e le altcoin desiderate.
"""

from __future__ import annotations
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from freqtrade.persistence import Trade
from pandas import DataFrame
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import math
import numpy as np
import logging
import os
import csv
try:
    import talib.abstract as ta
except Exception:
    ta = None
    
#docker compose run  --rm freqtrade hyperopt --hyperopt-loss ProfitDrawDownHyperOptLoss --space buy sell --epochs 100 --config user_data/Hyper_02.json --strategy AceVault_Hyper04  --timerange 20250801-20250805 
#docker compose run  --rm freqtrade backtesting --config user_data/Hyper_02.json --strategy AceVault_Hyper04 --export trades --timerange 20250801-20250805


class AceVault_Hyper_22(IStrategy):
    INTERFACE_VERSION = 3

    # Timeframe minimale (non usiamo indicatori)
    timeframe = '1h'
    startup_candle_count = 1

    # Disattiva ROI/Stop/trailing come richiesto
    minimal_roi = {"0": 1000}
    stoploss = -1000.0
    trailing_stop = False

    # Attiva motore di uscita e regolazioni posizione
    use_exit_signal = True
    position_adjustment_enable = True

    # Futures / short
    can_short = True
    # Per memorizzare stato/griglia per pair
    custom_info = {}

    # -------------------- Parametri principali --------------------
    # Target di utilizzo del capitale: margin totale ~40% -> long ~20%, short ~20%
    total_margin_target = DecimalParameter(0.20, 0.50, default=0.30, decimals=2, space='buy', optimize=True)
    long_margin_fraction_of_total = DecimalParameter(0.30, 0.70, default=0.50, decimals=2, space='buy', optimize=True)

    # Ripartizione tra le 3 core long (BTC, ETH, HYPE) — pesi normalizzati
    core_weight_btc = DecimalParameter(0.05, 2.00, default=1.0, decimals=2, space='buy', optimize=False)
    core_weight_eth = DecimalParameter(0.05, 2.00, default=1.0, decimals=2, space='buy', optimize=False)
    core_weight_hype = DecimalParameter(0.05, 2.00, default=1.0, decimals=2, space='buy', optimize=False)

    # Leva fissa 5x richiesta
    fixed_leverage = DecimalParameter(1.0, 10.0, default=10.0, decimals=1, space='buy', optimize=False)

    # Dimensione parziali (stake currency): frazione dell'initial_stake della posizione
    partial_entry_frac = DecimalParameter(0.05, 0.30, default=0.10, decimals=2, space='buy', optimize=True)
    partial_exit_frac = DecimalParameter(0.05, 0.30, default=0.10, decimals=2, space='sell', optimize=True)

    # Gate temporali per la cadenza (stile CSV: cluster circa ogni ~5 minuti)
    add_interval_minutes = DecimalParameter(60.0, 120.0, default=60.0, decimals=1, space='buy', optimize=True)
    # Crescita progressiva dell'intervallo tra entrate parziali: es. 0.10 => +10% per ogni add
    add_interval_growth_pct = DecimalParameter(0.00, 0.1, default=0.0, decimals=2, space='buy', optimize=True)
    reduce_interval_minutes = DecimalParameter(60.0, 120.0, default=60.0, decimals=1, space='sell', optimize=True)
    # Tetto massimo al numero di entrate parziali per trade
    max_partial_entries = IntParameter(0, 100, default=10000, space='buy', optimize=False)

    # Chiusura completa se la posizione è diventata microscopica
    min_close_stake = DecimalParameter(1.0, 50.0, default=5.0, decimals=1, space='sell', optimize=True)
    # Percentuali del total balance sotto le quali chiudere il trade
    # Core: 2% (0.0200), Alt: 0.75% (0.0075)
    min_close_pct_core = DecimalParameter(0.0050, 0.1000, default=0.0200, decimals=4, space='sell', optimize=True)
    min_close_pct_alt = DecimalParameter(0.0010, 0.0500, default=0.0075, decimals=4, space='sell', optimize=True)
    # Soglie per gating dinamico delle partial entry basato su used_pct e lato perdente
    used_pct_double_thr = DecimalParameter(0.50, 0.95, default=0.70, decimals=2, space='buy', optimize=False)
    used_pct_triple_thr = DecimalParameter(0.60, 0.99, default=0.85, decimals=2, space='buy', optimize=False)
    losing_side_share_thr = DecimalParameter(0.50, 0.90, default=0.70, decimals=2, space='buy', optimize=False)
    
    # Equity logging / guardia
    equity_log_interval_minutes = DecimalParameter(1.0, 60.0, default=5.0, decimals=1, space='sell', optimize=True)
    equity_warn_ratio = DecimalParameter(0.05, 0.80, default=0.20, decimals=2, space='sell', optimize=True)
    
    # Parametri leva dinamica basata su volatilità (ATR/close)
    max_leverage = IntParameter(5, 40, default=25, space='sell')
    btc_leverage_cap = IntParameter(5, 40, default=25, space='sell')
    non_btc_leverage_cap = IntParameter(3, 25, default=15, space='sell')
    
    atr_window = IntParameter(10, 50, default=14, space='buy')

    # Stato per pair
    _state: Dict[str, Dict[str, Any]] = {}

    # Pairs core (verranno risolti su pairlist attuale in bot_start)
    CORE_LONG_SYMBOL_HINTS = [
        'BTC/',  # BTC
        'ETH/',  # ETH
        'HYPE/'  # HYPE (assicurarsi che il naming corrisponda all'exchange)
    ]
    _core_pairs: Dict[str, str] = {}
    _initial_wallet: float = 0.0
    _last_equity_log_time: Optional[datetime] = None

    def bot_start(self, *args, **kwargs) -> None:
        """Risolvi i 3 pair core dai suggerimenti rispetto alla whitelist corrente."""
        wl = []
        try:
            if hasattr(self, 'dp') and self.dp is not None:
                wl = list(getattr(self.dp, 'current_whitelist', []) or [])
        except Exception:
            wl = []

        mapped = {}
        for hint in self.CORE_LONG_SYMBOL_HINTS:
            cand = next((p for p in wl if p.upper().startswith(hint.replace('/', '/').upper())), None)
            if cand:
                mapped[hint] = cand
        # Fallback: usa naming comune se non trovati
        self._core_pairs = {
            'BTC': mapped.get('BTC/', 'BTC/USDC:USDC'),
            'ETH': mapped.get('ETH/', 'ETH/USDC:USDC'),
            'HYPE': mapped.get('HYPE/', 'HYPE/USDC:USDC'),
        }

        # Memorizza wallet iniziale per soglia di alert equity
        try:
            if hasattr(self, 'wallets') and self.wallets is not None:
                self._initial_wallet = float(self.wallets.get_total_stake_amount())
        except Exception:
            self._initial_wallet = 0.0

    # ----------------------------- Segnali -----------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=int(self.atr_window.value))
        # Percentuale ATR rispetto al prezzo
        dataframe['atr_pct'] = (
            dataframe['atr'] / dataframe['close']
        ).replace([np.inf, -np.inf], np.nan).ffill()
        # Salva ultimo atr_pct per la leva dinamica
        try:
            pair = metadata.get('pair') if metadata else None
            if pair is not None and len(dataframe) > 0:
                st = self._pair_state(pair)
                last_ap = float(dataframe['atr_pct'].iloc[-1]) if not math.isnan(dataframe['atr_pct'].iloc[-1]) else None
                st['last_atr_pct'] = last_ap
        except Exception:
            pass
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # Core LONG sempre attivi
        if pair in self._core_pairs.values():
            dataframe.loc[:, 'enter_long'] = 1
        else:
            # Tutto il resto: SHORT
            dataframe.loc[:, 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    # ------------------------- Leverage per futures -------------------------
    def _dynamic_leverage(self, pair: str) -> float:
        st = self._pair_state(pair)
        atr_pct = st.get('last_atr_pct', None)
        if atr_pct is None or math.isnan(atr_pct) or atr_pct <= 0:
            return 3.0

        is_btc = 'BTC' in pair.upper()
        is_eth = 'ETH' in pair.upper()
        # Mappatura soglie: low vol → leva alta; high vol → leva bassa
        if atr_pct <= 0.003:  # ~0.3% giornaliero su 5m medio (bassa volatilità)
            lev = 40 if is_btc else 30 if is_eth else 20
        elif atr_pct <= 0.005:
            lev = 30 if is_btc else 20 if is_eth else 15
        elif atr_pct <= 0.010:
            lev = 20 if is_btc else 15 if is_eth else 10
        elif atr_pct <= 0.020:
            lev = 15 if is_btc else 10 if is_eth else 7
        elif atr_pct <= 0.040:
            lev = 10 if is_btc else 8 if is_eth else 5
        else:
            lev = 7 if is_btc else 5 if is_eth else 3

        # Applica cap configurabili
        if is_btc:
            lev = float(min(lev, int(self.btc_leverage_cap.value)))
        else:
            lev = float(min(lev, int(self.non_btc_leverage_cap.value)))
        lev = float(min(lev, int(self.max_leverage.value)))
        # Arrotonda a intero per evitare leva con virgola
        return int(round(max(1.0, lev)))

    # Leverage dinamica per futures/short
    def leverage(self,pair: str,current_time: datetime,current_rate: float,proposed_leverage: float,max_leverage: float,entry_tag: str | None,side: str,**kwargs) -> float:
        # Usa la leva dinamica basata su ATR, limitata da max_leverage del pair
        dyn = float(self._dynamic_leverage(pair))
        return float(max(1.0, min(dyn, float(max_leverage))))
    
    def _pair_state(self, pair: str):
        st = self.custom_info.get(pair)
        if not st:
            st = {
                'grid': None,
                'next_add_idx': 0,
                'next_reduce_idx': 0,
            }
            self.custom_info[pair] = st
        return st

    # --------------------- Stake allocation (iniziale) ---------------------
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """Calcola lo stake iniziale per rispettare:
        - Margin totale target (es. 40%)
        - 50% del margin ai 3 LONG core (BTC/ETH/HYPE), 50% alle SHORT alt
        """
        # Bilancio disponibile (stake currency)
        wallet_balance = None
        try:
            if hasattr(self, 'wallets') and self.wallets is not None:
                wallet_balance = float(self.wallets.get_free(self.stake_currency))
        except Exception:
            wallet_balance = None

        # Target globali
        total_margin = float(self.total_margin_target.value)  # ~0.40
        long_share = float(self.long_margin_fraction_of_total.value)  # ~0.50 del margin totale
        short_share = max(0.0, 1.0 - long_share)

        # Stake target globali (in stake currency)
        if wallet_balance is not None and wallet_balance > 0:
            long_budget = wallet_balance * total_margin * long_share
            short_budget = wallet_balance * total_margin * short_share
        else:
            # Fallback in backtest: basati su proposed_stake
            long_budget = float(proposed_stake) * long_share
            short_budget = float(proposed_stake) * short_share

        # Conta alts
        alt_count = 10
        try:
            if hasattr(self, 'dp') and self.dp is not None:
                wl = list(getattr(self.dp, 'current_whitelist', []) or [])
                alt_count = max(1, len([p for p in wl if p not in self._core_pairs.values()]))
        except Exception:
            alt_count = max(1, alt_count)

        # Pesi core normalizzati
        w_btc = float(self.core_weight_btc.value)
        w_eth = float(self.core_weight_eth.value)
        w_hyp = float(self.core_weight_hype.value)
        w_sum = max(1e-9, (w_btc + w_eth + w_hyp))
        w_btc /= w_sum
        w_eth /= w_sum
        w_hyp /= w_sum

        # Stake per pair
        stake = proposed_stake
        if pair == self._core_pairs.get('BTC'):
            stake = long_budget * w_btc
        elif pair == self._core_pairs.get('ETH'):
            stake = long_budget * w_eth
        elif pair == self._core_pairs.get('HYPE'):
            stake = long_budget * w_hyp
        else:
            stake = short_budget / float(alt_count)

        # Minimo stake dinamico per l'open trade (non per le partial entry):
        # - Core coin (BTC/ETH/HYPE): 5% del capitale
        # - Alt coin: 1.5% del capitale
        # Il "capitale" viene stimato come totale wallet in stake currency, con fallback.
        try:
            capital = None
            try:
                if hasattr(self, 'wallets') and self.wallets is not None:
                    capital = float(self.wallets.get_total_stake_amount())
            except Exception:
                capital = None

            if capital is None or capital <= 0:
                capital = float(wallet_balance) if wallet_balance is not None and wallet_balance > 0 else None
            if capital is None or capital <= 0:
                capital = float(self._initial_wallet) if getattr(self, '_initial_wallet', 0.0) > 0 else None
            if capital is None or capital <= 0:
                capital = float(proposed_stake)

            is_core = pair in self._core_pairs.values()
            floor_pct = 0.05 if is_core else 0.015
            floor_amt = max(0.0, float(capital) * float(floor_pct))
            stake = max(float(stake), float(floor_amt))
        except Exception:
            pass

        # Clamping a min/max
        try:
            if min_stake is not None:
                stake = max(stake, float(min_stake))
        except Exception:
            pass
        try:
            stake = min(float(stake), float(max_stake))
        except Exception:
            pass

        # Memorizza la leverage usata per questo pair (se disponibile) per utilizzi successivi
        try:
            st = self._state.setdefault(pair, {})
            st['last_leverage'] = float(leverage) if leverage is not None else float(self._dynamic_leverage(pair))
        except Exception:
            try:
                st = self._state.setdefault(pair, {})
                st['last_leverage'] = float(self.fixed_leverage.value)
            except Exception:
                pass

        return float(max(0.0, stake))

    def _get_trade_leverage(self, tr: Trade) -> float:
        """Restituisce la leverage da usare per i calcoli.
        Ordine di priorità:
        1) Attribute del Trade (se presente)
        2) Stato della strategia per il pair (last_leverage)
        3) Leverage dinamica calcolata dal pair
        4) Fallback a fixed_leverage
        """
        try:
            if hasattr(tr, 'leverage') and tr.leverage is not None:
                return max(1.0, float(tr.leverage))
        except Exception:
            pass

        try:
            st = self._pair_state(tr.pair)
            lev_st = st.get('last_leverage')
            if lev_st is not None:
                return max(1.0, float(lev_st))
        except Exception:
            pass

        try:
            dyn = float(self._dynamic_leverage(tr.pair))
            return max(1.0, dyn)
        except Exception:
            return max(1.0, float(self.fixed_leverage.value))

    def _get_last_order_time(self, trade: Trade, side: str) -> Optional[datetime]:
        """Recupera l'ultimo timestamp di ordine per il trade dato lato ('buy' o 'sell').
        Cerca di inferire i campi datetime disponibili negli oggetti ordine.
        Ritorna None se non disponibile.
        """
        side_l = (side or '').lower()
        try:
            orders = getattr(trade, 'orders', None)
        except Exception:
            orders = None
        if not orders:
            return None

        def _get_dt(o) -> Optional[datetime]:
            for attr in ['order_date', 'open_date', 'timestamp', 'created_at', 'date', 'time', 'filled_time']:
                try:
                    val = getattr(o, attr, None)
                except Exception:
                    val = None
                if isinstance(val, datetime):
                    return val
            return None

        def _get_side(o) -> Optional[str]:
            for attr in ['side', 'ft_order_side', 'action', 'order_side']:
                try:
                    val = getattr(o, attr, None)
                except Exception:
                    val = None
                if isinstance(val, str) and val:
                    return val.lower()
            # Heuristic: if amount is positive -> buy, negative -> sell
            try:
                amt = getattr(o, 'amount', None)
                if isinstance(amt, (int, float)):
                    return 'buy' if float(amt) >= 0 else 'sell'
            except Exception:
                pass
            return None

        last_dt: Optional[datetime] = None
        for o in list(orders):
            oside = _get_side(o)
            if oside is None:
                continue
            if oside != side_l:
                continue
            odt = _get_dt(o)
            if odt is None:
                continue
            if (last_dt is None) or (odt > last_dt):
                last_dt = odt
        return last_dt

    def _normalize_dt(self, dt: Optional[datetime], ref: datetime) -> Optional[datetime]:
        """Normalizza un datetime alla stessa timezone del riferimento.
        - Se dt è naive (senza tzinfo), applica la tz di ref (o UTC se non disponibile).
        - Se dt ha una tz diversa, lo converte alla tz di ref.
        - Se dt è None o non è un datetime, ritorna None.
        """
        if not isinstance(dt, datetime):
            return None
        try:
            ref_tz = getattr(ref, 'tzinfo', None)
            # Naive -> assegna tz di ref (o UTC)
            if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                return dt.replace(tzinfo=ref_tz or timezone.utc)
            # Aware con tz diversa -> converti
            if ref_tz and dt.tzinfo != ref_tz:
                return dt.astimezone(ref_tz)
        except Exception:
            pass
        return dt

    # ----------------------- Position adjustment (DCA) ----------------------
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """Gestione di entrate/uscite parziali basata su profit/loss.
        - Se la posizione è in perdita (current_profit < 0) e l'intervallo minimo è trascorso:
          aggiungi una frazione fissa dell'initial_stake.
        - Se la posizione è in profitto (current_profit > 0) e l'intervallo minimo è trascorso:
          riduci di una frazione fissa dell'initial_stake.
        """
        if not trade.is_open:
            return None

        pair = trade.pair
        st = self._state.setdefault(pair, {})

        # Reset dello stato se è un nuovo trade (evita carry-over tra posizioni diverse sullo stesso pair)
        try:
            prev_tid = st.get('trade_id')
            cur_tid = getattr(trade, 'id', None)
            if prev_tid is None or prev_tid != cur_tid:
                # Nuovo trade o ripristino dopo riavvio: inizializza stato
                st['trade_id'] = cur_tid
                st['initial_stake'] = float(getattr(trade, 'stake_amount', 0.0))
                st['add_count'] = int(0)
                # Recupera tempi da ordini precedenti, altrimenti usa open_date o current_time
                try:
                    add_dt = self._get_last_order_time(trade, 'buy')
                    red_dt = self._get_last_order_time(trade, 'sell')
                except Exception:
                    add_dt = None
                    red_dt = None
                if add_dt is None:
                    try:
                        add_dt = getattr(trade, 'open_date', None)
                    except Exception:
                        add_dt = None
                if red_dt is None:
                    try:
                        red_dt = getattr(trade, 'open_date', None)
                    except Exception:
                        red_dt = None
                st['last_add_time'] = add_dt if isinstance(add_dt, datetime) else current_time
                st['last_reduce_time'] = red_dt if isinstance(red_dt, datetime) else current_time
                st.pop('fixed_exit_stake', None)
        except Exception:
            pass

        # Traccia initial_stake (stake della prima entrata) per dimensionare le parziali
        if 'initial_stake' not in st or st.get('initial_stake', 0.0) <= 0.0:
            try:
                st['initial_stake'] = float(getattr(trade, 'stake_amount', 0.0))
            except Exception:
                st['initial_stake'] = float(0.0)
            # Inizializza il contatore delle entrate parziali per la crescita dell'intervallo
            st['add_count'] = int(0)

        # Calcola importo base per parziali: frazione dell'initial_stake (importo costante, non moltiplicativo)
        base_partial_stake = max(0.0, float(self.partial_entry_frac.value) * float(st.get('initial_stake', 0.0)))
        if base_partial_stake <= 0.0:
            return None

        # Gate temporale
        # Se mancano (ad es. dopo ripartenza), prova a recuperare da ordini
        if 'last_add_time' not in st or st.get('last_add_time') is None:
            rec_add = None
            try:
                rec_add = self._get_last_order_time(trade, 'buy')
            except Exception:
                rec_add = None
            if rec_add is None:
                try:
                    rec_add = getattr(trade, 'open_date', None)
                except Exception:
                    rec_add = None
            norm_add = self._normalize_dt(rec_add, current_time) if isinstance(rec_add, datetime) else None
            st['last_add_time'] = norm_add or current_time
        if 'last_reduce_time' not in st or st.get('last_reduce_time') is None:
            rec_red = None
            try:
                rec_red = self._get_last_order_time(trade, 'sell')
            except Exception:
                rec_red = None
            if rec_red is None:
                try:
                    rec_red = getattr(trade, 'open_date', None)
                except Exception:
                    rec_red = None
            norm_red = self._normalize_dt(rec_red, current_time) if isinstance(rec_red, datetime) else None
            st['last_reduce_time'] = norm_red or current_time
        # Assicura timezone coerente anche se lo stato era già popolato
        last_add: Optional[datetime] = st.get('last_add_time')
        last_red: Optional[datetime] = st.get('last_reduce_time')
        last_add = self._normalize_dt(last_add, current_time) if isinstance(last_add, datetime) else None
        last_red = self._normalize_dt(last_red, current_time) if isinstance(last_red, datetime) else None
        # Intervallo dinamico per le entrate parziali: base * (1 + growth_pct * add_count)
        add_count = int(st.get('add_count', 0))
        base_add = float(self.add_interval_minutes.value)
        growth = float(self.add_interval_growth_pct.value)
        add_gap_minutes = base_add * (1.0 + growth * add_count)
        add_gap = timedelta(minutes=add_gap_minutes)
        red_gap = timedelta(minutes=float(self.reduce_interval_minutes.value))

        # Safety su NaN
        if current_profit is None or math.isnan(current_profit) or math.isinf(current_profit):
            return None

        # Distinzione long/short (profit segno già gestito da Freqtrade)
        is_short = getattr(trade, 'is_short', False)
        _ = is_short  # non serve cambiare segno: usiamo profit già coerente col lato

        # Gating dinamico delle partial entry quando il lato perdente assorbe la maggior parte del capitale
        # e l'utilizzo totale è elevato. Manteniamo invariati gli intervalli di presa profitto (red_gap).
        try:
            # Bilancio corrente (stake currency)
            total_wallet = 0.0
            if hasattr(self, 'wallets') and self.wallets is not None:
                total_wallet = float(self.wallets.get_total_stake_amount())
            if total_wallet <= 0.0 and hasattr(self, 'wallets') and self.wallets is not None:
                total_wallet = float(self.wallets.get_free(self.stake_currency))
            if total_wallet <= 0.0:
                total_wallet = float(getattr(self, '_initial_wallet', 0.0) or 0.0)

            used_margin = self._compute_used_margin(current_time)
            used_pct = (used_margin / total_wallet) if total_wallet > 0 else 0.0

            # Split per lato
            try:
                used_long, used_short = self._compute_used_margin_by_side(current_time)
            except Exception:
                used_long, used_short = 0.0, 0.0
            used_total = float(used_long + used_short)
            share_long = (used_long / used_total) if used_total > 0 else 0.0
            share_short = (used_short / used_total) if used_total > 0 else 0.0

            # PnL non realizzato per lato, per stimare quale lato stia perdendo
            try:
                pnl_long, pnl_short = self._compute_unrealized_pnl_by_side(current_time)
            except Exception:
                pnl_long, pnl_short = 0.0, 0.0
            losing_side = None
            if pnl_long < pnl_short:
                losing_side = 'long'
            elif pnl_short < pnl_long:
                losing_side = 'short'

            # Applica moltiplicatore di difesa solo alle entrate del lato perdente
            risk_mult = 1.0
            share_thr = float(self.losing_side_share_thr.value)
            double_thr = float(self.used_pct_double_thr.value)
            triple_thr = float(self.used_pct_triple_thr.value)
            if losing_side and used_total > 0:
                if losing_side == 'long' and not is_short and share_long >= share_thr:
                    if used_pct >= triple_thr:
                        risk_mult = 3.0
                    elif used_pct >= double_thr:
                        risk_mult = 2.0
                elif losing_side == 'short' and is_short and share_short >= share_thr:
                    if used_pct >= triple_thr:
                        risk_mult = 3.0
                    elif used_pct >= double_thr:
                        risk_mult = 2.0

            # Aggiorna l'intervallo di add con il moltiplicatore di difesa
            add_gap_minutes = base_add * (1.0 + growth * add_count) * risk_mult
            add_gap = timedelta(minutes=add_gap_minutes)

            # Log informativo (silenzioso su eccezioni)
            try:
                logging.getLogger(__name__).debug(
                    (
                        f"[AddIntervalDynamic] time={current_time} side={'SHORT' if is_short else 'LONG'} "
                        f"used_pct={used_pct:.4%} used_total={used_total:.4f} share_long={share_long:.4%} share_short={share_short:.4%} "
                        f"losing_side={losing_side} risk_mult={risk_mult:.1f} add_gap_minutes={add_gap_minutes:.2f}"
                    )
                )
            except Exception:
                pass
        except Exception:
            # In caso di problemi, lascia l'intervallo base
            add_gap_minutes = base_add * (1.0 + growth * add_count)
            add_gap = timedelta(minutes=add_gap_minutes)

        # Regola: in perdita -> add (positivo); in profitto -> reduce (negativo)
        if current_profit < 0.0:
            # Cap sul numero massimo di entrate parziali
            try:
                max_adds = int(self.max_partial_entries.value)
            except Exception:
                max_adds = 20
            if add_count >= max_adds:
                return None
            # Check tempo
            if last_add and (current_time - last_add) < add_gap:
                return None
            
            # Calcola la dimensione attuale della posizione in stake currency
            try:
                current_amount_tokens = float(getattr(trade, 'amount', 0.0))
                lev = self._get_trade_leverage(trade)
                current_position_size = current_amount_tokens * float(current_rate) / max(1.0, float(lev))
            except Exception:
                current_position_size = 0.0
            
            # Calcola il limite massimo di posizione in base al tipo (SHORT: 10%, LONG: 20%)
            is_short = getattr(trade, 'is_short', False)
            max_position_pct = 0.10 if is_short else 0.30  # 10% per SHORT, 25% per LONG
            max_position_size = total_wallet * max_position_pct
            
            # Se la posizione attuale è già al limite, non aggiungere
            # ma permette ad altre operazioni (reduce, exit) di continuare
            if current_position_size >= max_position_size:
                return None
                
            # Calcola quanto si può ancora aggiungere senza superare il limite
            available_to_add = max(0.0, max_position_size - current_position_size)
            stake_to_add = min(float(base_partial_stake), available_to_add)
            
            # Respect min/max
            if min_stake is not None and stake_to_add < float(min_stake):
                stake_to_add = float(min_stake)
            stake_to_add = float(min(stake_to_add, float(max_stake)))
            if stake_to_add <= 0.0:
                return None
            st['last_add_time'] = current_time
            # Incrementa il contatore delle entrate parziali
            try:
                st['add_count'] = int(add_count + 1)
            except Exception:
                st['add_count'] = int(1)
            # Reset pattern di uscita fissa
            st.pop('fixed_exit_stake', None)
            return stake_to_add

        elif current_profit > 0.0:
            # Check tempo
            if last_red and (current_time - last_red) < red_gap:
                return None
            # Dimensione riduzione: usa frazione fissa di initial_stake, con clamp alla size corrente
            try:
                current_amount_tokens = float(getattr(trade, 'amount', 0.0))
                lev = self._get_trade_leverage(trade)
            except Exception:
                current_amount_tokens = 0.0
                try:
                    lev = self._get_trade_leverage(trade)
                except Exception:
                    lev = float(self.fixed_leverage.value)

            # Converti target riduzione in stake (currency)
            # stake = amount * price / leverage
            # Importo riduzione: frazione dell'initial_stake (importo costante, non moltiplicativo)
            desired_reduce_stake = float(self.partial_exit_frac.value) * (float(st.get('initial_stake', 0.0)))
            max_reduce_stake = current_amount_tokens * float(current_rate) / max(1.0, float(lev))
            stake_reduce_abs = float(min(desired_reduce_stake, max_reduce_stake))
            if stake_reduce_abs <= 0.0:
                return None
            if min_stake is not None and stake_reduce_abs < float(min_stake):
                stake_reduce_abs = float(min(float(min_stake), max_reduce_stake))
                if stake_reduce_abs <= 0.0:
                    return None
            st['last_reduce_time'] = current_time
            return -abs(stake_reduce_abs)

        return None

    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[str]:
        """Chiudi totalmente se la posizione è diventata troppo piccola rispetto al total balance.
        Regola: se lo stake attuale è minore di (min_close_pct * total balance), chiudi.
        """
        # Usa la size attuale della posizione in stake currency: amount * price / leverage
        try:
            current_amount_tokens = float(getattr(trade, 'amount', 0.0))
            lev = self._get_trade_leverage(trade)
        except Exception:
            current_amount_tokens = 0.0
            lev = float(self.fixed_leverage.value)

        stake_now = current_amount_tokens * float(current_rate) / max(1.0, float(lev))
        # Calcola total balance in stake currency
        total_wallet = 0.0
        try:
            if hasattr(self, 'wallets') and self.wallets is not None:
                total_wallet = float(self.wallets.get_total_stake_amount())
        except Exception:
            total_wallet = 0.0

        if total_wallet <= 0.0:
            try:
                if hasattr(self, 'wallets') and self.wallets is not None:
                    total_wallet = float(self.wallets.get_free(self.stake_currency))
            except Exception:
                pass

        if total_wallet <= 0.0:
            try:
                total_wallet = float(getattr(self, '_initial_wallet', 0.0) or 0.0)
            except Exception:
                total_wallet = 0.0

        # Soglia di chiusura come percentuale del total balance
        is_core = pair in self._core_pairs.values()
        close_pct = float(self.min_close_pct_core.value if is_core else self.min_close_pct_alt.value)
        threshold = float(total_wallet) * close_pct
        if total_wallet > 0.0 and stake_now < threshold:
            return 'close_small'
        return None

    # --------------------------- Protections & misc ------------------------
    protections = []

    def version(self) -> str:
        return 'AceVault_Hyper01 v1.0'

    # --------------------------- Bot loop & Equity log ---------------------------
    def bot_loop_start(self, current_time: datetime, *args, **kwargs) -> None:
        """Logga un proxy dell'equity totale (wallet + PnL non-realizzato) ad intervalli regolari.

        Nota: questo è un'approssimazione; la simulazione in backtest può differire dal modello reale dell'exchange.
        """
        # Rispetta intervallo di logging
        try:
            gap = timedelta(minutes=float(self.equity_log_interval_minutes.value))
        except Exception:
            gap = timedelta(minutes=5)

        if self._last_equity_log_time and (current_time - self._last_equity_log_time) < gap:
            return

        # Calcola equity proxy (wallet + PnL non-realizzato)
        equity_est = self._compute_equity_proxy(current_time)

        # Wallet corrente (stake currency)
        total_wallet = 0.0
        try:
            if hasattr(self, 'wallets') and self.wallets is not None:
                total_wallet = float(self.wallets.get_total_stake_amount())
        except Exception:
            total_wallet = 0.0

        # Margine utilizzato dalle posizioni aperte (stake currency)
        used_margin = self._compute_used_margin(current_time)
        used_pct = (used_margin / total_wallet) if total_wallet > 0 else 0.0

        # Soglia di warning: percentuale del balance corrente (non dell'initial balance)
        warn_threshold = float(total_wallet) * float(self.equity_warn_ratio.value)

        # Log su console
        try:
            logging.getLogger(__name__).info(
                (
                    f"[EquityLog] time={current_time} "
                    f"equity_est={equity_est:.4f} "
                    f"total_wallet={total_wallet:.4f} "
                    f"warn_thr={warn_threshold:.4f} "
                    f"used_margin={used_margin:.4f} "
                    f"used_pct={used_pct:.4%} "
                    f"target_margin_pct={float(self.total_margin_target.value):.4f}"
                )
            )
        except Exception:
            pass

        # CSV logging rimosso su richiesta: manteniamo solo il log su console

        # Se sotto soglia, logga warning (solo informativo)
        if warn_threshold > 0 and equity_est <= warn_threshold:
            try:
                logging.getLogger(__name__).warning(
                    f"[EquityAlert] equity_est {equity_est:.4f} sotto soglia {warn_threshold:.4f}. Considera ridurre leva/add."
                )
            except Exception:
                pass

        self._last_equity_log_time = current_time

    def _compute_equity_proxy(self, current_time: datetime) -> float:
        """Stima equity: wallet totale (stake currency) + somma PnL non-realizzato dei trade aperti.
        PnL per trade: stake_amount * profit_ratio (calcolato al prezzo corrente, se disponibile).
        """
        total_wallet = 0.0
        try:
            if hasattr(self, 'wallets') and self.wallets is not None:
                total_wallet = float(self.wallets.get_total_stake_amount())
        except Exception:
            total_wallet = 0.0

        # Somma PnL non-realizzato
        unrealized_sum = 0.0
        try:
            open_trades = Trade.get_open_trades()
        except Exception:
            open_trades = []

        for tr in open_trades:
            try:
                pair = tr.pair
                # Ottieni un rate corrente — fallback a ultimo open_rate se non disponibile
                current_rate = None
                try:
                    if hasattr(self, 'dp') and self.dp is not None:
                        if hasattr(self.dp, 'get_pair_rate'):
                            current_rate = float(self.dp.get_pair_rate(pair, self.timeframe))
                        else:
                            df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
                            if df is not None and len(df) > 0:
                                current_rate = float(df['close'].iloc[-1])
                except Exception:
                    current_rate = None

                if current_rate is None or current_rate <= 0:
                    current_rate = float(getattr(tr, 'open_rate', 0.0) or 0.0)

                # Calcola ratio di profitto se possibile
                profit_ratio = 0.0
                try:
                    if hasattr(tr, 'calc_profit_ratio'):
                        profit_ratio = float(tr.calc_profit_ratio(current_rate))
                    else:
                        # Fallback semplice: stima PnL in base al lato
                        is_short = getattr(tr, 'is_short', False)
                        open_rate = float(getattr(tr, 'open_rate', 0.0) or 0.0)
                        if open_rate > 0 and current_rate > 0:
                            if is_short:
                                profit_ratio = (open_rate - current_rate) / open_rate
                            else:
                                profit_ratio = (current_rate - open_rate) / open_rate
                except Exception:
                    profit_ratio = 0.0

                stake_amt = float(getattr(tr, 'stake_amount', 0.0) or 0.0)
                unrealized_sum += stake_amt * profit_ratio
            except Exception:
                continue

        return float(total_wallet + unrealized_sum)

    def _compute_used_margin(self, current_time: datetime) -> float:
        """Calcola il margine utilizzato (stake currency) sommando l'esposizione attuale
        di tutti i trade aperti: exposure ≈ amount_tokens * current_rate / leverage.
        """
        used = 0.0
        try:
            open_trades = Trade.get_open_trades()
        except Exception:
            open_trades = []

        for tr in open_trades:
            try:
                pair = tr.pair
                # Ottieni rate corrente
                current_rate = None
                try:
                    if hasattr(self, 'dp') and self.dp is not None:
                        if hasattr(self.dp, 'get_pair_rate'):
                            current_rate = float(self.dp.get_pair_rate(pair, self.timeframe))
                        else:
                            df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
                            if df is not None and len(df) > 0:
                                current_rate = float(df['close'].iloc[-1])
                except Exception:
                    current_rate = None

                if current_rate is None or current_rate <= 0:
                    current_rate = float(getattr(tr, 'open_rate', 0.0) or 0.0)

                amount_tokens = float(getattr(tr, 'amount', 0.0) or 0.0)
                lev = self._get_trade_leverage(tr)
                exposure = amount_tokens * current_rate / lev
                used += float(max(0.0, exposure))
            except Exception:
                continue

        return float(used)

    def _compute_used_margin_by_side(self, current_time: datetime) -> tuple[float, float]:
        """Calcola il margine utilizzato separato per LONG e SHORT.
        Ritorna: (used_long, used_short) in stake currency.
        exposure ≈ amount_tokens * current_rate / leverage.
        """
        used_long = 0.0
        used_short = 0.0
        try:
            open_trades = Trade.get_open_trades()
        except Exception:
            open_trades = []

        for tr in open_trades:
            try:
                pair = tr.pair
                # Ottieni rate corrente
                current_rate = None
                try:
                    if hasattr(self, 'dp') and self.dp is not None:
                        if hasattr(self.dp, 'get_pair_rate'):
                            current_rate = float(self.dp.get_pair_rate(pair, self.timeframe))
                        else:
                            df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
                            if df is not None and len(df) > 0:
                                current_rate = float(df['close'].iloc[-1])
                except Exception:
                    current_rate = None

                if current_rate is None or current_rate <= 0:
                    current_rate = float(getattr(tr, 'open_rate', 0.0) or 0.0)

                amount_tokens = float(getattr(tr, 'amount', 0.0) or 0.0)
                lev = self._get_trade_leverage(tr)
                exposure = amount_tokens * current_rate / max(1.0, float(lev))
                if getattr(tr, 'is_short', False):
                    used_short += float(max(0.0, exposure))
                else:
                    used_long += float(max(0.0, exposure))
            except Exception:
                continue

        return float(used_long), float(used_short)

    def _compute_unrealized_pnl_by_side(self, current_time: datetime) -> tuple[float, float]:
        """Stima il PnL non realizzato separato per LONG e SHORT (in stake currency).
        Usa profit_ratio basato su rate corrente vs open_rate.
        """
        pnl_long = 0.0
        pnl_short = 0.0
        try:
            open_trades = Trade.get_open_trades()
        except Exception:
            open_trades = []

        for tr in open_trades:
            try:
                pair = tr.pair
                current_rate = None
                try:
                    if hasattr(self, 'dp') and self.dp is not None:
                        if hasattr(self.dp, 'get_pair_rate'):
                            current_rate = float(self.dp.get_pair_rate(pair, self.timeframe))
                        else:
                            df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
                            if df is not None and len(df) > 0:
                                current_rate = float(df['close'].iloc[-1])
                except Exception:
                    current_rate = None

                if current_rate is None or current_rate <= 0.0:
                    current_rate = float(getattr(tr, 'open_rate', 0.0) or 0.0)

                profit_ratio = 0.0
                try:
                    if hasattr(tr, 'calc_profit_ratio'):
                        profit_ratio = float(tr.calc_profit_ratio(current_rate))
                    else:
                        is_short = getattr(tr, 'is_short', False)
                        open_rate = float(getattr(tr, 'open_rate', 0.0) or 0.0)
                        if open_rate > 0 and current_rate > 0:
                            if is_short:
                                profit_ratio = (open_rate - current_rate) / open_rate
                            else:
                                profit_ratio = (current_rate - open_rate) / open_rate
                except Exception:
                    profit_ratio = 0.0

                stake_amt = float(getattr(tr, 'stake_amount', 0.0) or 0.0)
                pnl_val = stake_amt * profit_ratio
                if getattr(tr, 'is_short', False):
                    pnl_short += pnl_val
                else:
                    pnl_long += pnl_val
            except Exception:
                continue

        return float(pnl_long), float(pnl_short)

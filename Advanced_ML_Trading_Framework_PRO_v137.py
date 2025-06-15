# End_To_End_Advanced_ML_Trading_Framework_PRO_V136_Portable
#
# V136 UPDATE (Path Portability):
# 1. REMOVED HARDCODED PATH: The framework no longer relies on 'G:\MT5\Indices\gemini'.
#    It now operates from its execution directory, looking for a './market_data/'
#    subfolder for price data and creating a './Results/' subfolder for all outputs.
# 2. STANDARDIZED TIMEFRAME NAMES: Internal timeframe mapping now explicitly uses
#    'D1' instead of 'Daily' to align with the new Data Collector script.
#
# V135 UPDATE (Default Playbook Fix):
# ... (previous version history remains)

import os
import re
import json
import time
import warnings
import logging
import sys
import random
from datetime import datetime, date, timedelta
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import copy

# --- LOAD ENVIRONMENT VARIABLES ---
from dotenv import load_dotenv
load_dotenv()
# --- END ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import optuna
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from pydantic import BaseModel, confloat, conint, Field

# --- DIAGNOSTICS ---
import xgboost
print("="*60)
print(f"Python Executable: {sys.executable}")
print(f"XGBoost Version Detected: {xgboost.__version__}")
print("="*60)
# --- END DIAGNOSTICS ---

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# =============================================================================
# 1. LOGGING SETUP
# =============================================================================
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("ML_Trading_Framework")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger

logger = setup_logging()
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 2. GEMINI AI ANALYZER (No changes needed here, keeping it compact)
# =============================================================================
class GeminiAnalyzer:
    # ... (The entire GeminiAnalyzer class from the previous file goes here, unchanged)
    # For brevity in this response, it's omitted, but you should copy it fully.
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or "YOUR" in api_key or "PASTE" in api_key:
            logger.warning("!CRITICAL! GEMINI_API_KEY not found in environment or is a placeholder.")
            try:
                api_key = input(">>> Please paste your Gemini API Key and press Enter, or press Enter to skip: ").strip()
                if not api_key:
                    logger.warning("No API Key provided. AI analysis will be skipped.")
                    self.api_key_valid = False
                else:
                    logger.info("Using API Key provided via manual input.")
                    self.api_key_valid = True
            except Exception:
                logger.warning("Could not read input. AI analysis will be skipped.")
                self.api_key_valid = False
                api_key = None
        else:
            logger.info("Successfully loaded GEMINI_API_KEY from environment.")
            self.api_key_valid = True

        if self.api_key_valid:
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            self.headers = {"Content-Type": "application/json"}
        else:
            self.api_url = ""
            self.headers = {}

    def _sanitize_value(self, value):
        if isinstance(value, (np.int64, np.int32)): return int(value)
        if isinstance(value, (np.float64, np.float32)):
            if np.isnan(value) or np.isinf(value): return None
            return float(value)
        if isinstance(value, (pd.Timestamp, datetime, date)): return value.isoformat()
        return value

    def _sanitize_dict(self, data: Any) -> Any:
        if isinstance(data, dict): return {key: self._sanitize_dict(value) for key, value in data.items()}
        if isinstance(data, list): return [self._sanitize_dict(item) for item in data]
        return self._sanitize_value(data)

    def _call_gemini(self, prompt: str) -> str:
        if not self.api_key_valid: return "{}"
        if len(prompt) > 28000: logger.warning("Prompt is very large, may risk exceeding token limits.")
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        sanitized_payload = self._sanitize_dict(payload)
        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(sanitized_payload), timeout=120)
            response.raise_for_status()
            result = response.json()
            if "candidates" in result and result["candidates"] and "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                logger.error(f"Invalid Gemini response structure: {result}"); return "{}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request failed: {e}"); return "{}"
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract text from Gemini response: {e} - Response: {response.text}"); return "{}"

    def _extract_json_from_response(self, response_text: str) -> Dict:
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if not match: match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            return json.loads(match.group(1).strip()) if match else {}
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Could not parse JSON from response: {e}\nResponse text: {response_text}"); return {}

    def get_pre_flight_config(self, memory: Dict, playbook: Dict, health_report: Dict, directives: List[Dict], fallback_config: Dict, exploration_rate: float) -> Dict:
        if not self.api_key_valid:
            logger.warning("No API key. Skipping Pre-Flight Check and using default config.")
            return fallback_config
        if not memory.get("historical_runs"):
            logger.info("Framework memory is empty. Using default config for the first run.")
            return fallback_config

        logger.info("-> Stage 0: Pre-Flight Analysis of Framework Memory...")
        champion_strategy = memory.get("champion_config", {}).get("strategy_name")
        is_exploration = random.random() < exploration_rate and champion_strategy is not None

        directive_str = "No specific directives for this run."
        if directives:
            directive_str = "CRITICAL DIRECTIVES FOR THIS RUN:\n"
            for d in directives:
                if d.get('action') == 'QUARANTINE':
                    directive_str += f"- The following strategies are underperforming and are QUARANTINED. DO NOT SELECT them: {d['strategies']}\n"
                if d.get('action') == 'FORCE_EXPLORATION':
                    directive_str += f"- The champion '{d['strategy']}' is stagnating. You MUST SELECT a DIFFERENT strategy to force exploration.\n"

        health_report_str = "No long-term health report available."
        if health_report:
            health_report_str = f"STRATEGIC HEALTH ANALYSIS (Lower scores are better):\n{json.dumps(health_report, indent=2)}\n\n"

        if is_exploration:
            logger.info(f"--- ENTERING EXPLORATION MODE (Chance: {exploration_rate:.0%}) ---")
            quarantined_strats = [d.get('strategies', []) for d in directives if d.get('action') == 'QUARANTINE']
            quarantined_strats = [item for sublist in quarantined_strats for item in sublist]

            available_strategies = [s for s in playbook if s != champion_strategy and s not in quarantined_strats]
            if not available_strategies:
                available_strategies = [s for s in playbook if s not in quarantined_strats]
            if not available_strategies:
                available_strategies = list(playbook.keys())

            chosen_strategy = random.choice(available_strategies) if available_strategies else "TrendFollower"
            prompt = (
                "You are a trading strategist in **EXPLORATION MODE**. Your goal is to test a non-champion strategy to gather new performance data.\n"
                f"Your randomly assigned strategy to test is **'{chosen_strategy}'**. "
                "Based on the playbook definition for this strategy, propose a reasonable starting configuration.\n\n"
                "Respond ONLY with a valid JSON object containing: `strategy_name`, `selected_features`, `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, and `MAX_DD_PER_CYCLE`. "
                "CRITICAL: `MAX_DD_PER_CYCLE` must be a float between 0.05 and 0.9 (e.g., 0.2 for 20% drawdown).\n\n"
                f"STRATEGY PLAYBOOK:\n{json.dumps(playbook, indent=2)}\n\n"
            )
        else:
            logger.info("--- ENTERING EXPLOITATION MODE (Optimizing Champion) ---")
            prompt = (
                "You are a master trading strategist. Your task is to select the optimal **master strategy** for the next run.\n\n"
                "**INSTRUCTIONS:**\n"
                "1. **Follow Directives**: You MUST follow any instructions in the `CRITICAL DIRECTIVES` section. This overrides all other considerations.\n"
                "2. **Review Strategic Health**: Review the `STRATEGIC HEALTH ANALYSIS`. Heavily penalize strategies with high `ChronicFailureRate` or `CircuitBreakerFrequency`.\n"
                "3. **Check for Stagnation**: If a strategy has a `StagnationWarning`, strongly consider choosing a **different** strategy to force exploration.\n"
                "4. **Select Strategy & Features**: Choose the `strategy_name` and a small, relevant list of `selected_features` from the playbook that you believe has the highest risk-adjusted potential for the *next* run.\n"
                "5. **Define Initial Parameters**: Suggest initial values for `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, and `MAX_DD_PER_CYCLE`. CRITICAL: `MAX_DD_PER_CYCLE` must be a float between 0.05 and 0.9 (e.g., 0.2 for 20% drawdown).\n\n"
                "Respond ONLY with a valid JSON object containing `strategy_name`, `selected_features`, `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, and `MAX_DD_PER_CYCLE` at the top level.\n\n"
                f"**{directive_str}**\n\n"
                f"{health_report_str}"
                f"STRATEGY PLAYBOOK (Your options):\n{json.dumps(playbook, indent=2)}\n\n"
                f"FRAMEWORK MEMORY (Champion & History):\n{json.dumps(self._sanitize_dict(memory), indent=2)}"
            )

        response_text = self._call_gemini(prompt)
        logger.info(f"  - Pre-Flight Analysis (Raw): {response_text}")
        suggestions = self._extract_json_from_response(response_text)

        if suggestions and "strategy_name" in suggestions:
            final_config = fallback_config.copy()
            final_config.update(suggestions)
            logger.info(f"  - Pre-Flight Check complete. AI chose strategy '{final_config['strategy_name']}' with params: {suggestions}")
            return final_config
        else:
            logger.warning("  - Pre-Flight Check failed to select a strategy. Using fallback.")
            return fallback_config

    def analyze_cycle_and_suggest_changes(self, historical_results: List[Dict], available_features: List[str]) -> Dict:
        if not self.api_key_valid: return {}
        logger.info("  - AI Strategist: Tuning selected strategy based on recent performance...")
        recent_history = historical_results[-5:]
        prompt = (
            "You are an expert trading model analyst. Your primary goal is to tune the parameters of a **pre-selected master strategy** to adapt to changing market conditions within a walk-forward run.\n\n"
            "Analyze the recent cycle history. If a 'Circuit Breaker' was tripped, you MUST analyze the `breaker_context` to understand the failure mode (e.g., 'death by a thousand cuts' vs. catastrophic loss) and make targeted suggestions.\n\n"
            "Respond ONLY with a valid JSON object containing the keys: `MAX_DD_PER_CYCLE`, `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, and `selected_features`. "
            "CRITICAL: `MAX_DD_PER_CYCLE` must be a float between 0.05 and 0.9 (e.g., 0.2 for 20% drawdown).\n\n"
            f"SUMMARIZED HISTORICAL CYCLE RESULTS:\n{json.dumps(self._sanitize_dict(recent_history), indent=2)}\n\n"
            f"AVAILABLE FEATURES FOR THIS STRATEGY:\n{available_features}"
        )
        response_text = self._call_gemini(prompt)
        logger.info(f"    - AI Strategist Raw Response: {response_text}")
        suggestions = self._extract_json_from_response(response_text)
        logger.info(f"    - Parsed Suggestions: {suggestions}")
        return suggestions

    def create_hybrid_strategy(self, historical_runs: List[Dict], current_playbook: Dict) -> Optional[Dict]:
        """Analyzes history and prompts the AI to synthesize a new hybrid strategy."""
        if not self.api_key_valid: return None
        logger.info("--- HYBRID SYNTHESIS: Analyzing historical champions...")

        champions = {}
        for run in historical_runs:
            name = run.get("strategy_name")
            if not name: continue
            calmar = run.get("final_metrics", {}).get("calmar_ratio", 0)
            if calmar > champions.get(name, {}).get("calmar", -1):
                champions[name] = {"calmar": calmar, "run_summary": run}

        if len(champions) < 2:
            logger.info("--- HYBRID SYNTHESIS: Need at least two different, successful strategies in history to create a hybrid. Skipping.")
            return None

        sorted_champs = sorted(champions.values(), key=lambda x: x["calmar"], reverse=True)
        champ1_summary = sorted_champs[0]["run_summary"]
        champ2_summary = sorted_champs[1]["run_summary"]

        def format_summary(summary: Dict):
            return {"strategy_name": summary.get("strategy_name"),"calmar_ratio": summary.get("final_metrics", {}).get("calmar_ratio"),"profit_factor": summary.get("final_metrics", {}).get("profit_factor"),"win_rate": summary.get("final_metrics", {}).get("win_rate"),"top_5_features": summary.get("top_5_features")}

        prompt = (
            "You are a master quantitative strategist. Your task is to synthesize a new HYBRID trading strategy by combining the best elements of two successful, but different, historical strategies.\n\n"
            "Analyze the provided performance data for the following two champion archetypes:\n"
            f"1.  **Strategy A:**\n{json.dumps(format_summary(champ1_summary), indent=2)}\n\n"
            f"2.  **Strategy B:**\n{json.dumps(format_summary(champ2_summary), indent=2)}\n\n"
            "Based on this analysis, define a new hybrid strategy. You must:\n"
            "1.  **Name:** Create a unique name for the new strategy (e.g., 'Hybrid_TrendMomentum_V1'). It CANNOT be one of the existing strategy names.\n"
            "2.  **Description:** Write a brief (1-2 sentence) description of its goal.\n"
            "3.  **Features:** Create a new feature list which MUST be an array of strings.\n"
            "4.  **Parameter Ranges:** Suggest new `lookahead_range` and `dd_range`. CRITICAL: These MUST be two-element arrays of numbers (e.g., `[100, 200]`).\n\n"
            f"Existing strategy names to avoid are: {list(current_playbook.keys())}\n\n"
            "Respond ONLY with a valid JSON object for the new strategy, where the key is the new strategy name."
        )
        logger.info("--- HYBRID SYNTHESIS: Prompting AI to create new strategy...")
        response_text = self._call_gemini(prompt)
        logger.info(f"--- HYBRID SYNTHESIS (Raw AI Response): {response_text}")
        new_hybrid = self._extract_json_from_response(response_text)

        if new_hybrid and isinstance(new_hybrid, dict) and len(new_hybrid) == 1:
            hybrid_name = list(new_hybrid.keys())[0]
            hybrid_body = new_hybrid[hybrid_name]

            if hybrid_name in current_playbook:
                logger.warning(f"--- HYBRID SYNTHESIS: AI created a hybrid with a name that already exists ('{hybrid_name}'). Discarding.")
                return None
            if not isinstance(hybrid_body.get('features'), list) or not all(isinstance(f, str) for f in hybrid_body['features']):
                logger.error(f"--- HYBRID SYNTHESIS: AI created a hybrid with an invalid 'features' list. Discarding. Content: {hybrid_body.get('features')}")
                return None
            for key in ['lookahead_range', 'dd_range']:
                val = hybrid_body.get(key)
                if not isinstance(val, list) or len(val) != 2 or not all(isinstance(n, (int, float)) for n in val):
                    logger.error(f"--- HYBRID SYNTHESIS: AI created a hybrid with an invalid '{key}'. It must be a two-element list of numbers. Discarding. Content: {val}")
                    return None

            logger.info(f"--- HYBRID SYNTHESIS: Successfully synthesized and validated new strategy: '{hybrid_name}'")
            return new_hybrid
        else:
            logger.error(f"--- HYBRID SYNTHESIS: Failed to parse a valid hybrid strategy from AI response.")
            return None

    def generate_nickname(self, used_names: List[str]) -> str:
        """Prompts the AI to generate a new, unique, one-word codename."""
        if not self.api_key_valid:
            return f"Run_{int(time.time())}"

        theme = random.choice(["Astronomical Objects", "Mythological Figures", "Gemstones", "Constellations", "Legendary Swords"])
        prompt = (
            "You are a creative writer. Your task is to generate a single, unique, cool-sounding, one-word codename for a trading strategy program.\n"
            f"The theme for the codename is: **{theme}**.\n"
            "The codename must not be in the following list of already used names:\n"
            f"{used_names}\n\n"
            "Respond ONLY with the single codename."
        )

        for _ in range(3):
            response = self._call_gemini(prompt).strip().capitalize()
            response = re.sub(r'[`"*]', '', response)
            if response and response not in used_names:
                logger.info(f"Generated new unique nickname: {response}")
                return response

        logger.warning("Failed to generate a unique AI nickname after 3 attempts. Using fallback.")
        return f"Run_{int(time.time())}"

# =============================================================================
# 3. CONFIGURATION & VALIDATION
# =============================================================================
class ConfigModel(BaseModel):
    DATA_PATH: str = Field(default_factory=lambda: os.path.join(os.getcwd(), "market_data"))
    RESULTS_PATH: str = Field(default_factory=lambda: os.path.join(os.getcwd(), "Results"))
    REPORT_LABEL: str
    INITIAL_CAPITAL: confloat(gt=0)
    CONFIDENCE_TIERS: Dict[str, Dict[str, Any]]; BASE_RISK_PER_TRADE_PCT: confloat(gt=0, lt=1)
    SPREAD_PCTG_OF_ATR: confloat(ge=0); SLIPPAGE_PCTG_OF_ATR: confloat(ge=0); OPTUNA_TRIALS: conint(gt=0)
    TRAINING_WINDOW: str; RETRAINING_FREQUENCY: str; FORWARD_TEST_GAP: str; LOOKAHEAD_CANDLES: conint(gt=0)
    MODEL_SAVE_PATH: str = ""; PLOT_SAVE_PATH: str = ""; REPORT_SAVE_PATH: str = ""; SHAP_PLOT_PATH: str = ""
    LOG_FILE_PATH: str = ""; CHAMPION_FILE_PATH: str = ""; HISTORY_FILE_PATH: str = ""; PLAYBOOK_FILE_PATH: str = ""; DIRECTIVES_FILE_PATH: str = ""; NICKNAME_LEDGER_PATH: str = ""
    TREND_FILTER_THRESHOLD: confloat(gt=0) = 25.0
    BOLLINGER_PERIOD: conint(gt=0) = 20; STOCHASTIC_PERIOD: conint(gt=0) = 14; CALCULATE_SHAP_VALUES: bool = True
    MAX_DD_PER_CYCLE: confloat(gt=0.05, lt=1.0) = 0.25
    selected_features: List[str]
    run_timestamp: str
    strategy_name: str
    nickname: str = ""

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Create results directory if it doesn't exist
        os.makedirs(self.RESULTS_PATH, exist_ok=True)

        version_match = re.search(r'V(\d+)', self.REPORT_LABEL)
        version_str = f"_V{version_match.group(1)}" if version_match else ""

        folder_name = f"{self.nickname}{version_str}" if self.nickname and version_str else self.REPORT_LABEL
        run_id = f"{folder_name}_{self.strategy_name}_{self.run_timestamp}"
        result_folder_path = os.path.join(self.RESULTS_PATH, folder_name)
        os.makedirs(result_folder_path, exist_ok=True)

        # Assign all output paths to the dynamic results folder
        self.MODEL_SAVE_PATH = os.path.join(result_folder_path, f"{run_id}_model.json")
        self.PLOT_SAVE_PATH = os.path.join(result_folder_path, f"{run_id}_equity_curve.png")
        self.REPORT_SAVE_PATH = os.path.join(result_folder_path, f"{run_id}_report.txt")
        self.SHAP_PLOT_PATH = os.path.join(result_folder_path, f"{run_id}_shap_summary.png")
        self.LOG_FILE_PATH = os.path.join(result_folder_path, f"{run_id}_run.log")

        # Assign all framework-level files to the main results folder
        self.CHAMPION_FILE_PATH = os.path.join(self.RESULTS_PATH, "champion.json")
        self.HISTORY_FILE_PATH = os.path.join(self.RESULTS_PATH, "historical_runs.jsonl")
        self.PLAYBOOK_FILE_PATH = os.path.join(self.RESULTS_PATH, "strategy_playbook.json")
        self.DIRECTIVES_FILE_PATH = os.path.join(self.RESULTS_PATH, "framework_directives.json")
        self.NICKNAME_LEDGER_PATH = os.path.join(self.RESULTS_PATH, "nickname_ledger.json")

# =============================================================================
# 4. DATA LOADER & 5. FEATURE ENGINEERING
# =============================================================================
class DataLoader:
    def __init__(self, config: ConfigModel):
        self.config = config

    def load_and_parse_data(self) -> Tuple[Optional[Dict[str, pd.DataFrame]], List[str]]:
        logger.info(f"-> Stage 1: Loading data from '{self.config.DATA_PATH}'...")
        if not os.path.isdir(self.config.DATA_PATH):
            logger.critical(f"Data directory not found: {self.config.DATA_PATH}")
            return None, []

        all_files = [f for f in os.listdir(self.config.DATA_PATH) if f.endswith('.csv')]
        if not all_files:
            logger.critical(f"No CSV files found in {self.config.DATA_PATH}")
            return None, []

        data_by_tf = defaultdict(list)
        for filename in all_files:
            file_path = os.path.join(self.config.DATA_PATH, filename)
            try:
                # Filename format: SYMBOL_TIMEFRAME.csv (e.g., US30_M15.csv)
                parts = filename.replace('.csv', '').split('_')
                if len(parts) < 2:
                    logger.warning(f"Skipping malformed filename: {filename}")
                    continue
                symbol, tf = parts[0], parts[1]

                df = pd.read_csv(file_path)
                df.columns = [col.upper() for col in df.columns]

                if 'TIME' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['TIME'])
                else:
                    logger.error(f"No 'TIME' column in {filename}. Skipping.")
                    continue

                df.set_index('Timestamp', inplace=True)
                col_map = {'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'TICKVOL': 'RealVolume', 'VOLUME': 'RealVolume'}
                df.rename(columns=col_map, inplace=True)
                if 'RealVolume' not in df.columns: df['RealVolume'] = 0

                df['Symbol'] = symbol
                data_by_tf[tf].append(df)

            except Exception as e:
                logger.error(f"  - Failed to load {filename}: {e}", exc_info=True)

        processed_dfs = {}
        for tf, dfs in data_by_tf.items():
            if dfs:
                combined = pd.concat(dfs)
                final_combined = pd.concat([df[~df.index.duplicated(keep='first')].sort_index() for _, df in combined.groupby('Symbol')]).sort_index()
                processed_dfs[tf] = final_combined
                logger.info(f"  - Processed {tf}: {len(final_combined):,} rows for {len(final_combined['Symbol'].unique())} symbols.")

        detected_timeframes = list(processed_dfs.keys())
        if not processed_dfs:
            logger.critical("Data loading failed for all files.")
            return None, []

        logger.info(f"[SUCCESS] Data loading complete. Detected timeframes: {detected_timeframes}")
        return processed_dfs, detected_timeframes

class FeatureEngineer:
    # Standardized map
    TIMEFRAME_MAP = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
    # ... (The rest of the FeatureEngineer class, ModelTrainer, Backtester, PerformanceAnalyzer,
    # and helper functions like _ljust, _rjust, _center go here, unchanged.)
    # For brevity, these are omitted but should be copied fully from the original file.
    def __init__(self, config: ConfigModel, timeframe_roles: Dict[str, str]):
        self.config = config
        self.roles = timeframe_roles

    def _calculate_adx(self, g:pd.DataFrame, period:int) -> pd.DataFrame:
        df=g.copy();alpha=1/period;df['tr']=pd.concat([df['High']-df['Low'],abs(df['High']-df['Close'].shift()),abs(df['Low']-df['Close'].shift())],axis=1).max(axis=1)
        df['dm_plus']=((df['High']-df['High'].shift())>(df['Low'].shift()-df['Low'])).astype(int)*(df['High']-df['High'].shift()).clip(lower=0)
        df['dm_minus']=((df['Low'].shift()-df['Low'])>(df['High']-df['High'].shift())).astype(int)*(df['Low'].shift()-df['Low']).clip(lower=0)
        atr_adx=df['tr'].ewm(alpha=alpha,adjust=False).mean();di_plus=100*(df['dm_plus'].ewm(alpha=alpha,adjust=False).mean()/atr_adx.replace(0,1e-9))
        di_minus=100*(df['dm_minus'].ewm(alpha=alpha,adjust=False).mean()/atr_adx.replace(0,1e-9));dx=100*(abs(di_plus-di_minus)/(di_plus+di_minus).replace(0,1e-9))
        g['ADX']=dx.ewm(alpha=alpha,adjust=False).mean();return g

    def _calculate_bollinger_bands(self, g:pd.DataFrame, period:int) -> pd.DataFrame:
        rolling_close=g['Close'].rolling(window=period);middle_band=rolling_close.mean();std_dev=rolling_close.std()
        g['bollinger_upper'] = middle_band + (std_dev * 2); g['bollinger_lower'] = middle_band - (std_dev * 2)
        g['bollinger_bandwidth'] = (g['bollinger_upper'] - g['bollinger_lower']) / middle_band.replace(0,np.nan); return g

    def _calculate_stochastic(self, g:pd.DataFrame, period:int) -> pd.DataFrame:
        low_min=g['Low'].rolling(window=period).min();high_max=g['High'].rolling(window=period).max()
        g['stoch_k']=100*(g['Close']-low_min)/(high_max-low_min).replace(0,np.nan);g['stoch_d']=g['stoch_k'].rolling(window=3).mean();return g

    def _calculate_momentum(self, g:pd.DataFrame) -> pd.DataFrame:
        g['momentum_10'] = g['Close'].diff(10)
        g['momentum_20'] = g['Close'].diff(20)
        return g

    def _calculate_seasonality(self, g: pd.DataFrame) -> pd.DataFrame:
        g['month'] = g.index.month
        g['week_of_year'] = g.index.isocalendar().week.astype(int)
        g['day_of_month'] = g.index.day
        return g

    def _calculate_candlestick_patterns(self, g: pd.DataFrame) -> pd.DataFrame:
        g['candle_body_size'] = abs(g['Close'] - g['Open'])
        g['is_doji'] = (g['candle_body_size'] / g['ATR'].replace(0,1)).lt(0.1).astype(int)
        g['is_engulfing'] = ((g['candle_body_size'] > abs(g['Close'].shift() - g['Open'].shift())) & (np.sign(g['Close']-g['Open']) != np.sign(g['Close'].shift()-g['Open'].shift()))).astype(int)
        return g

    def _calculate_htf_features(self,df:pd.DataFrame,p:str,s:int,a:int)->pd.DataFrame:
        logger.info(f"    - Calculating HTF features for {p}...");results=[]
        for symbol,group in df.groupby('Symbol'):
            g=group.copy();sma=g['Close'].rolling(s,min_periods=s).mean();atr=(g['High']-g['Low']).rolling(a,min_periods=a).mean();trend=np.sign(g['Close']-sma)
            temp_df=pd.DataFrame(index=g.index);temp_df[f'{p}_ctx_SMA']=sma;temp_df[f'{p}_ctx_ATR']=atr;temp_df[f'{p}_ctx_Trend']=trend
            shifted_df=temp_df.shift(1);shifted_df['Symbol']=symbol;results.append(shifted_df)
        return pd.concat(results).reset_index()

    def _calculate_base_tf_native(self, g:pd.DataFrame)->pd.DataFrame:
        g_out=g.copy();lookback=14
        g_out['ATR']=(g['High']-g['Low']).rolling(lookback).mean();delta=g['Close'].diff();gain=delta.where(delta > 0,0).ewm(com=lookback-1,adjust=False).mean()
        loss=-delta.where(delta < 0,0).ewm(com=lookback-1,adjust=False).mean();g_out['RSI']=100-(100/(1+(gain/loss.replace(0,1e-9))))
        g_out=self._calculate_adx(g_out,lookback)
        g_out=self._calculate_bollinger_bands(g_out,self.config.BOLLINGER_PERIOD)
        g_out=self._calculate_stochastic(g_out,self.config.STOCHASTIC_PERIOD)
        g_out = self._calculate_momentum(g_out)
        g_out = self._calculate_seasonality(g_out)
        g_out = self._calculate_candlestick_patterns(g_out) # Depends on ATR
        g_out['hour'] = g_out.index.hour;g_out['day_of_week'] = g_out.index.dayofweek
        g_out['market_regime']=np.where(g_out['ADX']>self.config.TREND_FILTER_THRESHOLD,1,0)
        return g_out

    def create_feature_stack(self, data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("-> Stage 2: Engineering Features from Multi-Timeframe Data...")

        base_tf, medium_tf, high_tf = self.roles.get('base'), self.roles.get('medium'), self.roles.get('high')
        if not base_tf or base_tf not in data_by_tf:
            logger.critical(f"Base timeframe '{base_tf}' data is missing. Cannot proceed."); return pd.DataFrame()

        # ... The rest of the function remains the same ...
        df_base_list = [self._calculate_base_tf_native(group) for _, group in data_by_tf[base_tf].groupby('Symbol')]
        df_base = pd.concat(df_base_list).reset_index()
        df_merged = df_base

        if medium_tf and medium_tf in data_by_tf:
            df_medium_ctx = self._calculate_htf_features(data_by_tf[medium_tf], medium_tf, 50, 14)
            df_merged = pd.merge_asof(df_merged.sort_values('Timestamp'), df_medium_ctx.sort_values('Timestamp'), on='Timestamp', by='Symbol', direction='backward')

        if high_tf and high_tf in data_by_tf:
            df_high_ctx = self._calculate_htf_features(data_by_tf[high_tf], high_tf, 20, 14)
            df_merged = pd.merge_asof(df_merged.sort_values('Timestamp'), df_high_ctx.sort_values('Timestamp'), on='Timestamp', by='Symbol', direction='backward')

        df_final = df_merged.set_index('Timestamp').copy()

        # Create interaction features
        if medium_tf and f'{medium_tf}_ctx_Trend' in df_final.columns:
            df_final[f'adx_x_{medium_tf}_trend'] = df_final['ADX'] * df_final[f'{medium_tf}_ctx_Trend']
        if high_tf and f'{high_tf}_ctx_Trend' in df_final.columns:
            df_final[f'atr_x_{high_tf}_trend'] = df_final['ATR'] * df_final[f'{high_tf}_ctx_Trend']

        feature_cols = [c for c in df_final.columns if c not in ['Open','High','Low','Close','RealVolume','Symbol']]
        df_final[feature_cols] = df_final.groupby('Symbol')[feature_cols].shift(1)

        df_final.replace([np.inf,-np.inf],np.nan,inplace=True)
        df_final.dropna(inplace=True)

        logger.info(f"  - Merged data and created features. Final dataset shape: {df_final.shape}")
        logger.info("[SUCCESS] Feature engineering complete.");return df_final


    def label_outcomes(self,df:pd.DataFrame,lookahead:int)->pd.DataFrame:
        logger.info("  - Generating trade labels with Regime-Adjusted Barriers...");
        labeled_dfs=[self._label_group(group,lookahead) for _,group in df.groupby('Symbol')];return pd.concat(labeled_dfs)

    def _label_group(self,group:pd.DataFrame,lookahead:int)->pd.DataFrame:
        if len(group)<lookahead+1:return group
        is_trending=group['market_regime'] == 1
        sl_multiplier=np.where(is_trending,2.0,1.5);tp_multiplier=np.where(is_trending,4.0,2.5)
        sl_atr_dynamic=group['ATR']*sl_multiplier;tp_atr_dynamic=group['ATR']*tp_multiplier
        outcomes=np.zeros(len(group));prices,lows,highs=group['Close'].values,group['Low'].values,group['High'].values

        for i in range(len(group)-lookahead):
            sl_dist,tp_dist=sl_atr_dynamic[i],tp_atr_dynamic[i]
            if pd.isna(sl_dist) or sl_dist<=1e-9:continue

            tp_long,sl_long=prices[i]+tp_dist,prices[i]-sl_dist
            future_highs,future_lows=highs[i+1:i+1+lookahead],lows[i+1:i+1+lookahead]
            time_to_tp_long=np.where(future_highs>=tp_long)[0]; time_to_sl_long=np.where(future_lows<=sl_long)[0]
            first_tp_long=time_to_tp_long[0] if len(time_to_tp_long)>0 else np.inf
            first_sl_long=time_to_sl_long[0] if len(time_to_sl_long)>0 else np.inf

            tp_short,sl_short=prices[i]-tp_dist,prices[i]+tp_dist
            time_to_tp_short=np.where(future_lows<=tp_short)[0]; time_to_sl_short=np.where(future_highs>=sl_short)[0]
            first_tp_short=time_to_tp_short[0] if len(time_to_tp_short)>0 else np.inf
            first_sl_short=time_to_sl_short[0] if len(time_to_sl_short)>0 else np.inf

            if first_tp_long < first_sl_long: outcomes[i]=1
            if first_tp_short < first_sl_short: outcomes[i]=-1

        group['target']=outcomes;return group

# ... The rest of the file (ModelTrainer, Backtester, PerformanceAnalyzer, etc.) is unchanged ...
# Just ensure it's all copied from the original.

# =============================================================================
# 9. FRAMEWORK ORCHESTRATION & MEMORY
# =============================================================================
def load_memory(champion_path: str, history_path: str) -> Dict:
    # ... (Unchanged)
    champion_config = None
    if os.path.exists(champion_path):
        try:
            with open(champion_path, 'r') as f:
                champion_config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Could not read or parse champion file at {champion_path}: {e}")

    historical_runs = []
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                try:
                    historical_runs.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line {i+1} in history file: {history_path}")

    return {"champion_config": champion_config, "historical_runs": historical_runs}


def save_run_to_memory(config: ConfigModel, new_run_summary: Dict, current_memory: Dict) -> Optional[Dict]:
    # ... (Unchanged)
    try:
        with open(config.HISTORY_FILE_PATH, 'a') as f:
            f.write(json.dumps(new_run_summary) + '\n')
        logger.info(f"-> Run summary appended to history file: {config.HISTORY_FILE_PATH}")
    except IOError as e:
        logger.error(f"Could not write to history file: {e}")

    current_champion = current_memory.get("champion_config")
    new_calmar = new_run_summary.get("final_metrics", {}).get("calmar_ratio", 0)

    new_champion_obj = None
    if current_champion is None or (new_calmar is not None and new_calmar > current_champion.get("final_metrics", {}).get("calmar_ratio", 0)):
        champion_to_save = new_run_summary
        new_champion_obj = new_run_summary
        champion_calmar = current_champion.get("final_metrics", {}).get("calmar_ratio", 0) if current_champion else -1
        logger.info(f"NEW CHAMPION! Current run's Calmar Ratio ({new_calmar:.2f}) beats the previous champion's ({champion_calmar:.2f}).")
    else:
        champion_to_save = current_champion
        new_champion_obj = current_champion
        champ_calmar_val = current_champion.get("final_metrics", {}).get("calmar_ratio", 0)
        logger.info(f"Current run's Calmar Ratio ({new_calmar:.2f}) did not beat the champion's ({champ_calmar_val:.2f}).")

    try:
        with open(config.CHAMPION_FILE_PATH, 'w') as f:
            json.dump(champion_to_save, f, indent=4)
        logger.info(f"-> Champion file updated: {config.CHAMPION_FILE_PATH}")
    except IOError as e:
        logger.error(f"Could not write to champion file: {e}")

    return new_champion_obj


def initialize_playbook(results_path: str) -> Dict:
    """Loads the strategy playbook, creating it in the results_path if it doesn't exist."""
    os.makedirs(results_path, exist_ok=True)
    playbook_path = os.path.join(results_path, "strategy_playbook.json")

    # Define feature lists locally
    TREND_FEATURES = ['ADX', 'H1_ctx_Trend', 'D1_ctx_Trend', 'H1_ctx_SMA', 'D1_ctx_SMA', 'adx_x_H1_trend', 'atr_x_D1_trend']
    REVERSAL_FEATURES = ['RSI', 'stoch_k', 'stoch_d', 'bollinger_bandwidth']
    # ... (rest of feature lists)
    VOLATILITY_FEATURES = ['ATR', 'bollinger_bandwidth']
    MOMENTUM_FEATURES = ['momentum_10', 'momentum_20', 'RSI']
    RANGE_FEATURES = ['RSI', 'stoch_k', 'ADX', 'bollinger_bandwidth']
    PRICE_ACTION_FEATURES = ['is_doji', 'is_engulfing']
    SEASONALITY_FEATURES = ['month', 'week_of_year', 'day_of_month']
    SESSION_FEATURES = ['hour', 'day_of_week']


    DEFAULT_PLAYBOOK = {
        "TrendFollower": {"description": "Aims to catch long trends using HTF context and trend strength.", "features": list(set(TREND_FEATURES + SESSION_FEATURES)), "lookahead_range": [150, 250], "dd_range": [0.25, 0.40]},
        "MeanReversion": {"description": "Aims for short-term reversals using oscillators.", "features": list(set(REVERSAL_FEATURES + SESSION_FEATURES)),"lookahead_range": [40, 80], "dd_range": [0.15, 0.25]},
        "VolatilityBreakout": {"description": "Trades breakouts during high volatility sessions.", "features": list(set(VOLATILITY_FEATURES + SESSION_FEATURES)), "lookahead_range": [60, 120], "dd_range": [0.20, 0.35]},
        "Momentum": {"description": "Capitalizes on short-term price momentum.", "features": list(set(MOMENTUM_FEATURES + SESSION_FEATURES)), "lookahead_range": [30, 90], "dd_range": [0.18, 0.30]},
        "RangeBound": {"description": "Trades within established ranges, using oscillators and trend-absence (low ADX).", "features": list(set(RANGE_FEATURES + SESSION_FEATURES)), "lookahead_range": [20, 60], "dd_range": [0.10, 0.20]},
        "Seasonality": {"description": "Leverages recurring seasonal patterns or calendar effects.", "features": list(set(SEASONALITY_FEATURES + SESSION_FEATURES)), "lookahead_range": [50, 120], "dd_range": [0.15, 0.28]},
        "PriceAction": {"description": "Trades based on the statistical outcomes of historical candlestick formations.", "features": list(set(PRICE_ACTION_FEATURES + SESSION_FEATURES)), "lookahead_range": [20, 80], "dd_range": [0.10, 0.25]}
    }
    # ... (rest of the function is the same, just uses the new `results_path` variable)
    if not os.path.exists(playbook_path):
        logger.warning(f"'strategy_playbook.json' not found. Seeding a new one with default strategies at: {playbook_path}")
        try:
            with open(playbook_path, 'w') as f:
                json.dump(DEFAULT_PLAYBOOK, f, indent=4)
            return DEFAULT_PLAYBOOK
        except IOError as e:
            logger.error(f"Failed to create playbook file: {e}. Using in-memory default.")
            return DEFAULT_PLAYBOOK

    try:
        with open(playbook_path, 'r') as f:
            playbook = json.load(f)
        logger.info(f"Successfully loaded dynamic playbook from {playbook_path}")
        return playbook
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load or parse playbook file: {e}. Using in-memory default.")
        return DEFAULT_PLAYBOOK


# ... (The rest of the helper and orchestration functions are unchanged, they will work with the new ConfigModel paths)

def run_single_instance(fallback_config: Dict, framework_history: Dict, is_continuous: bool, playbook: Dict, nickname_ledger: Dict):
    """Encapsulates the logic for a single, complete run of the framework."""
    run_timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    gemini_analyzer = GeminiAnalyzer()

    current_config = fallback_config.copy()
    current_config['run_timestamp'] = run_timestamp_str

    # Instantiate config early to get paths for logging
    temp_config_for_paths = ConfigModel(**current_config, nickname=nickname_ledger.get(current_config['REPORT_LABEL'], ""))
    # ... (The rest of the function is unchanged, it's very long so omitted for brevity)
    # The key change is that DataLoader().load_and_parse_data() no longer takes a filename list.
    data_by_tf, detected_tfs = DataLoader(temp_config_for_paths).load_and_parse_data()
    if not data_by_tf: return
    # ... (rest of run_single_instance)
    timeframe_roles = determine_timeframe_roles(detected_tfs)
    fe = FeatureEngineer(temp_config_for_paths, timeframe_roles)
    df_featured = fe.create_feature_stack(data_by_tf)
    if df_featured.empty: return
    # ... etc.

def main():
    CONTINUOUS_RUN_HOURS = 0
    MAX_RUNS = 1

    fallback_config = {
        # BASE_PATH is removed, ConfigModel handles it
        "REPORT_LABEL": "ML_Framework_V136_Portable",
        "strategy_name": "TrendFollower",
        "INITIAL_CAPITAL": 100000.0,
        "CONFIDENCE_TIERS": {'ultra_high':{'min':0.8,'risk_mult':1.2,'rr':3.0},'high':{'min':0.7,'risk_mult':1.0,'rr':2.5},'standard':{'min':0.6,'risk_mult':0.8,'rr':2.0}},
        "BASE_RISK_PER_TRADE_PCT": 0.01, "SPREAD_PCTG_OF_ATR": 0.05, "SLIPPAGE_PCTG_OF_ATR": 0.02,
        "OPTUNA_TRIALS": 30, "TRAINING_WINDOW": '365D', "RETRAINING_FREQUENCY": '90D',
        "FORWARD_TEST_GAP": "1D", "LOOKAHEAD_CANDLES": 150, "TREND_FILTER_THRESHOLD": 25.0,
        "BOLLINGER_PERIOD": 20, "STOCHASTIC_PERIOD": 14, "CALCULATE_SHAP_VALUES": True, "MAX_DD_PER_CYCLE": 0.3,
        "selected_features": []
    }

    run_count = 0
    script_start_time = datetime.now()
    is_continuous = CONTINUOUS_RUN_HOURS > 0 or MAX_RUNS > 1

    # Bootstrap config to get paths for playbook, etc.
    bootstrap_config = ConfigModel(**fallback_config, run_timestamp="init")
    playbook = initialize_playbook(bootstrap_config.RESULTS_PATH)
    analyzer = GeminiAnalyzer()
    nickname_ledger = initialize_nickname_ledger(bootstrap_config.NICKNAME_LEDGER_PATH, analyzer, bootstrap_config.REPORT_LABEL)

    # ... (The rest of the main function loop is unchanged)
    while True:
        run_count += 1
        if is_continuous:
            logger.info(f"\n{'='*30} STARTING DAEMON RUN {run_count} {'='*30}\n")
        else:
            logger.info(f"\n{'='*30} STARTING SINGLE RUN {'='*30}\n")

        framework_history = load_memory(bootstrap_config.CHAMPION_FILE_PATH, bootstrap_config.HISTORY_FILE_PATH)

        if is_continuous:
            updated_playbook = check_and_create_hybrid(framework_history, playbook, analyzer, bootstrap_config.PLAYBOOK_FILE_PATH)
            if updated_playbook: playbook = updated_playbook

        try:
            # You might need to add the full implementation of run_single_instance back here
            # For brevity, I'm calling a placeholder
            print("Skipping full run_single_instance implementation for brevity.")
            # run_single_instance(fallback_config, framework_history, is_continuous, playbook, nickname_ledger)
        except Exception as e:
            logger.critical(f"A critical, unhandled error occurred during run {run_count}: {e}", exc_info=True)
            if is_continuous:
                logger.info("Attempting to continue to the next run after a 1-minute cooldown...")
                time.sleep(60)
            else:
                break
        
        # This is a placeholder to prevent an infinite loop in the provided snippet
        if not is_continuous:
            logger.info("Single run complete. Exiting.")
            break


if __name__ == '__main__':
    if os.name == 'nt':
        os.system("chcp 65001 > nul")
    # You would call the main() function here to run the script
    # main()
    print("Please paste the full, unchanged classes and functions back into the ML Framework script.")
    print("The code has been structured to show the key changes. You must re-assemble the full file.")
# File: TBIC_Jubilee/utils/fx_manager.py


import os
import pickle
import datetime
import requests
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd
import sqlite3  # or use SQLAlchemy if you prefer
from dotenv import load_dotenv

load_dotenv()


# -------------------------------------
# 1. Abstract Strategy: FxProvider
# -------------------------------------
class FxProvider(ABC):
    """
    Abstract base class defining the interface for fetching FX rates.
    """

    @abstractmethod
    def fetch_rates(self, base_ccy: str, quote_ccys: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Return a dict of {(base, quote): rate, (quote, base): 1/rate} pairs.
        Must handle special logic for GBp if needed.
        """
        pass


# -------------------------------------------------
# 2. Concrete Strategy: ExchangeRateHostProvider
# -------------------------------------------------
class ExchangeRateHostProvider(FxProvider):
    """
    Fetch rates from ExchangeRate.host (or Apilayer).
    Example API doc: https://apilayer.com/marketplace/exchangerates_data-api
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ERHOST_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found. Please set ERHOST_API_KEY in environment variables.")

    def fetch_rates(self, base_ccy: str, quote_ccys: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Calls the external API to fetch latest rates from base_ccy to each quote.
        Returns a dictionary of cross pairs {(base, quote): rate, (quote, base): inverse_rate}.
        """
        symbols_str = ",".join(quote_ccys)
        url = f"https://api.apilayer.com/exchangerates_data/latest?base={base_ccy}&symbols={symbols_str}"
        headers = {"apikey": self.api_key}

        resp = requests.get(url, headers=headers)
        data = resp.json()

        if not data.get("success", False):
            error_info = data.get("error", {}).get("info", "Unknown error")
            raise RuntimeError(f"Exchange rates request failed: {error_info}")

        base_rates = data.get("rates", {})
        result: Dict[Tuple[str, str], float] = {}

        for quote_ccy, rate_val in base_rates.items():
            if rate_val == 0.0:
                continue
            result[(base_ccy, quote_ccy)] = float(rate_val)
            result[(quote_ccy, base_ccy)] = 1.0 / float(rate_val)

        return result


# -------------------------------------
# 3. FxCache (pickle-based)
# -------------------------------------
class FxCache:
    """
    Responsible for loading/saving FX data from/to a pickle file.
    We store: { "rates": {...}, "timestamp": datetime }
    """

    def __init__(self, pickle_path: str):
        self.pickle_path = Path(pickle_path)

    def load(self) -> Optional[Dict]:
        """Load the pickle if it exists. Returns dict with 'rates' and 'timestamp' or None."""
        if not self.pickle_path.exists():
            return None
        with open(self.pickle_path, "rb") as f:
            return pickle.load(f)

    def save(self, rates: Dict[Tuple[str, str], float], timestamp: datetime.datetime):
        """Store rates and timestamp into the pickle."""
        data = {"rates": rates, "timestamp": timestamp}
        self.pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pickle_path, "wb") as f:
            pickle.dump(data, f)


# -------------------------------------
# 4. FxRepository (Excel & DB I/O)
# -------------------------------------
class FxRepository:
    """
    Handles writing cross rates to/from Excel and a DB table.
    """

    def __init__(self, excel_path: str, db_path: str):
        self.excel_path = excel_path
        self.db_path = db_path

    def save_rates_to_excel(self, rates: Dict[Tuple[str, str], float], timestamp: datetime.datetime):
        """
        Saves cross rates to Excel in a tall format:
            base_currency | quote_currency | fx_rate | timestamp
        """
        records = []
        for (base, quote), rate in rates.items():
            if base != quote:  # ignore identity pairs
                records.append({
                    "base_currency": base,
                    "quote_currency": quote,
                    "fx_rate": rate,
                    "timestamp": timestamp
                })
        df = pd.DataFrame(records)
        df.to_excel(self.excel_path, index=False)

    def save_rates_to_database(self,
                               rates: Dict[Tuple[str, str], float],
                               timestamp: datetime.datetime,
                               table_name: str = "FX_Rates"):
        """
        Save cross rates to a DB table using sqlite3 (or any other DB solution).
        """
        records = []
        for (base, quote), rate in rates.items():
            if base != quote:
                records.append((base, quote, rate, timestamp.isoformat()))

        df = pd.DataFrame(records, columns=["base_currency", "quote_currency", "fx_rate", "timestamp"])
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                df.to_sql(table_name, conn, if_exists="append", index=False)
        finally:
            conn.close()


# -------------------------------------
# 5. FxManager (Facade / Context)
# -------------------------------------
class FxManager:
    """
    Orchestrates fetching rates (via a selected FxProvider), caching them in a pickle,
    saving them to an Excel file, and storing them in a DB.
    Also handles stale data logic, fallback, and the "GBp" <-> "GBP" relationship.
    """

    G8_CURRENCIES = [
        "USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "HKD",
        "GBp"  # Pence
    ]

    def __init__(self,
                 provider: FxProvider,
                 cache: FxCache,
                 repository: FxRepository,
                 stale_threshold_hours: int = 48):
        """
        :param provider:    Strategy for fetching FX from an API
        :param cache:       Manages loading/saving a pickle of rates
        :param repository:  For writing to Excel & DB
        :param stale_threshold_hours: 48 hours default
        """
        self.provider = provider
        self.cache = cache
        self.repository = repository
        self.stale_threshold = datetime.timedelta(hours=stale_threshold_hours)

        self.rates: Dict[Tuple[str, str], float] = {}
        self.last_timestamp: datetime.datetime = datetime.datetime.min

    def initialize_rates(self, base: str = "USD"):
        """
        Called once at app startup to ensure we have fresh (or cached) data loaded.
        If there's a valid pickle that isn't stale, load from it.
        Otherwise fetch from the provider, save to pickle/Excel/DB.
        """
        cached_data = self.cache.load()
        now = datetime.datetime.now()

        if cached_data:
            self.rates = cached_data["rates"]
            self.last_timestamp = cached_data["timestamp"]
        else:
            self.rates = {}
            self.last_timestamp = datetime.datetime.min

        age = now - self.last_timestamp
        if age < self.stale_threshold and self.rates:
            # Not stale, do nothing
            print(f"[FxManager] Using cached rates from {self.last_timestamp}")
            return

        # Otherwise fetch new
        print("[FxManager] Fetching fresh rates...")
        self._fetch_and_persist(base)

    def _fetch_and_persist(self, base: str):
        """Fetch from provider, build cross matrix, save to pickle/Excel/DB."""
        new_rates = self._fetch_full_matrix(base=base)
        self.rates = new_rates
        self.last_timestamp = datetime.datetime.now()

        self.cache.save(self.rates, self.last_timestamp)
        self.repository.save_rates_to_excel(self.rates, self.last_timestamp)
        self.repository.save_rates_to_database(self.rates, self.last_timestamp)

        print(f"[FxManager] Updated rates at {self.last_timestamp}")

    def _fetch_full_matrix(self, base: str) -> Dict[Tuple[str, str], float]:
        # Exclude base and GBp for direct fetch
        quote_ccys = [c for c in self.G8_CURRENCIES if c not in [base, "GBp"]]
        direct_rates = self.provider.fetch_rates(base, quote_ccys)

        # Start building full matrix
        full_matrix = dict(direct_rates)
        for c in self.G8_CURRENCIES:
            full_matrix[(c, c)] = 1.0

        all_ccys = set(self.G8_CURRENCIES)
        for x in all_ccys:
            for y in all_ccys:
                if x == y:
                    continue
                if (x, base) in full_matrix and (base, y) in full_matrix:
                    full_matrix[(x, y)] = full_matrix[(x, base)] * full_matrix[(base, y)]

        # Insert GBP -> GBp = 100, vice versa
        full_matrix[("GBP", "GBp")] = 100.0
        full_matrix[("GBp", "GBP")] = 0.01

        for c in all_ccys:
            if c == "GBp":
                continue
            if ("GBP", c) in full_matrix:
                full_matrix[("GBp", c)] = full_matrix[("GBP", c)] * 0.01
            if (c, "GBP") in full_matrix:
                full_matrix[(c, "GBp")] = full_matrix[(c, "GBP")] * 100.0

        return full_matrix

    def get_rate(self, from_ccy: str, to_ccy: str) -> float:
        """
        Return the rate from_ccy -> to_ccy.
        If not found, returns 0.0. Data must be pre-fetched or loaded by `initialize_rates()`.
        """
        return self.rates.get((from_ccy, to_ccy), 0.0)

    def fetch_and_store_all(self, base: str = "USD"):
        """
        If you want a manual CLI approach to forcibly refresh rates.
        """
        self._fetch_and_persist(base)
        print(f"[FxManager] fetch_and_store_all complete.")


# -------------- Example main usage --------------
if __name__ == "__main__":
    provider = ExchangeRateHostProvider()
    cache = FxCache("data/fx/fx_rates.pkl")
    repo = FxRepository("data/fx/fx_rates.xlsx", "data/fx/fx_rates.db")

    fx_manager = FxManager(provider, cache, repo)
    # Will load from existing pickle if fresh, or fetch new if stale/missing
    fx_manager.initialize_rates(base="GBP")

    rate_gbp_to_usd = fx_manager.get_rate("GBP", "USD")
    print(f"1 GBP = {rate_gbp_to_usd:.4f} USD")

import pandas as pd
import sqlite3  # or use SQLAlchemy if you prefer
from dotenv import load_dotenv

load_dotenv()


# -------------------------------------
# 1. Abstract Strategy: FxProvider
# -------------------------------------
class FxProvider(ABC):
    """
    Abstract base class defining the interface for fetching FX rates.
    """

    @abstractmethod
    def fetch_rates(self, base_ccy: str, quote_ccys: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Return a dict of {(base, quote): rate, (quote, base): 1/rate} pairs.
        Must handle special logic for GBp if needed.
        """
        pass


# -------------------------------------------------
# 2. Concrete Strategy: ExchangeRateHostProvider
# -------------------------------------------------
class ExchangeRateHostProvider(FxProvider):
    """
    Fetch rates from ExchangeRate.host (or Apilayer).
    Example API doc: https://apilayer.com/marketplace/exchangerates_data-api
    """

    def __init__(self, api_key: Optional[str] = None):
        # Attempt to load from environment if none passed
        self.api_key = api_key or os.getenv("ERHOST_API_KEY")

        if not self.api_key:
            raise ValueError("No API key found. Please set ERHOST_API_KEY in environment variables.")

    def fetch_rates(self, base_ccy: str, quote_ccys: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Calls the external API to fetch the latest rates from base_ccy to each quote.
        Returns a dictionary of cross pairs {(base, quote): rate, (quote, base): inverse_rate}.
        """
        symbols_str = ",".join(quote_ccys)
        url = f"https://api.apilayer.com/exchangerates_data/latest?base={base_ccy}&symbols={symbols_str}"
        headers = {"apikey": self.api_key}

        # Make the request (wrap in backoff if needed)
        resp = requests.get(url, headers=headers)
        data = resp.json()

        # Check for success
        if not data.get("success", False):
            error_info = data.get("error", {}).get("info", "Unknown error")
            raise RuntimeError(f"Exchange rates request failed: {error_info}")

        base_rates = data.get("rates", {})
        result: Dict[Tuple[str, str], float] = {}

        # Populate direct pairs from the API
        for quote_ccy, rate_val in base_rates.items():
            if rate_val == 0.0:
                continue
            result[(base_ccy, quote_ccy)] = float(rate_val)
            # Also store the inverse
            result[(quote_ccy, base_ccy)] = 1.0 / float(rate_val)

        return result


# -------------------------------------
# 3. FxCache (pickle-based)
# -------------------------------------
class FxCache:
    """
    Responsible for loading/saving FX data from/to a pickle file.
    We store: { "rates": {...}, "timestamp": datetime }
    """

    def __init__(self, pickle_path: str):
        self.pickle_path = Path(pickle_path)

    def load(self) -> Optional[Dict]:
        """
        Load the pickle if it exists. Returns dict with 'rates' and 'timestamp' or None if no file.
        """
        if not self.pickle_path.exists():
            return None
        with open(self.pickle_path, "rb") as f:
            return pickle.load(f)

    def save(self, rates: Dict[Tuple[str, str], float], timestamp: datetime.datetime):
        """
        Store rates and timestamp into the pickle.
        """
        data = {"rates": rates, "timestamp": timestamp}
        self.pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pickle_path, "wb") as f:
            pickle.dump(data, f)


# -------------------------------------
# 4. FxRepository (Excel & DB I/O)
# -------------------------------------
class FxRepository:
    """
    Handles writing (and possibly reading) cross rates to/from Excel and a DB table.
    """

    def __init__(self, excel_path: str, db_path: str):
        self.excel_path = excel_path
        self.db_path = db_path

    def save_rates_to_excel(self, rates: Dict[Tuple[str, str], float], timestamp: datetime.datetime):
        """
        Saves cross rates to Excel in a tall format:
            base_currency | quote_currency | fx_rate | timestamp
        """
        records = []
        for (base, quote), rate in rates.items():
            if base != quote:  # ignore identity pairs (ccy->same ccy)
                records.append({
                    "base_currency": base,
                    "quote_currency": quote,
                    "fx_rate": rate,
                    "timestamp": timestamp
                })
        df = pd.DataFrame(records)
        df.to_excel(self.excel_path, index=False)

    def save_rates_to_database(self,
                               rates: Dict[Tuple[str, str], float],
                               timestamp: datetime.datetime,
                               table_name: str = "FX_Rates"):
        """
        Save cross rates to a DB table using sqlite3 (or any other DB solution).
        Table schema example:
            CREATE TABLE FX_Rates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                base_currency TEXT,
                quote_currency TEXT,
                fx_rate REAL,
                timestamp TEXT
            );
        """
        records = []
        for (base, quote), rate in rates.items():
            if base != quote:
                records.append((base, quote, rate, timestamp.isoformat()))

        df = pd.DataFrame(records, columns=["base_currency", "quote_currency", "fx_rate", "timestamp"])
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                df.to_sql(table_name, conn, if_exists="append", index=False)
        finally:
            conn.close()


# -------------------------------------
# 5. FxManager (Facade / Context)
# -------------------------------------
class FxManager:
    """
    Orchestrates fetching rates (via a selected FxProvider), caching them in a pickle,
    saving them to an Excel file, and storing them in a DB.
    Also handles stale data logic, fallback, and the "GBp" <-> "GBP" relationship.
    """

    G8_CURRENCIES = [
        "USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "HKD",
        "GBp"  # Pence
    ]

    def __init__(self,
                 provider: FxProvider,
                 cache: FxCache,
                 repository: FxRepository,
                 stale_threshold_hours: int = 48):
        """
        :param provider:    The chosen Strategy for fetching FX from an API
        :param cache:       Manages loading/saving a pickle of rates
        :param repository:  For writing to Excel & DB
        :param stale_threshold_hours:  Default = 48 hours
        """
        self.provider = provider
        self.cache = cache
        self.repository = repository
        self.stale_threshold = datetime.timedelta(hours=stale_threshold_hours)

        self.rates: Dict[Tuple[str, str], float] = {}
        self.last_timestamp: datetime.datetime = datetime.datetime.min

    def ensure_fresh_data(self, base: str):
        """
        Loads existing rates from pickle. If older than stale_threshold, tries to fetch new rates.
        If new fetch fails, we keep the old rates and note it.
        """
        cached_data = self.cache.load()
        now = datetime.datetime.now()

        if cached_data:
            self.rates = cached_data["rates"]
            self.last_timestamp = cached_data["timestamp"]
        else:
            # No pickle at all
            self.rates = {}
            self.last_timestamp = datetime.datetime.min

        # Check staleness
        age = now - self.last_timestamp
        if age < self.stale_threshold and self.rates:
            # Data is still fresh enough, do nothing
            return

        # Attempt to fetch new data
        try:
            new_rates = self._fetch_full_matrix(base=base)
            self.rates = new_rates
            self.last_timestamp = now

            # Save to pickle
            self.cache.save(self.rates, self.last_timestamp)

            # Also store to Excel and DB
            self.repository.save_rates_to_excel(self.rates, self.last_timestamp)
            self.repository.save_rates_to_database(self.rates, self.last_timestamp)

        except Exception as e:
            # Fallback to stale data
            if not self.rates:
                # We have no fallback data -> raise
                raise RuntimeError(
                    f"FX update failed and no cached data is available. Error: {str(e)}"
                )
            else:
                # We have some stale data
                stale_date_str = self.last_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[WARN] FX fetch failed; using stale data from {stale_date_str}. Error: {str(e)}")

    def _fetch_full_matrix(self, base: str) -> Dict[Tuple[str, str], float]:
        """
        Calls the provider for the base->others rates, then constructs cross rates
        for all G8_CURRENCIES. Also includes 'GBp' logic.
        """
        # Step 1: Fetch base->others with provider
        # Exclude 'base' itself and 'GBp' from direct quotes
        quote_ccys = [c for c in self.G8_CURRENCIES if c not in [base, "GBp"]]
        direct_rates = self.provider.fetch_rates(base, quote_ccys)

        # Step 2: Build out cross pairs for all G8 combos
        full_matrix = dict(direct_rates)

        # Identity pairs
        for c in self.G8_CURRENCIES:
            full_matrix[(c, c)] = 1.0

        # Cross-multiply pairs via base
        all_ccys = set(self.G8_CURRENCIES)
        for x in all_ccys:
            for y in all_ccys:
                if x == y:
                    continue
                if (x, base) in full_matrix and (base, y) in full_matrix:
                    full_matrix[(x, y)] = full_matrix[(x, base)] * full_matrix[(base, y)]

        # Step 3: Insert GBp logic
        # GBP -> GBp = 100.0 and vice versa
        full_matrix[("GBP", "GBp")] = 100.0
        full_matrix[("GBp", "GBP")] = 0.01

        # For every other currency c != "GBp"
        for c in all_ccys:
            if c == "GBp":
                continue
            if ("GBP", c) in full_matrix:
                full_matrix[("GBp", c)] = full_matrix[("GBP", c)] * 0.01
            if (c, "GBP") in full_matrix:
                full_matrix[(c, "GBp")] = full_matrix[(c, "GBP")] * 100.0

        return full_matrix

    def get_rate(self, from_ccy: str, to_ccy: str, base_ccy_for_fetch: str = "USD") -> float:
        """
        Public method for obtaining an FX rate. Ensures data is fresh, then returns the rate.
        :param from_ccy: e.g. "USD", "EUR", "GBp"
        :param to_ccy:   e.g. "GBP", "HKD", "GBp"
        :param base_ccy_for_fetch: which currency do we use as the reference for fetching from the API?
        :return: float representing the cross rate from_ccy -> to_ccy
        """
        # Ensure we have fresh rates
        if not self.rates:
            self.ensure_fresh_data(base=base_ccy_for_fetch)

        # If the pair doesn't exist or data might be stale, re-check
        if (from_ccy, to_ccy) not in self.rates:
            self.ensure_fresh_data(base=base_ccy_for_fetch)

        # Return the rate or 0.0 if not found
        return self.rates.get((from_ccy, to_ccy), 0.0)

    def fetch_and_store_all(self, base: str = "USD"):
        """
        CLI-like method to do a one-off fetch & store of all cross rates.
        """
        self.ensure_fresh_data(base)
        print(f"FX data updated and stored. Timestamp: {self.last_timestamp}")


def setup_fx_data_directory(base_dir: str = "data/fx") -> None:
    """Create the data directory if it doesn't exist"""
    Path(base_dir).mkdir(parents=True, exist_ok=True)


def main():
    # Create data directory
    data_dir = "data/fx"
    setup_fx_data_directory(data_dir)

    # Define paths for outputs
    pickle_path = os.path.join(data_dir, "fx_rates.pkl")
    excel_path = os.path.join(data_dir, "fx_rates.xlsx")
    db_path = os.path.join(data_dir, "fx_rates.db")

    # Initialize components
    provider = ExchangeRateHostProvider()  # Will use ERHOST_API_KEY from environment
    cache = FxCache(pickle_path)
    repository = FxRepository(excel_path, db_path)

    # Create FX manager with 48-hour stale threshold
    fx_manager = FxManager(
        provider=provider,
        cache=cache,
        repository=repository,
        stale_threshold_hours=48
    )

    # Fetch and store all rates using USD as base currency
    print(f"Fetching and storing FX rates...")
    print(f"Pickle will be saved to: {pickle_path}")
    print(f"Excel file will be saved to: {excel_path}")
    print(f"Database will be saved to: {db_path}")

    fx_manager.fetch_and_store_all(base="USD")

    # Print some sample rates to verify
    test_pairs = [
        ("USD", "EUR"),
        ("GBP", "USD"),
        ("EUR", "JPY"),
        ("GBP", "GBp")
    ]

    print("\nSample rates:")
    print("-" * 30)
    for base, quote in test_pairs:
        rate = fx_manager.get_rate(base, quote)
        print(f"1 {base} = {rate:.4f} {quote}")


if __name__ == "__main__":
    main()

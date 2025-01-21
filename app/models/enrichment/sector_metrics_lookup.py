# File: app/models/enrichment/sector_metrics_lookup.py

import logging
from typing import Dict, Any, List
import pandas as pd
import os

logger = logging.getLogger(__name__)

class SectorMetricsLookup:
    """
    Loads a spreadsheet of 'industry' -> up to 20 metrics, plus textual columns like 'Methodology' or
    'Ranking Justification'. Allows retrieving a list of metrics for an industry, plus any textual
    commentary if desired.
    """

    def __init__(self, excel_path: str = "app/assets/sector_valuation_metrics_methodology.xlsx"):
        """
        :param excel_path: Path to the Excel file containing columns like:
          - Industry
          - Metric Rank 1 ... Metric Rank 20
          - Methodology
          - Ranking Justification
        """
        self.excel_path = excel_path
        self.industry_data: Dict[str, Dict[str, Any]] = {}
        self._load_data()

    def _load_data(self) -> None:
        """
        Reads the entire spreadsheet at `self.excel_path` and populates `self.industry_data`.
        Each row must have 'Industry'. We'll gather up to 20 metrics plus optional text fields.
        """
        if not os.path.isfile(self.excel_path):
            logger.error(f"[SectorMetricsLookup] File not found: {self.excel_path}")
            return

        try:
            df = pd.read_excel(self.excel_path)
            if "Industry" not in df.columns:
                logger.error(f"[SectorMetricsLookup] 'Industry' column not found in {self.excel_path}")
                return

            # We'll check each row for up to 20 metrics
            metric_columns = [f"Metric Rank {i}" for i in range(1, 21)]

            for idx, row in df.iterrows():
                industry_name = str(row["Industry"]).strip()

                # Gather all non-empty Metric Rank columns
                metrics_list: List[str] = []
                for mc in metric_columns:
                    if mc in df.columns:
                        cell_val = row.get(mc)
                        if pd.notna(cell_val):
                            metrics_list.append(str(cell_val).strip())

                # We also read these textual columns if present
                methodology = ""
                if "Methodology" in df.columns and pd.notna(row.get("Methodology")):
                    methodology = str(row["Methodology"]).strip()

                ranking_justification = ""
                if "Ranking Justification" in df.columns and pd.notna(row.get("Ranking Justification")):
                    ranking_justification = str(row["Ranking Justification"]).strip()

                self.industry_data[industry_name] = {
                    "metrics": metrics_list,
                    "methodology": methodology,
                    "justification": ranking_justification
                }

            print(f"[DEBUG] Loaded sector metrics for {len(self.industry_data)} industries from {self.excel_path}")
        except Exception as e:
            logger.error(f"[SectorMetricsLookup] Error reading {self.excel_path}: {e}")

    def get_metrics_for_industry(self, industry: str, top_n: int = 5) -> List[str]:
        """
        Return up to 'top_n' metrics for the specified industry.
        If not found, returns an empty list or fallback.
        """
        entry = self.industry_data.get(industry, {})
        all_metrics = entry.get("metrics", [])
        # If top_n is bigger than the length, slicing won't hurt
        return all_metrics[:top_n]

    def get_methodology_text(self, industry: str) -> str:
        """
        Return the 'Methodology' text for the specified industry, or "" if not found.
        """
        entry = self.industry_data.get(industry, {})
        return entry.get("methodology", "")

    def get_justification_text(self, industry: str) -> str:
        """
        Return the 'Ranking Justification' text for the specified industry, or "" if not found.
        """
        entry = self.industry_data.get(industry, {})
        return entry.get("justification", "")

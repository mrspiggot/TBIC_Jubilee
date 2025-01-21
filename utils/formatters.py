import pandas as pd
def format_number(value):
    """Format numbers with K, M, Bn, Tn suffixes and 3 decimal places."""
    if pd.isna(value) or value is None:
        return "N/A"

    try:
        value = float(value)
        if abs(value) >= 1e12:
            return f"{value / 1e12:.3f}Tn"
        if abs(value) >= 1e9:
            return f"{value / 1e9:.3f}Bn"
        if abs(value) >= 1e6:
            return f"{value / 1e6:.3f}M"
        if abs(value) >= 1e3:
            return f"{value / 1e3:.3f}K"
        return f"{value:.3f}"
    except (ValueError, TypeError):
        return str(value)
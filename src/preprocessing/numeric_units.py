# preprocessing/numeric_units.py

import pandas as pd
import numpy as np
import regex as re
from typing import Dict, Tuple, List, Optional, Any



# Regex patterns to identify columns to exclude from the cleaning process (case-insensitive)
BASIC_EXCLUDE_SUBSTRINGS = ["description", "summary", "title", "name", "url", "desc"]

# slightly smarter ID patterns (case-insensitive)
ID_PATTERNS = [
    r"^id$",              # id
    r"^.*_id$",           # product_id, item_id, etc.
    r"^id_.*$",           # id_product, etc.
    r".*product.?id.*",   # productid, product_id, product-id
    r".*sku.*",           # sku
    r".*item.?id.*",      # itemid, item_id
]

def find_exclude_columns(df: pd.DataFrame) -> list[str]:
    exclude_cols: list[str] = []

    for col in df.columns:
        name = col.strip().lower()

        # 1) match simple substrings
        if any(sub in name for sub in BASIC_EXCLUDE_SUBSTRINGS):
            exclude_cols.append(col)
            continue

        # 2) match ID-like patterns
        for pat in ID_PATTERNS:
            if re.match(pat, name):
                exclude_cols.append(col)
                break  # no need to test other patterns for this col

    return exclude_cols

#exclude_cols = find_exclude_columns(df1)

def convert_numeric_strings(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert numeric-like string cells to floats/ints for all columns
    except those in exclude_cols.

    Rules:
    - Treat as numeric if:
        * Pure integer/float: [+/-]digits[.digits], e.g. "123", "-0.5", "3.14"
        * Thousands-style:    digits with groups of 3 separated by commas,
                              e.g. "1,234", "12,345,678", optionally with .decimals
    - Leave strings like "23,42" or "1,2,3" alone (these are likely list-like).
    """
    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = set(exclude_cols)

    # Pure integer/float: no commas, optional decimal point
    decimal_pattern = re.compile(r'^[+-]?\d+(?:\.\d+)?$')

    # Thousands pattern: 1–3 digits, then one or more ",ddd" groups, optional .decimals
    # Examples: "1,234", "12,345,678", "-1,234.56"
    thousands_pattern = re.compile(r'^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$')

    def to_number_if_numeric_string(x: Any) -> Any:
        # Leave non-strings alone
        if not isinstance(x, str):
            return normalize_attribute_value(x)

        s = x.strip()
        if not s:
            return normalize_attribute_value(x)

        # Case 1: pure decimal (no commas)
        if decimal_pattern.match(s):
            try:
                num = float(s)
            except ValueError:
                return normalize_attribute_value(x)
            return int(num) if num.is_integer() else num

        # Case 2: thousands-style (commas as thousands separators)
        if thousands_pattern.match(s):
            s_no_commas = s.replace(',', '')
            try:
                num = float(s_no_commas)
            except ValueError:
                return normalize_attribute_value(x)
            return int(num) if num.is_integer() else num

        # Anything else (e.g. "23,42", "1,2,3", "10 mm, 12 mm") -> keep as string
        return normalize_attribute_value(x)

    df_out = df.copy()
    for col in df_out.columns:
        if col in exclude_cols:
            continue
        df_out[col] = df_out[col].apply(to_number_if_numeric_string)

    return df_out


def safe_to_float(x: Any) -> float:
    """
    Best-effort conversion to float for comparison purposes.
    Handles ints/floats, decimal strings, and thousands-style strings.
    Returns np.nan if not convertible.
    """
    if isinstance(x, (int, float, np.number)):
        return float(x)

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return np.nan

        # Match the same patterns as convert_numeric_strings
        decimal_pattern = re.compile(r'^[+-]?\d+(?:\.\d+)?$')
        thousands_pattern = re.compile(r'^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$')

        if thousands_pattern.match(s):
            s = s.replace(',', '')
        elif not decimal_pattern.match(s):
            # As a last resort, try a generic float; if that fails, give up
            try:
                return float(s)
            except ValueError:
                return np.nan

        try:
            return float(s)
        except ValueError:
            return np.nan

    return np.nan


# =========================
# 2. Inline "value unit" parser
# =========================

# Reuse the same numeric patterns as convert_numeric_strings
_NUM_DECIMAL = re.compile(r'^[+-]?\d+(?:[.,]\d+)?$')
_NUM_THOUSANDS = re.compile(r'^[+-]?\d{1,3}(?:,\d{3})+(?:[.,]\d+)?$')

# Unit must start with a letter, to avoid ".5" / "0" etc.
_INLINE_VALUE_UNIT_PATTERN = re.compile(
    r'^\s*([-+]?\d+(?:[.,]\d+)?)\s*([A-Za-z][^\s,;]*)\s*$'
)


def parse_inline_value_unit(x: Any) -> Tuple[Optional[float], Optional[str]]:
    """
    If x looks like 'number unit' (e.g. '132 mm', '45mm', '84 in'),
    return (value: float, unit: str). Otherwise return (None, None).

    - Skips pure numeric strings and thousands-style numeric strings
      (those are handled by convert_numeric_strings instead).
    """
    if not isinstance(x, str):
        return None, None

    s = x.strip()
    if not s:
        return None, None

    # 1) Skip pure numeric / thousands-style -> let convert_numeric_strings handle these
    if _NUM_DECIMAL.match(s) or _NUM_THOUSANDS.match(s):
        return None, None

    # 2) Now try to match "number + unit" where unit starts with a letter
    m = _INLINE_VALUE_UNIT_PATTERN.match(s)
    if not m:
        return None, None

    raw_val, unit = m.groups()
    raw_val = raw_val.replace(',', '.')  # decimal comma -> dot
    try:
        value = float(raw_val)
    except ValueError:
        return None, None

    return value, unit


def extract_inline_units_from_values(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    For each non-excluded column, look for simple 'number unit' strings.
    If found in a cell, convert that cell to numeric and store the unit
    in a new {col}_unit column. Other values are left unchanged.

    Example:
        "132 mm" -> 132.0, unit "mm"
        "45mm"   -> 45.0,  unit "mm"
        "84 in"  -> 84.0,  unit "in"
    """
    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = set(exclude_cols)

    df_out = df.copy()
    unit_cols_data: Dict[str, List[Any]] = {}

    for col in df_out.columns:
        if col in exclude_cols:
            continue

        units: List[Any] = []
        new_vals: List[Any] = []

        for x in df_out[col]:
            value, unit = parse_inline_value_unit(x)
            if unit is not None:
                # Inline "number unit" detected
                new_vals.append(value)
                units.append(unit)
            else:
                new_vals.append(normalize_attribute_value(x))
                units.append(np.nan)

        df_out[col] = new_vals

        # Only create a unit column if we actually saw any units
        if any(pd.notna(units)):
            unit_cols_data[f"{col}_unit"] = units

    if unit_cols_data:
        df_out = df_out.assign(**unit_cols_data)

    return df_out


# =========================
# 3. Units from description text
# =========================

def extract_attr_units(description: Any, attr_labels: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    For a single description string, return:
        { attr_label: {'value': float, 'unit': 'mm'} }
    based on patterns like:
        "Package depth: 230.5 mm"
        "Package depth 230.5 mm"
    """
    if pd.isna(description):
        return {}

    text = str(description)
    result: Dict[str, Dict[str, Any]] = {}

    for label in attr_labels:
        pattern = re.compile(
            rf'{re.escape(label)}\s*[:=]?\s*'   # label + optional ":" or "="
            r'([-+]?\d+(?:[.,]\d+)?)\s*'        # numeric value
            r'([^\s,.;]+)',                     # unit (until space/comma/period/semicolon)
            flags=re.IGNORECASE
        )

        m = pattern.search(text)
        if m:
            raw_val, unit = m.groups()
            raw_val = raw_val.replace(',', '.')  # normalize decimal comma
            try:
                value = float(raw_val)
            except ValueError:
                value = np.nan

            result[label] = {'value': value, 'unit': unit}

    return result


def add_unit_columns_from_description(
    df,
    desc_col='description',
    exclude_cols=None,
    atol=1e-3
):
    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = set(exclude_cols) | {desc_col}

    df_out = df.copy()

    # 1) Which numeric columns to process
    attr_cols = [
        c for c in df_out.columns
        if (
            c not in exclude_cols
            and c != desc_col
            and not c.endswith('_unit')
            and pd.api.types.is_numeric_dtype(df_out[c])
        )
    ]

    # 2) Parse description into dict per row
    df_out['_parsed_attrs'] = df_out[desc_col].apply(
        extract_attr_units,
        attr_labels=attr_cols
    )

    # 3) Build unit columns into a dict first
    new_unit_cols = {}

    for col in attr_cols:
        unit_col = f'{col}_unit'

        def get_unit_for_row(row, col=col, unit_col=unit_col):
            # keep existing unit if present (from inline parsing)
            existing = row.get(unit_col, np.nan)
            if pd.notna(existing):
                return existing

            d = row['_parsed_attrs']
            if not isinstance(d, dict) or col not in d:
                return np.nan

            parsed_val = d[col].get('value')
            unit = d[col].get('unit')
            col_val = normalize_attribute_value(row[col])
            col_val_float = safe_to_float(row[col_val])

            if (
                parsed_val is not None
                and not np.isnan(parsed_val)
                and pd.notna(col_val_float)
            ):
                if not np.isclose(col_val_float, parsed_val, atol=atol):
                    return np.nan

            return unit

        series = df_out.apply(get_unit_for_row, axis=1)

        # only keep this *_unit column if at least one non-NaN
        if series.notna().any():
            new_unit_cols[unit_col] = series

    # attach all *_unit columns at once
    if new_unit_cols:
        df_out = df_out.assign(**new_unit_cols)

    # 4) Clean up helper column and return
    df_out = df_out.drop(columns=['_parsed_attrs'])

    return df_out, attr_cols

def detect_description_like_column(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    min_label_hits: int = 2, # minimum number of column labels that appear in the possible description column
    min_median_len: int = 20, # ignore columns whose typical value is shorter than 20 characters
    max_rows: int = 1500,
) -> Optional[str]:
    """
    Heuristically detect a 'description-like' column.

    Idea:
      - Candidate columns are text-ish and relatively long.
      - A description-like column is the one whose cell values
        contain the names of many *other* columns (attribute labels).
    """
    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = set(exclude_cols)

    if df.empty:
        return None

    # 1) Candidate text-like columns
    text_cols = [
        c for c in df.columns
        if (
            c not in exclude_cols
            and not c.endswith("_unit")
            and (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object")
        )
    ]
    if not text_cols:
        return None

    # Sample rows for performance
    if len(df) > max_rows:
        sample = df.sample(n=max_rows, random_state=0)
    else:
        sample = df

    sample_str = {c: sample[c].astype(str).str.lower() for c in text_cols}
    all_labels = [c for c in df.columns if not c.endswith("_unit")]

    scores: Dict[str, Dict[str, Any]] = {}

    for cand in text_cols:
        series = sample_str[cand]
        median_len = series.str.len().median()
        if median_len < min_median_len:
            continue

        label_hits = set()
        total_hits = 0

        other_labels = [lbl for lbl in all_labels if lbl != cand]

        for lbl in other_labels:
            lbl_clean = str(lbl).strip()
            if not lbl_clean:
                continue
            lbl_lc = lbl_clean.lower()
            # avoid super-short labels like "id"
            if len(lbl_lc) < 4:
                continue

            contains = series.str.contains(lbl_lc, regex=False, na=False)
            if contains.any():
                label_hits.add(lbl)
                total_hits += contains.sum()

        if label_hits:
            scores[cand] = {
                "distinct_labels": len(label_hits),
                "total_hits": total_hits,
                "median_len": median_len,
            }

    if not scores:
        return None

    best_col = None
    best_key = (-1, -1)  # (distinct_labels, total_hits)

    for col, info in scores.items():
        k = (info["distinct_labels"], info["total_hits"])
        if k > best_key:
            best_col = col
            best_key = k

    if best_col is None:
        return None

    if scores[best_col]["distinct_labels"] < min_label_hits:
        return None

    return best_col


# =========================
# 4. High-level convenience
# =========================

def process_units(
    df: pd.DataFrame,
    desc_col: Optional[str] = 'description',
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    High-level pipeline to:
      1) Extract inline "number unit" values from attribute columns.
      2) Convert remaining numeric-like strings to numbers.
      3) (If available) Fill missing units from a description-like column.

    If desc_col is None or desc_col is not present in df, we try to
    auto-detect a description-like column. If none is found, step (3)
    is skipped and all units come from inline values.
    """
    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = list(exclude_cols)  # ensure list

    # --- Detect description column if needed ---
    has_desc = desc_col in df.columns if desc_col is not None else False

    if not has_desc:
        detected = detect_description_like_column(df, exclude_cols=exclude_cols)
        if detected is not None:
            desc_col = detected
            has_desc = True
        else:
            desc_col = None
            has_desc = False

    # 1) Inline "number unit" in the attribute columns
    exclude_for_inline = exclude_cols.copy()
    if has_desc:
        exclude_for_inline.append(desc_col)

    df_step1 = extract_inline_units_from_values(df, exclude_cols=exclude_for_inline)

    # 2) Convert remaining numeric-like strings in non-excluded columns
    exclude_for_numeric = exclude_cols.copy()
    if has_desc:
        exclude_for_numeric.append(desc_col)

    df_step2 = convert_numeric_strings(df_step1, exclude_cols=exclude_for_numeric)

    # 3) If a description column exists, fill unit gaps from description text.
    if has_desc:
        df_final, attr_cols = add_unit_columns_from_description(
            df_step2,
            desc_col=desc_col,
            exclude_cols=exclude_cols
        )
    else:
        df_final = df_step2
        attr_cols = [
            c for c in df_final.columns
            if (
                c not in exclude_cols
                and not c.endswith('_unit')
                and pd.api.types.is_numeric_dtype(df_final[c])
            )
        ]

    return df_final, attr_cols


# Detect if a string value is "list-like"
def is_list_like_cell(x) -> bool:
    """
    Heuristic for a single cell: does this value look like a list?

    Returns True for things like:
    - "CX 542 / CX 422 / CX 124"
    - "CX 54/24, CX 13/45, CX 15/22"
    - "10 mm, 12 mm, 14 mm"
    """
    if pd.isna(x):
        return False
    
    s = str(x).strip()
    if not s:
        return False
    
    # obvious list delimiters
    if re.search(r'[;,\|\n]', s): # old pattern: r'[;,|]'
        return True
    
    # spaced slash as separator
    if ' / ' in s:
        return True
    
    # multiple slashes without spaces (like "CX1/CX2/CX3")
    if s.count('/') >= 2:
        return True
    
    return False

# Convert a list-like string into a list of items
def parse_list_value(s: str):
    
    s = str(s).strip()
    if not s:
        return []
    
    items = []
    
    # 1) split on commas as outer separator
    top_parts = re.split(r'\s*,\s*', s)
    
    for part in top_parts:
        part = part.strip()
        if not part:
            continue
        
        # 2) split on spaced slashes; keep tight slashes intact
        if ' / ' in part:
            subparts = re.split(r'\s*/\s*', part)
            for sp in subparts:
                sp = sp.strip()
                if sp:
                    items.append(sp)
        else:
            items.append(normalize_attribute_value(part))
    
    return items

# Transformer at the cell-level - only converts when string is list-like
def convert_cell_to_list_if_needed(x):
    """
    If x looks list-like, return a list of values.
    Otherwise, return x unchanged.
    """
    if not is_list_like_cell(x):
        return normalize_attribute_value(x)
    
    return parse_list_value(x)


def normalize_attribute_value(val):
    """
    Normalize any attribute value into a consistent, comparable form
    so that cluster naming (TF-IDF and purity checks) works better.

    Handles:
    - NaN / empty
    - multi-value lists: sort values, unify separators
    - lowercase normalization
    - punctuation removal
    - freetext token sorting
    - de-duplication
    """

    if val is None:
        return None

    # convert to string
    s = str(val).strip()
    if s == "" or s.lower() in ["nan", "none", "null", "n/a", "unknown"]:
        return None

    # ------------------------------
    # 1) Normalize common multi-value separators
    # ------------------------------
    # Convert commas, semicolons, pipes into semicolons
    s = re.sub(r"[,\|]", ";", s)

    # Remove duplicate semicolons or spaces around them
    s = re.sub(r"\s*;\s*", ";", s)

    # ------------------------------
    # 2) Split multi-values on semicolon if present
    # ------------------------------
    if ";" in s:
        parts = s.split(";")
        cleaned = []

        for p in parts:
            p = p.strip().lower()
            p = re.sub(r"[^a-z0-9 ]+", "", p)  # only alphanumeric + space
            if p not in ["", "nan", "none"]:
                cleaned.append(p)

        if len(cleaned) == 0:
            return None

        # sort values for consistency
        cleaned = sorted(set(cleaned))

        return "; ".join(cleaned)

    # ------------------------------
    # 3) Freetext normalization
    # ------------------------------
    # lowercase
    s = s.lower()

    # remove punctuation
    s = re.sub(r"[^a-z0-9 ]+", "", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # tokenize → sort → dedupe
    tokens = s.split()
    if len(tokens) > 1:
        tokens = sorted(set(tokens))
        return " ".join(tokens)

    return s


def clean_office_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the office_products CSV.

    - Detects columns to exclude from numeric/unit parsing
    - Runs the numeric + unit extraction logic
    - Converts list-like strings (e.g. 'A / B, C') into Python lists
    """
    df = df.copy()

    # 1) Decide which columns to exclude from numeric/unit parsing
    exclude_cols = find_exclude_columns(df)

    # 2) Run the units + numeric parsing pipeline
    df_units, _ = process_units(df, exclude_cols=exclude_cols)

    # 3) Convert list-like cells into real Python lists
    processed_df = df_units.copy()

    for col in processed_df.columns:
        if col in exclude_cols:
            continue

        # First, run the existing logic
        processed_df[col] = processed_df[col].apply(convert_cell_to_list_if_needed)

        # Then, if the column has ANY lists, wrap all non-list, non-null values
        col_series = processed_df[col]
        has_list = col_series.apply(lambda v: isinstance(v, list)).any()

        if has_list:
            processed_df[col] = col_series.apply(
                lambda v: v if isinstance(v, list) or pd.isna(v) else [ normalize_attribute_value(v) ]
            )


    return processed_df
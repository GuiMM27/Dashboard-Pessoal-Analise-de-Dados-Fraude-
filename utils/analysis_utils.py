import numpy as np
import pandas as pd
from scipy import stats

# ----------------------- Data helpers -----------------------
def load_data(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    # Normaliza nomes de colunas (por garantia)
    df.columns = [c.strip() for c in df.columns]
    return df

def describe_variables(df: pd.DataFrame):
    dtypes = df.dtypes.astype(str).to_dict()
    summary = df.describe(include='all', datetime_is_numeric=True).T
    return dtypes, summary

def class_balance(df: pd.DataFrame, target_col='Class'):
    counts = df[target_col].value_counts().sort_index()
    proportions = counts / counts.sum()
    return counts, proportions

# ----------------------- EDA metrics ------------------------
def central_tendency_and_dispersion(series: pd.Series):
    series = pd.to_numeric(series, errors='coerce').dropna()
    mean = series.mean()
    median = series.median()
    try:
        mode = series.mode().iloc[0]
    except Exception:
        mode = np.nan
    variance = series.var(ddof=1)
    std = series.std(ddof=1)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return {
        'mean': mean, 'median': median, 'mode': mode,
        'variance': variance, 'std': std, 'q1': q1, 'q3': q3, 'iqr': iqr
    }

def correlation_with_target(df: pd.DataFrame, target_col='Class', top_k=10):
    # Correlação ponto-bisserial para numéricas vs binária
    out = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    y = df[target_col].values
    for col in numeric_cols:
        x = df[col].values
        # Remover NaNs
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() > 2 and len(np.unique(y[mask])) == 2:
            r, p = stats.pointbiserialr(y[mask], x[mask])
            out.append((col, r, p))
    out.sort(key=lambda t: abs(t[1]), reverse=True)
    top = out[:top_k]
    return pd.DataFrame(top, columns=['feature', 'pointbiserial_r', 'p_value'])

# ---------------- Confidence Intervals ---------------------
def ci_for_proportion(successes, n, confidence=0.95, method='normal'):
    p_hat = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    if method == 'normal':
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        return p_hat, p_hat - z * se, p_hat + z * se
    elif method == 'wilson':
        # Wilson score interval
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denom
        half_width = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denom
        return p_hat, center - half_width, center + half_width
    else:
        raise ValueError("method must be 'normal' or 'wilson'")

def bootstrap_ci_mean(series: pd.Series, confidence=0.95, n_boot=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    x = pd.to_numeric(series, errors='coerce').dropna().values
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo = np.percentile(boots, (1 - confidence) / 2 * 100)
    hi = np.percentile(boots, (1 + confidence) / 2 * 100)
    return x.mean(), lo, hi

# ---------------- Hypothesis Tests -------------------------
def test_difference_in_means(x, y):
    # Welch's t-test (variâncias possivelmente diferentes)
    x = pd.to_numeric(x, errors='coerce').dropna().values
    y = pd.to_numeric(y, errors='coerce').dropna().values
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_ind(x, y, equal_var=False)
    return t, p

def ztest_proportions(successes_a, n_a, successes_b, n_b):
    # Teste z para diferença de proporções
    p1 = successes_a / n_a
    p2 = successes_b / n_b
    p = (successes_a + successes_b) / (n_a + n_b)
    se = np.sqrt(p * (1 - p) * (1/n_a + 1/n_b))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, pval

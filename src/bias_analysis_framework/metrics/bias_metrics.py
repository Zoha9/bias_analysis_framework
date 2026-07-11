"""Shared bias metric primitives.

This module contains low-level metric functions used by the workflow-specific
analysis modules. Keep these functions formula-focused and avoid user-path,
file-writing, or plotting logic here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import (
    anderson_ksamp,
    chi2_contingency,
    entropy,
    f_oneway,
    kendalltau,
    kruskal,
    ks_2samp,
    pearsonr,
    pointbiserialr,
    spearmanr,
)
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve

# --- Stability guards for DI/DIlog2 ---
MIN_GROUP_COUNT = 20       # ignore groups with fewer samples than this
LAPLACE_ALPHA = 1.0        # Laplace smoothing for rates: (pos+α)/(n+2α)
MIN_RATE_FLOOR = 1e-3      # floor very small rates to avoid blow-ups
MAX_ABS_LOG2 = 4.0         # mask |log2(r_i/r_j)| beyond this in heatmaps


class PretrainingBiasMetrics:
    """
    A class for pre-training bias detection metrics.
    """
    # ----------------------------
    # Class Distribution Metrics
    # ----------------------------
    @staticmethod
    def class_imbalance(data, target_column):
        
        #Compute Class Imbalance (CI) for a target column.

        counts = data[target_column].value_counts(normalize=True)
        imbalance = counts.max() / counts.min()
        return imbalance

    @staticmethod
    def gini_coefficient(data, target_column):

        #Compute Gini Coefficient for class distribution.

        class_counts = data[target_column].value_counts().values
        probs = class_counts / class_counts.sum()
        gini = 1 - np.sum(probs ** 2)
        return gini

    @staticmethod
    def rare_case_detection(data, column, threshold=0.05):

        #Identify rare subgroups or classes based on a threshold.

        counts = data[column].value_counts()
        rare_cases = counts[counts < threshold * len(data)]
        return rare_cases

    # ----------------------------
    # Demographic Metrics
    # ----------------------------
    @staticmethod
    def difference_in_positive_proportions(data, subgroup_column, target_column):
        
        #Compute Difference in Positive Proportions in Observed Labels (DPL).
        
        positive_rates = (
            data[data[target_column] == 1]
            .groupby(subgroup_column)
            .size()
            .div(data.groupby(subgroup_column).size())
        )
        dpl = positive_rates.max() - positive_rates.min()
        return dpl

    @staticmethod
    def statistical_parity_difference(data, subgroup_column, target_column):
        """
        Compute Statistical Parity Difference (SPD).
        """
        positive_rate_by_group = (
            data[data[target_column] == 1]
            .groupby(subgroup_column)
            .size()
            .div(data.groupby(subgroup_column).size())
        )
        spd = positive_rate_by_group.max() - positive_rate_by_group.min()
        return spd

    @staticmethod
    def _stabilized_positive_rates(data, group_col, target_col,
                                   min_count=MIN_GROUP_COUNT,
                                   alpha=LAPLACE_ALPHA,
                                   min_rate_floor=MIN_RATE_FLOOR) -> pd.Series:
        """
        Returns Laplace-smoothed positive rate per subgroup with basic stability:
        - drop groups with total count < min_count
        - smoothed rate = (pos + alpha) / (n + 2*alpha)
        - clip to [min_rate_floor, 1]
        """
        d = data[[group_col, target_col]].dropna()
        if d.empty:
            return pd.Series(dtype=float)

        grp_tot = d.groupby(group_col)[target_col].count()
        kept = grp_tot[grp_tot >= min_count].index
        if len(kept) < 2:
            return pd.Series(dtype=float)

        d = d[d[group_col].isin(kept)]
        grp_pos = d.groupby(group_col)[target_col].sum()
        grp_tot = d.groupby(group_col)[target_col].count()

        rates = (grp_pos + alpha) / (grp_tot + 2.0 * alpha)
        return rates.clip(lower=min_rate_floor, upper=1.0)

    @staticmethod
    def disparate_impact(data, subgroup_column, target_column):
        """
        Stable Disparate Impact (DI) = min(r_g)/max(r_g) using stabilized rates.
        Returns np.nan if fewer than two valid groups remain after filtering.
        """
        rates = PretrainingBiasMetrics._stabilized_positive_rates(
            data, subgroup_column, target_column
        )
        if rates.empty or rates.nunique() < 2:
            return np.nan
        return float(rates.min() / rates.max())

    """
    @staticmethod
    def disparate_impact(data, subgroup_column, target_column):

        #Compute Disparate Impact (DI).
 
        positive_rate_by_group = (
            data[data[target_column] == 1]
            .groupby(subgroup_column)
            .size()
            .div(data.groupby(subgroup_column).size())
        )
        di = positive_rate_by_group.min() / positive_rate_by_group.max()
        return di
        """
    
    @staticmethod
    def conditional_demographic_disparity(data, subgroup_column, target_column):
        """
        Compute Conditional Demographic Disparity in Labels (CDDL).
        If the target column is non-numeric, encode it numerically first.
        """
        if not pd.api.types.is_numeric_dtype(data[target_column]):
            # Encode non-numeric target column
            data[target_column] = pd.factorize(data[target_column])[0]
        subgroup_means = data.groupby(subgroup_column)[target_column].mean()
        cddl = subgroup_means.max() - subgroup_means.min()
        return cddl


    # ----------------------------
    # Distributional Metrics
    # ----------------------------
    @staticmethod
    def kl_divergence(p, q):
        """
        Compute Kullback-Leibler (KL) Divergence between two distributions.
        """
        return entropy(p, q)

    @staticmethod
    def js_divergence(p, q):
        """
        Compute Jensen-Shannon (JS) Divergence between two distributions.
        """
        return jensenshannon(p, q) ** 2

    @staticmethod
    def lp_norm(p, q, p_norm=2):
        """
        Compute LP Norm between two distributions.
        """
        return np.linalg.norm(np.array(p) - np.array(q), ord=p_norm)

    @staticmethod
    def total_variation_distance(p, q):
        """
        Compute Total Variation Distance (TVD) between two distributions.
        """
        return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))

    @staticmethod
    def kolmogorov_smirnov(data, group_column, target_column):
        """
        Compute Kolmogorov-Smirnov (KS) statistic for two groups.
        """
        groups = data[group_column].unique()
        if len(groups) != 2:
            raise ValueError("KS test requires exactly two groups.")
        group1 = data[data[group_column] == groups[0]][target_column].values
        group2 = data[data[group_column] == groups[1]][target_column].values
        ks_stat, p_value = ks_2samp(group1, group2)
        return ks_stat, p_value

    # ----------------------------
    # Temporal Metrics
    # ----------------------------
    @staticmethod
    def temporal_trend_analysis(data, time_column, target_column):
        """
        Analyze trends over time using Kendall Tau correlation.
        """
        kendall_tau, p_value = kendalltau(data[time_column], data[target_column])
        return kendall_tau, p_value

    # ----------------------------
    # Outcome Metrics
    # ----------------------------
    @staticmethod
    def average_odds_difference(y_true, y_pred, subgroup_column, data):
        """
        Compute Average Odds Difference (AOD).
        """
        groups = data[subgroup_column].unique()
        if len(groups) != 2:
            raise ValueError("AOD requires exactly two groups.")
        group1 = data[data[subgroup_column] == groups[0]]
        group2 = data[data[subgroup_column] == groups[1]]

        fpr1, tpr1, _ = roc_curve(group1[y_true], group1[y_pred])
        fpr2, tpr2, _ = roc_curve(group2[y_true], group2[y_pred])
        avg_odds_diff = 0.5 * (abs(tpr1 - tpr2).mean() + abs(fpr1 - fpr2).mean())
        return avg_odds_diff

    @staticmethod
    def equal_opportunity_difference(y_true, y_pred, subgroup_column, data):
        """
        Compute Equal Opportunity Difference (EOD).
        """
        groups = data[subgroup_column].unique()
        if len(groups) != 2:
            raise ValueError("EOD requires exactly two groups.")
        group1 = data[data[subgroup_column] == groups[0]]
        group2 = data[data[subgroup_column] == groups[1]]

        tpr1 = roc_curve(group1[y_true], group1[y_pred])[1].mean()
        tpr2 = roc_curve(group2[y_true], group2[y_pred])[1].mean()
        eod = abs(tpr1 - tpr2)
        return eod
    
    # =========================
    # Multi-subgroup extensions
    # =========================
    @staticmethod
    def pairwise_spd_matrix(data, group_col, target_col):
        """
        Pairwise Statistical Parity Difference (SPD) matrix across all subgroups.
        SPD_ij = P(Y=1|G=i) - P(Y=1|G=j)
        Returns a DataFrame (index=groups, columns=groups).
        """
        d = data[[group_col, target_col]].dropna()
        pos_rates = (
            d[d[target_col] == 1].groupby(group_col).size()
            .div(d.groupby(group_col).size())
        )
        groups = pos_rates.index.tolist()
        mat = pd.DataFrame(index=groups, columns=groups, dtype=float)
        for i in groups:
            for j in groups:
                mat.loc[i, j] = pos_rates.get(i, 0) - pos_rates.get(j, 0)
        return mat

    @staticmethod
    def pairwise_di_matrix(data, group_col, target_col):
        """
        Pairwise Disparate Impact (DI) across valid subgroups.
        Uses stabilized rates so no infinite/huge artifacts.
        """
        rates = PretrainingBiasMetrics._stabilized_positive_rates(
            data, group_col, target_col
        )
        groups = rates.index.tolist()
        if len(groups) < 2:
            return pd.DataFrame()

        mat = pd.DataFrame(np.nan, index=groups, columns=groups, dtype=float)
        for i in groups:
            for j in groups:
                if i == j:
                    mat.loc[i, j] = 1.0
                else:
                    mat.loc[i, j] = float(min(rates[i], rates[j]) / max(rates[i], rates[j]))
        return mat

    @staticmethod
    def pairwise_dilog2_matrix(data, group_col, target_col, max_abs_log2=MAX_ABS_LOG2):
        """
        Heatmap-ready DIlog2 matrix: log2(r_i / r_j) using stabilized rates.
        Cells with |log2| > max_abs_log2 are set to NaN to avoid dominating the colormap.
        """
        rates = PretrainingBiasMetrics._stabilized_positive_rates(
            data, group_col, target_col
        )
        groups = rates.index.tolist()
        if len(groups) < 2:
            return pd.DataFrame()

        mat = pd.DataFrame(0.0, index=groups, columns=groups, dtype=float)
        for i in groups:
            for j in groups:
                if i == j:
                    mat.loc[i, j] = 0.0
                else:
                    val = np.log2(rates[i] / rates[j])
                    mat.loc[i, j] = val if np.isfinite(val) and abs(val) <= max_abs_log2 else np.nan
        return mat
    
    """
    @staticmethod
    def pairwise_di_matrix(data, group_col, target_col):
        
        Pairwise Disparate Impact (DI) matrix across all subgroups.
        DI_ij = P(Y=1|G=i) / P(Y=1|G=j).
        Returns a DataFrame (index=groups, columns=groups).
        
        d = data[[group_col, target_col]].dropna()
        pos_rates = (
            d[d[target_col] == 1].groupby(group_col).size()
            .div(d.groupby(group_col).size())
        )
        groups = pos_rates.index.tolist()
        mat = pd.DataFrame(index=groups, columns=groups, dtype=float)
        for i in groups:
            for j in groups:
                pj = pos_rates.get(j, np.nan)
                mat.loc[i, j] = np.nan if pj in [0, np.nan] else pos_rates.get(i, np.nan) / pj
        return mat
        """
    
    @staticmethod
    def max_pairwise_spd(data, group_col, target_col):
        """
        Maximum absolute pairwise SPD across groups (worst-case disparity).
        """
        mat = PretrainingBiasMetrics.pairwise_spd_matrix(data, group_col, target_col)
        return np.nanmax(np.abs(mat.values))

    @staticmethod
    def max_pairwise_di_spread(data, group_col, target_col):
        """
        Max DI spread across groups, defined as max(DI_ij) - min(DI_ij) over valid pairs.
        Useful when >2 groups so a single DI doesn't hide extremes.
        """
        mat = PretrainingBiasMetrics.pairwise_di_matrix(data, group_col, target_col)
        arr = mat.values.astype(float)
        arr = arr[np.isfinite(arr)]
        return np.nan if arr.size == 0 else (np.nanmax(arr) - np.nanmin(arr))

    @staticmethod
    def coefficient_of_variation_across_groups(data, group_col, target_col):
        """
        Coefficient of Variation (CV) of positive rates across groups.
        CV = std(pos_rates) / mean(pos_rates).
        """
        d = data[[group_col, target_col]].dropna()
        grp_sizes = d.groupby(group_col).size()
        if grp_sizes.min() == 0:
            return np.nan
        pos_rates = (
            d[d[target_col] == 1].groupby(group_col).size()
            .div(d.groupby(group_col).size())
        )
        m = pos_rates.mean()
        return np.nan if m == 0 else float(pos_rates.std(ddof=0) / m)

    @staticmethod
    def range_of_positive_rates(data, group_col, target_col):
        """
        Range (max - min) of positive rates across groups.
        Equivalent to SPD when computed across all groups.
        """
        d = data[[group_col, target_col]].dropna()
        pos_rates = (
            d[d[target_col] == 1].groupby(group_col).size()
            .div(d.groupby(group_col).size())
        )
        return float(pos_rates.max() - pos_rates.min()) if len(pos_rates) else np.nan

    @staticmethod
    def entropy_of_positive_rates(data, group_col, target_col):
        """
        Entropy of normalized positive rates across groups.
        Normalizes positive rates to sum to 1, then computes Shannon entropy.
        Higher entropy ~ more even positive-rate distribution across groups.
        """
        d = data[[group_col, target_col]].dropna()
        pos_rates = (
            d[d[target_col] == 1].groupby(group_col).size()
            .div(d.groupby(group_col).size())
        )
        pr = pos_rates.values
        s = pr.sum()
        if s <= 0:
            return np.nan
        p = pr / s
        return float(entropy(p))

    @staticmethod
    def gini_across_groups_of_positive_rates(data, group_col, target_col):
        """
        Gini-like index across group positive rates: 1 - sum(p_i^2),
        where p_i are normalized positive rates across groups.
        """
        d = data[[group_col, target_col]].dropna()
        pos_rates = (
            d[d[target_col] == 1].groupby(group_col).size()
            .div(d.groupby(group_col).size())
        )
        pr = pos_rates.values
        s = pr.sum()
        if s <= 0:
            return np.nan
        p = pr / s
        return float(1 - np.sum(p ** 2))

    # ===========================================
    # Categorical association (multi-class ready)
    # ===========================================
    @staticmethod
    def chi_square_independence(data, group_col, target_col):
        """
        χ² test of independence between group_col and target_col (both categorical).
        Returns (chi2, p_value, dof, expected_table).
        """
        contingency = pd.crosstab(data[group_col], data[target_col])
        chi2, p, dof, expected = chi2_contingency(contingency)
        return float(chi2), float(p), int(dof), expected

    @staticmethod
    def cramers_v(data, group_col, target_col, bias_correction=True):
        """
        Cramér’s V association between two categorical variables (group, label).
        Range [0,1]; higher => stronger association.
        """
        contingency = pd.crosstab(data[group_col], data[target_col]).values
        chi2 = chi2_contingency(contingency)[0]
        n = contingency.sum()
        r, k = contingency.shape
        phi2 = chi2 / max(n, 1)
        if bias_correction:
            phi2corr = max(0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
            rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
            kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
            denom = min((kcorr - 1), (rcorr - 1))
            return float(np.sqrt(phi2corr / max(denom, 1e-12)))
        else:
            return float(np.sqrt(phi2 / max(min(k - 1, r - 1), 1e-12)))

    # ===================================
    # Multi-group distributional distance
    # ===================================
    @staticmethod
    def js_divergence_across_groups(data, group_col, target_col):
        """
        Jensen–Shannon divergence across multiple groups for a categorical target.
        1) Compute p_g(y) for each group g over all observed classes y (auto-discovered).
        2) Compute the mean distribution m(y) across groups.
        3) Return the mean JSD between each p_g and m.
        Works for binary and multi-class targets.
        """
        d = data[[group_col, target_col]].dropna()
        groups = d[group_col].unique().tolist()
        classes = d[target_col].dropna().unique().tolist()
        if len(groups) < 2 or len(classes) < 1:
            return np.nan
        dists = []
        for g in groups:
            dist = (
                d.loc[d[group_col] == g, target_col]
                .value_counts(normalize=True)
                .reindex(classes, fill_value=0.0)
                .values
            )
            dists.append(dist)
        dists = np.asarray(dists, dtype=float)
        mean_dist = dists.mean(axis=0)
        # Mean JSD to the centroid (squared because scipy returns sqrt(JS))
        js_vals = [(jensenshannon(di, mean_dist) ** 2) for di in dists]
        return float(np.mean(js_vals))

    @staticmethod
    def total_variation_across_groups(data, group_col, target_col):
        """
        Average Total Variation Distance between group label distributions and the mean distribution.
        TVD(p, q) = 0.5 * ||p - q||_1. Returns the mean TVD across groups.
        """
        d = data[[group_col, target_col]].dropna()
        groups = d[group_col].unique().tolist()
        classes = d[target_col].dropna().unique().tolist()
        if len(groups) < 2 or len(classes) < 1:
            return np.nan
        dists = []
        for g in groups:
            dist = (
                d.loc[d[group_col] == g, target_col]
                .value_counts(normalize=True)
                .reindex(classes, fill_value=0.0)
                .values
            )
            dists.append(dist)
        dists = np.asarray(dists, dtype=float)
        mean_dist = dists.mean(axis=0)
        tv_vals = [0.5 * np.sum(np.abs(di - mean_dist)) for di in dists]
        return float(np.mean(tv_vals))

    # ==========================================
    # Multi-group tests for continuous variables
    # ==========================================
    @staticmethod
    def anderson_darling_k_sample(data, group_col, value_col):
        """
        Anderson–Darling k-sample test for equality of continuous distributions across groups.
        Returns (statistic, critical_values, significance_level_estimate, p_value_approx)
        Note: scipy returns a statistic and critical values; p-value is approximate.
        """
        d = data[[group_col, value_col]].dropna()
        groups = [grp[value_col].values for _, grp in d.groupby(group_col)]
        if len(groups) < 2:
            return np.nan, None, None, None
        res = anderson_ksamp(groups)
        # res.statistic, res.critical_values, res.significance_level
        # SciPy provides a significance level (%) instead of exact p.
        return float(res.statistic), res.critical_values, float(res.significance_level), None

    @staticmethod
    def anova_f_oneway(data, group_col, value_col):
        """
        One-way ANOVA for comparing group means of a continuous variable.
        Returns (F_stat, p_value).
        """
        d = data[[group_col, value_col]].dropna()
        groups = [grp[value_col].values for _, grp in d.groupby(group_col)]
        if len(groups) < 2:
            return np.nan, np.nan
        F, p = f_oneway(*groups)
        return float(F), float(p)

    @staticmethod
    def kruskal_wallis(data, group_col, value_col):
        """
        Kruskal–Wallis H-test (non-parametric) for comparing group distributions of a continuous variable.
        Returns (H_stat, p_value).
        """
        d = data[[group_col, value_col]].dropna()
        groups = [grp[value_col].values for _, grp in d.groupby(group_col)]
        if len(groups) < 2:
            return np.nan, np.nan
        H, p = kruskal(*groups)
        return float(H), float(p)

    # =================================
    # Continuous-bias association tests
    # =================================
    @staticmethod
    def correlation_with_continuous_bias(data, bias_col, target_col, method="spearman"):
        """
        Correlation between a continuous bias factor (e.g., Age) and the target.
        For binary targets, this becomes a rank/linear association test.
        Returns (coefficient, p_value).
        method in {"pearson","spearman","kendall"}.
        """
        d = data[[bias_col, target_col]].dropna()
        x = pd.to_numeric(d[bias_col], errors="coerce")
        y = pd.to_numeric(d[target_col], errors="coerce")
        mask = x.notna() & y.notna()
        x, y = x[mask], y[mask]
        if len(x) < 3:
            return np.nan, np.nan
        if method == "pearson":
            r, p = pearsonr(x, y)
        elif method == "kendall":
            r, p = kendalltau(x, y)
        else:
            r, p = spearmanr(x, y)
        return float(r), float(p)

    @staticmethod
    def point_biserial_with_continuous_bias(data, bias_col, binary_target_col):
        """
        Point-biserial correlation between a continuous bias factor and a binary target.
        Equivalent to Pearson between continuous X and {0,1} Y.
        Returns (r_pb, p_value).
        """
        d = data[[bias_col, binary_target_col]].dropna()
        x = pd.to_numeric(d[bias_col], errors="coerce")
        y = pd.to_numeric(d[binary_target_col], errors="coerce")
        mask = x.notna() & y.isin([0, 1])
        x, y = x[mask], y[mask]
        if len(x) < 3:
            return np.nan, np.nan
        r, p = pointbiserialr(y, x)  # order (y, x) or (x, y) does not change r
        return float(r), float(p)

    # ======================================
    # Multi-class outcome distribution tools
    # ======================================
    @staticmethod
    def multiclass_distribution_by_group(data, group_col, target_col):
        """
        Returns a DataFrame where rows=groups and columns=classes of target_col,
        each cell = P(Y=class | group) (row-normalized).
        """
        d = data[[group_col, target_col]].dropna()
        classes = d[target_col].unique().tolist()
        tbl = (
            d.groupby(group_col)[target_col]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
            .reindex(columns=classes, fill_value=0.0)
        )
        return tbl

    @staticmethod
    def js_divergence_between_groups_for_multiclass(data, group_col, target_col):
        """
        Returns a symmetric matrix of JSD between group-specific multiclass distributions.
        Good for PAM50 or any multi-class label.
        """
        dist_tbl = PretrainingBiasMetrics.multiclass_distribution_by_group(data, group_col, target_col)
        groups = dist_tbl.index.tolist()
        mat = pd.DataFrame(index=groups, columns=groups, dtype=float)
        for i, gi in enumerate(groups):
            pi = dist_tbl.loc[gi].values
            for j, gj in enumerate(groups):
                pj = dist_tbl.loc[gj].values
                mat.loc[gi, gj] = jensenshannon(pi, pj) ** 2
        return mat


    # ----------------------------
    # Visualization Utilities
    # ----------------------------
    @staticmethod
    def plot_class_distribution(data, column, title="Class Distribution"):
        """
        Plot class distribution as a bar chart.
        """
        counts = data[column].value_counts()
        counts.plot(kind="bar", title=title)
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.show()

    @staticmethod
    def plot_age_distribution(data, age_column):
        """
        Plot age distribution as a histogram.
        """
        plt.hist(data[age_column].dropna(), bins=20, alpha=0.7, color="blue")
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.show()

class PosttrainingBiasMetrics:
    """
    A class for post-training bias detection metrics, measuring model bias.
    """
    # ----------------------------
    # Prediction-Based Bias Metrics
    # ----------------------------
    @staticmethod
    def difference_in_positive_proportions(data, subgroup_column, pred_column):
        """Compute Difference in Positive Proportions in Predicted Labels (DPPL)."""
        positive_rates = (
            data[data[pred_column] == 1]
            .groupby(subgroup_column)
            .size()
            .div(data.groupby(subgroup_column).size())
        )
        return positive_rates.max() - positive_rates.min()

    @staticmethod
    def disparate_impact_predictions(data, subgroup_column, pred_column):
        """Compute Disparate Impact (DI) in predicted labels."""
        positive_rate_by_group = (
            data[data[pred_column] == 1]
            .groupby(subgroup_column)
            .size()
            .div(data.groupby(subgroup_column).size())
        )
        return positive_rate_by_group.min() / positive_rate_by_group.max()

    @staticmethod
    def conditional_demographic_disparity_predictions(data, subgroup_column, pred_column):
        """Compute Conditional Demographic Disparity of Predicted Labels (CDDPL)."""
        subgroup_means = data.groupby(subgroup_column)[pred_column].mean()
        return subgroup_means.max() - subgroup_means.min()

    @staticmethod
    def recall_difference(data, subgroup_column, true_column, pred_column):
        """Compute Recall Difference (RD) between subgroups."""
        recall_rates = (
            data.groupby(subgroup_column).apply(
                lambda g: recall_score(g[true_column], g[pred_column], zero_division=0)
            )
        )
        return recall_rates.max() - recall_rates.min()

    @staticmethod
    def accuracy_difference(data, subgroup_column, true_column, pred_column):
        """Compute Accuracy Difference (AD) between subgroups."""
        accuracy_rates = (
            data.groupby(subgroup_column).apply(
                lambda g: accuracy_score(g[true_column], g[pred_column])
            )
        )
        return accuracy_rates.max() - accuracy_rates.min()

    # ----------------------------
    # Fairness Metrics Using ROC Curves
    # ----------------------------
    @staticmethod
    def auc_difference(data, subgroup_column, true_column, pred_probs):
        """Compute AUC Difference across subgroups."""
        aucs = (
            data.groupby(subgroup_column).apply(
                lambda g: roc_auc_score(g[true_column], g[pred_probs])
            )
        )
        return aucs.max() - aucs.min()

    @staticmethod
    def roc_curve_plot(data, true_column, pred_probs, subgroup_column):
        """Plot ROC Curves for different subgroups."""
        plt.figure(figsize=(8, 6))
        for group in data[subgroup_column].unique():
            subgroup_data = data[data[subgroup_column] == group]
            fpr, tpr, _ = roc_curve(subgroup_data[true_column], subgroup_data[pred_probs])
            plt.plot(fpr, tpr, label=f"{group} (AUC = {roc_auc_score(subgroup_data[true_column], subgroup_data[pred_probs]):.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Different Subgroups")
        plt.legend()
        plt.show()

    # ----------------------------
    # Advanced Bias Metrics
    # ----------------------------
    @staticmethod
    def counterfactual_difference(original_data, counterfactual_data, pred_column):
        """Compute Counterfactual Difference (CDn)."""
        return (original_data[pred_column] != counterfactual_data[pred_column]).mean()

    @staticmethod
    def fliptest_difference(original_preds, perturbed_preds):
        """Compute FlipTest (FTn), which measures prediction stability."""
        return (original_preds != perturbed_preds).mean()


class BiasComparisonMetrics:
    """
    A class to compare pre-training and post-training biases.
    """

    @staticmethod
    def compute_comparison(pre_train_metrics, post_train_metrics):
        """Compute bias shifts between pre-training and post-training."""
        results = {}
        for metric in pre_train_metrics.keys():
            if metric in post_train_metrics:
                results[f"Bias Shift ({metric})"] = BiasComparisonMetrics.bias_shift(pre_train_metrics[metric], post_train_metrics[metric])
                results[f"Fairness Improvement (%) ({metric})"] = BiasComparisonMetrics.fairness_improvement(pre_train_metrics[metric], post_train_metrics[metric])
        return results

    @staticmethod
    def bias_shift(pre_training_bias, post_training_bias):
        """Compute the numerical difference between pre-training and post-training bias metrics."""
        return post_training_bias - pre_training_bias

    @staticmethod
    def fairness_improvement(pre_training_bias, post_training_bias):
        """Calculate the percentage reduction (or increase) in bias values."""
        if pre_training_bias == 0:
            return "No initial bias" if post_training_bias == 0 else "Bias introduced"
        reduction = ((pre_training_bias - post_training_bias) / abs(pre_training_bias)) * 100
        return f"{reduction:.2f}% {'Improvement' if reduction > 0 else 'Worsening'}"

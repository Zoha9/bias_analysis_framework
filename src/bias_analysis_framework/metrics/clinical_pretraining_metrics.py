from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bias_analysis_framework.metrics.bias_metrics import PretrainingBiasMetrics as BM


TISSUE_BIAS_NAME = "Tissue Source Site"
TISSUE_SITE_ALIASES = ["Tissue Source Site", "Source Site"]
RACE_CANONICAL = "Race"
RACE_ALIASES = ["Race", "Race Category"]

AGE_CONTINUOUS_CANDIDATES = ["Age", "Diagnosis Age", "Age_Onset_Continuous"]

MIN_TISSUE_COUNT = 20

GLOBAL_LABEL_MAP = {
    "MUT": 1,
    "MUTATED": 1,
    "POS": 1,
    "POSITIVE": 1,
    "MSI-H": 1,
    "MSI_HIGH": 1,
    "MSI_H": 1,
    "MSI": 1,
    "YES": 1,
    "Positive": 1,
    "positive": 1,
    "WT": 0,
    "WILDTYPE": 0,
    "NEG": 0,
    "NEGATIVE": 0,
    "MSS": 0,
    "NO": 0,
    "Negative": 0,
    "negative": 0,
}


SUMMARY_MARKERS = (
    "Summary Metrics",
    "Multiclass Summary",
    "Age (continuous) — Summary",
)


SECTION_RE = re.compile(
    r"^(?P<target>.+?) "
    r"\[(?P<classblock>Class=(?P<classname>.+?)|MULTICLASS)\]"
    r"\s+—\s+by\s+(?P<factor>.+?)\s+—\s+(?P<section>.+)$"
)


def compute_clinical_pretraining_metrics(
    clinical_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Compute Step 1 pretraining bias metric tables for one configured dataset.

    This is the config-driven replacement for the old hardcoded
    pretraining_metrics.py workflow.

    It preserves the old logic:
    - binary biomarker fairness metrics
    - multiclass one-vs-rest metrics
    - multiclass distribution metrics
    - tissue-source-site metrics
    - continuous-age association metrics
    - sectioned output tables
    """
    cohort = config.get("cohort", {})
    organ = str(cohort.get("organ", "UNKNOWN"))
    dataset = str(cohort.get("dataset", "DATASET"))

    df = clinical_df.copy()
    df = standardize_column_aliases(df)

    target_map = get_targets_from_config(config, df)
    bias_factors = get_bias_factors_from_config(config, df)

    if not target_map:
        return {
            "No configured targets": pd.DataFrame(
                [{"Info": "No configured target columns were found in the clinical table."}]
            )
        }

    outputs: dict[str, pd.DataFrame] = OrderedDict()

    age_col = choose_continuous_age_column(df)
    df_tissue_filtered = filter_tissue_sites(df)

    for biomarker, classes in target_map.items():
        if biomarker not in df.columns:
            continue

        is_binary = len({str(c).casefold() for c in classes}) == 2

        df_local = df.copy()
        df_tissue_local = df_tissue_filtered.copy()

        if is_binary:
            y_col = f"y__{biomarker}"
            df_local[y_col] = map_to_binary(df_local[biomarker], classes=classes)
            df_tissue_local[y_col] = map_to_binary(df_tissue_local[biomarker], classes=classes)
            class_targets = {"ALL": y_col}
        else:
            class_targets = {}
            for class_label in classes:
                y_col = f"y__{biomarker}__{class_label}"
                df_local[y_col] = map_to_binary(df_local[biomarker], positive_label=class_label)
                df_tissue_local[y_col] = map_to_binary(
                    df_tissue_local[biomarker],
                    positive_label=class_label,
                )
                class_targets[str(class_label)] = y_col
        
        for factor_name, factor_column in bias_factors:
            if factor_column not in df_local.columns:
                continue

            if is_binary:
                title_prefix = (
                    f"{biomarker} [Class=ALL] — by {factor_name}"
                )
                tables = build_binary_group_metrics(
                    df_local,
                    factor_column,
                    class_targets["ALL"],
                )
                add_section_tables(
                    outputs,
                    title_prefix,
                    tables,
                )
            else:
                for class_label, y_col in class_targets.items():
                    title_prefix = (
                        f"{biomarker} [Class={class_label}] "
                        f"— by {factor_name}"
                    )
                    tables = build_binary_group_metrics(
                        df_local,
                        factor_column,
                        y_col,
                    )
                    add_section_tables(
                        outputs,
                        title_prefix,
                        tables,
                    )

                title_prefix = (
                    f"{biomarker} [MULTICLASS] "
                    f"— by {factor_name}"
                )
                tables = build_multiclass_group_metrics(
                    df_local,
                    factor_column,
                    biomarker,
                )
                add_section_tables(
                    outputs,
                    title_prefix,
                    tables,
                )

        if TISSUE_BIAS_NAME in df_tissue_local.columns:
            if is_binary:
                title_prefix = f"{biomarker} [Class=ALL] — by {TISSUE_BIAS_NAME}"
                tables = build_binary_group_metrics(
                    df_tissue_local,
                    TISSUE_BIAS_NAME,
                    class_targets["ALL"],
                )
                add_section_tables(outputs, title_prefix, tables)
            else:
                for class_label, y_col in class_targets.items():
                    title_prefix = f"{biomarker} [Class={class_label}] — by {TISSUE_BIAS_NAME}"
                    tables = build_binary_group_metrics(df_tissue_local, TISSUE_BIAS_NAME, y_col)
                    add_section_tables(outputs, title_prefix, tables)

                title_prefix = f"{biomarker} [MULTICLASS] — by {TISSUE_BIAS_NAME}"
                tables = build_multiclass_group_metrics(
                    df_tissue_local,
                    TISSUE_BIAS_NAME,
                    biomarker,
                )
                add_section_tables(outputs, title_prefix, tables)

        if age_col is not None:
            if is_binary:
                title_prefix = f"{biomarker} [Class=ALL] — by Age (continuous)"
                tables = build_age_continuous_metrics(
                    df_local,
                    age_col,
                    class_targets["ALL"],
                    label_is_binary=True,
                )
                add_section_tables(outputs, title_prefix, tables)
            else:
                for class_label, y_col in class_targets.items():
                    title_prefix = f"{biomarker} [Class={class_label}] — by Age (continuous)"
                    tables = build_age_continuous_metrics(
                        df_local,
                        age_col,
                        y_col,
                        label_is_binary=True,
                    )
                    add_section_tables(outputs, title_prefix, tables)

                dmc = df_local[[age_col, biomarker]].dropna()
                try:
                    f_stat, p_value = BM.anova_f_oneway(dmc, biomarker, age_col)
                except Exception:
                    f_stat, p_value = np.nan, np.nan

                try:
                    h_stat, kruskal_p = BM.kruskal_wallis(dmc, biomarker, age_col)
                except Exception:
                    h_stat, kruskal_p = np.nan, np.nan

                outputs[
                    f"{biomarker} [MULTICLASS] — by Age (continuous) — Group Tests"
                ] = pd.DataFrame(
                    [
                        ("One-way ANOVA F (Label groups -> Age)", f_stat),
                        ("ANOVA p-value", p_value),
                        ("Kruskal–Wallis H (Label groups -> Age)", h_stat),
                        ("Kruskal–Wallis p-value", kruskal_p),
                    ],
                    columns=["Metric", "Value"],
                )

    if not outputs:
        outputs["No metric outputs"] = pd.DataFrame(
            [
                {
                    "Info": (
                        f"No pretraining metrics were generated for organ={organ}, "
                        f"dataset={dataset}. Check configured targets and bias factors."
                    )
                }
            ]
        )

    return outputs


def get_targets_from_config(
    config: dict[str, Any],
    clinical_df: pd.DataFrame,
) -> OrderedDict[str, list[str]]:
    """Return configured targets as {target_name: class_labels}.

    Supports either:
    - labels: ["WT", "MUT"]
    - positive_label / negative_label
    - automatic class discovery from the clinical table
    """
    output: OrderedDict[str, list[str]] = OrderedDict()

    for target in config.get("targets", []):
        name = target.get("name")
        if not name or name not in clinical_df.columns:
            continue

        if "labels" in target and target["labels"]:
            labels = [str(x) for x in target["labels"]]
        elif target.get("negative_label") is not None and target.get("positive_label") is not None:
            labels = [str(target["negative_label"]), str(target["positive_label"])]
        elif target.get("positive_label") is not None:
            raw_labels = clinical_df[name].dropna().astype(str).unique().tolist()
            labels = raw_labels
        else:
            labels = clinical_df[name].dropna().astype(str).unique().tolist()

        labels = [label for label in labels if str(label).strip() != ""]
        output[name] = labels

    return output


def get_bias_factors_from_config(
    config: dict[str, Any],
    clinical_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Return configured non-tissue bias factors.

    Each result contains:

    - display name used in reports and plot titles;
    - resolved dataframe column used for calculations.

    Tissue Source Site is handled separately because the original workflow
    applied a minimum-count filter before computing tissue-site metrics.
    """
    output: list[tuple[str, str]] = []

    for factor in config.get("bias_factors", []):
        display_name = str(
            factor.get("name")
            or factor.get("column")
            or ""
        ).strip()

        requested_column = str(
            factor.get("column")
            or factor.get("name")
            or ""
        ).strip()

        if not display_name or not requested_column:
            continue

        resolved_column = resolve_bias_factor_column(
            requested_column=requested_column,
            clinical_df=clinical_df,
        )

        if resolved_column is None:
            continue

        if (
            display_name == TISSUE_BIAS_NAME
            or resolved_column == TISSUE_BIAS_NAME
        ):
            continue

        pair = (
            display_name,
            resolved_column,
        )

        if pair not in output:
            output.append(pair)

    return output


def resolve_bias_factor_column(
    *,
    requested_column: str,
    clinical_df: pd.DataFrame,
) -> str | None:
    """Resolve a configured bias-factor column after alias normalization."""
    if requested_column in clinical_df.columns:
        return requested_column

    requested_casefold = requested_column.casefold()

    for existing_column in clinical_df.columns:
        if str(existing_column).strip().casefold() == requested_casefold:
            return str(existing_column)

    if (
        requested_column in RACE_ALIASES
        and RACE_CANONICAL in clinical_df.columns
    ):
        return RACE_CANONICAL

    if (
        requested_column in TISSUE_SITE_ALIASES
        and TISSUE_BIAS_NAME in clinical_df.columns
    ):
        return TISSUE_BIAS_NAME

    return None


def standardize_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize known clinical aliases used by old scripts."""
    df = df.copy()

    for candidate in RACE_ALIASES:
        if candidate in df.columns and candidate != RACE_CANONICAL:
            df.rename(columns={candidate: RACE_CANONICAL}, inplace=True)
            break

    for candidate in TISSUE_SITE_ALIASES:
        if candidate in df.columns and candidate != TISSUE_BIAS_NAME:
            df.rename(columns={candidate: TISSUE_BIAS_NAME}, inplace=True)
            break

    return df


def casefold_value(value: object) -> str | None:
    """Casefold a value, preserving missing values as None."""
    if pd.isna(value):
        return None
    return str(value).strip().casefold()


def map_to_binary(
    series: pd.Series,
    classes: list[str] | None = None,
    positive_label: str | None = None,
) -> pd.Series:
    """Map raw target labels to {0, 1}.

    For one-vs-rest multiclass metrics, pass positive_label.
    For configured binary classes, the second class is treated as positive
    unless GLOBAL_LABEL_MAP can recognize the labels.
    """
    if positive_label is not None:
        positive_cf = casefold_value(positive_label)
        return series.apply(
            lambda value: np.nan
            if pd.isna(value)
            else int(casefold_value(value) == positive_cf)
        )

    global_map = {key.casefold(): value for key, value in GLOBAL_LABEL_MAP.items()}

    mapped = series.map(
        lambda value: np.nan
        if pd.isna(value)
        else global_map.get(str(value).strip().casefold(), np.nan)
    )

    if mapped.notna().any():
        return mapped

    if classes and len(classes) == 2:
        negative_label, positive_label = classes[0], classes[1]
        negative_cf = casefold_value(negative_label)
        positive_cf = casefold_value(positive_label)

        return series.apply(
            lambda value: np.nan
            if pd.isna(value)
            else (
                1
                if casefold_value(value) == positive_cf
                else 0
                if casefold_value(value) == negative_cf
                else np.nan
            )
        )

    return mapped


def choose_continuous_age_column(df: pd.DataFrame) -> str | None:
    """Choose the first available continuous age column."""
    for column in AGE_CONTINUOUS_CANDIDATES:
        if column in df.columns:
            return column
    return None


def filter_tissue_sites(df: pd.DataFrame) -> pd.DataFrame:
    """Filter tissue-source-site groups below MIN_TISSUE_COUNT, preserving old behavior."""
    if TISSUE_BIAS_NAME not in df.columns:
        return df

    tissue = df[TISSUE_BIAS_NAME].astype(str).str.strip().replace({"<NA>": pd.NA})
    tissue_cf = tissue.map(lambda value: casefold_value(value) if pd.notna(value) else value)
    counts = tissue_cf.value_counts(dropna=True)
    kept = set(counts[counts >= MIN_TISSUE_COUNT].index.tolist())

    if not kept:
        return df

    return df[tissue_cf.isin(list(kept))].copy()


def positive_rate_by_subgroup(
    df: pd.DataFrame,
    subgroup_col: str,
    y_col: str,
) -> pd.Series:
    """Compute positive rate by subgroup."""
    return df.groupby(subgroup_col)[y_col].mean()


def build_binary_group_metrics(
    df: pd.DataFrame,
    subgroup_col: str,
    y_col: str,
) -> dict[str, pd.DataFrame]:
    """Build binary-target metrics against one categorical subgroup."""
    outputs: dict[str, pd.DataFrame] = {}

    d = df[[subgroup_col, y_col]].dropna()

    if d.empty or d[y_col].nunique() < 2 or d[subgroup_col].nunique() < 2:
        outputs["Summary Metrics"] = pd.DataFrame(
            [{"Metric": "INSUFFICIENT_VARIATION", "Value": np.nan}]
        )
        return outputs

    counts = d.groupby(subgroup_col).size().rename("Count")
    positive_rates = positive_rate_by_subgroup(d, subgroup_col, y_col).rename("PositiveRate")

    metrics = []

    safe_metric(metrics, "Statistical Parity Difference", BM.statistical_parity_difference, d, subgroup_col, y_col)
    safe_metric(metrics, "Disparate Impact (min/max)", BM.disparate_impact, d, subgroup_col, y_col)
    safe_metric(metrics, "Conditional Demographic Disparity", BM.conditional_demographic_disparity, d, subgroup_col, y_col)
    safe_metric(metrics, "Difference in Positive Proportions", BM.difference_in_positive_proportions, d, subgroup_col, y_col)

    safe_metric(metrics, "Max Pairwise SPD (abs)", BM.max_pairwise_spd, d, subgroup_col, y_col)
    safe_metric(metrics, "DI Spread (max - min DI)", BM.max_pairwise_di_spread, d, subgroup_col, y_col)
    safe_metric(metrics, "Coefficient of Variation (pos rates)", BM.coefficient_of_variation_across_groups, d, subgroup_col, y_col)
    safe_metric(metrics, "Range of Positive Rates", BM.range_of_positive_rates, d, subgroup_col, y_col)
    safe_metric(metrics, "Entropy of Positive Rates", BM.entropy_of_positive_rates, d, subgroup_col, y_col)
    safe_metric(metrics, "Gini across Groups (pos rates)", BM.gini_across_groups_of_positive_rates, d, subgroup_col, y_col)

    safe_metric(metrics, "Class Imbalance (global)", BM.class_imbalance, d, y_col)
    safe_metric(metrics, "Gini Coefficient (global)", BM.gini_coefficient, d, y_col)

    try:
        chi2, p_value, dof, _ = BM.chi_square_independence(
            d.assign(**{y_col: d[y_col].astype(int)}),
            subgroup_col,
            y_col,
        )
    except Exception:
        chi2, p_value, dof = np.nan, np.nan, np.nan

    metrics.append(("Chi-square (group x label)", chi2))
    metrics.append(("Chi-square p-value", p_value))
    metrics.append(("Chi-square dof", dof))

    try:
        cramers_v = BM.cramers_v(
            d.assign(**{y_col: d[y_col].astype(int)}),
            subgroup_col,
            y_col,
        )
    except Exception:
        cramers_v = np.nan

    metrics.append(("Cramér's V (group x label)", cramers_v))

    try:
        max_group = positive_rates.idxmax()
        min_group = positive_rates.idxmin()
        p_dist = (
            d[d[subgroup_col] == max_group][y_col]
            .value_counts(normalize=True)
            .reindex([0, 1], fill_value=0)
            .values
        )
        q_dist = (
            d[d[subgroup_col] == min_group][y_col]
            .value_counts(normalize=True)
            .reindex([0, 1], fill_value=0)
            .values
        )

        try:
            metrics.append(("KL Divergence (max_vs_min)", BM.kl_divergence(p_dist, q_dist)))
        except Exception:
            metrics.append(("KL Divergence (max_vs_min)", np.nan))

        try:
            metrics.append(("JS Divergence (max_vs_min)", BM.js_divergence(p_dist, q_dist)))
        except Exception:
            metrics.append(("JS Divergence (max_vs_min)", np.nan))

        try:
            metrics.append(("Total Variation (max_vs_min)", BM.total_variation_distance(p_dist, q_dist)))
        except Exception:
            metrics.append(("Total Variation (max_vs_min)", np.nan))

        try:
            metrics.append(("L2 Distance (max_vs_min)", BM.lp_norm(p_dist, q_dist, p_norm=2)))
        except Exception:
            metrics.append(("L2 Distance (max_vs_min)", np.nan))
    except Exception:
        pass

    summary_table = pd.DataFrame(metrics, columns=["Metric", "Value"])
    subgroup_summary = (
        pd.concat([counts, positive_rates], axis=1)
        .reset_index()
        .rename(columns={subgroup_col: "Subgroup"})
    )

    outputs["Summary Metrics"] = pd.concat(
        [
            summary_table,
            pd.DataFrame([{"Metric": "—", "Value": "Subgroup summaries below"}]),
            subgroup_summary.rename(columns={"Count": "Value"})
            .assign(Metric=lambda _: "Subgroup Count")
            .loc[:, ["Metric", "Subgroup", "Value"]],
            subgroup_summary.rename(columns={"PositiveRate": "Value"})
            .assign(Metric=lambda _: "Subgroup PositiveRate")
            .loc[:, ["Metric", "Subgroup", "Value"]],
        ],
        ignore_index=True,
    )

    try:
        outputs["Pairwise SPD Matrix"] = BM.pairwise_spd_matrix(d, subgroup_col, y_col)
    except Exception:
        pass

    try:
        outputs["Pairwise DI Matrix"] = BM.pairwise_di_matrix(d, subgroup_col, y_col)
    except Exception:
        pass

    try:
        outputs["JSD Across Groups (to centroid)"] = pd.DataFrame(
            {
                "Metric": ["JSD_mean_to_centroid"],
                "Value": [
                    BM.js_divergence_across_groups(
                        d.assign(**{y_col: d[y_col].astype(int)}),
                        subgroup_col,
                        y_col,
                    )
                ],
            }
        )
    except Exception:
        pass

    try:
        outputs["TVD Across Groups (to centroid)"] = pd.DataFrame(
            {
                "Metric": ["TVD_mean_to_centroid"],
                "Value": [
                    BM.total_variation_across_groups(
                        d.assign(**{y_col: d[y_col].astype(int)}),
                        subgroup_col,
                        y_col,
                    )
                ],
            }
        )
    except Exception:
        pass

    return outputs


def build_multiclass_group_metrics(
    df: pd.DataFrame,
    subgroup_col: str,
    target_col: str,
) -> dict[str, pd.DataFrame]:
    """Build multiclass-target distribution and association metrics."""
    outputs: dict[str, pd.DataFrame] = {}

    d = df[[subgroup_col, target_col]].dropna()

    if d.empty or d[subgroup_col].nunique() < 2 or d[target_col].nunique() < 2:
        outputs["Multiclass Summary"] = pd.DataFrame(
            [{"Metric": "INSUFFICIENT_VARIATION", "Value": np.nan}]
        )
        return outputs

    try:
        outputs["Class Distribution by Subgroup"] = BM.multiclass_distribution_by_group(
            d,
            subgroup_col,
            target_col,
        )
    except Exception:
        pass

    try:
        chi2, p_value, dof, _ = BM.chi_square_independence(d, subgroup_col, target_col)
    except Exception:
        chi2, p_value, dof = np.nan, np.nan, np.nan

    try:
        cramers_v = BM.cramers_v(d, subgroup_col, target_col)
    except Exception:
        cramers_v = np.nan

    try:
        js_centroid = BM.js_divergence_across_groups(d, subgroup_col, target_col)
    except Exception:
        js_centroid = np.nan

    try:
        tv_centroid = BM.total_variation_across_groups(d, subgroup_col, target_col)
    except Exception:
        tv_centroid = np.nan

    outputs["Multiclass Summary"] = pd.DataFrame(
        [
            ("Chi-square (group x label)", chi2),
            ("Chi-square p-value", p_value),
            ("Chi-square dof", dof),
            ("Cramér's V (group x label)", cramers_v),
            ("JSD mean to centroid", js_centroid),
            ("TVD mean to centroid", tv_centroid),
        ],
        columns=["Metric", "Value"],
    )

    try:
        outputs["JSD Matrix (group vs group)"] = BM.js_divergence_between_groups_for_multiclass(
            d,
            subgroup_col,
            target_col,
        )
    except Exception:
        pass

    return outputs


def build_age_continuous_metrics(
    df: pd.DataFrame,
    age_col: str,
    y_col: str,
    label_is_binary: bool,
) -> dict[str, pd.DataFrame]:
    """Build continuous-age metrics against configured label."""
    outputs: dict[str, pd.DataFrame] = {}

    d = df[[age_col, y_col]].dropna()

    if d.empty:
        outputs["Age (continuous) — Summary"] = pd.DataFrame(
            [{"Metric": "INSUFFICIENT_VARIATION", "Value": np.nan}]
        )
        return outputs

    rows = []

    for method, label in [
        ("pearson", "Pearson r (Age, Y)"),
        ("spearman", "Spearman rho (Age, Y)"),
        ("kendall", "Kendall tau (Age, Y)"),
    ]:
        try:
            r_value, p_value = BM.correlation_with_continuous_bias(
                d,
                age_col,
                y_col,
                method=method,
            )
        except Exception:
            r_value, p_value = np.nan, np.nan

        rows.append((label, r_value))
        rows.append((f"{label.split()[0]} p-value", p_value))

    if label_is_binary:
        try:
            rpb, p_value = BM.point_biserial_with_continuous_bias(d, age_col, y_col)
        except Exception:
            rpb, p_value = np.nan, np.nan

        rows.append(("Point-biserial r (Age, Ybin)", rpb))
        rows.append(("Point-biserial p-value", p_value))

        try:
            f_stat, p_value = BM.anova_f_oneway(
                d.assign(**{y_col: pd.to_numeric(d[y_col])}),
                y_col,
                age_col,
            )
        except Exception:
            f_stat, p_value = np.nan, np.nan

        rows.append(("One-way ANOVA F (Y groups -> Age)", f_stat))
        rows.append(("ANOVA p-value", p_value))

        try:
            h_stat, p_value = BM.kruskal_wallis(
                d.assign(**{y_col: pd.to_numeric(d[y_col])}),
                y_col,
                age_col,
            )
        except Exception:
            h_stat, p_value = np.nan, np.nan

        rows.append(("Kruskal–Wallis H (Y groups -> Age)", h_stat))
        rows.append(("Kruskal–Wallis p-value", p_value))

    outputs["Age (continuous) — Summary"] = pd.DataFrame(rows, columns=["Metric", "Value"])
    return outputs


def safe_metric(
    rows: list[tuple[str, Any]],
    name: str,
    function,
    *args,
    **kwargs,
) -> None:
    """Append a metric value, using NaN if calculation fails."""
    try:
        rows.append((name, function(*args, **kwargs)))
    except Exception:
        rows.append((name, np.nan))


def add_section_tables(
    output: dict[str, pd.DataFrame],
    title_prefix: str,
    tables: dict[str, pd.DataFrame],
) -> None:
    """Add named tables to the output section map."""
    for section_name, table in tables.items():
        output[f"{title_prefix} — {section_name}"] = table


def write_sectioned_pretraining_metrics_excel(
    section_map: dict[str, pd.DataFrame],
    output_path: str | Path,
    dataset_name: str,
) -> Path:
    """Write sectioned metric tables to one Excel workbook.

    Preserves the old script's layout:
    - one sheet per dataset
    - section title row
    - table below each title
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sheet_name = safe_excel_sheet_name(dataset_name)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        startrow = 0

        if not section_map:
            pd.DataFrame({"Info": [f"No biomarkers available for {dataset_name}"]}).to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )
            return output_path

        _ = (
            writer.book.create_sheet(sheet_name)
            if sheet_name not in writer.book.sheetnames
            else None
        )
        worksheet = writer.book[sheet_name]

        for section_title in sorted(section_map.keys()):
            table = section_map[section_title]
            worksheet.cell(row=startrow + 1, column=1, value=f"{dataset_name} — {section_title}")

            if isinstance(table, pd.DataFrame):
                write_index = not table.index.equals(pd.RangeIndex(len(table)))
                table.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=startrow + 1,
                    index=write_index,
                )
                startrow += len(table) + 4
            else:
                pd.DataFrame({"Value": [str(table)]}).to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=startrow + 1,
                    index=False,
                )
                startrow += 6

    return output_path


def write_pretraining_summary_excel(
    section_map: dict[str, pd.DataFrame],
    output_path: str | Path,
    config: dict[str, Any],
) -> Path:
    """Write compact long/wide summary workbook."""
    long_df, wide_df = flatten_summary_metrics_only(section_map, config)

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_order = [
        "Statistical Parity Difference",
        "Disparate Impact (min/max)",
        "Conditional Demographic Disparity",
        "Difference in Positive Proportions",
        "Max Pairwise SPD (abs)",
        "DI Spread (max - min DI)",
        "Coefficient of Variation (pos rates)",
        "Range of Positive Rates",
        "Entropy of Positive Rates",
        "Gini across Groups (pos rates)",
        "Class Imbalance (global)",
        "Gini Coefficient (global)",
        "Chi-square (group x label)",
        "Chi-square p-value",
        "Chi-square dof",
        "Cramér's V (group x label)",
        "KL Divergence (max_vs_min)",
        "JS Divergence (max_vs_min)",
        "Total Variation (max_vs_min)",
        "L2 Distance (max_vs_min)",
    ]

    if not long_df.empty:
        long_df["MetricOrder"] = long_df["Metric"].apply(
            lambda metric: metric_order.index(metric)
            if metric in metric_order
            else len(metric_order) + 1
        )
        long_df = (
            long_df.sort_values(
                ["Organ", "Dataset", "Target", "Class", "Factor", "Section", "MetricOrder", "Metric"],
                kind="mergesort",
            )
            .drop(columns="MetricOrder")
            .reset_index(drop=True)
        )

    if not wide_df.empty:
        id_cols = ["Organ", "Dataset", "Dataset_Display", "Target", "Class", "TargetFull", "Factor"]
        metric_cols = [column for column in wide_df.columns if column not in id_cols]
        ordered_metrics = [metric for metric in metric_order if metric in metric_cols]
        remaining_metrics = [metric for metric in metric_cols if metric not in metric_order]
        wide_df = wide_df[id_cols + ordered_metrics + remaining_metrics]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="summary_long", index=False)
        wide_df.to_excel(writer, sheet_name="summary_wide", index=False)

    return output_path


def flatten_summary_metrics_only(
    section_map: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten only summary-like metric tables into long and wide summary frames."""
    cohort = config.get("cohort", {})
    organ = str(cohort.get("organ", ""))
    dataset = str(cohort.get("dataset", ""))
    dataset_display = dataset.split("_", 1)[1] if "_" in dataset else dataset

    rows = []

    for section_title, section_df in section_map.items():
        if not isinstance(section_df, pd.DataFrame):
            continue

        parsed = parse_section_title(section_title)
        section = parsed.get("Section", "")

        if not is_summary_like(section):
            continue

        if "Metric" not in section_df.columns or "Value" not in section_df.columns:
            continue

        usable = section_df.copy()
        usable = usable[usable["Metric"].astype(str).str.strip() != "—"]
        
        if "Subgroup" in usable.columns:
            # The full metric workbook keeps subgroup count/rate rows.
            # The compact summary workbook should keep metric rows only.
            usable = usable[usable["Subgroup"].isna()]

        target = parsed.get("Target", "")
        class_name = parsed.get("Class", "") or "ALL"
        factor = parsed.get("Factor", "")
        target_full = f"{target} [Class={class_name}]"

        for _, row in usable.iterrows():
            rows.append(
                {
                    "Organ": organ,
                    "Dataset": dataset,
                    "Dataset_Display": dataset_display,
                    "Target": target,
                    "Class": class_name,
                    "TargetFull": target_full,
                    "Factor": factor,
                    "Section": section,
                    "Metric": str(row["Metric"]),
                    "Value": pd.to_numeric(row["Value"], errors="coerce"),
                }
            )

    long_df = pd.DataFrame(
        rows,
        columns=[
            "Organ",
            "Dataset",
            "Dataset_Display",
            "Target",
            "Class",
            "TargetFull",
            "Factor",
            "Section",
            "Metric",
            "Value",
        ],
    )

    if long_df.empty:
        wide_df = pd.DataFrame(
            columns=[
                "Organ",
                "Dataset",
                "Dataset_Display",
                "Target",
                "Class",
                "TargetFull",
                "Factor",
            ]
        )
        return long_df, wide_df

    long_df = (
        long_df.sort_values(
            ["Organ", "Dataset", "Target", "Class", "Factor", "Section", "Metric"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    wide_df = (
        long_df.pivot_table(
            index=["Organ", "Dataset", "Dataset_Display", "Target", "Class", "TargetFull", "Factor"],
            columns="Metric",
            values="Value",
            aggfunc="first",
        )
        .reset_index()
    )

    id_cols = ["Organ", "Dataset", "Dataset_Display", "Target", "Class", "TargetFull", "Factor"]
    metric_cols = [column for column in wide_df.columns if column not in id_cols]
    wide_df = wide_df[id_cols + metric_cols]

    return long_df, wide_df


def parse_section_title(section_title: str) -> dict[str, str]:
    """Extract target, class, factor, and section from a metric section title."""
    match = SECTION_RE.match(section_title)

    if match:
        class_name = match.group("classname") if match.group("classname") else "MULTICLASS"
        return {
            "Target": match.group("target").strip(),
            "Class": class_name.strip(),
            "Factor": match.group("factor").strip(),
            "Section": match.group("section").strip(),
        }

    parts = [part.strip() for part in section_title.split("—")]
    target = ""
    class_name = ""
    factor = ""
    section = ""

    try:
        head = parts[0]
        target = head.split("[")[0].strip()

        if "[" in head and "]" in head:
            inside = head[head.find("[") + 1 : head.rfind("]")]
            class_name = inside.replace("Class=", "").strip()

        if len(parts) > 1:
            by_part = parts[1]
            factor = by_part[3:].strip() if by_part.lower().startswith("by ") else by_part

        if len(parts) > 2:
            section = parts[2]
    except Exception:
        pass

    return {
        "Target": target,
        "Class": class_name,
        "Factor": factor,
        "Section": section,
    }


def is_summary_like(section: str) -> bool:
    """Return whether a section name should be included in compact summary output."""
    section = section.strip()
    return section in SUMMARY_MARKERS or "Summary" in section


def safe_excel_sheet_name(value: str) -> str:
    """Return Excel-compatible sheet name."""
    sheet_name = str(value)
    for char in [":", "\\", "/", "?", "*", "[", "]"]:
        sheet_name = sheet_name.replace(char, "_")
    return sheet_name[:31]
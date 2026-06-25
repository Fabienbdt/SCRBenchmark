"""
Statistical tests and significance grouping for clustering comparison.
Provides both parametric (ANOVA + Tukey HSD) and non-parametric
(Kruskal-Wallis + Dunn) approaches with Compact Letter Display (CLD).
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def kruskal_wallis_test(data_groups: Dict[str, List[float]]) -> Tuple[float, float]:
    """
    Perform Kruskal-Wallis H-test for independent samples.

    Args:
        data_groups: Dictionary mapping group names to their values

    Returns:
        Tuple of (H-statistic, p-value)
    """
    groups = list(data_groups.values())
    if len(groups) < 2:
        return 0.0, 1.0

    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return 0.0, 1.0

    try:
        statistic, pvalue = stats.kruskal(*groups)
        return float(statistic), float(pvalue)
    except Exception:
        return 0.0, 1.0


def anova_test(data_groups: Dict[str, List[float]]) -> Tuple[float, float]:
    """
    Perform one-way ANOVA test.

    Args:
        data_groups: Dictionary mapping group names to their values

    Returns:
        Tuple of (F-statistic, p-value)
    """
    groups = list(data_groups.values())
    if len(groups) < 2:
        return 0.0, 1.0

    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return 0.0, 1.0

    try:
        statistic, pvalue = stats.f_oneway(*groups)
        return float(statistic), float(pvalue)
    except Exception:
        return 0.0, 1.0


def dunn_posthoc_test(data_groups: Dict[str, List[float]],
                      p_adjust: str = 'bonferroni') -> Dict[Tuple[str, str], float]:
    """
    Perform Dunn's post-hoc test for pairwise comparisons after Kruskal-Wallis.
    Implementation based on the Dunn (1964) procedure.

    Args:
        data_groups: Dictionary mapping group names to their values
        p_adjust: P-value adjustment method ('bonferroni', 'holm', 'none')

    Returns:
        Dictionary mapping (group1, group2) tuples to adjusted p-values
    """
    group_names = list(data_groups.keys())
    n_groups = len(group_names)

    if n_groups < 2:
        return {}

    # Combine all data and compute ranks
    all_data = []
    group_labels = []
    for name in group_names:
        values = data_groups[name]
        all_data.extend(values)
        group_labels.extend([name] * len(values))

    all_data = np.array(all_data)
    n_total = len(all_data)

    if n_total == 0:
        return {}

    # Compute ranks (average ranks for ties)
    ranks = stats.rankdata(all_data, method='average')

    # Compute mean rank for each group
    mean_ranks = {}
    group_sizes = {}
    for name in group_names:
        mask = np.array([l == name for l in group_labels])
        group_sizes[name] = np.sum(mask)
        if group_sizes[name] > 0:
            mean_ranks[name] = np.mean(ranks[mask])
        else:
            mean_ranks[name] = 0

    # Compute tie correction
    _, tie_counts = np.unique(all_data, return_counts=True)
    tie_correction = 1 - np.sum(tie_counts**3 - tie_counts) / (n_total**3 - n_total)
    if tie_correction == 0:
        tie_correction = 1

    # Variance of rank sums
    variance = (n_total * (n_total + 1) / 12) * tie_correction

    # Compute pairwise z-statistics and p-values
    pvalues = {}
    n_comparisons = n_groups * (n_groups - 1) // 2

    for i, name1 in enumerate(group_names):
        for name2 in group_names[i+1:]:
            n1 = group_sizes[name1]
            n2 = group_sizes[name2]

            if n1 == 0 or n2 == 0:
                pvalues[(name1, name2)] = 1.0
                continue

            # Standard error
            se = np.sqrt(variance * (1/n1 + 1/n2))

            if se == 0:
                pvalues[(name1, name2)] = 1.0
                continue

            # Z-statistic
            z = (mean_ranks[name1] - mean_ranks[name2]) / se

            # Two-tailed p-value
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            pvalues[(name1, name2)] = p

    # Apply p-value adjustment
    if p_adjust == 'bonferroni' and n_comparisons > 0:
        for key in pvalues:
            pvalues[key] = min(1.0, pvalues[key] * n_comparisons)
    elif p_adjust == 'holm' and n_comparisons > 0:
        # Sort p-values
        sorted_pairs = sorted(pvalues.items(), key=lambda x: x[1])
        for i, (key, p) in enumerate(sorted_pairs):
            pvalues[key] = min(1.0, p * (n_comparisons - i))
        # Enforce monotonicity
        sorted_pairs = sorted(pvalues.items(), key=lambda x: x[1])
        max_p = 0
        for key, p in sorted_pairs:
            max_p = max(max_p, pvalues[key])
            pvalues[key] = max_p

    return pvalues


def tukey_hsd_test(data_groups: Dict[str, List[float]]) -> Dict[Tuple[str, str], float]:
    """
    Perform Tukey's HSD (Honestly Significant Difference) post-hoc test.

    Args:
        data_groups: Dictionary mapping group names to their values

    Returns:
        Dictionary mapping (group1, group2) tuples to p-values
    """
    group_names = list(data_groups.keys())
    n_groups = len(group_names)

    if n_groups < 2:
        return {}

    # Prepare data for scipy
    all_data = []
    group_labels = []
    for name in group_names:
        values = data_groups[name]
        all_data.extend(values)
        group_labels.extend([name] * len(values))

    if len(all_data) == 0:
        return {}

    try:
        # Use scipy's tukey_hsd
        groups = [np.array(data_groups[name]) for name in group_names if len(data_groups[name]) > 0]
        valid_names = [name for name in group_names if len(data_groups[name]) > 0]

        if len(groups) < 2:
            return {}

        result = stats.tukey_hsd(*groups)

        pvalues = {}
        for i, name1 in enumerate(valid_names):
            for j, name2 in enumerate(valid_names):
                if i < j:
                    pvalues[(name1, name2)] = float(result.pvalue[i, j])

        return pvalues
    except Exception:
        # Fallback: use pairwise t-tests with Bonferroni correction
        return _pairwise_ttest_bonferroni(data_groups)


def _pairwise_ttest_bonferroni(data_groups: Dict[str, List[float]]) -> Dict[Tuple[str, str], float]:
    """
    Fallback: pairwise t-tests with Bonferroni correction.
    """
    group_names = list(data_groups.keys())
    n_groups = len(group_names)
    n_comparisons = n_groups * (n_groups - 1) // 2

    pvalues = {}
    for i, name1 in enumerate(group_names):
        for name2 in group_names[i+1:]:
            g1 = data_groups[name1]
            g2 = data_groups[name2]

            if len(g1) < 2 or len(g2) < 2:
                pvalues[(name1, name2)] = 1.0
                continue

            try:
                _, p = stats.ttest_ind(g1, g2)
                pvalues[(name1, name2)] = min(1.0, p * n_comparisons)
            except Exception:
                pvalues[(name1, name2)] = 1.0

    return pvalues


def _pair_pvalue(
    pvalue_matrix: Dict[Tuple[str, str], float],
    name1: str,
    name2: str,
) -> float:
    """Return pairwise p-value regardless of tuple ordering."""
    return float(pvalue_matrix.get((name1, name2), pvalue_matrix.get((name2, name1), 1.0)))


def _absorb_letter_columns(columns: List[set]) -> List[set]:
    """Remove duplicate/subset columns and keep deterministic ordering."""
    unique_cols: List[set] = []
    seen = set()
    for col in columns:
        frozen = frozenset(col)
        if not frozen or frozen in seen:
            continue
        seen.add(frozen)
        unique_cols.append(set(col))

    # Absorption: if A is subset of B, A is redundant.
    keep_mask = [True] * len(unique_cols)
    for i, col_i in enumerate(unique_cols):
        if not keep_mask[i]:
            continue
        for j, col_j in enumerate(unique_cols):
            if i == j or not keep_mask[i]:
                continue
            if col_i <= col_j and col_i != col_j:
                keep_mask[i] = False
                break

    kept = [c for c, keep in zip(unique_cols, keep_mask) if keep]
    kept.sort(key=lambda s: (-len(s), tuple(sorted(str(x) for x in s))))
    return kept


def _letter_token(idx: int) -> str:
    """Excel-like sequence: a..z, aa..az, ba.."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    n = idx + 1
    token = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        token = letters[rem] + token
    return token


def generate_cld(
    pvalue_matrix: Dict[Tuple[str, str], float],
    group_names: List[str],
    alpha: float = 0.05,
) -> Dict[str, str]:
    """
    Generate Compact Letter Display (CLD) from pairwise p-values.

    Guarantees:
    - Significant pairs (p < alpha) do not share letters.
    - Non-significant pairs (p >= alpha) share at least one letter.
    """
    if not group_names:
        return {}
    if len(group_names) == 1:
        return {group_names[0]: "a"}

    groups = list(group_names)
    # Start with one letter shared by all groups.
    columns: List[set] = [set(groups)]

    sig_pairs = []
    non_sig_pairs = []
    for i, name1 in enumerate(groups):
        for name2 in groups[i + 1:]:
            p = _pair_pvalue(pvalue_matrix, name1, name2)
            if p < alpha:
                sig_pairs.append((name1, name2))
            else:
                non_sig_pairs.append((name1, name2))

    # Insert-absorb algorithm (Piepho-style):
    # split any column that contains a significant pair.
    for name1, name2 in sig_pairs:
        updated: List[set] = []
        for col in columns:
            if name1 in col and name2 in col:
                c1 = set(col)
                c2 = set(col)
                c1.discard(name1)
                c2.discard(name2)
                if c1:
                    updated.append(c1)
                if c2:
                    updated.append(c2)
            else:
                updated.append(set(col))
        columns = _absorb_letter_columns(updated)

    # Ensure every group gets at least one letter.
    covered = set().union(*columns) if columns else set()
    for name in groups:
        if name not in covered:
            columns.append({name})
    columns = _absorb_letter_columns(columns)

    # Ensure every non-significant pair shares at least one letter.
    for name1, name2 in non_sig_pairs:
        shares_letter = any(name1 in col and name2 in col for col in columns)
        if not shares_letter:
            columns.append({name1, name2})
            columns = _absorb_letter_columns(columns)

    # Build final mapping.
    result = {name: "" for name in groups}
    for idx, col in enumerate(columns):
        letter = _letter_token(idx)
        for name in groups:
            if name in col:
                result[name] += letter

    # Fallback safety (should not happen after coverage step).
    for name in groups:
        if not result[name]:
            result[name] = "a"

    return result


def compute_marker_genes(adata, 
                        labels, 
                        target_cluster, 
                        method='wilcoxon', 
                        n_genes=100) -> pd.DataFrame:
    """
    Compute marker genes for a specific cluster using differential expression analysis.
    
    Args:
        adata: AnnData object containing expression data (X should be normalized/log-transformed)
        labels: Array of cluster labels matching adata.shape[0]
        target_cluster: The cluster ID to find markers for (vs rest)
        method: Statistical test ('wilcoxon', 't-test', 't-test_overestim_var', 'logreg')
        n_genes: Number of top genes to return
        
    Returns:
        DataFrame with columns: Gene, Log2FC, P-val, Adj P-val, Score
    """
    import scanpy as sc
    import pandas as pd
    
    # Create a copy or view to avoid modifying the original significantly
    # We only need to add temporary labels
    adata_copy = adata.copy()
    
    # Ensure labels are string for scanpy
    adata_copy.obs['temp_clustering'] = labels.astype(str)
    target_cluster = str(target_cluster)
    
    # Check if target cluster exists
    if target_cluster not in adata_copy.obs['temp_clustering'].values:
        raise ValueError(f"Cluster {target_cluster} not found in labels.")
    
    # Run rank_genes_groups
    # reference='rest' compares target cluster vs all other cells
    try:
        sc.tl.rank_genes_groups(
            adata_copy, 
            groupby='temp_clustering', 
            groups=[target_cluster], 
            reference='rest', 
            method=method,
            n_genes=n_genes,
            use_raw=False 
        )
    except Exception as e:
        # Fallback for very small groups where wilcoxon might fail
        if method == 'wilcoxon':
            sc.tl.rank_genes_groups(
                adata_copy, 
                groupby='temp_clustering', 
                groups=[target_cluster], 
                reference='rest', 
                method='t-test',
                n_genes=n_genes,
                use_raw=False
            )
        else:
            raise e
            
    # Extract results
    # scanpy returns structured array, we want a clean DataFrame
    result = sc.get.rank_genes_groups_df(adata_copy, group=target_cluster)
    
    # Rename columns to be more user-friendly
    col_map = {
        'names': 'Gene',
        'logfoldchanges': 'Log2FC',
        'pvals': 'P-value',
        'pvals_adj': 'Adj P-value',
        'scores': 'Score'
    }
    result = result.rename(columns=col_map)
    
    # Filter columns
    cols = ['Gene', 'Log2FC', 'P-value', 'Adj P-value', 'Score']
    result = result[cols]
    
    # Add mean expression for context (optional but useful)
    # Calculate mean expression in target cluster
    mask = adata_copy.obs['temp_clustering'] == target_cluster
    
    # We need to map genes back to their indices
    gene_to_idx = {g: i for i, g in enumerate(adata_copy.var_names)}
    
    means = []
    for gene in result['Gene']:
        if gene in gene_to_idx:
            idx = gene_to_idx[gene]
            # Handle sparse matrix
            X_col = adata_copy.X[mask, idx]
            if hasattr(X_col, 'toarray'):
                X_col = X_col.toarray()
            mean_val = np.mean(X_col)
            means.append(mean_val)
        else:
            means.append(0.0)
            
    result['Mean Expr'] = means
    
    return result


def compute_significance_groups(data_by_algo: Dict[str, List[float]],
                                 method: str = 'nonparametric',
                                 alpha: float = 0.05) -> Tuple[Dict[str, str], float, str]:
    """
    Compute significance groups and CLD for algorithm comparison.

    Args:
        data_by_algo: Dictionary mapping algorithm names to their metric values
        method: 'nonparametric' (Kruskal-Wallis + Dunn) or 'parametric' (ANOVA + Tukey)
        alpha: Significance threshold

    Returns:
        Tuple of:
        - Dictionary mapping algorithm names to CLD letters
        - Global test p-value
        - Test name string
    """
    group_names = list(data_by_algo.keys())

    # Filter to finite numeric values only.
    valid_groups: Dict[str, List[float]] = {}
    for name, values in data_by_algo.items():
        clean_vals: List[float] = []
        for val in values:
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fval):
                clean_vals.append(fval)
        if clean_vals:
            valid_groups[name] = clean_vals

    if len(valid_groups) < 2:
        # Not enough groups for comparison
        return {name: 'a' for name in group_names}, 1.0, "N/A"

    if method == 'nonparametric':
        # Kruskal-Wallis + Dunn
        _, global_p = kruskal_wallis_test(valid_groups)
        test_name = "Kruskal-Wallis + Dunn"

        if global_p >= alpha:
            # No significant difference overall
            return {name: 'a' for name in group_names}, global_p, test_name

        pvalue_matrix = dunn_posthoc_test(valid_groups, p_adjust='bonferroni')

    else:  # parametric
        # ANOVA + Tukey HSD
        _, global_p = anova_test(valid_groups)
        test_name = "ANOVA + Tukey HSD"

        if global_p >= alpha:
            # No significant difference overall
            return {name: 'a' for name in group_names}, global_p, test_name

        pvalue_matrix = tukey_hsd_test(valid_groups)

    # Generate CLD
    valid_names = list(valid_groups.keys())
    cld = generate_cld(pvalue_matrix, valid_names, alpha)

    # Add back any excluded groups with 'a' (no comparison possible)
    for name in group_names:
        if name not in cld:
            cld[name] = '-'

    return cld, global_p, test_name


def compute_highly_expressed_genes_by_group(
    adata, 
    groupby: str,
    n_genes: int = 20,
    method: str = 'mean'
) -> Dict[str, pd.DataFrame]:
    """
    Compute the most highly expressed genes in each group (batch or cell type).
    
    This function calculates gene expression statistics for each group separately,
    identifying the genes with highest expression levels. Unlike marker gene analysis,
    this finds genes that are highly expressed within each group (not necessarily
    specific to that group).
    
    Args:
        adata: AnnData object containing expression data
        groupby: Column name in adata.obs to group by (e.g., 'batch', 'cell_type')
        n_genes: Number of top genes to return per group
        method: Method for ranking genes ('mean', 'median', or 'fraction_expressed')
        
    Returns:
        Dictionary mapping group names to DataFrames with columns:
        - Gene: gene name
        - Mean Expression: mean expression in the group
        - Median Expression: median expression in the group
        - Std Expression: standard deviation
        - Fraction Expressed: fraction of cells with expression > 0
        - Rank: rank by the selected method
        
    Example:
        >>> genes_by_batch = compute_highly_expressed_genes_by_group(
        ...     adata, groupby='batch', n_genes=10, method='mean'
        ... )
        >>> print(genes_by_batch['Batch1'].head())
    """
    import pandas as pd
    import numpy as np
    from scipy import sparse
    
    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")
    
    groups = adata.obs[groupby].unique()
    results = {}
    
    for group in groups:
        # Get cells in this group
        mask = adata.obs[groupby] == group
        group_cells = adata[mask, :]
        
        # Get expression matrix
        X = group_cells.X
        if sparse.issparse(X):
            X = X.toarray()
        
        # Calculate statistics for each gene
        gene_stats = []
        for gene_idx, gene_name in enumerate(group_cells.var_names):
            gene_expr = X[:, gene_idx]
            
            mean_expr = np.mean(gene_expr)
            median_expr = np.median(gene_expr)
            std_expr = np.std(gene_expr)
            frac_expressed = np.sum(gene_expr > 0) / len(gene_expr)
            
            gene_stats.append({
                'Gene': gene_name,
                'Mean Expression': mean_expr,
                'Median Expression': median_expr,
                'Std Expression': std_expr,
                'Fraction Expressed': frac_expressed
            })
        
        # Create DataFrame
        df = pd.DataFrame(gene_stats)
        
        # Sort by selected method
        if method == 'mean':
            df = df.sort_values('Mean Expression', ascending=False)
        elif method == 'median':
            df = df.sort_values('Median Expression', ascending=False)
        elif method == 'fraction_expressed':
            df = df.sort_values('Fraction Expressed', ascending=False)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'mean', 'median', or 'fraction_expressed'")
        
        # Add rank
        df['Rank'] = range(1, len(df) + 1)
        
        # Keep top N genes
        df = df.head(n_genes).reset_index(drop=True)
        
        results[str(group)] = df
    
    return results

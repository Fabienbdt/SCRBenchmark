#!/usr/bin/env python3
"""
Reconstruct the scIB Pancreas dataset with TRUE raw integer counts.

The OpenProblems/scIB pancreas dataset (Luecken et al. 2022) only provides
normalized counts. This script downloads the 6 original source datasets
with raw integer counts and reconstructs a compatible h5ad file.

Sources:
    - Baron et al. 2016 (inDrop)           - GSE84133 (via scrnaseq)
    - Muraro et al. 2016 (CEL-seq2)        - GSE85241 (via scrnaseq)
    - Segerstolpe et al. 2016 (Smart-seq2) - E-MTAB-5061 (via GEO/direct)
    - Lawlor et al. 2017 (Fluidigm C1)     - GSE86469 (via GEO/direct)
    - Grün et al. 2016 (CEL-seq)           - GSE81076 (via GEO/direct)
    - Xin et al. 2016 (SMARTer)            - GSE81608 (via GEO/direct)

Usage:
    pip install scrnaseq biocutils GEOparse mygene
    
    # Intersection (default) - only genes common to all datasets (~14,813 genes)
    python scripts/setup/reconstruct_pancreas_raw_counts.py --gene-strategy intersection
    
    # Union - all genes from all datasets (~51,501 genes, missing=0)
    python scripts/setup/reconstruct_pancreas_raw_counts.py --gene-strategy union

Output:
    data/pancreas_raw_counts.h5ad (intersection mode)
    data/pancreas_raw_counts_union.h5ad (union mode)
    data/pancreas_raw_before_filtering.h5ad (diagnostic: all cells before filtering)
"""

import os
import sys
import gzip
import json
import logging
import tempfile
import urllib.request
import argparse
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, '_pancreas_cache')
OUTPUT_FILE = os.path.join(DATA_DIR, 'pancreas_raw_counts.h5ad')
REFERENCE_FILE = os.path.join(DATA_DIR, 'pancreas_openproblems.h5ad')
BARON_LOCAL_DIR = os.path.join(DATA_DIR, 'GSE84133_RAW')  # Local Baron files

# Cell type mapping from each study's labels to scIB harmonized labels
CELLTYPE_MAPPING = {
    'baron': {
        'acinar': 'acinar',
        'activated_stellate': 'activated_stellate',
        'alpha': 'alpha',
        'beta': 'beta',
        'delta': 'delta',
        'ductal': 'ductal',
        'endothelial': 'endothelial',
        'epsilon': 'epsilon',
        'gamma': 'gamma',
        'macrophage': 'macrophage',
        'mast': 'mast',
        'quiescent_stellate': 'quiescent_stellate',
        'schwann': 'schwann',
        't_cell': 't_cell',
    },
    'muraro': {
        'acinar': 'acinar',
        'alpha': 'alpha',
        'beta': 'beta',
        'delta': 'delta',
        'duct': 'ductal',
        'ductal': 'ductal',
        'endothelial': 'endothelial',
        'epsilon': 'epsilon',
        'gamma': 'gamma',
        'pp': 'gamma',
        'mesenchymal': 'activated_stellate',
        'unclear': None,
    },
    'segerstolpe': {
        'acinar cell': 'acinar',
        'alpha cell': 'alpha',
        'beta cell': 'beta',
        'co-expression cell': None,
        'delta cell': 'delta',
        'ductal cell': 'ductal',
        'endothelial cell': 'endothelial',
        'epsilon cell': 'epsilon',
        'gamma cell': 'gamma',
        'mast cell': 'mast',
        'MHC class II cell': 'macrophage',
        'PSC cell': 'activated_stellate',
        'unclassified cell': None,
        'unclassified endocrine cell': None,
        'not applicable': None,
    },
    'lawlor': {
        'Acinar': 'acinar',
        'Alpha': 'alpha',
        'Beta': 'beta',
        'Delta': 'delta',
        'Ductal': 'ductal',
        'GCG/PPY': 'alpha',
        'Gamma/PP': 'gamma',
        'None/Other': None,
        'Stellate': 'activated_stellate',
    },
    'grun': {
        'acinar': 'acinar',
        'alpha': 'alpha',
        'beta': 'beta',
        'delta': 'delta',
        'duct': 'ductal',
        'ductal': 'ductal',
        'endothelial': 'endothelial',
        'epsilon': 'epsilon',
        'mesenchymal': 'activated_stellate',
        'unclear': None,
        'pp': 'gamma',
        'PSC': 'activated_stellate',
        'not applicable': None,
    },
    'xin': {
        'alpha': 'alpha',
        'beta': 'beta',
        'delta': 'delta',
        'pp': 'gamma',
        'PP': 'gamma',
    },
}

# Batch name mapping
BATCH_MAPPING = {
    'baron': {
        'human1': 'inDrop1',
        'human2': 'inDrop2',
        'human3': 'inDrop3',
        'human4': 'inDrop4',
    },
    'muraro': 'celseq2',
    'segerstolpe': 'smartseq2',
    'lawlor': 'fluidigmc1',
    'grun': 'celseq',
    'xin': 'smarter',
}


# =============================================================================
# Gene name harmonization
# =============================================================================

def normalize_gene_names(adata: ad.AnnData, study_name: str) -> ad.AnnData:
    """Normalize gene names to HGNC symbols.

    Different datasets use different gene identifiers:
    - Baron: HGNC symbols (A1BG, A1CF)
    - Muraro/Grün: symbol__chrN format (A1BG__chr19)
    - Lawlor: Ensembl IDs (ENSG00000229483)
    - Xin: Entrez Gene IDs (1, 2, 3)
    """
    var_names = list(adata.var_names)

    # Detect format
    sample = var_names[:10]
    is_ensembl = any(str(g).startswith('ENSG') for g in sample)
    is_entrez = all(str(g).isdigit() for g in sample if str(g) != '')
    has_chr = any('__chr' in str(g) for g in sample)

    if has_chr:
        # Muraro/Grün format: strip __chrN suffix
        logger.info(f"  [{study_name}] Converting symbol__chrN -> symbol")
        new_names = [str(g).split('__')[0] for g in var_names]
        adata.var_names = new_names
        adata.var_names_make_unique()

    elif is_ensembl:
        # Lawlor: Ensembl -> symbol via mygene
        logger.info(f"  [{study_name}] Converting Ensembl IDs -> symbols via mygene")
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            results = mg.querymany(var_names, scopes='ensembl.gene',
                                   fields='symbol', species='human',
                                   returnall=True, verbose=False)
            id_to_symbol = {}
            for r in results['out']:
                if 'symbol' in r and r.get('query'):
                    id_to_symbol[r['query']] = r['symbol']

            new_names = [id_to_symbol.get(g, g) for g in var_names]
            n_mapped = sum(1 for g in var_names if g in id_to_symbol)
            logger.info(f"  [{study_name}] Mapped {n_mapped}/{len(var_names)} Ensembl IDs to symbols")
            adata.var_names = new_names
            adata.var_names_make_unique()
        except ImportError:
            logger.warning(f"  [{study_name}] mygene not installed, keeping Ensembl IDs")

    elif is_entrez:
        # Xin: Entrez -> symbol via mygene
        logger.info(f"  [{study_name}] Converting Entrez IDs -> symbols via mygene")
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            str_ids = [str(g) for g in var_names]
            results = mg.querymany(str_ids, scopes='entrezgene',
                                   fields='symbol', species='human',
                                   returnall=True, verbose=False)
            id_to_symbol = {}
            for r in results['out']:
                if 'symbol' in r and r.get('query'):
                    id_to_symbol[r['query']] = r['symbol']

            new_names = [id_to_symbol.get(str(g), str(g)) for g in var_names]
            n_mapped = sum(1 for g in var_names if str(g) in id_to_symbol)
            logger.info(f"  [{study_name}] Mapped {n_mapped}/{len(var_names)} Entrez IDs to symbols")
            adata.var_names = new_names
            adata.var_names_make_unique()
        except ImportError:
            logger.warning(f"  [{study_name}] mygene not installed, keeping Entrez IDs")

    else:
        logger.info(f"  [{study_name}] Gene names appear to be HGNC symbols already")

    return adata


# =============================================================================
# SCE to AnnData conversion (for scrnaseq package)
# =============================================================================

def sce_to_anndata(sce, study_name: str) -> ad.AnnData:
    """Convert SingleCellExperiment to AnnData with raw counts.

    SCE format: genes × cells (rows = genes, cols = cells)
    AnnData format: cells × genes (rows = cells, cols = genes)
    """
    X = sce.assay('counts')
    logger.info(f"  [{study_name}] SCE assay shape (genes×cells): {X.shape}")

    # Transpose: SCE is genes×cells, AnnData is cells×genes
    if sp.issparse(X):
        X = X.T.tocsr()
    else:
        X = np.asarray(X).T

    # Get metadata
    row_names = list(sce.get_row_names()) if sce.get_row_names() is not None else None
    col_names = list(sce.get_column_names()) if sce.get_column_names() is not None else None

    # Build obs DataFrame (cells = SCE columns)
    col_data = sce.get_column_data()
    obs_dict = {}
    if col_data is not None:
        for col in col_data.get_column_names():
            vals = col_data.get_column(col)
            obs_dict[col] = list(vals) if hasattr(vals, '__iter__') else [vals]
    obs = pd.DataFrame(obs_dict)
    if col_names:
        obs.index = col_names

    # Build var DataFrame (genes = SCE rows)
    row_data = sce.get_row_data()
    var_dict = {}
    if row_data is not None:
        for col in row_data.get_column_names():
            vals = row_data.get_column(col)
            var_dict[col] = list(vals) if hasattr(vals, '__iter__') else [vals]
    var = pd.DataFrame(var_dict)
    if row_names:
        var.index = row_names

    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Verify and fix raw counts (some datasets store ints as float64)
    if sp.issparse(adata.X):
        sample = adata.X[:min(100, adata.n_obs), :min(100, adata.n_vars)].toarray()
    else:
        sample = adata.X[:min(100, adata.n_obs), :min(100, adata.n_vars)]
    nonzero = sample[sample != 0]
    is_int = len(nonzero) == 0 or np.allclose(nonzero, np.round(nonzero))

    if not is_int:
        logger.warning(f"  [{study_name}] Counts are float, rounding to integers")
    # Always round and convert - SCE counts should be integers
    if sp.issparse(adata.X):
        adata.X.data = np.round(adata.X.data).astype(np.float32)
    else:
        adata.X = np.round(adata.X).astype(np.float32)

    logger.info(f"  [{study_name}] AnnData shape (cells×genes): {adata.shape}, "
                f"Is integer: {is_int}")

    return adata


# =============================================================================
# Download helpers for GEO datasets
# =============================================================================

def download_file(url: str, dest: str, description: str = ""):
    """Download a file with progress indicator."""
    if os.path.exists(dest):
        logger.info(f"  Using cached: {dest}")
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info(f"  Downloading {description}: {url}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            if count % 100 == 0:
                print(f"\r  Progress: {pct}%", end='', flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print()
    return dest


# =============================================================================
# Dataset fetchers
# =============================================================================

def fetch_baron() -> ad.AnnData:
    """Fetch Baron et al. 2016 (inDrop, 4 human donors) via scrnaseq."""
    import scrnaseq
    logger.info("Downloading Baron et al. 2016 (inDrop)...")
    sce = scrnaseq.fetch_dataset("baron-pancreas-2016", "2023-12-14",
                                  path="human", realize_assays=True)
    adata = sce_to_anndata(sce, 'baron')

    # Map cell types
    label_col = 'label' if 'label' in adata.obs.columns else adata.obs.columns[0]
    adata.obs['original_celltype'] = adata.obs[label_col].astype(str)
    adata.obs['cell_type'] = adata.obs['original_celltype'].map(CELLTYPE_MAPPING['baron'])

    # Map batches (donor -> inDrop1-4)
    donor_col = 'donor' if 'donor' in adata.obs.columns else None
    if donor_col:
        adata.obs['batch'] = adata.obs[donor_col].map(BATCH_MAPPING['baron'])
    else:
        logger.warning("  [baron] No donor column, using inDrop1")
        adata.obs['batch'] = 'inDrop1'

    adata.obs['tech'] = adata.obs['batch']
    adata.obs['study'] = 'baron'
    adata = normalize_gene_names(adata, 'baron')
    return adata


def fetch_baron_local() -> ad.AnnData:
    """Fetch Baron et al. 2016 from local H5AD files in data/GSE84133_RAW.
    
    This is a fallback when scrnaseq package fails or local files are preferred.
    """
    logger.info("Loading Baron et al. (inDrop) from local files...")
    
    if not os.path.exists(BARON_LOCAL_DIR):
        raise FileNotFoundError(f"Local Baron directory not found: {BARON_LOCAL_DIR}")
    
    patterns = {
        'inDrop1': '*human1*_counts.h5ad',
        'inDrop2': '*human2*_counts*.h5ad',
        'inDrop3': '*human3*_counts.h5ad',
        'inDrop4': '*human4*_counts.h5ad',
    }
    
    adatas = []
    for batch_id, pattern in patterns.items():
        files = glob.glob(os.path.join(BARON_LOCAL_DIR, pattern))
        if not files:
            logger.warning(f"  No file found for {batch_id}")
            continue
        
        filepath = files[0]
        logger.info(f"  Loading {batch_id}: {os.path.basename(filepath)}")
        
        try:
            ad_batch = ad.read_h5ad(filepath)
            ad_batch.obs['batch'] = batch_id
            ad_batch.obs['tech'] = batch_id
            ad_batch.obs['study'] = 'baron'
            
            # Map cell types
            ct_col = next((c for c in ['label', 'cell_type', 'assigned_cluster'] 
                          if c in ad_batch.obs.columns), None)
            if ct_col:
                ad_batch.obs['original_celltype'] = ad_batch.obs[ct_col].astype(str)
                ad_batch.obs['cell_type'] = ad_batch.obs['original_celltype'].map(
                    CELLTYPE_MAPPING['baron']).fillna('unknown')
            else:
                ad_batch.obs['cell_type'] = 'unknown'
                ad_batch.obs['original_celltype'] = 'unknown'
            
            adatas.append(ad_batch)
        except Exception as e:
            logger.error(f"  Error loading {filepath}: {e}")
    
    if not adatas:
        raise FileNotFoundError(f"Could not load any Baron files from {BARON_LOCAL_DIR}")
    
    adata = ad.concat(adatas, join='outer', merge='same')
    adata.obs_names_make_unique()
    adata = normalize_gene_names(adata, 'baron')
    
    logger.info(f"  [baron_local] Shape: {adata.shape}")
    return adata

def fetch_muraro() -> ad.AnnData:
    """Fetch Muraro et al. 2016 (CEL-seq2) via scrnaseq."""
    import scrnaseq
    logger.info("Downloading Muraro et al. 2016 (CEL-seq2)...")
    sce = scrnaseq.fetch_dataset("muraro-pancreas-2016", "2023-12-19",
                                  realize_assays=True)
    adata = sce_to_anndata(sce, 'muraro')

    label_col = 'label' if 'label' in adata.obs.columns else adata.obs.columns[0]
    adata.obs['original_celltype'] = adata.obs[label_col].astype(str)
    adata.obs['cell_type'] = adata.obs['original_celltype'].map(CELLTYPE_MAPPING['muraro'])

    adata.obs['batch'] = BATCH_MAPPING['muraro']
    adata.obs['tech'] = BATCH_MAPPING['muraro']
    adata.obs['study'] = 'muraro'
    adata = normalize_gene_names(adata, 'muraro')
    return adata


def fetch_segerstolpe() -> ad.AnnData:
    """Fetch Segerstolpe et al. 2016 (Smart-seq2) from ArrayExpress E-MTAB-5061."""
    logger.info("Downloading Segerstolpe et al. 2016 (Smart-seq2) from ArrayExpress...")

    # The processed data file contains both RPKM and counts
    url = ("https://www.ebi.ac.uk/biostudies/files/E-MTAB-5061/"
           "pancreas_refseq_rpkms_counts_3514sc.txt")
    dest = os.path.join(CACHE_DIR, 'segerstolpe_counts.txt')

    download_file(url, dest, "Segerstolpe counts")

    # File format: 
    # Header: #samples <TAB> Cell1 ... CellN (N=3514 cells)
    # Data rows: Gene <TAB> Accession <TAB> RPKM1...RPKMN <TAB> Count1...CountN
    # So each row has: 2 ID columns + 3514 RPKM + 3514 Counts = 7030 total
    logger.info("  [segerstolpe] Parsing count matrix (column-based RPKM|Counts)...")

    # Read header separately (first line starts with #samples)
    with open(dest, 'r') as f:
        header_line = f.readline().strip()

    # Parse header to get cell names
    header_parts = header_line.split('\t')
    n_cells = len(header_parts) - 1  # Skip '#samples'
    cell_names = header_parts[1:]
    logger.info(f"  [segerstolpe] Found {n_cells} cells in header")

    # Read data: col0=Gene, col1=Accession, col2...(2+n_cells-1)=RPKM, col(2+n_cells)...(2+2*n_cells-1)=Counts
    df = pd.read_csv(dest, sep='\t', skiprows=1, header=None)
    gene_names = df[0].values.astype(str)
    
    # Extract counts (second half of data columns after Gene and Accession)
    # Columns: 0=Gene, 1=Acc, 2:(2+n_cells)=RPKM, (2+n_cells):(2+2*n_cells)=Counts
    counts_start = 2 + n_cells
    counts_end = 2 + 2 * n_cells
    counts_data = df.iloc[:, counts_start:counts_end].values
    logger.info(f"  [segerstolpe] Counts shape: {counts_data.shape}")

    # Transpose: cells × genes
    X = sp.csr_matrix(counts_data.astype(np.float32).T)
    adata = ad.AnnData(X=X)
    adata.obs_names = cell_names
    adata.var_names = gene_names.tolist()

    # Download cell annotations
    ann_url = ("https://www.ebi.ac.uk/biostudies/files/E-MTAB-5061/"
               "E-MTAB-5061.sdrf.txt")
    ann_dest = os.path.join(CACHE_DIR, 'segerstolpe_annotations.txt')
    download_file(ann_url, ann_dest, "Segerstolpe annotations")

    ann_df = pd.read_csv(ann_dest, sep='\t')
    # Find the inferred cell type column (NOT 'Characteristics [cell type]' which is tissue-level)
    ct_col = None
    for col in ann_df.columns:
        if 'inferred cell type' in col.lower():
            ct_col = col
            break
    if ct_col is None:
        for col in ann_df.columns:
            if 'cell type' in col.lower() or 'celltype' in col.lower():
                ct_col = col
                break

    if ct_col:
        # Map cell IDs to cell types
        id_col = ann_df.columns[0]  # Usually 'Source Name'
        ct_map = dict(zip(ann_df[id_col], ann_df[ct_col]))
        adata.obs['original_celltype'] = adata.obs_names.map(
            lambda x: ct_map.get(x, 'unknown')).astype(str)
    else:
        logger.warning("  [segerstolpe] No cell type column found in annotations")
        adata.obs['original_celltype'] = 'unknown'

    adata.obs['cell_type'] = adata.obs['original_celltype'].map(
        CELLTYPE_MAPPING['segerstolpe'])
    adata.obs['batch'] = BATCH_MAPPING['segerstolpe']
    adata.obs['tech'] = BATCH_MAPPING['segerstolpe']
    adata.obs['study'] = 'segerstolpe'
    adata = normalize_gene_names(adata, 'segerstolpe')

    logger.info(f"  [segerstolpe] Shape: {adata.shape}")
    return adata


def fetch_lawlor() -> ad.AnnData:
    """Fetch Lawlor et al. 2017 (Fluidigm C1) from GEO GSE86469."""
    logger.info("Downloading Lawlor et al. 2017 (Fluidigm C1) from GEO...")

    # GSE86469 is the single-cell companion series (GSE86473 has no supplementary files)
    url = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE86nnn/GSE86469/suppl/"
           "GSE86469_GEO.islet.single.cell.processed.data.RSEM.raw.expected.counts.csv.gz")
    dest = os.path.join(CACHE_DIR, 'lawlor_counts.csv.gz')

    download_file(url, dest, "Lawlor counts")

    logger.info("  [lawlor] Parsing count matrix...")
    df = pd.read_csv(dest, index_col=0, compression='gzip')
    # genes × cells -> transpose to cells × genes
    df = df.T

    X = sp.csr_matrix(df.values.astype(np.float32))
    adata = ad.AnnData(X=X)
    adata.obs_names = df.index.tolist()
    adata.var_names = df.columns.tolist()

    # Lawlor cell types are encoded in sample names: CellType_DonorID_CellNum
    logger.info("  [lawlor] Extracting cell types from sample names...")
    cell_types = []
    for name in adata.obs_names:
        parts = name.split('_')
        if len(parts) >= 1:
            cell_types.append(parts[0])
        else:
            cell_types.append('unknown')

    adata.obs['original_celltype'] = cell_types
    adata.obs['cell_type'] = adata.obs['original_celltype'].map(
        CELLTYPE_MAPPING['lawlor'])
    adata.obs['batch'] = BATCH_MAPPING['lawlor']
    adata.obs['tech'] = BATCH_MAPPING['lawlor']
    adata.obs['study'] = 'lawlor'
    adata = normalize_gene_names(adata, 'lawlor')

    logger.info(f"  [lawlor] Shape: {adata.shape}")
    return adata


def fetch_grun() -> ad.AnnData:
    """Fetch Grün et al. 2016 (CEL-seq) from GEO GSE81076."""
    logger.info("Downloading Grün et al. 2016 (CEL-seq) from GEO...")

    url = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE81nnn/GSE81076/suppl/"
           "GSE81076_D2_3_7_10_17.txt.gz")
    dest = os.path.join(CACHE_DIR, 'grun_counts.txt.gz')

    download_file(url, dest, "Grün counts")

    logger.info("  [grun] Parsing count matrix...")
    df = pd.read_csv(dest, sep='\t', index_col=0, compression='gzip')
    # genes × cells -> transpose to cells × genes
    df = df.T

    X = sp.csr_matrix(df.values.astype(np.float32))
    adata = ad.AnnData(X=X)
    adata.obs_names = df.index.tolist()
    adata.var_names = df.columns.tolist()

    # Grün dataset: cell names encode donor info (D2, D3, D7, D10, D17)
    # Cell types are not in the count file - need to match from reference
    adata.obs['original_celltype'] = 'unknown'
    adata.obs['cell_type'] = None  # Will be mapped from reference
    adata.obs['batch'] = BATCH_MAPPING['grun']
    adata.obs['tech'] = BATCH_MAPPING['grun']
    adata.obs['study'] = 'grun'
    adata = normalize_gene_names(adata, 'grun')

    logger.info(f"  [grun] Shape: {adata.shape}")
    return adata


def fetch_xin() -> ad.AnnData:
    """Fetch Xin et al. 2016 (SMARTer) from GEO GSE81608."""
    logger.info("Downloading Xin et al. 2016 (SMARTer) from GEO...")

    url = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE81nnn/GSE81608/suppl/"
           "GSE81608_human_islets_rpkm.txt.gz")
    dest = os.path.join(CACHE_DIR, 'xin_rpkm.txt.gz')

    download_file(url, dest, "Xin expression data")

    logger.info("  [xin] Parsing expression matrix...")
    df = pd.read_csv(dest, sep='\t', index_col=0, compression='gzip')
    df = df.T

    # Xin provides RPKM, not raw counts. We need to check for counts file.
    # If only RPKM is available, we'll note this limitation.
    logger.warning("  [xin] GEO only provides RPKM values, not raw counts. "
                   "Will attempt to use rounded values or exclude from ZINB models.")

    X = sp.csr_matrix(df.values.astype(np.float32))
    adata = ad.AnnData(X=X)
    adata.obs_names = [str(x) for x in df.index.tolist()]
    adata.var_names = [str(x) for x in df.columns.tolist()]

    # Xin cell types from GEO sample annotations
    # Cell names typically encode the cell type
    logger.info("  [xin] Extracting cell types from sample names...")
    cell_types = []
    for name in adata.obs_names:
        name_lower = name.lower()
        if 'alpha' in name_lower:
            cell_types.append('alpha')
        elif 'beta' in name_lower:
            cell_types.append('beta')
        elif 'delta' in name_lower:
            cell_types.append('delta')
        elif 'pp' in name_lower or 'gamma' in name_lower:
            cell_types.append('pp')
        else:
            cell_types.append('unknown')

    adata.obs['original_celltype'] = cell_types
    adata.obs['cell_type'] = adata.obs['original_celltype'].map(
        CELLTYPE_MAPPING['xin'])
    adata.obs['batch'] = BATCH_MAPPING['xin']
    adata.obs['tech'] = BATCH_MAPPING['xin']
    adata.obs['study'] = 'xin'
    adata.uns['is_rpkm'] = True  # Flag that this is RPKM, not raw counts
    adata = normalize_gene_names(adata, 'xin')

    logger.info(f"  [xin] Shape: {adata.shape}")
    return adata


# =============================================================================
# Cell type mapping from reference
# =============================================================================

def map_celltypes_from_reference(datasets: dict, ref_path: str):
    """Map cell types for datasets without annotations using OpenProblems reference.

    Uses obs_names matching between reconstructed and reference datasets.
    """
    if not os.path.exists(ref_path):
        logger.warning("Reference file not found. Cannot map cell types for Grün.")
        return

    logger.info("Mapping cell types from OpenProblems reference...")
    ref = ad.read_h5ad(ref_path)

    ct_col = 'cell_type' if 'cell_type' in ref.obs.columns else 'celltype'
    ref_celltypes = ref.obs[ct_col]
    ref_batches = ref.obs['batch']

    for name, adata in datasets.items():
        if adata.obs['cell_type'].isna().any() or (adata.obs['cell_type'] == 'unknown').any():
            # Try to match by obs_names
            batch_name = BATCH_MAPPING[name] if isinstance(BATCH_MAPPING[name], str) else None
            if batch_name:
                ref_mask = ref_batches == batch_name
                ref_subset = ref.obs.loc[ref_mask]

                # Try matching obs_names
                common = set(adata.obs_names) & set(ref_subset.index)
                if common:
                    logger.info(f"  [{name}] Matched {len(common)} cells by obs_names")
                    for cell_id in common:
                        if cell_id in ref_subset.index:
                            adata.obs.loc[cell_id, 'cell_type'] = ref_subset.loc[cell_id, ct_col]

            n_mapped = adata.obs['cell_type'].notna().sum()
            n_total = adata.n_obs
            logger.info(f"  [{name}] Cell types mapped: {n_mapped}/{n_total}")


# =============================================================================
# Validation
# =============================================================================

def validate_against_reference(adata_new: ad.AnnData, ref_path: str):
    """Compare reconstructed dataset against OpenProblems reference."""
    if not os.path.exists(ref_path):
        logger.warning(f"Reference file not found: {ref_path}")
        return

    logger.info("Validating against OpenProblems reference...")
    ref = ad.read_h5ad(ref_path)

    print("\n" + "=" * 70)
    print("VALIDATION: Reconstructed vs OpenProblems Reference")
    print("=" * 70)

    # Compare batch distribution
    ct_col_ref = 'cell_type' if 'cell_type' in ref.obs.columns else 'celltype'

    print("\n--- Cells per batch ---")
    print(f"{'Batch':<15} {'Reference':>10} {'Reconstructed':>15}")
    print("-" * 45)
    ref_batches = ref.obs['batch'].value_counts().sort_index()
    new_batches = adata_new.obs['batch'].value_counts().sort_index()
    for batch in sorted(set(list(ref_batches.index) + list(new_batches.index))):
        ref_n = ref_batches.get(batch, 0)
        new_n = new_batches.get(batch, 0)
        print(f"{batch:<15} {ref_n:>10} {new_n:>15}")

    print("\n--- Cells per type ---")
    print(f"{'Cell type':<25} {'Reference':>10} {'Reconstructed':>15}")
    print("-" * 55)
    ref_types = ref.obs[ct_col_ref].value_counts().sort_index()
    new_types = adata_new.obs['cell_type'].value_counts().sort_index()
    for ct in sorted(set(list(ref_types.index) + list(new_types.index))):
        ref_n = ref_types.get(ct, 0)
        new_n = new_types.get(ct, 0)
        print(f"{ct:<25} {ref_n:>10} {new_n:>15}")

    # Verify raw counts
    print("\n--- Data quality ---")
    if sp.issparse(adata_new.X):
        sample = adata_new.X[:1000, :min(1000, adata_new.n_vars)].toarray()
    else:
        sample = adata_new.X[:1000, :min(1000, adata_new.n_vars)]
    nonzero = sample[sample != 0]
    if len(nonzero) > 0:
        is_int = np.allclose(nonzero, np.round(nonzero))
        print(f"Raw counts are integers: {is_int}")
        print(f"Value range: [{nonzero.min():.1f}, {nonzero.max():.1f}]")
    print(f"Total cells: {adata_new.n_obs} (ref: {ref.n_obs})")
    print(f"Total genes: {adata_new.n_vars} (ref: {ref.n_vars})")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Reconstruct pancreas dataset with raw integer counts.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Gene Strategy Options:
  intersection  Only genes common to ALL datasets (~14,813 genes)
                More conservative, no missing values.
  
  union         ALL genes from any dataset (~51,501 genes)
                Larger than OpenProblems reference (18,771 genes).
                Missing values filled with 0.
        """
    )
    parser.add_argument(
        '--gene-strategy',
        choices=['intersection', 'union'],
        default='intersection',
        help='How to combine genes across datasets (default: intersection)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gene_strategy = args.gene_strategy
    
    # Set output file based on strategy
    if gene_strategy == 'union':
        output_file = os.path.join(DATA_DIR, 'pancreas_raw_counts_union.h5ad')
    else:
        output_file = OUTPUT_FILE  # pancreas_raw_counts.h5ad
    
    print("=" * 70)
    print("  Pancreas Dataset Reconstruction - Raw Integer Counts")
    print("  Based on scIB benchmark (Luecken et al. 2022)")
    print(f"  Gene strategy: {gene_strategy.upper()}")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Step 1: Download all 6 datasets
    logger.info("STEP 1: Downloading source datasets...")
    datasets = {}

    # Baron: Prefer local files (matching cell names with reference), fallback to scrnaseq
    logger.info("Loading Baron et al. 2016 (inDrop)...")
    if os.path.exists(BARON_LOCAL_DIR):
        try:
            datasets['baron'] = fetch_baron_local()
            logger.info(f"  OK baron (local): {datasets['baron'].n_obs} cells")
        except Exception as e:
            logger.warning(f"  Local Baron files failed: {e}")
            logger.info("  Trying scrnaseq as fallback...")
            try:
                datasets['baron'] = fetch_baron()
            except Exception as e2:
                logger.error(f"  FAILED baron (both methods): {e2}")
    else:
        try:
            datasets['baron'] = fetch_baron()
        except Exception as e:
            logger.error(f"  FAILED baron: {e}")

    # Other fetchers
    other_fetchers = [
        ('muraro', fetch_muraro),
        ('segerstolpe', fetch_segerstolpe),
        ('lawlor', fetch_lawlor),
        ('grun', fetch_grun),
        ('xin', fetch_xin),
    ]

    for name, fetcher in other_fetchers:
        try:
            datasets[name] = fetcher()
            n_cells = datasets[name].n_obs
            n_genes = datasets[name].n_vars
            logger.info(f"  OK {name}: {n_cells} cells, {n_genes} genes")

            # Print cell type distribution
            if 'cell_type' in datasets[name].obs.columns:
                cts = datasets[name].obs['cell_type'].value_counts()
                for ct, count in cts.items():
                    logger.info(f"    {ct}: {count}")
        except Exception as e:
            logger.error(f"  FAILED {name}: {e}")

            import traceback
            traceback.print_exc()

    if not datasets:
        logger.error("No datasets downloaded. Exiting.")
        sys.exit(1)

    logger.info(f"\nDownloaded {len(datasets)}/6 datasets")

    # Step 2: Map cell types from reference for datasets without annotations
    map_celltypes_from_reference(datasets, REFERENCE_FILE)

    # Step 3: Save raw (pre-filtering) diagnostic file, then filter
    logger.info("\nSTEP 3: Saving raw diagnostic file and filtering unmapped cells...")

    # Build pre-filtering summary and save raw h5ad
    raw_summary = {}
    raw_adata_list = []
    for name, adata in datasets.items():
        ct_col = 'cell_type'
        orig_col = 'original_celltype'
        n_total = adata.n_obs
        n_genes = adata.n_vars

        # Classify cells
        valid_mask = (adata.obs[ct_col].notna() &
                      (adata.obs[ct_col] != 'unknown') &
                      (adata.obs[ct_col] != ''))
        n_kept = valid_mask.sum()
        n_rejected = n_total - n_kept

        # Detail rejected labels
        rejected_cells = adata.obs[~valid_mask.values]
        if orig_col in rejected_cells.columns:
            rejected_labels = rejected_cells[orig_col].value_counts().to_dict()
        else:
            rejected_labels = {'(no label)': n_rejected}

        raw_summary[name] = {
            'total_cells': n_total,
            'total_genes': n_genes,
            'kept_cells': int(n_kept),
            'rejected_cells': n_rejected,
            'rejected_labels': rejected_labels,
        }

        logger.info(f"  [{name}] {n_total} raw cells, {n_genes} genes, "
                     f"{n_kept} kept, {n_rejected} rejected")
        if n_rejected > 0:
            for lab, cnt in rejected_labels.items():
                logger.info(f"    rejected label '{lab}': {cnt}")

        # Tag cells with filtering status for raw h5ad
        adata.obs['_will_be_kept'] = valid_mask.values
        raw_adata_list.append(adata.copy())

    # Save raw pre-filtering h5ad (union join to keep all genes)
    raw_output = os.path.join(DATA_DIR, 'pancreas_raw_before_filtering.h5ad')
    logger.info(f"  Saving raw (pre-filtering) data to {raw_output}...")
    raw_combined = ad.concat(raw_adata_list, join='outer', merge='same', fill_value=0)
    raw_combined.obs_names_make_unique()
    # Store raw summary in uns
    raw_combined.uns['raw_cell_summary'] = json.dumps(raw_summary, default=str)
    raw_combined.uns['description'] = (
        'Pre-filtering diagnostic file. Contains ALL cells before rejection. '
        'Column _will_be_kept=True for cells that pass filtering.'
    )
    raw_combined.write_h5ad(raw_output)
    raw_size = os.path.getsize(raw_output) / (1024 * 1024)
    logger.info(f"  Saved raw diagnostic: {raw_output} ({raw_size:.1f} MB)")

    # Print summary table
    print("\n" + "=" * 70)
    print("  RAW CELL COUNTS BEFORE FILTERING")
    print("=" * 70)
    print(f"  {'Study':<15} {'Raw':>8} {'Kept':>8} {'Rejected':>10} {'%Rejected':>10}")
    print("  " + "-" * 55)
    total_raw = total_kept = total_rejected = 0
    for name, info in raw_summary.items():
        pct = info['rejected_cells'] / info['total_cells'] * 100 if info['total_cells'] > 0 else 0
        print(f"  {name:<15} {info['total_cells']:>8} {info['kept_cells']:>8} "
              f"{info['rejected_cells']:>10} {pct:>9.1f}%")
        total_raw += info['total_cells']
        total_kept += info['kept_cells']
        total_rejected += info['rejected_cells']
    print("  " + "-" * 55)
    total_pct = total_rejected / total_raw * 100 if total_raw > 0 else 0
    print(f"  {'TOTAL':<15} {total_raw:>8} {total_kept:>8} {total_rejected:>10} {total_pct:>9.1f}%")
    print("=" * 70)

    # Now filter
    for name in list(datasets.keys()):
        adata = datasets[name]
        n_before = adata.n_obs
        mask = (adata.obs['cell_type'].notna() &
                (adata.obs['cell_type'] != 'unknown') &
                (adata.obs['cell_type'] != ''))
        datasets[name] = adata[mask.values].copy()
        # Remove diagnostic column
        if '_will_be_kept' in datasets[name].obs.columns:
            del datasets[name].obs['_will_be_kept']
        n_removed = n_before - datasets[name].n_obs
        if n_removed > 0:
            logger.info(f"  [{name}] Removed {n_removed} unmapped cells "
                        f"({n_before} -> {datasets[name].n_obs})")

    # Step 4: Harmonize gene names based on strategy
    logger.info(f"\nSTEP 4: Finding genes ({gene_strategy} mode)...")

    # Ensure unique gene names across all datasets
    for name, adata in datasets.items():
        adata.var_names_make_unique()

    # Collect gene sets
    gene_sets = {}
    for name, adata in datasets.items():
        genes = set(adata.var_names)
        gene_sets[name] = genes
        logger.info(f"  [{name}] {len(genes)} genes")

    if gene_strategy == 'intersection':
        # INTERSECTION: only genes common to ALL datasets
        final_genes = gene_sets[list(gene_sets.keys())[0]]
        for name, gs in gene_sets.items():
            prev = len(final_genes)
            final_genes = final_genes.intersection(gs)
            logger.info(f"  After intersecting {name}: {len(final_genes)} genes (was {prev})")
        final_genes = sorted(final_genes)
        logger.info(f"  Final gene count (intersection): {len(final_genes)}")
    else:
        # UNION: all genes from any dataset
        final_genes = set()
        for name, gs in gene_sets.items():
            prev = len(final_genes)
            final_genes = final_genes.union(gs)
            logger.info(f"  After adding {name}: {len(final_genes)} genes (was {prev})")
        final_genes = sorted(final_genes)
        logger.info(f"  Final gene count (union): {len(final_genes)}")
        
        # Show intersection for comparison
        intersection_genes = gene_sets[list(gene_sets.keys())[0]]
        for gs in gene_sets.values():
            intersection_genes = intersection_genes.intersection(gs)
        logger.info(f"  (Intersection would be: {len(intersection_genes)} genes)")

    # Step 5: Subset and concatenate
    logger.info("\nSTEP 5: Subsetting and concatenating...")
    adata_list = []
    
    if gene_strategy == 'intersection':
        # For intersection, subset each dataset to common genes
        for name, adata in datasets.items():
            sub = adata[:, adata.var_names.isin(final_genes)].copy()
            # Reorder genes to match
            sub = sub[:, final_genes].copy()
            # Keep only essential columns
            obs_keep = ['cell_type', 'batch', 'tech', 'study', 'original_celltype']
            obs_keep = [c for c in obs_keep if c in sub.obs.columns]
            sub.obs = sub.obs[obs_keep].copy()
            adata_list.append(sub)
            logger.info(f"  [{name}] {sub.n_obs} cells x {sub.n_vars} genes")
        
        adata_combined = ad.concat(adata_list, join='outer', merge='same')
    else:
        # For union, just clean obs columns and let concat handle the rest
        for name, adata in datasets.items():
            sub = adata.copy()
            # Keep only essential columns
            obs_keep = ['cell_type', 'batch', 'tech', 'study', 'original_celltype']
            obs_keep = [c for c in obs_keep if c in sub.obs.columns]
            sub.obs = sub.obs[obs_keep].copy()
            adata_list.append(sub)
            logger.info(f"  [{name}] {sub.n_obs} cells x {sub.n_vars} genes")
        
        # Concatenate with outer join and fill missing with 0
        adata_combined = ad.concat(adata_list, join='outer', merge='same', fill_value=0)
    
    adata_combined.obs_names_make_unique()
    logger.info(f"  Combined: {adata_combined.shape}")

    # Step 6: Finalize
    logger.info("\nSTEP 6: Finalizing...")

    # Ensure sparse CSR and handle NaN (for union mode)
    if sp.issparse(adata_combined.X):
        adata_combined.X = adata_combined.X.tocsr()
        # Replace any NaN with 0 (union mode may have some)
        if np.any(np.isnan(adata_combined.X.data)):
            logger.info("  Replacing NaN values with 0...")
            adata_combined.X.data = np.nan_to_num(adata_combined.X.data, nan=0.0)
        adata_combined.X.data = np.round(adata_combined.X.data).astype(np.float32)
    else:
        adata_combined.X = np.nan_to_num(adata_combined.X, nan=0.0)
        adata_combined.X = sp.csr_matrix(np.round(adata_combined.X).astype(np.float32))

    # Store in layers
    adata_combined.layers['counts'] = adata_combined.X.copy()

    # Metadata
    dataset_id = 'pancreas_raw_counts' if gene_strategy == 'intersection' else 'pancreas_raw_counts_union'
    adata_combined.uns['dataset_id'] = dataset_id
    adata_combined.uns['dataset_name'] = f'Human pancreas (raw counts, {gene_strategy})'
    adata_combined.uns['dataset_description'] = (
        f'Human pancreatic islet scRNA-seq data from 6 studies. '
        f'Reconstructed with raw integer counts from original sources. '
        f'Gene strategy: {gene_strategy}.'
    )
    adata_combined.uns['dataset_reference'] = 'luecken2022benchmarking'
    adata_combined.uns['dataset_organism'] = 'homo_sapiens'
    adata_combined.uns['gene_strategy'] = gene_strategy

    # Categorical columns
    for col in ['cell_type', 'batch', 'tech', 'study']:
        if col in adata_combined.obs.columns:
            adata_combined.obs[col] = adata_combined.obs[col].astype('category')

    # Compatibility columns
    adata_combined.obs['celltype'] = adata_combined.obs['cell_type']
    adata_combined.obs['labels'] = adata_combined.obs['cell_type']

    # Step 7: Save
    logger.info(f"\nSTEP 7: Saving to {output_file}...")
    adata_combined.write_h5ad(output_file)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    logger.info(f"  Saved: {output_file} ({file_size_mb:.1f} MB)")

    # Step 8: Validate
    validate_against_reference(adata_combined, REFERENCE_FILE)

    # Summary
    print("\n" + "=" * 70)
    print(f"  RECONSTRUCTION COMPLETE ({gene_strategy.upper()} MODE)")
    print("=" * 70)
    print(f"  Output: {output_file}")
    print(f"  Cells:  {adata_combined.n_obs}")
    print(f"  Genes:  {adata_combined.n_vars}")
    print(f"  Batches: {list(adata_combined.obs['batch'].cat.categories)}")
    print(f"  Cell types: {list(adata_combined.obs['cell_type'].cat.categories)}")

    # Check integer counts
    if sp.issparse(adata_combined.X):
        sample = adata_combined.X[:100, :100].toarray()
    else:
        sample = adata_combined.X[:100, :100]
    nz = sample[sample != 0]
    is_int = len(nz) == 0 or np.allclose(nz, np.round(nz))
    print(f"  Raw integer counts: {'YES' if is_int else 'NO (check warnings above)'}")
    print(f"  Gene strategy: {gene_strategy}")

    # Note about Xin
    if 'xin' in datasets and datasets['xin'].uns.get('is_rpkm', False):
        print("\n  WARNING: Xin et al. (smarter batch) only provides RPKM values.")
        print("  scDeepCluster should NOT use this batch for ZINB modeling.")

    print(f"\n  Usage: --data {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()

from anndata import read_h5ad
import scanpy as sc

normalize_per_cell = getattr(sc.pp, "normalize_per_cell", sc.pp.normalize_total)
highly_variable_genes = sc.pp.highly_variable_genes
log1p = sc.pp.log1p
scale = sc.pp.scale

from .test import run_desc_test
from .read import read_10X
from .write import write_desc_result
#from .downstream import run_tsne
#from .preprocessing import log1p








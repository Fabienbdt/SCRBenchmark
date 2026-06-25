#from . import original as og
from . import tools
from . import models
from . import datasets

from anndata import read_h5ad
import scanpy as sc

normalize_per_cell = getattr(sc.pp, "normalize_per_cell", sc.pp.normalize_total)
normalize_total = sc.pp.normalize_total
highly_variable_genes = sc.pp.highly_variable_genes
log1p = sc.pp.log1p
scale = sc.pp.scale

from .models.desc import train
from .tools.test import run_desc_test
from .tools.read import read_10X
from .tools.write import write_desc_result
from .tools.preprocessing import scale_bygroup
#from .tools.downstream import run_tsne
__version__ = '2.1.1'



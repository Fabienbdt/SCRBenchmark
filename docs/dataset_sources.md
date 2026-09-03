# Dataset Sources and Report/Repository Reconciliation

This is the provenance audit for the datasets used in Fabien Bidet's M2
internship report. It separates two operations that must not be confused:

- **exact report replay** uses the prepared H5AD files pinned by size, SHA256,
  dimensions, labels, and batch columns in
  `data/stable_generalist/download_manifest.csv`;
- **upstream reconstruction** downloads public study files with
  `scripts/setup/download_report_sources.py`, then applies the documented
  preparation steps.

Downloading an original GEO/Figshare file does not by itself reproduce the
bytes of a prepared H5AD. Use the pinned H5AD manifest whenever exact report
numbers or preserved scRAW checkpoints are required.

Audit date: **2026-07-11**.

## Report Set Versus Repository Set

The report and the repository are consistent once historical aliases and later
extensions are separated:

| Scope | Prepared dataset keys | Count |
| --- | --- | ---: |
| Report common-8 | `bbag094_zeisel`, `bbag094_spleen`, `baron_human_pancreas`, `gse112013_human_testis_raw_counts`, `kang_pbmc_gse96583_singlets_raw_counts`, `macaque_retina_gse118480_bipolar_raw_counts`, `paul15_bone_marrow_raw_counts`, `Tabula_Muris_liver_filtered_raw_counts` | 8 |
| Report external validation | `pancreas_raw_counts_four_batches_celseq_celseq2_fluidigmc1_smartseq2`, `Mouse_Pancreas_1_raw_counts` | 2 |
| Later stable-generalist extensions | `pancreas_raw_counts`, `Human_Pancreas_1_raw_counts`, `Human_Pancreas_2_raw_counts` | 3 |
| Full repository campaign | the 10 report entries plus the 3 extensions | 13 |

In particular, the report's 6,339-cell **Human Pancreas** external-validation
row is the four-technology pancreas object, not the 1,937-cell Baron donor
subset called `Human_Pancreas_1_raw_counts`. The report reproduction map now
uses the correct key.

## Audited Mapping and Public Sources

| Report name | Prepared dimensions | Public source(s) | Reconstruction status and important details |
| --- | ---: | --- | --- |
| BBAG094 Zeisel | 3,006 × 19,972 | [GEO GSE60361](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60361), [Hemberg dataset definitions](https://github.com/hemberg-lab/scRNA.seq.datasets) | GEO contains 3,005 high-quality cells. The historical prepared H5AD contains one additional `unknown` cell and uses the cell-type label as `batch`; see “Known discrepancies”. Exact replay therefore requires the pinned H5AD. |
| BBAG094 spleen | 9,552 × 23,341 | [scziDesk repository](https://github.com/xuebaliang/scziDesk), [direct H5 file](https://raw.githubusercontent.com/xuebaliang/scziDesk/master/dataset/Quake_10x_Spleen/data.h5) | Direct file was downloaded and verified: 9,552 cells, 23,341 genes, two donors, five labels. SHA256 is pinned in the source manifest. |
| Baron human pancreas | 8,569 × 20,125 | [GEO GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133) | `python scripts/setup/prepare_baron_dataset.py --download` reconstructs all four human donors and preserves donor batches. |
| Human testis | 6,490 × 27,477 | [GEO GSE112013](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE112013), [open-access article and supplements](https://pmc.ncbi.nlm.nih.gov/articles/PMC6274646/) | The manifest includes the UMI table, GEO series metadata, and official Supplementary Table S1 annotations. Preparation must retain all six donor/replicate samples and 13 report labels. |
| Kang PBMC | 24,679 × 35,635 | [GEO GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583) | Use singlets and the official metadata. The report batch is donor + condition (`confounded_batch`), giving 16 batches; a synthetic `single_batch` is not equivalent. |
| Macaque retina bipolar | 30,302 × 36,162 | [GEO GSE118480](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE118480), [scDML reproduction data](https://github.com/eleozzr/scDML_reproduce) | The LFS archive `macaque_raw.zip` was downloaded and verified. Its H5AD has the exact report dimensions and 30 samples; SHA256 is pinned. |
| Paul15 bone marrow | 2,730 × 3,451 | [Scanpy Paul15 documentation](https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.paul15.html), [direct upstream H5](https://falexwolf.de/data/paul15.h5) | This is the exact input used by `scanpy.datasets.paul15()`. SCRBenchmark assigns a single synthetic batch because the source has no report batch. |
| “Tabula Muris liver” | 2,859 × 22,966 | [Tabula Muris Senis data objects](https://figshare.com/articles/dataset/Tabula_Muris_Senis_Data_Objects/12654728), [direct FACS Liver H5AD](https://ndownloader.figshare.com/files/23872526) | The exact dimensions and 11 label counts match the official **Tabula Muris Senis** FACS Liver object. The similarly named original Tabula Muris v8 liver table has 23,433 genes and is not the report input. SCRBenchmark adds `single_batch`. |
| Human Pancreas, four technologies | 6,339 × 14,813 | [OpenProblems pancreas](https://openproblems.bio/datasets/openproblems_v1/pancreas), [OpenProblems datasets code](https://github.com/openproblems-bio/datasets) | The repo reconstruction combines CEL-seq, CEL-seq2, Fluidigm C1, and Smart-seq2 raw studies after harmonization. Use the four-batch key for the report external-validation row. |
| SCIB Mouse Pancreas 1 | 822 × 14,878 | [GEO GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133), sample GSM2230761 | Mouse sample 1 from Baron, filtered and harmonized to the prepared report object. It has no genuine batch column; SCRBenchmark uses a single batch. |

The raw-study URLs for the pancreas reconstruction are also pinned in the CSV:

- Muraro, CEL-seq2: [GSE85241](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85241)
- Segerstolpe, Smart-seq2: [E-MTAB-5061](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-5061)
- Lawlor, Fluidigm C1: [GSE86469](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE86469)
- Grün, CEL-seq: [GSE81076](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81076)
- Xin, SMARTer reference: [GSE81608](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81608)
- Baron, inDrop: [GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133)

## Download the Public Upstream Assets

List every audited asset without downloading:

```bash
python scripts/setup/download_report_sources.py --list
```

Download the sources for selected prepared datasets:

```bash
python scripts/setup/download_report_sources.py \
  --datasets bbag094_spleen,gse112013_human_testis_raw_counts
```

Download all upstream assets. This is large: the OpenProblems reference alone
is about 1.36 GB and several GEO archives are also substantial.

```bash
python scripts/setup/download_report_sources.py --all
```

Verify files already present in the default raw-source directory:

```bash
python scripts/setup/download_report_sources.py --all --verify-only
```

The default destination is `data/report_sources/raw/`, which is ignored by Git.
Use `--output-dir /large/disk/report_sources` to store the downloads elsewhere.
The authoritative URL/checksum inventory is
`data/report_sources/source_manifest.csv`.

## Exact Prepared H5AD Verification

If a local or hosted copy of the 13 prepared files exists, materialize it and
verify it separately:

```bash
python scripts/reproduction/download_datasets.py \
  --source-root /path/to/exact/prepared/h5ad/files

python scripts/reproduction/download_datasets.py --verify-only
```

This second command checks SHA256, byte size, dimensions, label column, batch
column, and number of labels. It is the required check before replaying a
preserved scRAW checkpoint.

## Known Discrepancies and Decisions

1. **Zeisel is a legacy non-canonical artifact.** GSE60361 states 3,005 cells,
   while the report H5AD and report table have 3,006. The additional row is
   labeled `unknown`. Its `batch` values duplicate the ten label values, which
   creates label leakage if that column is used for batch correction or DANN.
   Do not silently claim that a clean GEO reconstruction is byte-equivalent.
2. **The report's Human Pancreas alias was ambiguous.** Its 6,339 × 14,813
   dimensions identify the four-technology object. The previous reproduction
   map incorrectly pointed to the 1,937-cell `Human_Pancreas_1_raw_counts`
   donor subset; this audit corrects the map.
3. **The stable-generalist Pancreas row had a stale cell count.** The exact H5AD
   manifest and evaluated artifacts specify 14,908 cells, not 16,400. The
   dataset table now uses 14,908.
4. **The Lawlor accession is GSE86469.** GSE86473 is not the single-cell series
   used by the reconstruction. The setup script documentation now agrees with
   its working download implementation.
5. **“Tabula Muris liver” is a historical shorthand.** The exact dimensions and
   annotations identify the Tabula Muris Senis FACS Liver H5AD. The original
   Tabula Muris v8 liver matrix is retained in the source manifest only as
   provenance and must not replace the report object.

These discrepancies do not change the recorded report results. They explain
which exact prepared objects produced those results and prevent a future
reconstruction from being presented as identical when it is not.

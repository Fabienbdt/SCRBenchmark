#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
})

option_list <- list(
  make_option(c("--input-csv"), type = "character", dest = "input_csv"),
  make_option(c("--output-dir"), type = "character", dest = "output_dir"),
  make_option(c("--source-root"), type = "character", dest = "source_root", default = Sys.getenv("GINICLUST_ROOT", "")),
  make_option(c("--data-type"), type = "character", dest = "data_type", default = "RNA-seq"),
  make_option(c("--epsilon"), type = "double", default = NA),
  make_option(c("--minPts"), type = "integer", default = NA)
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$input_csv) || !file.exists(opt$input_csv)) {
  stop("--input-csv is required and must exist", call. = FALSE)
}
if (is.null(opt$output_dir)) {
  stop("--output-dir is required", call. = FALSE)
}
if (is.null(opt$source_root) || opt$source_root == "") {
  stop("--source-root is required", call. = FALSE)
}
if (is.null(opt$data_type)) {
  opt$data_type <- "RNA-seq"
}

out.folder <- normalizePath(opt$output_dir, mustWork = FALSE)
dir.create(out.folder, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out.folder, "figures"), recursive = TRUE, showWarnings = FALSE)

data.type <- opt$data_type
if (!(data.type %in% c("RNA-seq", "qPCR"))) {
  stop("--data-type must be RNA-seq or qPCR", call. = FALSE)
}

parsed_opt <- opt
opt <- list(
  epsilon = if (is.na(opt$epsilon)) NULL else opt$epsilon,
  MinPts = if (is.na(opt$minPts)) NULL else opt$minPts
)

source_root <- normalizePath(parsed_opt$source_root, mustWork = TRUE)
oldwd <- getwd()
on.exit(setwd(oldwd), add = TRUE)
setwd(source_root)

source("Rfunction/GiniClust_parameters.R")
source("Rfunction/GiniClust_packages.R")
source("Rfunction/GiniClust_Fitting.R")
source("Rfunction/GiniClust_Clustering.R")

exprimentID <- "giniclust_shared_cells_native_genes"
raw_counts <- read.csv(parsed_opt$input_csv, row.names = 1, check.names = FALSE)
gene_ids <- rownames(raw_counts)
ExprM.RawCounts <- as.data.frame(lapply(raw_counts, as.numeric), check.names = FALSE)
rownames(ExprM.RawCounts) <- gene_ids

if (nrow(ExprM.RawCounts) < 2 || ncol(ExprM.RawCounts) < 2) {
  stop("Input matrix must contain at least two genes and two cells", call. = FALSE)
}

ExpressedinCell_per_gene <- apply(ExprM.RawCounts, 1, function(x) length(x[x > expressed_cutoff]))
nonMir <- grep("MIR|Mir", rownames(ExprM.RawCounts), invert = TRUE)
Genelist <- intersect(
  rownames(ExprM.RawCounts)[nonMir],
  rownames(ExprM.RawCounts)[ExpressedinCell_per_gene >= minCellNum]
)
if (length(Genelist) < 2) {
  stop(sprintf("GiniClust native gene filter retained too few genes: %s", length(Genelist)), call. = FALSE)
}
ExprM.RawCounts.filter <- ExprM.RawCounts[Genelist, , drop = FALSE]
GeneList.final <- GiniClust_Fitting(data.type, ExprM.RawCounts.filter, out.folder, exprimentID)
GeneList.final <- intersect(as.character(GeneList.final), rownames(ExprM.RawCounts.filter))
if (length(GeneList.final) < 2) {
  stop(sprintf("GiniClust selected too few genes: %s", length(GeneList.final)), call. = FALSE)
}

Cluster.Results <- GiniClust_Clustering(
  data.type,
  ExprM.RawCounts.filter,
  GeneList.final,
  eps,
  MinPts,
  out.folder,
  exprimentID
)

labels <- Cluster.Results$clustering_membership_r
labels <- labels[match(colnames(ExprM.RawCounts), labels$cell.ID), ]
if (any(is.na(labels$cluster.ID))) {
  stop("GiniClust did not return labels for every input cell", call. = FALSE)
}

write.csv(
  data.frame(cell_id = labels$cell.ID, predicted_label = labels$cluster.ID),
  file = file.path(out.folder, "labels.csv"),
  row.names = FALSE,
  quote = TRUE
)
write.csv(
  data.frame(
    metric = c("n_input_genes", "n_gene_filtered_genes", "n_selected_genes", "gene_filter_mode"),
    value = c(
      nrow(ExprM.RawCounts),
      nrow(ExprM.RawCounts.filter),
      length(GeneList.final),
      "giniclust_native_gene_filter_no_cell_drop"
    )
  ),
  file = file.path(out.folder, "run_info.csv"),
  row.names = FALSE,
  quote = TRUE
)

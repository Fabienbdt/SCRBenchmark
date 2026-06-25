#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(CellSIUS)
  library(data.table)
})

option_list <- list(
  make_option(c("--input-csv"), type = "character", dest = "input_csv"),
  make_option(c("--base-labels-csv"), type = "character", dest = "base_labels_csv"),
  make_option(c("--output-dir"), type = "character", dest = "output_dir"),
  make_option(c("--mcl-path"), type = "character", dest = "mcl_path", default = Sys.getenv("MCL_PATH", ""))
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$input_csv) || !file.exists(opt$input_csv)) {
  stop("--input-csv is required and must exist", call. = FALSE)
}
if (is.null(opt$base_labels_csv) || !file.exists(opt$base_labels_csv)) {
  stop("--base-labels-csv is required and must exist", call. = FALSE)
}
if (is.null(opt$output_dir)) {
  stop("--output-dir is required", call. = FALSE)
}
if (!file.exists(opt$mcl_path)) {
  stop(sprintf("mcl executable not found: %s", opt$mcl_path), call. = FALSE)
}

out_dir <- normalizePath(opt$output_dir, mustWork = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
oldwd <- getwd()
on.exit(setwd(oldwd), add = TRUE)
setwd(out_dir)

mat <- as.matrix(read.csv(opt$input_csv, row.names = 1, check.names = FALSE))
storage.mode(mat) <- "numeric"
base <- read.csv(opt$base_labels_csv, check.names = FALSE, stringsAsFactors = FALSE)
if (!all(c("cell_id", "base_label") %in% colnames(base))) {
  stop("--base-labels-csv must contain cell_id and base_label columns", call. = FALSE)
}
base <- base[match(colnames(mat), base$cell_id), ]
if (any(is.na(base$base_label))) {
  stop("Base labels are missing for some input cells", call. = FALSE)
}
group_id <- as.character(base$base_label)
names(group_id) <- base$cell_id

cellsius_out <- CellSIUS(
  mat.norm = mat,
  group_id = group_id,
  min_n_cells = 10,
  min_fc = 2,
  corr_cutoff = NULL,
  iter = 0,
  max_perc_cells = 50,
  fc_between_cutoff = 1,
  mcl_path = opt$mcl_path
)

if (length(cellsius_out) == 1 && is.na(cellsius_out)) {
  final_labels <- group_id
  n_rows <- 0
} else {
  final_labels <- CellSIUS_final_cluster_assignment(
    CellSIUS.out = cellsius_out,
    group_id = group_id,
    min_n_genes = 3
  )
  n_rows <- nrow(cellsius_out)
}

write.csv(
  data.frame(cell_id = names(final_labels), predicted_label = as.character(final_labels)),
  file = file.path(out_dir, "labels.csv"),
  row.names = FALSE,
  quote = TRUE
)
write.csv(
  data.frame(metric = c("cellsius_rows"), value = c(n_rows)),
  file = file.path(out_dir, "run_info.csv"),
  row.names = FALSE,
  quote = TRUE
)

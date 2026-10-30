# Portable Greenwald CSV preparation. Never overwrites existing inputs.
# Rscript prepare_inputs.R --rds-dir PATH --output-dir PATH [--sample ID]
#                         [--check-against PATH] [--recipe PATH]
args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% args) {
  cat("--rds-dir PATH --output-dir PATH [--sample ID] [--check-against PATH] [--recipe PATH]\n")
  quit(status = 0)
}
if (length(args) %% 2 != 0) stop("Options require values; use --help")
opts <- setNames(as.list(args[seq(2, length(args), 2)]), args[seq(1, length(args), 2)])
allowed <- c("--rds-dir", "--output-dir", "--sample", "--check-against", "--recipe")
if (anyDuplicated(names(opts)) || !all(names(opts) %in% allowed)) stop("Unknown or repeated option")
if (is.null(opts[["--rds-dir"]]) || is.null(opts[["--output-dir"]])) stop("--rds-dir and --output-dir are required")
suppressPackageStartupMessages(library(Seurat))
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
recipe_path <- opts[["--recipe"]]
if (is.null(recipe_path)) recipe_path <- file.path(dirname(normalizePath(script)), "preparation.json")
recipe <- jsonlite::fromJSON(recipe_path, simplifyVector = TRUE)
stopifnot(recipe$layer == "data", length(recipe$programs) == 14)
files <- list.files(opts[["--rds-dir"]], pattern = "^seuratObject_denoised_clustered_.+_500\\.RDS$", full.names = TRUE)
samples <- sub("^seuratObject_denoised_clustered_(.+)_500\\.RDS$", "\\1", basename(files))
if (!is.null(opts[["--sample"]])) {
  keep <- samples == opts[["--sample"]]; files <- files[keep]; samples <- samples[keep]
}
if (!length(files) || anyDuplicated(samples)) stop("No samples or duplicate identities")
out <- opts[["--output-dir"]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
programs <- names(recipe$programs)
# Preserve the original YAML order for scoring AND physical pair orientation.
# Control-gene sampling and directional TileScope outputs depend on these.
pairs <- combn(programs, 2, simplify = FALSE)
all_targets <- unlist(lapply(samples, function(s) vapply(pairs, function(p)
  file.path(out, paste0(s, "_expression_data_", paste(p, collapse = "_"), ".csv")), character(1))))
if (any(file.exists(all_targets))) stop("Output files exist; choose a fresh output directory")
for (i in seq_along(files)) {
  sample <- samples[i]; message(sample, ": reading Seurat object")
  object <- readRDS(files[i])
  if (length(object@images) != 1) stop("Expected one spatial image")
  image <- object@images[[1]]
  coordinates <- image@coordinates
  barcodes <- rownames(coordinates)
  stopifnot(!is.null(barcodes), !anyDuplicated(barcodes), all(barcodes %in% rownames(object@meta.data)))
  genes <- rownames(object[[recipe$assay]])
  signatures <- lapply(recipe$programs, function(g) intersect(g, genes))
  if (any(lengths(signatures) == 0)) stop("A program has no covered genes")
  for (p in programs) if (length(signatures[[p]]) != length(recipe$programs[[p]]))
    message(sample, " ", p, ": missing ", paste(setdiff(recipe$programs[[p]], genes), collapse = ", "))
  object <- AddModuleScore(object, features = signatures, assay = recipe$assay,
    name = programs, nbin = recipe$nbin, ctrl = recipe$ctrl, seed = recipe$seed)
  scores <- object@meta.data[barcodes, paste0(programs, seq_along(programs)), drop = FALSE]
  colnames(scores) <- programs
  geometry <- data.frame(x = coordinates$imagecol * image@scale.factors$lowres,
                         y = coordinates$imagerow * image@scale.factors$lowres)
  labels <- object@meta.data[barcodes, recipe$roi_column]
  if (anyNA(labels) || any(!is.finite(as.matrix(scores)))) stop("Missing ROI labels or nonfinite scores")
  for (p in pairs) {
    frame <- data.frame(geometry, gene1 = scores[[p[1]]], gene2 = scores[[p[2]]], roi_id = labels)
    filename <- paste0(sample, "_expression_data_", paste(p, collapse = "_"), ".csv")
    if (!is.null(opts[["--check-against"]])) {
      baseline <- read.csv(file.path(opts[["--check-against"]], filename), check.names = FALSE)
      if (!identical(names(frame), names(baseline)) || nrow(frame) != nrow(baseline) ||
          !identical(as.character(frame$roi_id), as.character(baseline$roi_id))) stop("Identity mismatch: ", filename)
      for (column in c("x", "y", "gene1", "gene2")) {
        delta <- max(abs(frame[[column]] - baseline[[column]]))
        if (!is.finite(delta) || delta > 1e-12) stop("Numerical mismatch: ", filename, " ", column, " max delta=", delta)
      }
    }
    write.csv(frame, file.path(out, filename), row.names = FALSE)
  }
  message(sample, ": wrote 91 pairs", if (!is.null(opts[["--check-against"]])) " (baseline verified, absolute tolerance 1e-12)" else "")
  rm(object, image, coordinates, scores); gc()
}

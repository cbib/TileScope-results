#!/usr/bin/env Rscript

# Exact focus-ROI audit for competing mouse-GBM panel candidates.
#
# This is a pre-freeze diagnostic. It measures gene detection, same-spot
# co-detection and within-ROI coherence on LogNormalized source expression.
# It does not compute AddModuleScore and does not inspect TileScope pair
# outcomes. Single-gene candidates are handled as spatial-marker comparators.

suppressPackageStartupMessages({
  library(Seurat)
  library(yaml)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
  stop(
    "usage: phase5_candidate_panel_audit.R ",
    "candidates.yaml samples.csv focus_contexts.csv data_root output.csv"
  )
}

catalogue <- yaml::read_yaml(args[[1]])
samples <- read.csv(args[[2]], check.names = FALSE, stringsAsFactors = FALSE)
focus <- read.csv(args[[3]], stringsAsFactors = FALSE)
data_root <- args[[4]]
output_path <- args[[5]]
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

safe_cor <- function(x, y) {
  if (length(x) < 3L || sd(x) == 0 || sd(y) == 0) return(NA_real_)
  suppressWarnings(cor(x, y, method = "spearman"))
}

rows <- list()
row_i <- 0L

for (sample_i in seq_len(nrow(samples))) {
  sample_name <- samples$sample_dir[[sample_i]]
  object <- readRDS(file.path(data_root, samples$rds_relpath[[sample_i]]))
  expression <- LayerData(
    object,
    assay = samples$assay[[sample_i]],
    layer = "data"
  )
  metadata <- object[[]]
  if (!identical(colnames(expression), rownames(metadata))) {
    stop("expression and metadata are not aligned for ", sample_name)
  }
  roi <- as.character(metadata[[samples$bayesspace_column[[sample_i]]]])
  sample_focus <- focus[focus$sample_dir == sample_name, , drop = FALSE]

  for (candidate in catalogue$candidates) {
    intended <- unlist(candidate$intended_samples)
    if (!(sample_name %in% intended)) next
    genes <- unlist(candidate$genes)
    missing_genes <- setdiff(genes, rownames(expression))
    variable_genes <- genes[genes %in% rownames(expression)]
    if (length(variable_genes) > 0L) {
      variances <- apply(
        as.matrix(expression[variable_genes, , drop = FALSE]),
        1L,
        var
      )
      invariant_genes <- names(variances)[!is.finite(variances) | variances <= 1e-12]
    } else {
      invariant_genes <- character(0)
    }
    status <- if (length(missing_genes) > 0L) {
      "missing_gene"
    } else if (length(invariant_genes) > 0L) {
      "invariant_gene"
    } else {
      "complete"
    }

    for (focus_i in seq_len(nrow(sample_focus))) {
      roi_id <- as.character(sample_focus$roi_id[[focus_i]])
      comparison_roi_id <- as.character(sample_focus$comparison_roi_id[[focus_i]])
      spot_index <- which(roi == roi_id)
      comparison_index <- which(roi == comparison_roi_id)
      row_i <- row_i + 1L
      base <- data.frame(
        candidate = candidate$candidate,
        topic = candidate$topic,
        architecture = candidate$architecture,
        scope = candidate$scope,
        sample_dir = sample_name,
        roi_id = roi_id,
        comparison_roi_id = comparison_roi_id,
        biological_context = sample_focus$biological_context[[focus_i]],
        n_spots = length(spot_index),
        n_genes = length(genes),
        candidate_status = status,
        missing_genes = paste(missing_genes, collapse = "|"),
        invariant_genes = paste(invariant_genes, collapse = "|"),
        stringsAsFactors = FALSE
      )
      if (status != "complete") {
        rows[[row_i]] <- cbind(
          base,
          minimum_gene_detection = NA_real_,
          mean_gene_detection = NA_real_,
          mean_abs_gene_detection_delta = NA_real_,
          raw_gene_mean_contrast = NA_real_,
          fraction_spots_at_least_half_genes = NA_real_,
          fraction_spots_all_genes = NA_real_,
          mean_fraction_genes_detected_per_spot = NA_real_,
          mean_pairwise_spearman = NA_real_,
          minimum_pairwise_spearman = NA_real_,
          mean_pairwise_both_detected_fraction = NA_real_,
          minimum_pairwise_both_detected_fraction = NA_real_
        )
        next
      }

      values <- as.matrix(expression[genes, spot_index, drop = FALSE])
      comparison_values <- as.matrix(
        expression[genes, comparison_index, drop = FALSE]
      )
      detected <- values > 0
      comparison_detected <- comparison_values > 0
      detected_count <- colSums(detected)
      gene_detection <- rowMeans(detected)
      comparison_gene_detection <- rowMeans(comparison_detected)
      if (length(genes) >= 2L) {
        pair_indices <- combn(seq_along(genes), 2L, simplify = FALSE)
        pair_spearman <- vapply(
          pair_indices,
          function(pair) safe_cor(values[pair[[1]], ], values[pair[[2]], ]),
          numeric(1)
        )
        pair_codetection <- vapply(
          pair_indices,
          function(pair) mean(detected[pair[[1]], ] & detected[pair[[2]], ]),
          numeric(1)
        )
        finite_spearman <- pair_spearman[is.finite(pair_spearman)]
      } else {
        finite_spearman <- numeric(0)
        pair_codetection <- numeric(0)
      }
      rows[[row_i]] <- cbind(
        base,
        minimum_gene_detection = min(gene_detection),
        mean_gene_detection = mean(gene_detection),
        mean_abs_gene_detection_delta = mean(
          abs(gene_detection - comparison_gene_detection)
        ),
        raw_gene_mean_contrast = mean(values) - mean(comparison_values),
        fraction_spots_at_least_half_genes = mean(
          detected_count >= ceiling(length(genes) / 2)
        ),
        fraction_spots_all_genes = mean(detected_count == length(genes)),
        mean_fraction_genes_detected_per_spot = mean(detected_count / length(genes)),
        mean_pairwise_spearman = if (length(finite_spearman) > 0L) mean(finite_spearman) else NA_real_,
        minimum_pairwise_spearman = if (length(finite_spearman) > 0L) min(finite_spearman) else NA_real_,
        mean_pairwise_both_detected_fraction = if (length(pair_codetection) > 0L) mean(pair_codetection) else NA_real_,
        minimum_pairwise_both_detected_fraction = if (length(pair_codetection) > 0L) min(pair_codetection) else NA_real_
      )
    }
  }
  rm(object, expression, metadata)
  gc()
}

write.csv(do.call(rbind, rows), output_path, row.names = FALSE)

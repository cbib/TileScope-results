#!/usr/bin/env Rscript

# Score reduced mouse-GBM panel candidates under the frozen Seurat contract.
# Multi-gene panels use AddModuleScore separately with fixed parameters.
# Single-gene comparators use LogNormalized expression directly and are never
# labelled programmes. All comparisons are within sample and focus ROI.

suppressPackageStartupMessages({
  library(Seurat)
  library(yaml)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
  stop(
    "usage: phase5_score_candidate_panels.R ",
    "candidates.yaml samples.csv focus_contexts.csv data_root output_dir"
  )
}

catalogue <- yaml::read_yaml(args[[1]])
samples <- read.csv(args[[2]], check.names = FALSE, stringsAsFactors = FALSE)
focus <- read.csv(args[[3]], stringsAsFactors = FALSE)
data_root <- args[[4]]
output_dir <- args[[5]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

safe_cor <- function(x, y) {
  keep <- is.finite(x) & is.finite(y)
  if (sum(keep) < 3L || sd(x[keep]) == 0 || sd(y[keep]) == 0) return(NA_real_)
  suppressWarnings(cor(x[keep], y[keep], method = "spearman"))
}

same_nonzero_sign <- function(x, y, tolerance = 1e-12) {
  if (!is.finite(x) || !is.finite(y)) return(NA)
  sign_x <- if (abs(x) <= tolerance) 0L else sign(x)
  sign_y <- if (abs(y) <= tolerance) 0L else sign(y)
  sign_x == sign_y
}

score_panel <- function(object, genes, assay, prefix) {
  set.seed(1234)
  scored <- AddModuleScore(
    object = object,
    features = list(genes),
    pool = rownames(LayerData(object, assay = assay, layer = "data")),
    nbin = 24,
    ctrl = 50,
    seed = 1234,
    assay = assay,
    name = prefix
  )
  as.numeric(scored[[]][[paste0(prefix, "1")]])
}

roi_rows <- list()
loo_rows <- list()
spot_rows <- list()
similarity_rows <- list()
roi_i <- 0L
loo_i <- 0L
spot_i <- 0L
similarity_i <- 0L

for (sample_i in seq_len(nrow(samples))) {
  sample_name <- samples$sample_dir[[sample_i]]
  message("[score candidates] ", sample_name)
  object <- readRDS(file.path(data_root, samples$rds_relpath[[sample_i]]))
  assay <- samples$assay[[sample_i]]
  expression <- LayerData(object, assay = assay, layer = "data")
  metadata <- object[[]]
  if (!identical(colnames(expression), rownames(metadata))) {
    stop("expression and metadata are not aligned for ", sample_name)
  }
  roi <- as.character(metadata[[samples$bayesspace_column[[sample_i]]]])
  image_names <- names(object@images)
  if (length(image_names) != 1L) stop("expected one spatial image for ", sample_name)
  coordinates <- GetTissueCoordinates(object, image = image_names[[1]])
  if (!all(colnames(expression) %in% rownames(coordinates))) {
    stop("coordinates do not cover expression spots for ", sample_name)
  }
  coordinates <- coordinates[colnames(expression), c("x", "y"), drop = FALSE]
  sample_focus <- focus[focus$sample_dir == sample_name, , drop = FALSE]
  sample_candidates <- Filter(
    function(candidate) sample_name %in% unlist(candidate$intended_samples),
    catalogue$candidates
  )

  score_bank <- list()
  raw_bank <- list()
  for (candidate in sample_candidates) {
    candidate_name <- candidate$candidate
    genes <- unlist(candidate$genes)
    missing <- setdiff(genes, rownames(expression))
    if (length(missing) > 0L) {
      stop(candidate_name, " missing in ", sample_name, ": ", paste(missing, collapse = ", "))
    }
    raw_mean <- colMeans(as.matrix(expression[genes, , drop = FALSE]))
    score <- if (length(genes) == 1L) {
      raw_mean
    } else {
      score_panel(object, genes, assay, paste0("tscandidate_", candidate_name, "_"))
    }
    score_bank[[candidate_name]] <- score
    raw_bank[[candidate_name]] <- raw_mean

    spot_i <- spot_i + 1L
    spot_rows[[spot_i]] <- data.frame(
      sample_dir = sample_name,
      spot_id = colnames(expression),
      roi_id = roi,
      x = coordinates$x,
      y = coordinates$y,
      candidate = candidate_name,
      topic = candidate$topic,
      architecture = candidate$architecture,
      score_method = if (length(genes) == 1L) "single_gene_lognormalized" else "AddModuleScore",
      score = score,
      raw_gene_mean = raw_mean,
      stringsAsFactors = FALSE
    )

    for (focus_i in seq_len(nrow(sample_focus))) {
      target_roi <- as.character(sample_focus$roi_id[[focus_i]])
      comparison_roi <- as.character(sample_focus$comparison_roi_id[[focus_i]])
      target <- which(roi == target_roi)
      comparison <- which(roi == comparison_roi)
      score_contrast <- mean(score[target]) - mean(score[comparison])
      raw_contrast <- mean(raw_mean[target]) - mean(raw_mean[comparison])
      roi_i <- roi_i + 1L
      roi_rows[[roi_i]] <- data.frame(
        candidate = candidate_name,
        topic = candidate$topic,
        architecture = candidate$architecture,
        scope = candidate$scope,
        sample_dir = sample_name,
        roi_id = target_roi,
        comparison_roi_id = comparison_roi,
        biological_context = sample_focus$biological_context[[focus_i]],
        n_spots = length(target),
        score_method = if (length(genes) == 1L) "single_gene_lognormalized" else "AddModuleScore",
        score_target_mean = mean(score[target]),
        score_comparison_mean = mean(score[comparison]),
        score_contrast = score_contrast,
        raw_mean_target = mean(raw_mean[target]),
        raw_mean_comparison = mean(raw_mean[comparison]),
        raw_mean_contrast = raw_contrast,
        score_raw_spearman = safe_cor(score[target], raw_mean[target]),
        score_raw_contrast_sign_agreement = same_nonzero_sign(score_contrast, raw_contrast),
        score_variance = var(score[target]),
        raw_mean_variance = var(raw_mean[target]),
        stringsAsFactors = FALSE
      )
    }

    if (length(genes) >= 3L) {
      for (dropped_gene in genes) {
        loo_genes <- setdiff(genes, dropped_gene)
        loo_score <- score_panel(
          object,
          loo_genes,
          assay,
          paste0("tsloo_", candidate_name, "_", make.names(dropped_gene), "_")
        )
        for (focus_i in seq_len(nrow(sample_focus))) {
          target_roi <- as.character(sample_focus$roi_id[[focus_i]])
          comparison_roi <- as.character(sample_focus$comparison_roi_id[[focus_i]])
          target <- which(roi == target_roi)
          comparison <- which(roi == comparison_roi)
          full_contrast <- mean(score[target]) - mean(score[comparison])
          loo_contrast <- mean(loo_score[target]) - mean(loo_score[comparison])
          loo_i <- loo_i + 1L
          loo_rows[[loo_i]] <- data.frame(
            candidate = candidate_name,
            sample_dir = sample_name,
            roi_id = target_roi,
            dropped_gene = dropped_gene,
            full_vs_loo_spearman = safe_cor(score[target], loo_score[target]),
            full_contrast = full_contrast,
            loo_contrast = loo_contrast,
            contrast_sign_preserved = same_nonzero_sign(full_contrast, loo_contrast),
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }

  for (candidate in sample_candidates) {
    candidate_name <- candidate$candidate
    reference_name <- catalogue$reference_by_topic[[candidate$topic]]
    if (is.null(score_bank[[reference_name]])) next
    score <- score_bank[[candidate_name]]
    reference <- score_bank[[reference_name]]
    for (focus_i in seq_len(nrow(sample_focus))) {
      target_roi <- as.character(sample_focus$roi_id[[focus_i]])
      comparison_roi <- as.character(sample_focus$comparison_roi_id[[focus_i]])
      target <- which(roi == target_roi)
      comparison <- which(roi == comparison_roi)
      candidate_contrast <- mean(score[target]) - mean(score[comparison])
      reference_contrast <- mean(reference[target]) - mean(reference[comparison])
      similarity_i <- similarity_i + 1L
      similarity_rows[[similarity_i]] <- data.frame(
        candidate = candidate_name,
        reference_candidate = reference_name,
        sample_dir = sample_name,
        roi_id = target_roi,
        candidate_reference_spearman = safe_cor(score[target], reference[target]),
        candidate_contrast = candidate_contrast,
        reference_contrast = reference_contrast,
        contrast_difference = candidate_contrast - reference_contrast,
        contrast_sign_agreement = same_nonzero_sign(candidate_contrast, reference_contrast),
        stringsAsFactors = FALSE
      )
    }
  }
  rm(object, expression, metadata, score_bank, raw_bank)
  gc()
}

write.csv(do.call(rbind, roi_rows), file.path(output_dir, "candidate_score_roi_audit.csv"), row.names = FALSE)
write.csv(
  if (length(loo_rows) > 0L) do.call(rbind, loo_rows) else data.frame(),
  file.path(output_dir, "candidate_score_loo_audit.csv"),
  row.names = FALSE
)
write.csv(do.call(rbind, spot_rows), file.path(output_dir, "candidate_spot_scores.csv"), row.names = FALSE)
write.csv(do.call(rbind, similarity_rows), file.path(output_dir, "candidate_score_reference_similarity.csv"), row.names = FALSE)

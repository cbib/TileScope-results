#!/usr/bin/env Rscript

# Rebuild the Seurat marker views used by the mouse-GBM signature workflow.
#
# This script owns retrieval only.  Seurat marker p-values are descriptive in
# spatially autocorrelated data; later phases use marker ranks to retrieve
# candidates and independently evaluate same-spot and spatial coherence.

suppressPackageStartupMessages({
  library(Seurat)
})

args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(2L, 3L))) {
  stop(
    "usage: phase1_2_build_marker_views.R samples.csv data_root [output_root]"
  )
}

samples_path <- args[[1]]
data_root <- args[[2]]
output_root <- if (length(args) == 3L) {
  args[[3]]
} else {
  file.path(data_root, "mouse_gbm_spatial_programs")
}
phase1_root <- file.path(output_root, "phase1")
phase2_root <- file.path(output_root, "phase2")
dir.create(phase1_root, recursive = TRUE, showWarnings = FALSE)
dir.create(phase2_root, recursive = TRUE, showWarnings = FALSE)

samples <- read.csv(samples_path, check.names = FALSE, stringsAsFactors = FALSE)

phase1_marker_args <- list(
  assay = "Spatial",
  slot = "data",
  test.use = "wilcox",
  min.pct = 0.1,
  logfc.threshold = 0.25,
  min.diff.pct = 0.1
)

phase2_marker_args <- list(
  assay = "Spatial",
  slot = "data",
  test.use = "wilcox",
  min.pct = 0,
  logfc.threshold = 0,
  min.diff.pct = 0,
  verbose = FALSE
)

logfc_column <- function(table) {
  candidates <- c("avg_log2FC", "avg_logFC")
  found <- candidates[candidates %in% colnames(table)]
  if (length(found) != 1L) {
    stop("expected one Seurat log-fold-change column; got ", paste(found, collapse = ", "))
  }
  found[[1]]
}

standardize_markers <- function(
    table,
    positive_label,
    negative_label,
    ranking_groups = character(0)) {
  if (!("gene" %in% colnames(table))) {
    table$gene <- rownames(table)
  }
  fc <- logfc_column(table)
  if (fc != "avg_log2FC") {
    colnames(table)[colnames(table) == fc] <- "avg_log2FC"
  }
  table$pct_delta <- table$pct.1 - table$pct.2
  table$retrieval_score <- abs(table$avg_log2FC) * abs(table$pct_delta)
  table$marker_direction <- ifelse(
    table$avg_log2FC >= 0,
    positive_label,
    negative_label
  )
  group_columns <- c(ranking_groups, "marker_direction")
  group_key <- interaction(table[, group_columns, drop = FALSE], drop = TRUE)
  table$rank_within_direction <- ave(
    -table$retrieval_score,
    group_key,
    FUN = function(x) rank(x, ties.method = "first")
  )
  table$rank_within_direction <- as.integer(table$rank_within_direction)
  table
}

load_sample <- function(row) {
  object <- readRDS(file.path(data_root, row$rds_relpath))
  if (!(row$assay %in% Assays(object))) {
    stop(row$sample_dir, " lacks assay ", row$assay)
  }
  values <- LayerData(object, assay = row$assay, layer = "data")
  if (max(values) >= 30) {
    stop(row$sample_dir, " Spatial/data does not look LogNormalized")
  }
  object
}

find_pair_markers <- function(object, ident_column, ident1, ident2, parameters) {
  Idents(object) <- as.character(object[[]][[ident_column]])
  result <- do.call(
    FindMarkers,
    c(
      list(object = object, ident.1 = as.character(ident1), ident.2 = as.character(ident2)),
      parameters
    )
  )
  standardize_markers(result, "ident1_enriched", "ident2_enriched")
}

objects <- list()

# Phase 1: every BayesSpace ROI versus the rest, plus the two highlighted
# within-sample contrasts and the labelled ROI map.
for (sample_i in seq_len(nrow(samples))) {
  row <- samples[sample_i, , drop = FALSE]
  sample_name <- row$sample_dir[[1]]
  message("[phase1] ", sample_name)
  object <- load_sample(row)
  objects[[sample_name]] <- object
  roi_column <- row$bayesspace_column[[1]]
  Idents(object) <- factor(as.character(object[[]][[roi_column]]))

  all_markers <- do.call(
    FindAllMarkers,
    c(
      list(object = object, only.pos = FALSE, return.thresh = Inf),
      phase1_marker_args
    )
  )
  all_markers <- standardize_markers(
    all_markers,
    "roi_enriched",
    "roi_depleted",
    ranking_groups = "cluster"
  )
  all_markers$sample_dir <- sample_name

  sample_output <- file.path(phase1_root, sample_name)
  dir.create(sample_output, recursive = TRUE, showWarnings = FALSE)
  write.csv(
    all_markers,
    file.path(sample_output, "one_vs_rest_markers.csv"),
    row.names = FALSE
  )
  roi_summary <- as.data.frame(table(
    bayesspace_cluster = as.character(object[[]][[roi_column]])
  ))
  colnames(roi_summary)[[2]] <- "n_spots"
  write.csv(roi_summary, file.path(sample_output, "roi_summary.csv"), row.names = FALSE)

  pdf(file.path(sample_output, "roi_map.pdf"), width = 9, height = 7)
  print(SpatialDimPlot(object, group.by = roi_column, label = TRUE, repel = TRUE))
  dev.off()

  core <- row$highlighted_core_bayesspace[[1]]
  periphery <- row$highlighted_peri_bayesspace[[1]]
  if (!is.na(core) && !is.na(periphery) && core != "" && periphery != "") {
    highlighted <- find_pair_markers(
      object,
      roi_column,
      core,
      periphery,
      phase1_marker_args
    )
    highlighted$core_bayesspace <- as.character(core)
    highlighted$peri_bayesspace <- as.character(periphery)
    highlighted$sample_dir <- sample_name
    highlighted <- highlighted[, c(
      "p_val", "avg_log2FC", "pct.1", "pct.2", "p_val_adj", "gene",
      "pct_delta", "retrieval_score", "marker_direction",
      "rank_within_direction", "sample_dir", "core_bayesspace",
      "peri_bayesspace"
    )]
    write.csv(
      highlighted,
      file.path(sample_output, "highlighted_contrast_markers.csv"),
      row.names = FALSE
    )
  }
}

contrast_specs <- list(
  list("within_core_periphery", "CTL_2_vs_5", "MCT1_CTL", "2", "MCT1_CTL", "5"),
  list("within_core_periphery", "KO_4_vs_9", "MCT1_KO", "4", "MCT1_KO", "9"),
  list("matched_ctl_ko_compartment", "CTL_2_vs_KO_4_core", "MCT1_CTL", "2", "MCT1_KO", "4"),
  list("matched_ctl_ko_compartment", "CTL_5_vs_KO_9_periphery", "MCT1_CTL", "5", "MCT1_KO", "9"),
  list("v4_intratumour", "V4_10_vs_1", "V4_WT", "10", "V4_WT", "1"),
  list("v4_tumour_reactive", "V4_10_vs_2", "V4_WT", "10", "V4_WT", "2"),
  list("v4_tumour_reactive", "V4_1_vs_2", "V4_WT", "1", "V4_WT", "2")
)

phase2_rows <- list()
for (contrast_i in seq_along(contrast_specs)) {
  spec <- contrast_specs[[contrast_i]]
  family <- spec[[1]]
  label <- spec[[2]]
  sample1 <- spec[[3]]
  roi1 <- spec[[4]]
  sample2 <- spec[[5]]
  roi2 <- spec[[6]]
  message("[phase2] ", label)

  if (sample1 == sample2) {
    sample_row <- samples[samples$sample_dir == sample1, , drop = FALSE]
    markers <- find_pair_markers(
      objects[[sample1]],
      sample_row$bayesspace_column[[1]],
      roi1,
      roi2,
      phase2_marker_args
    )
  } else {
    row1 <- samples[samples$sample_dir == sample1, , drop = FALSE]
    row2 <- samples[samples$sample_dir == sample2, , drop = FALSE]
    object1 <- objects[[sample1]]
    object2 <- objects[[sample2]]
    cells1 <- rownames(object1[[]])[
      as.character(object1[[]][[row1$bayesspace_column[[1]]]]) == roi1
    ]
    cells2 <- rownames(object2[[]])[
      as.character(object2[[]][[row2$bayesspace_column[[1]]]]) == roi2
    ]
    shared_features <- intersect(
      rownames(object1[[row1$assay[[1]]]]),
      rownames(object2[[row2$assay[[1]]]])
    )
    subset1 <- subset(object1, cells = cells1, features = shared_features)
    subset2 <- subset(object2, cells = cells2, features = shared_features)
    subset1$contrast_ident <- "ident1"
    subset2$contrast_ident <- "ident2"
    merged <- merge(subset1, y = subset2, merge.data = TRUE)
    merged <- JoinLayers(merged, assay = "Spatial")
    markers <- find_pair_markers(
      merged,
      "contrast_ident",
      "ident1",
      "ident2",
      phase2_marker_args
    )
    rm(subset1, subset2, merged)
  }

  markers$contrast_family <- family
  markers$contrast_label <- label
  markers$sample_scope <- if (sample1 == sample2) sample1 else paste(sample1, sample2, sep = "+")
  markers$ident1_sample <- sample1
  markers$ident1_cluster <- roi1
  markers$ident2_sample <- sample2
  markers$ident2_cluster <- roi2
  phase2_rows[[contrast_i]] <- markers
}

phase2 <- do.call(rbind, phase2_rows)
phase2_columns <- c(
  "contrast_family", "contrast_label", "sample_scope",
  "ident1_sample", "ident1_cluster", "ident2_sample", "ident2_cluster",
  "gene", "p_val", "avg_log2FC", "pct.1", "pct.2", "p_val_adj",
  "marker_direction", "retrieval_score", "rank_within_direction"
)
write.csv(
  phase2[, phase2_columns],
  file.path(phase2_root, "phase2_markers.csv"),
  row.names = FALSE
)
summary_rows <- lapply(seq_along(contrast_specs), function(i) {
  spec <- contrast_specs[[i]]
  sample1 <- spec[[3]]
  roi1 <- spec[[4]]
  sample2 <- spec[[5]]
  roi2 <- spec[[6]]
  row1 <- samples[samples$sample_dir == sample1, , drop = FALSE]
  row2 <- samples[samples$sample_dir == sample2, , drop = FALSE]
  n1 <- sum(
    as.character(objects[[sample1]][[]][[row1$bayesspace_column[[1]]]]) == roi1
  )
  n2 <- sum(
    as.character(objects[[sample2]][[]][[row2$bayesspace_column[[1]]]]) == roi2
  )
  markers <- phase2_rows[[i]]
  data.frame(
    contrast_family = spec[[1]],
    contrast_label = spec[[2]],
    ident1_sample = sample1,
    ident1_cluster = roi1,
    ident1_n_spots = n1,
    ident2_sample = sample2,
    ident2_cluster = roi2,
    ident2_n_spots = n2,
    n_genes_total = nrow(markers),
    n_genes_ident1_enriched = sum(markers$marker_direction == "ident1_enriched"),
    n_genes_ident2_enriched = sum(markers$marker_direction == "ident2_enriched"),
    stringsAsFactors = FALSE
  )
})
summary <- do.call(rbind, summary_rows)
write.csv(summary, file.path(phase2_root, "phase2_run_summary.csv"), row.names = FALSE)

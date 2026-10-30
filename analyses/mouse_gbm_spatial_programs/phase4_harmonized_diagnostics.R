#!/usr/bin/env Rscript

# Harmonized mouse-GBM gene diagnostics.
#
# Candidate genes must already have been retrieved by Seurat in one of the
# predeclared focus views.  This script evaluates them only in the biological
# focus ROIs used to define the panels.  Other ROIs are held back for transfer
# evaluation after the panels are frozen.

suppressPackageStartupMessages({
  library(FNN)
  library(igraph)
  library(Matrix)
  library(Seurat)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop(
    "usage: phase4_harmonized_diagnostics.R ",
    "samples.csv candidates.csv data_root output_dir"
  )
}

samples_path <- args[[1]]
candidates_path <- args[[2]]
data_root <- args[[3]]
output_dir <- args[[4]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

samples <- read.csv(samples_path, check.names = FALSE, stringsAsFactors = FALSE)
candidates <- unique(read.csv(candidates_path, stringsAsFactors = FALSE))

focus <- data.frame(
  sample_dir = c(
    "MCT1_CTL", "MCT1_CTL", "MCT1_KO", "MCT1_KO",
    "V4_WT", "V4_WT", "V4_WT"
  ),
  roi_id = c("2", "5", "4", "9", "10", "2", "1"),
  comparison_roi_id = c("5", "2", "9", "4", "2", "10", "10"),
  biological_context = c(
    "highlighted_tumour_core",
    "highlighted_tumour_periphery",
    "highlighted_tumour_core",
    "highlighted_tumour_periphery",
    "tumour_core",
    "reactive_tumour_brain_interface",
    "proliferative_intratumour_subregion"
  ),
  stringsAsFactors = FALSE
)

safe_cor <- function(x, y) {
  if (length(x) < 3L || sd(x) == 0 || sd(y) == 0) return(NA_real_)
  suppressWarnings(cor(x, y, method = "spearman"))
}

jaccard <- function(x, y) {
  x_detected <- x > 0
  y_detected <- y > 0
  union_n <- sum(x_detected | y_detected)
  if (union_n == 0L) return(NA_real_)
  sum(x_detected & y_detected) / union_n
}

build_source_graph <- function(object, expression_columns) {
  image_names <- names(object@images)
  if (length(image_names) != 1L) {
    stop("expected exactly one spatial image; got ", length(image_names))
  }
  coordinates <- GetTissueCoordinates(object, image = image_names[[1]])
  if (!all(expression_columns %in% rownames(coordinates))) {
    stop("spatial coordinates do not cover every expression column")
  }
  xy <- as.matrix(coordinates[expression_columns, c("x", "y"), drop = FALSE])
  neighbour_count <- min(6L, nrow(xy) - 1L)
  nearest <- FNN::get.knn(xy, k = neighbour_count)
  distance_threshold <- 1.5 * median(nearest$nn.dist[, 1L])
  from <- rep(seq_len(nrow(xy)), each = neighbour_count)
  to <- as.vector(t(nearest$nn.index))
  distance <- as.vector(t(nearest$nn.dist))
  keep <- distance <= distance_threshold
  edges <- unique(data.frame(
    from = pmin(from[keep], to[keep]),
    to = pmax(from[keep], to[keep])
  ))
  graph_from_data_frame(
    edges,
    directed = FALSE,
    vertices = data.frame(name = as.character(seq_len(nrow(xy))))
  )
}

availability_rows <- list()
gene_rows <- list()
pair_rows <- list()
gene_row_i <- 0L
pair_row_i <- 0L

for (sample_i in seq_len(nrow(samples))) {
  sample_name <- samples$sample_dir[[sample_i]]
  message("[harmonized diagnostics] ", sample_name)
  object <- readRDS(file.path(data_root, samples$rds_relpath[[sample_i]]))
  expression <- LayerData(
    object,
    assay = samples$assay[[sample_i]],
    layer = "data"
  )
  metadata <- object[[]]
  if (!identical(colnames(expression), rownames(metadata))) {
    stop("expression and metadata rows are not aligned for ", sample_name)
  }
  roi <- as.character(metadata[[samples$bayesspace_column[[sample_i]]]])
  source_graph <- build_source_graph(object, colnames(expression))

  feature_mean <- Matrix::rowMeans(expression)
  feature_variance <- Matrix::rowMeans(expression ^ 2) - feature_mean ^ 2
  feature_detection <- Matrix::rowMeans(expression > 0)
  present_index <- match(candidates$gene, rownames(expression))
  availability_rows[[sample_i]] <- data.frame(
    candidates,
    sample_dir = sample_name,
    present = !is.na(present_index),
    variable = !is.na(present_index) & feature_variance[present_index] > 1e-12,
    sample_detection_fraction = ifelse(
      is.na(present_index), NA_real_, feature_detection[present_index]
    ),
    stringsAsFactors = FALSE
  )

  sample_focus <- focus[focus$sample_dir == sample_name, , drop = FALSE]
  for (focus_i in seq_len(nrow(sample_focus))) {
    target_roi <- sample_focus$roi_id[[focus_i]]
    comparison_roi <- sample_focus$comparison_roi_id[[focus_i]]
    target_index <- which(roi == target_roi)
    comparison_index <- which(roi == comparison_roi)
    if (length(target_index) == 0L || length(comparison_index) == 0L) {
      stop("empty focus ROI for ", sample_name, ": ", target_roi, " vs ", comparison_roi)
    }
    target_graph <- igraph::induced_subgraph(
      source_graph,
      vids = as.character(target_index)
    )
    component_membership <- igraph::components(target_graph)$membership
    component_count <- length(unique(component_membership))

    for (topic_name in unique(candidates$topic)) {
      topic_candidates <- candidates[candidates$topic == topic_name, , drop = FALSE]
      present_genes <- topic_candidates$gene[
        topic_candidates$gene %in% rownames(expression)
      ]
      if (length(present_genes) == 0L) next
      topic_expression <- as.matrix(expression[present_genes, , drop = FALSE])
      target_expression <- topic_expression[, target_index, drop = FALSE]
      comparison_expression <- topic_expression[, comparison_index, drop = FALSE]

      for (gene_i in seq_along(present_genes)) {
        gene_name <- present_genes[[gene_i]]
        target_values <- target_expression[gene_i, ]
        comparison_values <- comparison_expression[gene_i, ]
        detected_vertices <- as.character(target_index[target_values > 0])
        covered_components <- unique(component_membership[detected_vertices])
        gene_row_i <- gene_row_i + 1L
        gene_rows[[gene_row_i]] <- data.frame(
          topic = topic_name,
          sample_dir = sample_name,
          roi_id = target_roi,
          comparison_roi_id = comparison_roi,
          biological_context = sample_focus$biological_context[[focus_i]],
          gene = gene_name,
          annotation_confidence = topic_candidates$annotation_confidence[
            match(gene_name, topic_candidates$gene)
          ],
          n_spots_target = length(target_values),
          n_spots_comparison = length(comparison_values),
          detection_target = mean(target_values > 0),
          detection_comparison = mean(comparison_values > 0),
          mean_expression_target = mean(target_values),
          mean_expression_comparison = mean(comparison_values),
          variance_target = var(target_values),
          n_components_total = component_count,
          n_components_covered = length(covered_components),
          component_coverage_fraction = length(covered_components) / component_count,
          stringsAsFactors = FALSE
        )
      }

      if (length(present_genes) >= 2L) {
        combinations <- combn(present_genes, 2L, simplify = FALSE)
        for (genes in combinations) {
          x <- target_expression[match(genes[[1]], present_genes), ]
          y <- target_expression[match(genes[[2]], present_genes), ]
          pair_row_i <- pair_row_i + 1L
          pair_rows[[pair_row_i]] <- data.frame(
            topic = topic_name,
            sample_dir = sample_name,
            roi_id = target_roi,
            biological_context = sample_focus$biological_context[[focus_i]],
            gene_i = genes[[1]],
            gene_j = genes[[2]],
            spearman = safe_cor(x, y),
            jaccard = jaccard(x, y),
            both_detected_fraction = mean(x > 0 & y > 0),
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }
  rm(object, expression, metadata, source_graph)
  gc()
}

write.csv(
  do.call(rbind, availability_rows),
  file.path(output_dir, "gene_availability.csv"),
  row.names = FALSE
)
write.csv(
  do.call(rbind, gene_rows),
  file.path(output_dir, "focus_gene_diagnostics.csv"),
  row.names = FALSE
)
write.csv(
  do.call(rbind, pair_rows),
  file.path(output_dir, "focus_pairwise_diagnostics.csv"),
  row.names = FALSE
)
write.csv(focus, file.path(output_dir, "focus_contexts.csv"), row.names = FALSE)

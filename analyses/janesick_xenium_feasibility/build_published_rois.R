#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript build_published_rois.R <sample1_rep1_dir> <output_dir>")
}

suppressPackageStartupMessages(library(readxl))

sample_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

source_workbook <- file.path(
  sample_dir,
  "paper_source",
  "41467_2023_43458_MOESM4_ESM.xlsx"
)
cells_file <- file.path(sample_dir, "xenium", "outs", "cells.csv.gz")
annotations_file <- file.path(
  sample_dir,
  "annotations",
  "GSM7780153_Xenium_R1_Fig1-5_supervised.csv.gz"
)

paper <- read_excel(
  source_workbook,
  sheet = "Fig. 4b-d ",
  col_types = c("text", rep("guess", 37L))
)
cells <- read.csv(
  gzfile(cells_file),
  colClasses = c(cell_id = "character"),
  check.names = FALSE
)
annotations <- read.csv(
  gzfile(annotations_file),
  colClasses = c(Barcode = "character"),
  check.names = FALSE
)

# The published annotation CSV spells barcode 100000 as `1e+05`; normalize
# these numeric identifiers before validating exact membership.
normalize_cell_id <- function(values) {
  format(as.numeric(values), scientific = FALSE, trim = TRUE)
}
paper$Barcode <- normalize_cell_id(paper$Barcode)
cells$cell_id <- normalize_cell_id(cells$cell_id)
annotations$Barcode <- normalize_cell_id(annotations$Barcode)

required_paper_columns <- c(
  "Barcode", "ROI_Membership", "celltype", "x_centroid", "y_centroid"
)
if (!all(required_paper_columns %in% names(paper))) {
  stop("The expected Fig. 4b-d source-data columns are absent")
}
if (anyDuplicated(paper$Barcode) || anyDuplicated(cells$cell_id) ||
    anyDuplicated(annotations$Barcode)) {
  stop("Cell identifiers are not unique")
}

paper_to_cells <- match(paper$Barcode, cells$cell_id)
paper_to_annotations <- match(paper$Barcode, annotations$Barcode)
cells_to_paper <- match(cells$cell_id, paper$Barcode)
cells_to_annotations <- match(cells$cell_id, annotations$Barcode)

if (anyNA(paper_to_cells) || anyNA(paper_to_annotations) ||
    anyNA(cells_to_annotations)) {
  stop("Paper, Xenium, and annotation cell identifiers do not align")
}

max_x_difference <- max(
  abs(paper$x_centroid - cells$x_centroid[paper_to_cells]),
  na.rm = TRUE
)
max_y_difference <- max(
  abs(paper$y_centroid - cells$y_centroid[paper_to_cells]),
  na.rm = TRUE
)
cell_type_disagreements <- sum(
  paper$celltype != annotations$Cluster[paper_to_annotations],
  na.rm = TRUE
)
if (max_x_difference > 1e-6 || max_y_difference > 1e-6) {
  stop("Paper centroids do not match the downloaded Xenium cell centroids")
}
if (cell_type_disagreements != 0L) {
  stop("Paper cell types do not match the downloaded supervised annotations")
}

paper$roi_id <- ifelse(
  grepl("^DCIS1", paper$ROI_Membership),
  "DCIS_1",
  ifelse(
    grepl("^DCIS2", paper$ROI_Membership),
    "DCIS_2",
    ifelse(grepl("^Invasive", paper$ROI_Membership), "Invasive", NA_character_)
  )
)
if (anyNA(paper$roi_id)) {
  stop("An unrecognized published ROI membership was found")
}

inside <- !is.na(cells_to_paper)
roi_id <- rep("Outside_published_ROIs", nrow(cells))
roi_subregion_id <- rep("", nrow(cells))
roi_id[inside] <- paper$roi_id[cells_to_paper[inside]]
roi_subregion_id[inside] <- paper$ROI_Membership[cells_to_paper[inside]]

assignments <- data.frame(
  cell_id = cells$cell_id,
  x = cells$x_centroid,
  y = cells$y_centroid,
  roi_id = roi_id,
  roi_subregion_id = roi_subregion_id,
  cell_type = annotations$Cluster[cells_to_annotations],
  in_published_roi = inside,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

write_gzip_csv <- function(frame, path) {
  connection <- gzfile(path, open = "wt")
  on.exit(close(connection), add = TRUE)
  write.csv(frame, connection, row.names = FALSE, na = "")
}

write_gzip_csv(
  assignments,
  file.path(output_dir, "cell_roi_assignments.csv.gz")
)
write_gzip_csv(
  assignments[assignments$in_published_roi, , drop = FALSE],
  file.path(output_dir, "published_roi_cells.csv.gz")
)

selected <- assignments[assignments$in_published_roi, , drop = FALSE]
parent_ids <- c("DCIS_1", "DCIS_2", "Invasive")

parent_summary <- do.call(
  rbind,
  lapply(parent_ids, function(parent_id) {
    current <- selected[selected$roi_id == parent_id, , drop = FALSE]
    data.frame(
      roi_id = parent_id,
      n_cells = nrow(current),
      n_subregions = length(unique(current$roi_subregion_id)),
      fraction_all_cells = nrow(current) / nrow(assignments),
      min_x = min(current$x),
      max_x = max(current$x),
      min_y = min(current$y),
      max_y = max(current$y),
      stringsAsFactors = FALSE
    )
  })
)
write.csv(
  parent_summary,
  file.path(output_dir, "roi_parent_summary.csv"),
  row.names = FALSE
)

subregion_ids <- sort(unique(selected$roi_subregion_id))
subregion_summary <- do.call(
  rbind,
  lapply(subregion_ids, function(subregion_id) {
    current <- selected[selected$roi_subregion_id == subregion_id, , drop = FALSE]
    data.frame(
      roi_subregion_id = subregion_id,
      roi_id = unique(current$roi_id),
      n_cells = nrow(current),
      min_x = min(current$x),
      max_x = max(current$x),
      min_y = min(current$y),
      max_y = max(current$y),
      stringsAsFactors = FALSE
    )
  })
)
write.csv(
  subregion_summary,
  file.path(output_dir, "roi_subregion_summary.csv"),
  row.names = FALSE
)

cell_type_counts <- as.data.frame(
  table(selected$roi_id, selected$cell_type),
  stringsAsFactors = FALSE
)
names(cell_type_counts) <- c("roi_id", "cell_type", "n_cells")
cell_type_counts <- cell_type_counts[cell_type_counts$n_cells > 0L, ]
denominators <- setNames(parent_summary$n_cells, parent_summary$roi_id)
cell_type_counts$fraction_within_roi <- (
  cell_type_counts$n_cells / denominators[cell_type_counts$roi_id]
)
cell_type_counts <- cell_type_counts[
  order(cell_type_counts$roi_id, -cell_type_counts$n_cells),
]
write.csv(
  cell_type_counts,
  file.path(output_dir, "cell_type_by_roi.csv"),
  row.names = FALSE
)

qc <- data.frame(
  metric = c(
    "downloaded_xenium_cells",
    "published_roi_cells",
    "outside_published_roi_cells",
    "unique_published_barcodes",
    "published_ids_missing_from_xenium",
    "max_abs_x_centroid_difference_um",
    "max_abs_y_centroid_difference_um",
    "published_vs_annotation_cell_type_disagreements"
  ),
  value = c(
    nrow(assignments),
    nrow(selected),
    sum(!inside),
    length(unique(paper$Barcode)),
    sum(is.na(paper_to_cells)),
    format(max_x_difference, scientific = TRUE, digits = 8L),
    format(max_y_difference, scientific = TRUE, digits = 8L),
    cell_type_disagreements
  ),
  stringsAsFactors = FALSE
)
write.csv(qc, file.path(output_dir, "source_data_validation.csv"), row.names = FALSE)

png(
  file.path(output_dir, "roi_assignment_overview.png"),
  width = 2400,
  height = 1800,
  res = 240
)
par(mar = c(4.2, 4.2, 3.2, 1.0))
plot(
  assignments$x,
  assignments$y,
  pch = ".",
  col = "#D9D9D980",
  asp = 1,
  ylim = rev(range(assignments$y)),
  xlab = "Xenium x (microns)",
  ylab = "Xenium y (microns)",
  main = "Janesick Sample 1 Replicate 1: published Figure 4 ROIs"
)
colors <- c(DCIS_1 = "#E69F00", DCIS_2 = "#CC79A7", Invasive = "#D55E00")
for (parent_id in parent_ids) {
  current <- selected$roi_id == parent_id
  points(selected$x[current], selected$y[current], pch = ".", col = colors[parent_id])
}
legend(
  "topright",
  legend = sprintf(
    "%s (n=%s)",
    parent_ids,
    format(parent_summary$n_cells, big.mark = ",")
  ),
  col = colors[parent_ids],
  pch = 16,
  pt.cex = 1.1,
  bty = "n"
)
dev.off()

message("Wrote exact published ROI assignments for ", nrow(selected), " cells")

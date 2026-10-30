#!/usr/bin/env Rscript
# Prepare the retained ACTA2/FASN comparison in three published parent ROIs.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript build_tilescope_inputs.R <sample1_rep1_dir> <output_dir>")
}
suppressPackageStartupMessages(library(readxl))
sample_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
csv_dir <- file.path(output_dir, "CSVs")
workbook <- file.path(sample_dir, "paper_source", "41467_2023_43458_MOESM4_ESM.xlsx")
paper <- read_excel(workbook, sheet = "Fig. 4b-d ",
                   col_types = c("text", rep("guess", 37L)))
assignments <- read.csv(gzfile(file.path(sample_dir, "roi", "published_roi_cells.csv.gz")),
                        colClasses = c(cell_id = "character"), check.names = FALSE)
normal_id <- function(values) format(as.numeric(values), scientific = FALSE, trim = TRUE)
paper$Barcode <- normal_id(paper$Barcode)
assignments$cell_id <- normal_id(assignments$cell_id)
required <- c("Barcode", "ROI_Membership", "x_centroid", "y_centroid", "ACTA2", "FASN")
stopifnot(all(required %in% names(paper)), !anyDuplicated(paper$Barcode),
          !anyNA(paper[required]), nrow(paper) == 87770L)
lassos <- c("DCIS1A", "DCIS1B", "DCIS1C", "DCIS1D", "DCIS1E",
            "DCIS2A", "DCIS2B", "DCIS2C", "InvasiveA")
stopifnot(setequal(unique(paper$ROI_Membership), lassos))
index <- match(paper$Barcode, assignments$cell_id)
stopifnot(!anyNA(index),
          all(paper$ROI_Membership == assignments$roi_subregion_id[index]),
          max(abs(paper$x_centroid - assignments$x[index])) < 1e-6,
          max(abs(paper$y_centroid - assignments$y[index])) < 1e-6)
paper <- paper[order(as.numeric(paper$Barcode)), ]
parents <- ifelse(grepl("^DCIS1", paper$ROI_Membership), "DCIS_1",
                  ifelse(grepl("^DCIS2", paper$ROI_Membership), "DCIS_2", "Invasive"))
input <- data.frame(x = paper$x_centroid, y = paper$y_centroid,
                    gene1 = paper$ACTA2, gene2 = paper$FASN,
                    roi_id = parents, stringsAsFactors = FALSE)
filename <- "Janesick_S1R1_expression_data_ACTA2_FASN_parent_rois.csv"
dir.create(csv_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(input, file.path(csv_dir, filename), row.names = FALSE)
write.csv(data.frame(pair_id = "ACTA2_FASN", filename = file.path("CSVs", filename),
                     feature_a_id = "ACTA2", feature_b_id = "FASN",
                     value_definition = "raw Xenium transcript count per segmented cell",
                     n_cells = nrow(input), n_rois = 3L,
                     rationale = "Published myoepithelial and invasive-tumour expression contrast"),
           file.path(output_dir, "pair_manifest.csv"), row.names = FALSE)
rois <- aggregate(list(n_cells = paper$Barcode), by = list(roi_id = parents), FUN = length)
write.csv(rois, file.path(output_dir, "roi_manifest.csv"), row.names = FALSE)
message("Wrote ACTA2/FASN: ", nrow(input), " cells in three parent ROIs")

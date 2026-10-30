# Shared single-sample preparation; also sourced by prepare_cohort.R.
suppressPackageStartupMessages(library(Matrix))

prepare_sample <- function(sample, labels, out, definition_path, expected_cells=NULL) {
  definition <- jsonlite::fromJSON(definition_path)
  ids <- rownames(sample$metadata)
  sample_id <- unique(as.character(sample$metadata$orig.ident))
  stopifnot(length(sample_id) == 1L, length(ids) > 0L, !anyDuplicated(ids),
            !anyDuplicated(labels$cell_id), setequal(ids, labels$cell_id))
  if (!is.null(expected_cells)) stopifnot(length(ids) == expected_cells)
  labels <- labels[match(ids, labels$cell_id),]
  stopifnot(!anyNA(labels$ann_CN), all(labels$orig.ident == sample_id),
            max(abs(labels$x - sample$metadata$x)) < 1e-8,
            max(abs(labels$y - sample$metadata$y)) < 1e-8)
  genes <- unique(c(definition$genes_a, definition$genes_b))
  stopifnot(identical(colnames(sample$data), ids), all(genes %in% rownames(sample$data)))
  values <- as.matrix(sample$data[genes, ids, drop=FALSE])
  counts <- as.matrix(sample$counts[genes, ids, drop=FALSE])
  expected <- log1p(sweep(counts, 2, sample$metadata$nCount_RNA, "/") * 100)
  stopifnot(all(is.finite(values)), all(values >= 0), max(abs(values-expected)) < 1e-10)
  input <- data.frame(cell_id=ids, x=sample$metadata$x, y=sample$metadata$y,
                      gene1=colMeans(values[definition$genes_a,,drop=FALSE]),
                      gene2=colMeans(values[definition$genes_b,,drop=FALSE]), roi_id=labels$ann_CN)
  target <- file.path(out, paste0(sample_id, "_expression_data_", definition$pair, ".csv"))
  if (file.exists(target)) {
    old <- read.csv(target, check.names=FALSE)
    stopifnot(identical(old$cell_id, input$cell_id), identical(old$roi_id, input$roi_id),
              isTRUE(all.equal(old[,c("x","y","gene1","gene2")], input[,c("x","y","gene1","gene2")],
                               tolerance=1e-13, check.attributes=FALSE)))
    for (f in c("cell_annotations.csv.gz", "signature_gene_values.csv.gz", "pair.json")) {
      stopifnot(file.exists(file.path(out, f)))
    }
    cat("Verified existing input, unchanged:", sample_id, "\n"); flush.console()
    return(invisible(input))
  }
  if (dir.exists(out) && length(list.files(out, all.files=TRUE, no..=TRUE))) stop("Output must be empty")
  dir.create(out, recursive=TRUE, showWarnings=FALSE)
  options(digits=17)
  write.csv(input, target, row.names=FALSE)
  annot <- data.frame(input, published_cluster=labels$published_cluster,
                      reconstructed_cluster=labels$reconstructed_cluster,
                      author_cell_type=sample$metadata$tier1,
                      author_Tcell_state=sample$metadata$tier1_T)
  con <- gzfile(file.path(out,"cell_annotations.csv.gz"), "wt")
  write.csv(annot, con, row.names=FALSE); close(con)
  con <- gzfile(file.path(out,"signature_gene_values.csv.gz"), "wt")
  write.csv(data.frame(cell_id=ids, t(values), check.names=FALSE), con, row.names=FALSE); close(con)
  file.copy(definition_path, file.path(out,"pair.json"), overwrite=FALSE)
  writeLines(c(paste("Sample:", sample_id, "; all", length(ids), "deposited cells retained."),
               definition$origin, definition$calculation,
               "ROI labels: author neighbourhood reconstruction, matched to published Fig. S4D compositions.",
               "No crop, positive-cell filter, new normalization or sample-local reclustering.",
               "Enrichment reference: all other labelled neighbourhoods in the same complete sample."),
             file.path(out,"README.txt"))
  cat("Prepared", sample_id, nrow(input), "cells and", length(unique(input$roi_id)), "ROIs\n"); flush.console()
  invisible(input)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly=TRUE)
  stopifnot(length(args) == 4L)
  prepare_sample(readRDS(args[1]), read.csv(args[2]), args[3], args[4])
}

# Exact barcode join of author final zones to current Seurat image rows.
# Inputs are read-only. No classifier, expression processing or TileScope run.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 4)
zones <- readRDS(args[1])
mapping <- read.csv(args[2], stringsAsFactors = FALSE)
mapping$in_current_corpus <- tolower(as.character(mapping$in_current_corpus)) == "true"
rds_root <- args[3]
output <- args[4]
rows <- list()
for (i in seq_len(nrow(mapping))) {
  official <- mapping$author_sample_id[i]
  sample <- mapping$sample_id[i]
  if (!mapping$in_current_corpus[i] || !official %in% names(zones)) next
  path <- file.path(rds_root, paste0("seuratObject_denoised_clustered_", sample, "_500.RDS"))
  if (!file.exists(path)) stop(paste("Missing coordinate/barcode authority:", path))
  obj <- readRDS(path)
  stopifnot(length(obj@images) == 1)
  image <- obj@images[[1]]
  coords <- image@coordinates
  barcodes <- rownames(coords)
  stopifnot(!is.null(barcodes), !anyDuplicated(barcodes))
  if (anyDuplicated(names(zones[[official]]))) stop("Duplicate author zone barcodes")
  if (!all(names(zones[[official]]) %in% barcodes)) {
    stop(paste("Author-labelled observations are absent from current input:", sample))
  }
  labels <- unname(zones[[official]][barcodes])
  stopifnot(all(na.omit(labels) %in% c("st_mal", "st_norm", "dis_mal", "dis_norm", "Intermediate")))
  rows[[length(rows) + 1]] <- data.frame(
    author_sample_id = official, sample_id = sample,
    spot_index = seq_along(barcodes) - 1L, spot_barcode = barcodes,
    x = coords$imagecol * image@scale.factors$lowres,
    y = coords$imagerow * image@scale.factors$lowres,
    source_roi_id = as.character(obj@meta.data[barcodes, "BayesSpace"]),
    author_zone = labels, stringsAsFactors = FALSE
  )
  message(sample, ": ", length(barcodes), " rows; ", sum(!is.na(labels)), " author labels")
  rm(obj, image, coords)
}
write.csv(do.call(rbind, rows), output, row.names = FALSE, na = "")

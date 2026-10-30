# Usage: Rscript prepare_cohort.R STUDY_DIR CODE_DIR INVENTORY_CSV
# The coordinator validates published profiles before calling this script.
suppressPackageStartupMessages(library(SeuratObject))
suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly=TRUE); stopifnot(length(args) == 3L)
study <- args[1]; code <- args[2]; inventory <- read.csv(args[3])
source(file.path(code, "prepare.R"))
validated <- file.path(study, "preparation", "validated")
report <- jsonlite::fromJSON(file.path(validated, "annotation_validation.json"))
stopifnot(isTRUE(report$accepted), report$rounded_values_matching == 360)
mapping <- read.csv(file.path(validated, "cluster_mapping.csv"))
named <- read.csv(file.path(study, "preparation", "cohort", "published_neighbourhood_mapping.csv"))
meta <- readRDS(file.path(study, "preparation", "reconstruction", "reconstructed_metadata_UNVALIDATED.rds"))
meta$published_cluster <- mapping$published_cluster[match(meta$reconstructed_cluster, mapping$reconstructed_cluster)]
meta$ann_CN <- named$roi_id[match(meta$published_cluster, named$published_cluster)]
stopifnot(!anyNA(meta$ann_CN), !anyDuplicated(rownames(meta)),
          setequal(unique(meta$orig.ident), inventory$sample), nrow(meta) == sum(inventory$n_cells))
cat("Loading complete deposited expression object\n"); flush.console()
obj <- readRDS(file.path(study, "raw", "GBM_oHSV_seurat_Full_20260107.rds"))
original <- attr(obj, "meta.data")
stopifnot(identical(rownames(meta), rownames(original)), identical(meta$orig.ident, original$orig.ident))
definition <- file.path(code, "pair.json")
genes <- unique(unlist(jsonlite::fromJSON(definition)[c("genes_a", "genes_b")]))
for (i in seq_len(nrow(inventory))) {
  s <- inventory$sample[i]
  ids <- rownames(original)[original$orig.ident == s]
  stopifnot(length(ids) == inventory$n_cells[i])
  labels <- data.frame(cell_id=ids, meta[ids, c("orig.ident","x","y","reconstructed_cluster","published_cluster","ann_CN")], row.names=NULL)
  label_path <- file.path(validated, paste0(s, "_annotations.csv.gz"))
  if (file.exists(label_path)) {
    old <- read.csv(label_path, check.names=FALSE)
    old <- old[match(ids, old$cell_id),]
    stopifnot(identical(old$ann_CN, labels$ann_CN), identical(old$published_cluster, labels$published_cluster))
  } else {
    con <- gzfile(label_path, "wt"); write.csv(labels, con, row.names=FALSE); close(con)
  }
  sample <- list(metadata=original[ids,,drop=FALSE],
                 data=LayerData(obj[["RNA"]], layer="data", features=genes, cells=ids),
                 counts=LayerData(obj[["RNA"]], layer="counts", features=genes, cells=ids))
  prepare_sample(sample, labels, file.path(study, inventory$input_directory[i]), definition, length(ids))
  rm(sample); gc(FALSE)
}
write.csv(inventory, file.path(study,"preparation","cohort","prepared_inventory.csv"), row.names=FALSE)
cat("All cohort inputs prepared and checked\n")

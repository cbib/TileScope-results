# Reconstruct the released 20-neighbour / 18-cluster recipe. Computational
# labels are candidates, NOT accepted biological ROIs until composition and
# published maps have been checked. No TileScope analysis is performed here.
# Usage: Rscript reconstruct_neighbourhoods.R CORRECTION_INPUTS AUDIT_CSV OUT
suppressPackageStartupMessages(library(dbscan))
suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 3L)
out <- args[3]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
inputs <- readRDS(args[1]); meta <- inputs$metadata
rm(inputs); gc()
diag <- read.csv(args[2])
labels <- meta$tier1_T
idx <- match(diag$cell_id, rownames(meta)); stopifnot(!anyNA(idx))
keep <- !is.na(diag$proposed_label)
# This branch reproduces the literal published correction, not a repaired
# version. Its indexing issue and the affected cells are retained in the audit.
labels[idx[keep]] <- diag$proposed_label[keep]
meta$tier1_T_cor <- labels
cat("Literal correction changes", sum(meta$tier1_T != labels), "cells; P28_post changes", sum(meta$tier1_T != labels & meta$orig.ident == "P28_post"), "\n")
types <- unique(labels)
parts <- split(seq_len(nrow(meta)), meta$orig.ident)

freqs <- lapply(names(parts), function(s) {
  path <- file.path(out, paste0(s, "_composition.rds"))
  if (file.exists(path)) return(readRDS(path))
  idx <- parts[[s]]
  xy <- meta[idx,c("x","y")]
  rownames(xy) <- rownames(meta)[idx]
  neighbours <- dbscan::kNN(xy, k=20, sort=TRUE)
  ids <- rownames(xy); n <- length(idx)
  type_idx <- match(labels[idx], types)
  tab <- matrix(tabulate(rep(seq_len(n), times=20L) + n*(type_idx[as.vector(neighbours$id)]-1L), nbins=n*length(types)), nrow=n, dimnames=list(ids, types))
  # The author script uses table(), whose rows are sorted by cell identifier.
  tab <- tab[order(rownames(tab)),,drop=FALSE]
  stopifnot(all(rowSums(tab) == 20L))
  if (s == "P28_post") {
    # Exact check against the table/melt implementation in the released code.
    original <- data.frame(neighbours$id); original$id <- rownames(neighbours$id)
    melted <- suppressMessages(reshape2::melt(original, varnames="id"))
    melted$annot <- labels[idx][melted$value]
    ref <- table(melted$id, melted$annot)
    missing <- setdiff(types, colnames(ref))
    if (length(missing)) ref <- cbind(ref, matrix(0L, nrow=nrow(ref), ncol=length(missing), dimnames=list(rownames(ref),missing)))
    ref <- ref[,types,drop=FALSE]
    stopifnot(identical(rownames(ref), rownames(tab)), identical(colnames(ref), colnames(tab)), all(ref == tab))
    cat("P28 neighbour frequencies exactly match author implementation\n")
  }
  saveRDS(tab, path)
  cat(s, n, "cell composition profiles complete\n"); flush.console()
  tab
})
names(freqs) <- names(parts)
all_freq <- do.call(rbind, unname(freqs)); rm(freqs); gc()
stopifnot(nrow(all_freq) == nrow(meta), !anyDuplicated(rownames(all_freq)))
cat("Fitting published 18-cluster model, 25 starts, seed 35\n"); flush.console()
set.seed(35)
fit <- kmeans(all_freq, centers=18, iter.max=300, nstart=25)
saveRDS(fit, file.path(out,"author_recipe_kmeans.rds"))
write.csv(data.frame(cluster=paste0("CN-",seq_len(nrow(fit$centers))), fit$centers, check.names=FALSE), file.path(out,"cluster_neighbour_composition.csv"), row.names=FALSE)
meta$reconstructed_cluster <- paste0("CN-", fit$cluster[match(rownames(meta), names(fit$cluster))])
stopifnot(!anyNA(meta$reconstructed_cluster))
mapping <- c("CN-1"="CN-Tumor", "CN-2"="CN-Tumor - T cell interface", "CN-3"="CN-Tumor - T cell interface", "CN-4"="CN-Tumor - T cell interface", "CN-5"="CN-Tumor", "CN-6"="CN-Peri-necrotic", "CN-7"="CN-Lymphoid aggregate", "CN-8"="CN-Tumor", "CN-9"="CN-Necrotic", "CN-10"="CN-Tumor - T cell interface", "CN-11"="CN-Peri-necrotic", "CN-12"="CN-Vasculature", "CN-13"="CN-Tumor - T cell interface", "CN-14"="CN-Tumor", "CN-15"="CN-Tumor", "CN-16"="CN-Tumor", "CN-17"="CN-Plasma cells", "CN-18"="CN-Tumor")
meta$unvalidated_author_mapping <- unname(mapping[meta$reconstructed_cluster])
saveRDS(meta, file.path(out,"reconstructed_metadata_UNVALIDATED.rds"))
write.csv(as.data.frame(table(meta$reconstructed_cluster, meta$tier1_T_cor)), file.path(out,"cluster_cell_composition.csv"), row.names=FALSE)
sub <- meta[meta$orig.ident == "P28_post",]
con <- gzfile(file.path(out,"P28_post_candidates_UNVALIDATED.csv.gz"),"wt")
write.csv(data.frame(cell_id=rownames(sub), sub, check.names=FALSE), con, row.names=FALSE)
close(con)
writeLines(c(capture.output(sessionInfo()), paste("centers",18,"iter.max",300,"nstart",25,"seed",35), "No biological labels accepted automatically."), file.path(out,"settings.txt"))
cat("Candidate reconstruction complete; mapping remains UNVALIDATED\n")

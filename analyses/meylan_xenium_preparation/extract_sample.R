# Inspect remaining reconstruction inputs and extract one intact specimen.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 3L)
suppressPackageStartupMessages(library(SeuratObject))
suppressPackageStartupMessages(library(Matrix))
obj <- readRDS(args[1]); sample <- args[2]; out <- args[3]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
meta <- attr(obj, "meta.data")
ids <- rownames(meta)[meta$orig.ident == sample]
stopifnot(length(ids) > 0)
sink(file.path(out, "remaining_slots.txt"))
cat("Top-level slots:", slotNames(obj), "\n")
for (nm in c("graphs", "neighbors", "reductions", "tools", "commands", "misc")) {
  cat("\n", nm, "\n")
  slot <- attr(obj, nm)
  print(names(slot))
  if (nm == "graphs") print(lapply(slot, dim))
  if (nm == "reductions") print(lapply(slot, function(z) dim(attr(z, "cell.embeddings"))))
}
cat("\nSelected image:\n")
fov <- attr(obj, "images")[[sample]]
print(slotNames(fov))
cat("Boundaries:", names(attr(fov, "boundaries")), "\n")
cat("\nRNA genes:\n")
print(rownames(obj[["RNA"]]))
sink()
cat("Extracting", sample, length(ids), "cells\n"); flush.console()
counts <- LayerData(obj[["RNA"]], layer="counts", cells=ids)
expr <- LayerData(obj[["RNA"]], layer="data", cells=ids)
stopifnot(identical(colnames(counts), ids), identical(colnames(expr), ids))
saveRDS(list(metadata=meta[ids,,drop=FALSE], counts=counts, data=expr), file.path(out, paste0(sample, ".rds")))
genes <- intersect(c("CXCL13", "CCL19", "IL7R", "SELL", "TCF7", "CCR7", "FCMR", "CD27", "ITGAE", "CXCR6", "ZNF683", "ITGA1", "IFIT3", "ISG15", "MX1", "CXCL10", "CXCL11", "IFITM3", "STAT2", "MKI67", "CDK1", "CDK2", "CDK6"), rownames(counts))
write.csv(data.frame(gene=genes, cells_expressing=as.numeric(rowSums(counts[genes,,drop=FALSE] > 0)), mean_normalized=as.numeric(rowMeans(expr[genes,,drop=FALSE]))), file.path(out, "selected_gene_summary.csv"), row.names=FALSE)
if ("RNA_snn_T" %in% names(attr(obj, "graphs"))) {
  saveRDS(attr(obj, "graphs")[["RNA_snn_T"]], file.path(out, "author_Tcell_graph.rds"))
}
cat("Extraction complete\n")

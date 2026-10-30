# Audit the published correction using the deposited graph. Do not replace
# metadata or infer ROIs. Usage: Rscript audit_correction.R FULL_RDS AUTHOR_RMD OUT
suppressPackageStartupMessages(library(SeuratObject))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(dbscan))
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 3L)
out <- args[3]
dir.create(out, recursive=TRUE, showWarnings=FALSE)

# Evaluate only the named, inspected function definition from the author script.
lines <- readLines(args[2])
inside <- FALSE; code <- character()
for (line in lines) {
  if (grepl("^```\\{r", line)) { inside <- TRUE; next }
  if (grepl("^```", line)) { inside <- FALSE; next }
  if (inside) code <- c(code, line)
}
exprs <- parse(text=code)
target <- Filter(function(e) is.call(e) && identical(e[[1]], as.name("<-")) && identical(e[[2]], as.name("reassign_prolif_tcells")), as.list(exprs))
stopifnot(length(target) == 1L, identical(target[[1]][[3]][[1]], as.name("function")))
eval(target[[1]])

cat("Reading deposited object\n"); flush.console()
obj <- readRDS(args[1]); meta <- obj[[]]
g <- obj@graphs$RNA_snn_T
stopifnot(!is.null(g), all(rownames(g) %in% rownames(meta)))
cat("Auditing candidate corrections\n"); flush.console()
counts <- GetAssayData(obj, assay="RNA", layer="counts")
proliferation <- Matrix::colMeans(counts[c("MKI67","CDK1","CDK2","CDK6"),,drop=FALSE])
saveRDS(list(metadata=meta, proliferation=proliferation, graph=g), file.path(out, "correction_inputs.rds"))

suspects <- unlist(lapply(split(meta, meta$orig.ident), function(m) {
  tumour_ids <- rownames(m)[m$tier1_T == "Tumor cells"]
  t_ids <- rownames(m)[m$tier1_T == "CD8 Proliferative"]
  tumour_ids <- tumour_ids[proliferation[tumour_ids] > 2]
  if (!length(t_ids) || !length(tumour_ids)) return(NULL)
  k <- min(5L, length(t_ids)-1L, length(tumour_ids)-1L)
  stopifnot(k > 0)
  n <- dbscan::kNN(x=as.matrix(m[tumour_ids,c("x","y")]), k=k, query=as.matrix(m[t_ids,c("x","y")]))
  t_ids[matrixStats::rowAnys(n$dist < 30)]
}), use.names=FALSE)
stopifnot(all(suspects %in% rownames(g)))
sub <- g[suspects,,drop=FALSE]
valid_cells <- setdiff(colnames(sub), suspects)
valid_cells <- valid_cells[meta[valid_cells,"tier1_T"] != "CD8 Proliferative"]
literal_neighbors <- vapply(seq_along(suspects), function(i) {
  # Deliberately reproduce the public code's indexing, including its matrix
  # orientation, to test rather than silently repair it.
  start_idx <- sub@p[i] + 1L
  end_idx <- sub@p[i+1L]
  row_cols <- sub@i[start_idx:end_idx] + 1L
  valid_idx <- which(colnames(sub)[row_cols] %in% valid_cells)[1]
  if (is.na(valid_idx)) return(NA_character_)
  colnames(sub)[row_cols[valid_idx]]
}, character(1))
matched <- !is.na(literal_neighbors)
edge_value <- rep(NA_real_, length(suspects))
edge_value[matched] <- as.numeric(g[cbind(match(suspects[matched], rownames(g)), match(literal_neighbors[matched], colnames(g)))])
diag <- data.frame(cell_id=suspects, sample=meta[suspects,"orig.ident"], old_label=meta[suspects,"tier1_T"], literal_neighbor=literal_neighbors, proposed_label=meta[literal_neighbors,"tier1_T"], expression_graph_edge=edge_value)
write.csv(diag, file.path(out,"literal_correction_audit.csv"), row.names=FALSE)
sink(file.path(out,"audit_summary.txt"))
cat("Graph class:", class(g), "\n")
cat("Subset dimensions:", dim(sub), "\n")
cat("Suspected proliferative T cells:", length(suspects), "\n")
cat("Assigned by literal author indexing:", sum(matched), "\n")
cat("Assigned neighbour has a real graph edge:", sum(edge_value > 0, na.rm=TRUE), "\n")
cat("Assigned neighbour has no graph edge:", sum(edge_value == 0, na.rm=TRUE), "\n")
print(table(diag$sample, !is.na(diag$literal_neighbor)))
print(table(diag$proposed_label, useNA="ifany"))
sink()
cat("Audit saved; no corrected labels or ROIs accepted automatically\n")

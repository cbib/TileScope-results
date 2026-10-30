suppressPackageStartupMessages(library(Seurat))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("usage: audit_gene_detection.R samples.csv mouse_data_root output.csv")
root <- args[[2]]
samples <- read.csv(args[[1]])
genes <- c('Nkg7','Gzmb','Cd3d','Cd3g','Cd8a','Prf1','Ifng','Slc16a1','Mcpt1','Mcpt2','Cpa3','Igkc')
rows <- list()
for (i in seq_len(nrow(samples))) {
 sn <- samples$sample_dir[[i]]
 if(sn=='V4_WT') next
 obj <- readRDS(file.path(root,samples$rds_relpath[[i]]))
 x <- LayerData(obj,assay='Spatial',layer='data')
 roi <- as.character(obj[[]][[samples$bayesspace_column[[i]]]])
 for(r in if(sn=='MCT1_CTL') c('2','5') else c('4','9')) {
  for(g in genes) {
   available <- g %in% rownames(x)
   v <- if(available) as.numeric(x[g,roi==r]) else NA_real_
   rows[[length(rows)+1]] <- data.frame(sample=sn,roi=r,gene=g,n_spots=sum(roi==r),
     measurement_status=if(available)'present_in_object' else 'absent_from_object',
     fraction_detected=if(available)mean(v>0) else NA_real_,
     mean_lognormalized=if(available)mean(v) else NA_real_)
  }
 }
}
write.csv(do.call(rbind,rows),args[[3]],row.names=FALSE)

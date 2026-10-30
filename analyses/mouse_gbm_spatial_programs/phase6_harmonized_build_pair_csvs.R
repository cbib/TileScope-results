#!/usr/bin/env Rscript

# Build each frozen mouse-GBM signature score once per sample, then reuse the
# score bank to materialize every unordered TileScope pair. The final YAML is
# the execution authority; the review workbook is provenance only.

suppressPackageStartupMessages({
  library(Seurat)
  library(yaml)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop(
    "usage: phase6_harmonized_build_pair_csvs.R ",
    "signatures.yaml samples.csv data_root output_root"
  )
}

contract <- yaml::read_yaml(args[[1]])
samples <- read.csv(args[[2]], check.names = FALSE, stringsAsFactors = FALSE)
data_root <- args[[3]]
output_root <- args[[4]]
pair_root <- file.path(output_root, "pair_csvs")
dir.create(pair_root, recursive = TRUE, showWarnings = FALSE)

if (!identical(contract$status, "frozen_for_tilescope")) {
  stop("signature contract is not frozen_for_tilescope")
}

score_panel <- function(object, genes, assay, prefix, scoring) {
  set.seed(as.integer(scoring$seed))
  scored <- AddModuleScore(
    object = object,
    features = list(genes),
    pool = rownames(LayerData(object, assay = assay, layer = scoring$layer)),
    nbin = as.integer(scoring$nbin),
    ctrl = as.integer(scoring$ctrl),
    seed = as.integer(scoring$seed),
    assay = assay,
    name = prefix
  )
  as.numeric(scored[[]][[paste0(prefix, "1")]])
}

safe_spearman <- function(x, y) {
  keep <- is.finite(x) & is.finite(y)
  if (sum(keep) < 3L || sd(x[keep]) == 0 || sd(y[keep]) == 0) return(NA_real_)
  suppressWarnings(cor(x[keep], y[keep], method = "spearman"))
}

manifest_rows <- list()
audit_rows <- list()
manifest_i <- 0L
audit_i <- 0L

signature_names <- names(contract$signatures)
if (length(signature_names) != 14L || anyDuplicated(signature_names)) {
  stop("expected exactly 14 uniquely named frozen signatures")
}

for (sample_i in seq_len(nrow(samples))) {
  sample_name <- samples$sample_dir[[sample_i]]
  message("[harmonized pair builder] ", sample_name)
  object <- readRDS(file.path(data_root, samples$rds_relpath[[sample_i]]))
  assay <- samples$assay[[sample_i]]
  expression <- LayerData(
    object,
    assay = assay,
    layer = contract$scoring_rule$layer
  )
  metadata <- object[[]]
  if (!identical(colnames(expression), rownames(metadata))) {
    stop("expression and metadata are not aligned for ", sample_name)
  }
  roi <- as.character(metadata[[samples$bayesspace_column[[sample_i]]]])

  image_names <- names(object@images)
  if (length(image_names) != 1L) {
    stop("expected exactly one spatial image for ", sample_name)
  }
  coordinates <- GetTissueCoordinates(object, image = image_names[[1]])
  if (!all(colnames(expression) %in% rownames(coordinates))) {
    stop("coordinates do not cover every expression spot for ", sample_name)
  }
  coordinates <- coordinates[
    colnames(expression),
    c("x", "y"),
    drop = FALSE
  ]

  applicable <- signature_names[vapply(
    contract$signatures,
    function(signature) sample_name %in% unlist(signature$intended_samples),
    logical(1)
  )]
  applicable <- sort(applicable)
  score_bank <- list()
  raw_bank <- list()

  for (signature_name in applicable) {
    signature <- contract$signatures[[signature_name]]
    genes <- unlist(signature$genes)
    missing <- setdiff(genes, rownames(expression))
    if (length(missing) > 0L) {
      stop(signature_name, " missing in ", sample_name, ": ", paste(missing, collapse = ", "))
    }
    values <- as.matrix(expression[genes, , drop = FALSE])
    variances <- apply(values, 1L, var)
    invariant <- names(variances)[!is.finite(variances) | variances <= 1e-12]
    if (length(invariant) > 0L) {
      stop(signature_name, " invariant in ", sample_name, ": ", paste(invariant, collapse = ", "))
    }

    raw_mean <- colMeans(values)
    score <- score_panel(
      object,
      genes,
      assay,
      paste0("tsharmonized_", signature_name, "_"),
      contract$scoring_rule
    )
    if (length(score) != ncol(expression) || any(!is.finite(score))) {
      stop("non-finite or incomplete score for ", signature_name, " in ", sample_name)
    }
    score_bank[[signature_name]] <- score
    raw_bank[[signature_name]] <- raw_mean
    audit_i <- audit_i + 1L
    audit_rows[[audit_i]] <- data.frame(
      sample_dir = sample_name,
      signature = signature_name,
      topic = signature$topic,
      kind = signature$kind,
      genes = paste(genes, collapse = "|"),
      n_spots = length(score),
      score_mean = mean(score),
      score_sd = sd(score),
      raw_mean = mean(raw_mean),
      raw_sd = sd(raw_mean),
      score_raw_spearman = safe_spearman(score, raw_mean),
      stringsAsFactors = FALSE
    )
  }

  sample_pair_root <- file.path(pair_root, sample_name)
  dir.create(sample_pair_root, recursive = TRUE, showWarnings = FALSE)

  score_frame <- data.frame(
    spot_id = colnames(expression),
    x = coordinates$x,
    y = coordinates$y,
    roi_id = roi,
    stringsAsFactors = FALSE
  )
  for (signature_name in applicable) {
    score_frame[[paste0(signature_name, "__score")]] <- score_bank[[signature_name]]
    score_frame[[paste0(signature_name, "__raw_mean")]] <- raw_bank[[signature_name]]
  }
  write.csv(
    score_frame,
    file.path(output_root, paste0(sample_name, "_signature_score_bank.csv")),
    row.names = FALSE
  )

  pairs <- combn(applicable, 2L, simplify = FALSE)
  expected_pairs <- if (sample_name %in% c("MCT1_CTL", "MCT1_KO")) 78L else 66L
  if (length(pairs) != expected_pairs) {
    stop(sample_name, " expected ", expected_pairs, " pairs, found ", length(pairs))
  }
  expected_paths <- character(0)
  for (pair in pairs) {
    signature_a <- pair[[1]]
    signature_b <- pair[[2]]
    pair_path <- file.path(
      sample_pair_root,
      paste0(signature_a, "__", signature_b, ".csv")
    )
    pair_frame <- data.frame(
      x = coordinates$x,
      y = coordinates$y,
      gene1 = score_bank[[signature_a]],
      gene2 = score_bank[[signature_b]],
      roi_id = roi,
      stringsAsFactors = FALSE
    )
    write.csv(pair_frame, pair_path, row.names = FALSE)
    expected_paths <- c(expected_paths, normalizePath(pair_path))
    manifest_i <- manifest_i + 1L
    manifest_rows[[manifest_i]] <- data.frame(
      sample_dir = sample_name,
      signature_a = signature_a,
      signature_b = signature_b,
      signature_a_kind = contract$signatures[[signature_a]]$kind,
      signature_b_kind = contract$signatures[[signature_b]]$kind,
      contains_technical_locator = (
        contract$signatures[[signature_a]]$kind == "technical_locator" ||
        contract$signatures[[signature_b]]$kind == "technical_locator"
      ),
      pair_csv = normalizePath(pair_path),
      n_spots = nrow(pair_frame),
      stringsAsFactors = FALSE
    )
  }

  observed_paths <- normalizePath(
    list.files(sample_pair_root, pattern = "\\.csv$", full.names = TRUE)
  )
  if (!setequal(expected_paths, observed_paths)) {
    stop("stale or missing pair CSVs under ", sample_pair_root)
  }
  rm(object, expression, metadata, score_bank, raw_bank, score_frame)
  gc()
}

manifest <- do.call(rbind, manifest_rows)
audit <- do.call(rbind, audit_rows)
if (nrow(manifest) != 222L || anyDuplicated(manifest[c("sample_dir", "signature_a", "signature_b")])) {
  stop("manifest closure failed")
}
write.csv(manifest, file.path(output_root, "phase6_harmonized_pair_manifest.csv"), row.names = FALSE)
write.csv(audit, file.path(output_root, "phase6_harmonized_score_audit.csv"), row.names = FALSE)
message("[harmonized pair builder] wrote ", nrow(manifest), " pairs and ", nrow(audit), " sample-signature audits")

# Install and load RadioGx
if (!require(BiocManager, quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("RadioGx", ask = FALSE)
library(RadioGx)

# Download the Cleveland dataset
cleveland <- downloadRSet("Cleveland")

# Get sensitivity raw data and info
sensRaw <- sensitivityRaw(cleveland)
sensInfo <- sensitivityInfo(cleveland)

# Get the number of cell lines
n_cell_lines <- dim(sensRaw)[1]

auc_results <- numeric(n_cell_lines)
rsquare_results <- numeric(n_cell_lines)
sf2_results <- numeric(n_cell_lines)
lq_params <- matrix(NA, nrow = n_cell_lines, ncol = 2)
colnames(lq_params) <- c("alpha", "beta")

for (i in 1:n_cell_lines) {
  radiationDoses <- sensRaw[i, , 'Dose']
  survivalFractions <- sensRaw[i, , 'Viability']

  valid_idx <- !is.na(radiationDoses) & !is.na(survivalFractions)
  radiationDoses <- radiationDoses[valid_idx]
  survivalFractions <- survivalFractions[valid_idx]

  tryCatch({
    LQmodel <- linearQuadraticModel(D = radiationDoses,
                                    SF = survivalFractions)

    # Compute AUC
    auc_results[i] <- RadioGx::computeAUC(D = radiationDoses,
                                          pars = LQmodel,
                                          lower = min(radiationDoses, na.rm = TRUE),
                                          upper = max(radiationDoses, na.rm = TRUE))

    lq_params[i, ] <- LQmodel

    rsquare_results[i] <- attr(LQmodel, "Rsquare")

    # Extract SF2: surviving fraction at D = 2 Gy (doses3)
    sf2_idx <- which(radiationDoses == 2)
    if (length(sf2_idx) > 0) {
      sf2_results[i] <- survivalFractions[sf2_idx[1]]
    } else {
      sf2_results[i] <- NA
    }

  }, error = function(e) {
    warning(paste("Failed to fit cell line", i, ":", e$message))
    auc_results[i] <- NA
    rsquare_results[i] <- NA
    sf2_results[i] <- NA
  })
}

results <- data.frame(
  cell_line = sensInfo$sampleid,
  alpha = lq_params[, 1],
  beta = lq_params[, 2],
  Rsquare = rsquare_results,
  AUC = auc_results,
  SF2 = sf2_results
)

duplicates <- results$cell_line[duplicated(results$cell_line)]

if (length(duplicates) > 0) {
  cat("Number of duplicates:", length(duplicates), "\n")
  cat("Duplicate cell lines:\n")
  print(unique(duplicates))

  cat("\nFull data for duplicates:\n")
  duplicate_rows <- results[results$cell_line %in% duplicates, ]
  print(duplicate_rows[order(duplicate_rows$cell_line), ])

  dup_counts <- table(results$cell_line[results$cell_line %in% duplicates])
  cat("\nDuplicate counts:\n")
  print(dup_counts)
} else {
  cat("\nNo duplicate cell lines found. All cell lines are unique.\n")
}

args <- commandArgs(trailingOnly = TRUE)
output_dir <- args[1]
output_path <- file.path(output_dir, "cleveland.csv")

if (length(duplicates) > 0) {
  # For each duplicate, keep the one with highest Rsquare
  results_unique <- results[order(results$cell_line, -results$Rsquare), ]
  results_unique <- results_unique[!duplicated(results_unique$cell_line), ]
  write.csv(results_unique, output_path, row.names = FALSE)
} else {
  write.csv(results, output_path, row.names = FALSE)
}

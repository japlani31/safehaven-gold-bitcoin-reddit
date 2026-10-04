# 10_robustness.R
# Robustness suite: 7 spesifikasi. Output ringkasan satu tabel:
#   output/tables/robustness_summary.csv
# Kolom: TCI rata-rata + NET rata-rata tiap variabel per spesifikasi.
# Jalankan dari folder scripts/ (Rscript 10_robustness.R). ~15-30 menit.

library(ConnectednessApproach)
library(zoo)

base <- ".."
panel <- read.csv(file.path(base, "data_processed", "panel_tvpvar.csv"))
panel$date <- as.Date(panel$date)

run_spec <- function(name, vars, data, nlag = 1, nfore = 10) {
  Y <- zoo(data[, vars], order.by = data$date)
  dca <- ConnectednessApproach(
    Y, model = "TVP-VAR", connectedness = "Time",
    nlag = nlag, nfore = nfore,
    VAR_config = list(TVPVAR = list(kappa1 = 0.99, kappa2 = 0.96,
                                    prior = "BayesPrior")))
  tab <- dca$TABLE
  net <- as.numeric(tab["NET", seq_along(vars)])
  names(net) <- vars
  tci <- mean(as.numeric(dca$TCI), na.rm = TRUE)
  cat(sprintf("[%s] TCI=%.2f | NET: %s\n", name, tci,
              paste(sprintf("%s=%.2f", vars, net), collapse = " ")))
  out <- data.frame(spec = name, TCI = tci, t(net))
  # samakan nama kolom X3 agar bisa digabung
  colnames(out) <- sub("^(FED|BEI|EPU)$", "X3", colnames(out))
  out
}

V <- function(x3) c("GOLD", "BTC", "SENT_G", "SENT_C", x3, "VIX", "DXY")
res <- list()

res[[1]] <- run_spec("Baseline (FED, lag1, H10)", V("FED"), panel)
res[[2]] <- run_spec("X3 = Breakeven Inflation",  V("BEI"), panel)
res[[3]] <- run_spec("X3 = ln EPU",               V("EPU"), panel)
res[[4]] <- run_spec("Lag 2",                     V("FED"), panel, nlag = 2)
res[[5]] <- run_spec("Horizon H=5",               V("FED"), panel, nfore = 5)
res[[6]] <- run_spec("Horizon H=20",              V("FED"), panel, nfore = 20)
res[[7]] <- run_spec("Subsample 2019+",           V("FED"),
                     panel[panel$date >= as.Date("2019-01-01"), ])

summary_df <- do.call(rbind, res)
write.csv(summary_df,
          file.path(base, "output", "tables", "robustness_summary.csv"),
          row.names = FALSE)
cat("\nSelesai -> output/tables/robustness_summary.csv\n")
print(summary_df, digits = 3)

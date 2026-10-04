# 12_kappa_network.R
# (a) R9: sensitivitas forgetting factors (kappa1=0.97, kappa2=0.94) -> 1 baris robustness
# (b) R6: network plot dari objek baseline -> output/figures/fig6_network.pdf/png
# Jalankan dari folder scripts/.

library(ConnectednessApproach)
library(zoo)

base <- ".."
panel <- read.csv(file.path(base, "data_processed", "panel_tvpvar.csv"))
dates <- as.Date(panel$date)
vars <- c("GOLD", "BTC", "SENT_G", "SENT_C", "FED", "VIX", "DXY")
Y <- zoo(panel[, vars], order.by = dates)

## (a) kappa sensitivity -----------------------------------------------------
dk <- ConnectednessApproach(
  Y, model = "TVP-VAR", connectedness = "Time", nlag = 1, nfore = 10,
  VAR_config = list(TVPVAR = list(kappa1 = 0.97, kappa2 = 0.94,
                                  prior = "BayesPrior")))
net <- as.numeric(dk$TABLE["NET", seq_along(vars)])
cat(sprintf("[Kappa 0.97/0.94] TCI=%.2f | NET: %s\n",
            mean(as.numeric(dk$TCI), na.rm = TRUE),
            paste(sprintf("%s=%.2f", vars, net), collapse = " ")))

## (b) network plot ----------------------------------------------------------
dca <- readRDS(file.path(base, "data_processed", "dca_baseline.rds"))
figdir <- file.path(base, "output", "figures")
if (exists("PlotNetwork")) {
  pdf(file.path(figdir, "fig6_network.pdf"), width = 7, height = 7)
  PlotNetwork(dca, method = "NPDC")
  dev.off()
  png(file.path(figdir, "fig6_network.png"), width = 2100, height = 2100, res = 300)
  PlotNetwork(dca, method = "NPDC")
  dev.off()
  cat("Network plot tersimpan (PlotNetwork).\n")
} else {
  cat("PlotNetwork tidak tersedia — pakai fallback Python.\n")
}

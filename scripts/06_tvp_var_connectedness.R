# 06_tvp_var_connectedness.R
# TVP-VAR Dynamic Connectedness (Antonakakis, Chatziantoniou & Gabauer 2020)
# Paket: ConnectednessApproach (Gabauer) — install.packages("ConnectednessApproach")
#
# Input : data_processed/panel_tvpvar.csv
# Output: output/tables/connectedness_table.csv
#         output/figures/fig2_TCI.pdf, fig3_NET.pdf, fig4_pairwise.pdf

library(ConnectednessApproach)
library(zoo)

# Jalankan dari folder scripts/:  Rscript 06_tvp_var_connectedness.R
base <- ".."

panel <- read.csv(file.path(base, "data_processed", "panel_tvpvar.csv"))
dates <- as.Date(panel$date)

# Sistem utama 2Y-3X-2C: Y={GOLD,BTC}, X={SENT_G,SENT_C,FED}, C={VIX,DXY}
# Robustness X3: ganti "FED" dengan "BEI" (breakeven inflation) atau "EPU"
vars <- c("GOLD", "BTC", "SENT_G", "SENT_C", "FED", "VIX", "DXY")
Y <- zoo(panel[, vars], order.by = dates)

## --- Uji pra-estimasi (laporkan di Tabel 1) -------------------------------
for (v in vars) {
  adf <- tseries::adf.test(na.omit(panel[[v]]))
  cat(sprintf("%-8s ADF p = %.4f\n", v, adf$p.value))
}

## --- TVP-VAR(1), GFEVD H=10, forgetting factors baku ----------------------
dca <- ConnectednessApproach(
  Y,
  model  = "TVP-VAR",
  connectedness = "Time",
  nlag   = 1,       # robustness: 2 (BIC)
  nfore  = 10,      # robustness: 5, 20
  VAR_config = list(TVPVAR = list(kappa1 = 0.99, kappa2 = 0.96,
                                  prior = "BayesPrior"))
)

## --- Tabel connectedness rata-rata (Tabel 2) ------------------------------
tab <- dca$TABLE
write.csv(tab, file.path(base, "output", "tables", "connectedness_table.csv"))
print(tab)

## --- Figur ----------------------------------------------------------------
figdir <- file.path(base, "output", "figures")

pdf(file.path(figdir, "fig2_TCI.pdf"), width = 9, height = 4)
PlotTCI(dca, ylim = c(0, 100))
dev.off()

pdf(file.path(figdir, "fig3_NET.pdf"), width = 9, height = 7)
PlotNET(dca)
dev.off()

pdf(file.path(figdir, "fig4_pairwise.pdf"), width = 9, height = 7)
PlotNPDC(dca)   # net pairwise: fokus BTC<->SENT_C dan GOLD<->SENT_G
dev.off()

cat("Selesai. Cek output/figures & output/tables.\n")

## --- Robustness (jalankan terpisah, ganti argumen): -----------------------
## 1. nlag = 2; nfore = 5 / 20
## 2. Ganti EPU dengan GPR harian
## 3. Ganti return dengan |return| atau vol GARCH(1,1) (paket rugarch)
## 4. Sub-sampel: 2015-2019 / 2020-2023 / 2024-2026
## 5. Model 9-variabel: + HSHARE_G, HSHARE_C (uji H3)

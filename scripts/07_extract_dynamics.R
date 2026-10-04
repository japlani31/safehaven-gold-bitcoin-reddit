# 07_extract_dynamics.R
# Re-estimasi TVP-VAR (spec identik 06) lalu simpan deret dinamis harian:
#   - TCI harian                       -> output/tables/dyn_tci.csv
#   - NET directional per variabel     -> output/tables/dyn_net.csv
#   - TO / FROM per variabel           -> output/tables/dyn_to.csv / dyn_from.csv
#   - Objek penuh                      -> data_processed/dca_baseline.rds
# Jalankan dari folder scripts/.

library(ConnectednessApproach)
library(zoo)

base <- ".."
panel <- read.csv(file.path(base, "data_processed", "panel_tvpvar.csv"))
dates <- as.Date(panel$date)
vars <- c("GOLD", "BTC", "SENT_G", "SENT_C", "FED", "VIX", "DXY")
Y <- zoo(panel[, vars], order.by = dates)

dca <- ConnectednessApproach(
  Y, model = "TVP-VAR", connectedness = "Time",
  nlag = 1, nfore = 10,
  VAR_config = list(TVPVAR = list(kappa1 = 0.99, kappa2 = 0.96,
                                  prior = "BayesPrior")))

cat("Komponen objek dca:", paste(names(dca), collapse = ", "), "\n")

tab <- file.path(base, "output", "tables")

# --- TCI harian ---
tci <- data.frame(date = dates, TCI = as.numeric(dca$TCI))
write.csv(tci, file.path(tab, "dyn_tci.csv"), row.names = FALSE)

# --- NET, TO, FROM harian ---
write.csv(data.frame(date = dates, dca$NET),
          file.path(tab, "dyn_net.csv"), row.names = FALSE)
write.csv(data.frame(date = dates, dca$TO),
          file.path(tab, "dyn_to.csv"), row.names = FALSE)
write.csv(data.frame(date = dates, dca$FROM),
          file.path(tab, "dyn_from.csv"), row.names = FALSE)

# --- NPDC pasangan kunci (net pairwise: >0 berarti baris MENERIMA dari kolom?) ---
# Konvensi paket: NPDC[i,j,t] = C_{i<-j,t} - C_{j<-i,t}. Kita simpan 4 pasangan.
pair_df <- data.frame(date = dates)
get_pair <- function(a, b) {
  ia <- which(vars == a); ib <- which(vars == b)
  as.numeric(dca$NPDC[ia, ib, ])
}
ok <- tryCatch({
  pair_df$NPDC_BTC_SENTC  <- get_pair("BTC", "SENT_C")
  pair_df$NPDC_GOLD_SENTG <- get_pair("GOLD", "SENT_G")
  pair_df$NPDC_BTC_VIX    <- get_pair("BTC", "VIX")
  pair_df$NPDC_GOLD_VIX   <- get_pair("GOLD", "VIX")
  TRUE
}, error = function(e) { cat("NPDC gagal:", conditionMessage(e), "\n"); FALSE })
if (ok) write.csv(pair_df, file.path(tab, "dyn_npdc_pairs.csv"), row.names = FALSE)

saveRDS(dca, file.path(base, "data_processed", "dca_baseline.rds"))
cat("Selesai. Deret dinamis tersimpan di output/tables/dyn_*.csv\n")

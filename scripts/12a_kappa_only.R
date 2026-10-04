# 12a_kappa_only.R: hanya bagian (a) dari 12_kappa_network.R, hasil disimpan ke output/kappa_log.txt
library(ConnectednessApproach)
library(zoo)
base <- ".."
panel <- read.csv(file.path(base, "data_processed", "panel_tvpvar.csv"))
dates <- as.Date(panel$date)
vars <- c("GOLD", "BTC", "SENT_G", "SENT_C", "FED", "VIX", "DXY")
Y <- zoo(panel[, vars], order.by = dates)
dk <- ConnectednessApproach(
  Y, model = "TVP-VAR", connectedness = "Time", nlag = 1, nfore = 10,
  VAR_config = list(TVPVAR = list(kappa1 = 0.97, kappa2 = 0.94, prior = "BayesPrior")))
net <- as.numeric(dk$TABLE["NET", seq_along(vars)])
baris <- sprintf("[Kappa 0.97/0.94] TCI=%.2f | NET: %s",
                 mean(as.numeric(dk$TCI), na.rm = TRUE),
                 paste(sprintf("%s=%.2f", vars, net), collapse = " "))
cat(baris, "\n")
writeLines(baris, file.path(base, "output", "kappa_log.txt"))

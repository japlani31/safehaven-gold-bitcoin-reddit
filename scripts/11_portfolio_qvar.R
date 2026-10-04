# 11_portfolio_qvar.R
# M1: Portfolio implications (Kroner-Ng bivariate weights + hedging effectiveness,
#     minimum connectedness portfolio) untuk pasangan GOLD-BTC.
# M2: Quantile connectedness (QVAR) tau = 0.05 / 0.50 / 0.95 sebagai robustness.
# Output: output/tables/portfolio_bivariate.csv, portfolio_weights_daily.csv,
#         qvar_summary.csv
# Jalankan dari folder scripts/.

library(ConnectednessApproach)
library(zoo)

base <- ".."
panel <- read.csv(file.path(base, "data_processed", "panel_tvpvar.csv"))
dates <- as.Date(panel$date)
vars <- c("GOLD", "BTC", "SENT_G", "SENT_C", "FED", "VIX", "DXY")
Y <- zoo(panel[, vars], order.by = dates)
tab <- file.path(base, "output", "tables")

## ================= M1: PORTFOLIO (pasangan GOLD-BTC) =====================
x <- Y[, c("GOLD", "BTC")]
dca2 <- ConnectednessApproach(
  x, model = "TVP-VAR", connectedness = "Time", nlag = 1, nfore = 10,
  VAR_config = list(TVPVAR = list(kappa1 = 0.99, kappa2 = 0.96,
                                  prior = "BayesPrior")))

cat("--- BivariatePortfolio (Kroner-Ng) ---\n")
bp <- tryCatch(
  BivariatePortfolio(x, dca2, method = "cumsum", statistics = "Fisher"),
  error = function(e) { cat("BP error:", conditionMessage(e), "\n"); NULL })
if (!is.null(bp)) {
  cat("komponen bp:", paste(names(bp), collapse = ", "), "\n")
  print(bp$TABLE)
  write.csv(bp$TABLE, file.path(tab, "portfolio_bivariate.csv"))
  # simpan deret bobot harian utk analisis episode
  w <- tryCatch(data.frame(date = dates, weight_gold = as.numeric(bp$portfolio_weights[, 1])),
                error = function(e) NULL)
  if (is.null(w))
    w <- tryCatch(data.frame(date = dates, weight_gold = as.numeric(bp$weights[, 1])),
                  error = function(e) NULL)
  if (!is.null(w)) write.csv(w, file.path(tab, "portfolio_weights_daily.csv"),
                             row.names = FALSE)
}

cat("\n--- MinimumConnectednessPortfolio ---\n")
mcp <- tryCatch(
  MinimumConnectednessPortfolio(x, dca2, statistics = "Fisher"),
  error = function(e) { cat("MCP error:", conditionMessage(e), "\n"); NULL })
if (!is.null(mcp)) {
  print(mcp$TABLE)
  write.csv(mcp$TABLE, file.path(tab, "portfolio_minconn.csv"))
}

## ================= M2: QVAR (quantile connectedness) =====================
qrows <- list()
for (tau in c(0.05, 0.50, 0.95)) {
  cat(sprintf("\n--- QVAR tau = %.2f ---\n", tau))
  dq <- tryCatch(
    ConnectednessApproach(Y, model = "QVAR", connectedness = "Time",
                          nlag = 1, nfore = 10,
                          VAR_config = list(QVAR = list(tau = tau))),
    error = function(e) { cat("QVAR error:", conditionMessage(e), "\n"); NULL })
  if (is.null(dq)) next
  tb <- dq$TABLE
  net <- as.numeric(tb["NET", seq_along(vars)])
  tci <- mean(as.numeric(dq$TCI), na.rm = TRUE)
  cat(sprintf("TCI=%.2f | NET: %s\n", tci,
              paste(sprintf("%s=%.2f", vars, net), collapse = " ")))
  qrows[[length(qrows) + 1]] <- data.frame(
    spec = sprintf("QVAR tau=%.2f", tau), TCI = tci,
    t(setNames(net, vars)))
}
if (length(qrows)) {
  qdf <- do.call(rbind, qrows)
  write.csv(qdf, file.path(tab, "qvar_summary.csv"), row.names = FALSE)
  print(qdf, digits = 3)
}
cat("\nSelesai 11_portfolio_qvar.R\n")

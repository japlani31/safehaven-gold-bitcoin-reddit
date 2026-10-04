# 11c_portfolio_direct.R
# Portfolio implications GOLD-BTC langsung dari DCC-GARCH (rmgarch):
#   w_t  = (hBB - hGB) / (hGG - 2 hGB + hBB)   [Kroner-Ng, clip 0..1]
#   b_t  = hBG / hGG                            [Kroner-Sultan]
#   HE   = 1 - Var(strategi) / Var(BTC unhedged), uji Fisher var-ratio
# Output: output/tables/portfolio_summary.csv, portfolio_daily.csv

library(rmgarch)
library(zoo)

base <- ".."
panel <- read.csv(file.path(base, "data_processed", "panel_tvpvar.csv"))
panel$date <- as.Date(panel$date)
r <- panel[, c("GOLD", "BTC")]
tab <- file.path(base, "output", "tables")

cat("Fitting DCC-GARCH(1,1) ...\n")
uspec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                    mean.model = list(armaOrder = c(0, 0)))
dspec <- dccspec(uspec = multispec(replicate(2, uspec)),
                 dccOrder = c(1, 1), distribution = "mvnorm")
fit <- dccfit(dspec, data = r)
H <- rcov(fit)                       # 2 x 2 x T

hGG <- H[1, 1, ]; hBB <- H[2, 2, ]; hGB <- H[1, 2, ]
w <- pmin(pmax((hBB - hGB) / (hGG - 2 * hGB + hBB), 0), 1)   # bobot emas
beta <- hGB / hGG                                            # hedge ratio

rp_w <- w * r$GOLD + (1 - w) * r$BTC     # portofolio Kroner-Ng
rp_h <- r$BTC - beta * r$GOLD            # BTC di-hedge dgn emas

he <- function(strategy, benchmark) {
  v1 <- var(strategy, na.rm = TRUE); v0 <- var(benchmark, na.rm = TRUE)
  f <- v1 / v0
  n <- sum(!is.na(strategy))
  p <- 2 * min(pf(f, n - 1, n - 1), 1 - pf(f, n - 1, n - 1))
  c(HE = 1 - f, p = p)
}

episodes <- list(
  "Full sample"  = c("2017-01-01", "2025-12-31"),
  "COVID crash"  = c("2020-02-20", "2020-04-30"),
  "Fed tightening" = c("2022-03-01", "2022-12-31"),
  "LUNA collapse" = c("2022-05-01", "2022-05-31"),
  "FTX collapse" = c("2022-11-01", "2022-11-30"),
  "ETF period"   = c("2024-01-01", "2025-12-31"))

rows <- list()
for (nm in names(episodes)) {
  idx <- panel$date >= as.Date(episodes[[nm]][1]) &
         panel$date <= as.Date(episodes[[nm]][2])
  he_w <- he(rp_w[idx], r$BTC[idx])
  he_h <- he(rp_h[idx], r$BTC[idx])
  rows[[nm]] <- data.frame(
    episode = nm, n = sum(idx),
    mean_weight_gold = mean(w[idx]),
    mean_hedge_ratio = mean(beta[idx]),
    HE_portfolio = he_w["HE"], p_portfolio = he_w["p"],
    HE_hedge = he_h["HE"], p_hedge = he_h["p"])
}
res <- do.call(rbind, rows)
write.csv(res, file.path(tab, "portfolio_summary.csv"), row.names = FALSE)
print(res, digits = 3, row.names = FALSE)

write.csv(data.frame(date = panel$date, weight_gold = w, hedge_ratio = beta),
          file.path(tab, "portfolio_daily.csv"), row.names = FALSE)
cat("\nSelesai 11c.\n")

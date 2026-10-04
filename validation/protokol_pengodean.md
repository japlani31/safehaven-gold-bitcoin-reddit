# Protokol anotasi validasi (ditetapkan 4 Oktober 2026, sebelum membaca sampel)

Pengode: model bahasa besar (large language model), buta terhadap label FinBERT dan label topik BERTopic.
Sampel: 600 komentar, acak bertingkat 2 aset x 3 label FinBERT x 100 (seed 20261004), urutan diacak (seed 7).
Unit: satu komentar (maksimal 600 karakter, seperti input model).

## Dimensi 1: SENT (mengikuti definisi Financial PhraseBank, Malo et al., 2014)
Sudut pandang investor: apakah komentar menyiratkan pengaruh positif, negatif, atau tidak ada terhadap harga/nilai aset yang dibahas (kripto atau emas).
- POS: pandangan naik/bullish, keuntungan, prospek baik, rekomendasi membeli karena nilai akan naik.
- NEG: pandangan turun/bearish, kerugian, penipuan, kejatuhan, risiko yang ditekankan, kekecewaan atas nilai.
- NEU: tidak ada implikasi arah harga: pertanyaan, informasi faktual, logistik (penyimpanan, pengiriman, toko), lelucon atau keluhan yang tidak terkait arah nilai aset.

## Dimensi 2: NARR
- H (hedging): penyimpan nilai, lindung inflasi atau pelemahan mata uang, safe haven, perlindungan kekayaan jangka panjang, menabung/menumpuk aset fisik sebagai simpanan.
- S (spekulatif): target harga, keuntungan cepat, timing beli-jual, leverage, judi, pump, "to the moon", membandingkan return jangka pendek.
- G (umum): membahas aset tetapi tanpa narasi hedging atau spekulatif (produk, teknologi, regulasi, tokoh, pertanyaan umum).
- X: tidak membahas aset atau tidak dapat dipahami.

## Aturan
- Label diberikan hanya dari teks komentar; tidak ada konteks utas.
- Jika ragu antara arah dan netral, pilih NEU; jika ragu antara H/S dan G, pilih G.
- Hasil tidak dikutip verbatim dalam naskah.

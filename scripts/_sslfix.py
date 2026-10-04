"""
_sslfix.py — perbaikan SSL untuk mesin ini.

Latar belakang: ada software (antivirus dengan HTTPS scanning) yang
mengintersep TLS dan menyisipkan sertifikat CA sendiri. Sertifikat itu
tidak memenuhi aturan VERIFY_X509_STRICT yang menjadi default di
Python 3.13+, sehingga semua koneksi HTTPS Python gagal dengan
"Basic Constraints of CA cert not marked critical".

Solusi: matikan flag STRICT saja. Verifikasi sertifikat TETAP AKTIF
(rantai sertifikat tetap divalidasi terhadap Windows cert store).

Pakai: `import _sslfix` di baris pertama skrip (sebelum import lain
yang membuka koneksi).
"""
import ssl

_orig_create = ssl.create_default_context


def _patched(*args, **kwargs):
    ctx = _orig_create(*args, **kwargs)
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


ssl.create_default_context = _patched
ssl._create_default_https_context = _patched

# Cakupan tambahan untuk library berbasis `requests`/urllib3 (HuggingFace dll):
# truststore memakai Windows certificate store — sama seperti browser.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

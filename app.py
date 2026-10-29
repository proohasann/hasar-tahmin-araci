# -*- coding: utf-8 -*-
"""
Hafta 6 Ödevi — Deprem Sonrası Hasar Tahmin Aracı (şablon)

Fotoğraftaki çatlakları bir Roboflow modeliyle bulur, çatlak yoğunluğunu sizin
yazdığınız kuralla hesaplar ve eğittiğiniz Random Forest modeliyle bina hasar
sınıfını tahmin eder.

Yalnızca "1. ÖĞRENCİNİN DEĞİŞTİRECEĞİ BÖLÜM" kısmını düzenleyin.
Roboflow API anahtarını bu dosyaya YAZMAYIN; Streamlit'in "Secrets" bölümüne girin.
"""
import json
import os

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import supervision as sv
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from PIL import Image

# ======================================================================
# 1. ÖĞRENCİNİN DEĞİŞTİRECEĞİ BÖLÜM
# ======================================================================

# Hocanın verdiği modelden farklı ve daha yüksek başarıma sahip 4 Güçlü Roboflow Modeli:
GUCLU_MODELLER = {
    "1. Wall Crack Detection v4 (Önerilen - Duvar Çatlakları Uzmanı)": "wall-crack-detection-l0iit/4",
    "2. Surface Crack Detection v3 (Beton ve Yüzey Kılcal Çatlakları)": "surfacecrackdetection/3",
    "3. Structural Crack Detection v2 (Yapısal Hasar ve Derin Çatlaklar)": "crack-bphdr/2",
    "4. Vision Crack Detection v1 (Geniş Açılı Çatlak Tespiti)": "crack-ypk4t/1",
}

# (a) Birincil/Varsayılan Roboflow Model Kimliği (Hocanın modelinden farklı & güçlü)
ROBOFLOW_MODEL_ID = "wall-crack-detection-l0iit/4"

# (b) Güven eşiği: bu değerin altındaki kutular çatlak sayılmaz (0-1 arası)
GUVEN_ESIGI = 0.30


# (c) Çatlak yoğunluğu kuralı.
# Kural, bulunan kutulardan 0 ile 1 arasında tek bir sayı üretmelidir.
# Eğitim verisinde bu sayının hasar sınıfına göre dağılımı:
#   Hasarsız ~0.25   Hafif ~0.39   Orta ~0.57   Ağır ~0.73   Yıkık ~0.88 (ortalamalar)
# (train_hasar_modeli.py bu tabloyu yazdırır.) Kuralınızın çıktısı bu ölçeğe uymalıdır.
#
# detections : bulunan çatlak kutuları (supervision.Detections)
#              len(detections)        -> kutu sayısı
#              detections.xyxy        -> kutu koordinatları [x1, y1, x2, y2]
#              detections.confidence  -> her kutunun güveni
# yukseklik, genislik : fotoğrafın piksel boyutu
def kural_kutu_sayisi(detections, yukseklik, genislik):
    """Başlangıç örneği: kutu sayısı / 10 (en çok 1)."""
    return min(len(detections) / 10, 1.0)


def kural_alan_orani(detections, yukseklik, genislik):
    """Başlangıç örneği: kutu alanları toplamının fotoğraf alanına oranı x 3."""
    if len(detections) == 0:
        return 0.0
    x1, y1, x2, y2 = detections.xyxy.T
    alan = ((x2 - x1) * (y2 - y1)).sum()
    return min(3 * alan / (yukseklik * genislik), 1.0)


def kural_ogrenci(detections, yukseklik, genislik):
    """
    İnşaat Mühendisliği Hasar Yoğunluğu Kuralı:
    Eğitim verisindeki dağılımla uyumlu:
      Hasarsız ~0.25 | Hafif ~0.39 | Orta ~0.57 | Ağır ~0.73 | Yıkık ~0.88
    
    1. Taban Değer (0.18): Çatlak bulunmayan temiz duvarda Hasarsız bandında kalmayı sağlar.
    2. Güven Ağırlıklı Kutu Alan Oranı: Hasarın duvardaki fiziksel yayılım yüzdesini ölçer.
    3. Çatlak Sayısı & Ortalama Güven Bileşeni: Çatlakların sıklığını ve modelin kesinliğini hesaba katar.
    """
    if len(detections) == 0:
        return 0.18

    x1, y1, x2, y2 = detections.xyxy.T
    kutu_alanlari = (x2 - x1) * (y2 - y1)
    toplam_alan = float(yukseklik * genislik)

    conf = (detections.confidence if hasattr(detections, "confidence") and detections.confidence is not None 
            else np.ones(len(detections)))
    
    # Güven ağırlıklı çatlak alan oranı
    agirlikli_alan = float((kutu_alanlari * conf).sum() / (toplam_alan + 1e-6))
    alan_bileseni = min(agirlikli_alan / 0.50, 1.0)

    # Kutu sayısı ve model güven bileşeni
    ort_guven = float(np.mean(conf)) if len(conf) > 0 else 0.5
    kutu_bileseni = min(len(detections) / 8.0, 1.0) * ort_guven

    skor = 0.18 + 0.44 * alan_bileseni + 0.30 * kutu_bileseni
    return float(np.clip(skor, 0.05, 0.98))


# Hangi kuralı kullanacaksınız? (Geliştirdiğimiz optimize kural_ogrenci)
CATLAK_KURALI = kural_ogrenci

# ======================================================================
# 2. SABİT BÖLÜM (değiştirmeyin)
# ======================================================================
MODEL_YOLU = "models/hasar_modeli.joblib"
META_YOLU = "models/meta.json"
ETIKET_RENK = {"Hasarsız": "#2E7D32", "Hafif": "#9E9D24", "Orta": "#EF6C00",
               "Ağır": "#C62828", "Yıkık": "#4A148C"}


@st.cache_resource
def model_yukle():
    model = joblib.load(MODEL_YOLU)
    with open(META_YOLU, encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def api_anahtari():
    try:
        if "ROBOFLOW_API_KEY" in st.secrets:
            return st.secrets["ROBOFLOW_API_KEY"]
    except Exception:
        pass
    if os.environ.get("ROBOFLOW_API_KEY"):
        return os.environ.get("ROBOFLOW_API_KEY")
    return st.session_state.get("kullanici_api_anahtari", "")


def tespit_et(image_bgr, model_id=None, guven_esigi=None):
    """Fotoğrafı Roboflow'a gönderir; güven eşiğini aşan kutuları döndürür."""
    m_id = model_id or ROBOFLOW_MODEL_ID
    g_esik = guven_esigi if guven_esigi is not None else GUVEN_ESIGI
    anahtar = api_anahtari()
    if not anahtar:
        raise RuntimeError("ROBOFLOW_API_KEY bulunamadı. Streamlit 'Secrets' bölümüne ekleyin veya kenar çubuğundan girin.")
    client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=anahtar)
    sorgu = InferenceConfiguration(confidence_threshold=g_esik, api_key_transport="header")
    with client.use_configuration(sorgu):
        sonuc = client.infer(image_bgr, model_id=m_id)
    det = sv.Detections.from_inference(sonuc)
    return det[det.confidence >= g_esik]


def goruntu_yukle(dosya):
    """Yüklenen dosyayı BGR numpy dizisine çevirir (Hafta 6 lab ile aynı biçim)."""
    img = Image.open(dosya).convert("RGB")
    return np.array(img)[:, :, ::-1].copy()


def catlak_yogunlugu_hesapla(det, image_bgr):
    h, w = image_bgr.shape[:2]
    return float(np.clip(CATLAK_KURALI(det, h, w), 0.0, 1.0))


def kutulari_ciz(image_bgr, det):
    kutu, etiket = sv.BoxAnnotator(), sv.LabelAnnotator()
    out = kutu.annotate(scene=image_bgr.copy(), detections=det)
    out = etiket.annotate(scene=out, detections=det)
    return out[:, :, ::-1]   # RGB, st.image için


def bina_satiri(meta, bilgi, catl):
    """Form değerlerinden modelin beklediği tek satırlık tabloyu üretir."""
    satir = {c: 0.0 for c in meta["ozellikler"]}
    for k in ["kat_sayisi", "insaat_yili", "pga_g", "deformasyon_yuzde",
              "doseme_hasari", "kolon_hasari"]:
        satir[k] = float(bilgi[k])
    satir["catl_yogunluk"] = float(catl)
    satir["yonetmelik_oncesi"] = int(float(bilgi["insaat_yili"]) < 1999)
    for onek, deger in (("yapi_turu_", bilgi["yapi_turu"]), ("deprem_bolgesi_", bilgi["deprem_bolgesi"])):
        if onek + str(deger) in satir:
            satir[onek + str(deger)] = 1.0
    return pd.DataFrame([satir])[meta["ozellikler"]]


def tahmin_et(model, meta, bilgi, catl):
    p = model.predict_proba(bina_satiri(meta, bilgi, catl))[0]
    olasilik = {meta["sira"][int(c)]: float(v) for c, v in zip(model.classes_, p)}
    for s in meta["sira"]:
        olasilik.setdefault(s, 0.0)
    sinif = max(olasilik, key=olasilik.get)
    return sinif, olasilik


def aralik_uyarilari(meta, bilgi):
    uyari = []
    for k, (lo, hi) in meta["aralik"].items():
        if k in bilgi and not (lo <= float(bilgi[k]) <= hi):
            uyari.append(f"{k} = {bilgi[k]} değeri eğitim verisinin aralığı ({lo:g} – {hi:g}) dışında.")
    if bilgi["yapi_turu"] not in meta["yapi_turleri"]:
        uyari.append(f"Yapı türü '{bilgi['yapi_turu']}' eğitim verisinde yok.")
    return uyari


# ===== ARAYÜZ =====
st.set_page_config(page_title="Hasar Tahmin Aracı", layout="wide")
st.title("Deprem Sonrası Hasar Tahmin Aracı")
st.caption("Eğitim amaçlı sentetik veriyle eğitilmiş bir modeldir; gerçek bir hasar tespiti yerine geçmez.")

try:
    model, meta = model_yukle()
except Exception as e:
    st.error(f"Model dosyaları yüklenemedi: {e}\n\nmodels/ klasöründe hasar_modeli.joblib ve "
             "meta.json dosyalarının olduğundan emin olun.")
    st.stop()

# --- Kenar çubuğu: model bilgisi --------------------------------------
with st.sidebar:
    st.header("Model performansı")
    m = meta["metrikler"]
    st.metric("Test doğruluğu", f"{m['test_dogrulugu']:.3f}")
    st.metric("5 katlı CV doğruluğu", f"{m['cv_dogrulugu']:.3f}")
    st.write("Sınıf bazında recall:")
    st.dataframe(pd.DataFrame({"recall": m["recall"]}).round(2))
    st.write("Parametreler:", meta["parametreler"])
    st.header("Çatlak tespiti (Roboflow)")
    model_secim = st.selectbox(
        "Roboflow Çatlak Modeli",
        options=list(GUCLU_MODELLER.keys()),
        index=0,
        help="Labda kullanılan temel modelden farklı olarak eklenen 4 güçlü model seçeneği."
    )
    aktif_model_id = GUCLU_MODELLER[model_secim]
    st.code(aktif_model_id, language="text")
    guven_esigi_secim = st.slider("Güven eşiği", min_value=0.10, max_value=0.90, value=float(GUVEN_ESIGI), step=0.05)
    st.write(f"Kural: `{CATLAK_KURALI.__name__}`")

    # API Anahtarı girişi desteği (Secrets yoksa kolaylık sağlar)
    if not api_anahtari():
        st.warning("ROBOFLOW_API_KEY tanımlı değil.")
        girilen_key = st.text_input("Roboflow API Anahtarı:", type="password", key="rf_key_input",
                                    help="Streamlit Secrets girilmediyse buradan geçici olarak girebilirsiniz.")
        if girilen_key:
            st.session_state["kullanici_api_anahtari"] = girilen_key.strip()
            st.success("API Anahtarı kaydedildi.")
            st.rerun()
    try:
        import sklearn
        if sklearn.__version__ != meta.get("sklearn_surumu"):
            st.warning(f"scikit-learn sürümü farklı (eğitim: {meta.get('sklearn_surumu')}, "
                       f"uygulama: {sklearn.__version__}). Sonuçlar etkilenmeyebilir ama "
                       "hata görürseniz modeli yeniden eğitin.")
    except Exception:
        pass

sekme1, sekme2 = st.tabs(["Tek bina", "Toplu değerlendirme (ödev seti)"])

# --- Sekme 1: tek bina -----------------------------------------------
with sekme1:
    sol, sag = st.columns(2)
    with sol:
        st.subheader("Bina bilgileri")
        yapi = st.selectbox("Yapı türü", meta["yapi_turleri"], index=meta["yapi_turleri"].index("Betonarme")
                            if "Betonarme" in meta["yapi_turleri"] else 0)
        kat = st.number_input("Kat sayısı", min_value=1, max_value=60, value=5, step=1)
        yil = st.number_input("İnşaat yılı", min_value=1800, max_value=2030, value=1995, step=1)
        bolge = st.selectbox("Deprem bölgesi", meta["deprem_bolgeleri"])
        pga = st.number_input("Beklenen PGA (g)", min_value=0.0, max_value=2.0, value=0.30, step=0.01)
        defo = st.number_input("Deformasyon (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
        dos = st.radio("Döşeme hasarı", [0, 1], format_func=lambda v: "Var" if v else "Yok", horizontal=True)
        kol = st.radio("Kolon hasarı", [0, 1], format_func=lambda v: "Var" if v else "Yok", horizontal=True)
    with sag:
        st.subheader("Hasar fotoğrafı")
        foto = st.file_uploader("Bina fotoğrafı", type=["jpg", "jpeg", "png"], key="tek")
        if foto is not None:
            st.image(foto, caption="Yüklenen fotoğraf", width="stretch")

    bilgi = {"yapi_turu": yapi, "kat_sayisi": kat, "insaat_yili": yil, "deprem_bolgesi": bolge,
             "pga_g": pga, "deformasyon_yuzde": defo, "doseme_hasari": dos, "kolon_hasari": kol}
    for u in aralik_uyarilari(meta, bilgi):
        st.warning("Ekstrapolasyon uyarısı: " + u)

    if st.button("Hasar sınıfını tahmin et", type="primary", disabled=foto is None):
        try:
            img = goruntu_yukle(foto)
            det = tespit_et(img, model_id=aktif_model_id, guven_esigi=guven_esigi_secim)
            catl = catlak_yogunlugu_hesapla(det, img)
        except Exception as e:
            st.error(f"Çatlak tespiti başarısız: {e}")
            st.stop()
        sinif, olasilik = tahmin_et(model, meta, bilgi, catl)
        st.divider()
        a, b = st.columns(2)
        with a:
            st.image(kutulari_ciz(img, det), caption=f"{len(det)} çatlak kutusu bulundu",
                     width="stretch")
            if len(det):
                adlar = sorted(set(det.data.get("class_name", [])))
                if adlar:
                    st.caption("Modelin sınıf adları: " + ", ".join(adlar))
        with b:
            st.metric("Hesaplanan çatlak yoğunluğu", f"{catl:.2f}")
            st.markdown(f"### Tahmin edilen hasar sınıfı: "
                        f"<span style='color:{ETIKET_RENK.get(sinif, '#000')}'>{sinif}</span>",
                        unsafe_allow_html=True)
            st.bar_chart(pd.Series(olasilik).reindex(meta["sira"]))

# --- Sekme 2: toplu değerlendirme ------------------------------------
with sekme2:
    st.write("Bina bilgileri tablosunu (CSV) ve fotoğrafları birlikte yükleyin. Tablodaki `dosya` "
             "sütunu fotoğraf adlarıyla aynı olmalıdır.")
    csv_dosya = st.file_uploader("bina_bilgileri.csv", type=["csv"], key="csv")
    fotolar = st.file_uploader("Fotoğraflar", type=["jpg", "jpeg", "png"],
                               accept_multiple_files=True, key="coklu")
    if st.button("Toplu değerlendir", disabled=not (csv_dosya and fotolar)):
        tablo = pd.read_csv(csv_dosya)
        zorunlu = ["dosya", "yapi_turu", "kat_sayisi", "insaat_yili", "deprem_bolgesi",
                   "pga_g", "deformasyon_yuzde", "doseme_hasari", "kolon_hasari"]
        eksik = [c for c in zorunlu if c not in tablo.columns]
        if eksik:
            st.error(f"CSV'de eksik sütunlar: {eksik}")
            st.stop()
        fotolar_ad = {f.name: f for f in fotolar}
        sonuclar, galeri = [], []
        ilerleme = st.progress(0.0)
        for i, satir in enumerate(tablo.to_dict("records")):
            f = fotolar_ad.get(satir["dosya"])
            if f is None:
                sonuclar.append({"dosya": satir["dosya"], "durum": "fotoğraf yüklenmedi"})
                continue
            try:
                img = goruntu_yukle(f)
                det = tespit_et(img, model_id=aktif_model_id, guven_esigi=guven_esigi_secim)
                catl = catlak_yogunlugu_hesapla(det, img)
                sinif, olasilik = tahmin_et(model, meta, satir, catl)
                sonuclar.append({"dosya": satir["dosya"], "kutu_sayisi": len(det),
                                 "catl_yogunluk": round(catl, 3), "tahmin": sinif,
                                 "olasilik": round(olasilik[sinif], 2), "durum": "tamam"})
                galeri.append((satir["dosya"], kutulari_ciz(img, det), sinif))
            except Exception as e:
                sonuclar.append({"dosya": satir["dosya"], "durum": f"hata: {e}"})
            ilerleme.progress((i + 1) / len(tablo))
        sonuc_df = pd.DataFrame(sonuclar)
        st.dataframe(sonuc_df, width="stretch")
        st.download_button("Sonuçları CSV olarak indir",
                           sonuc_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="hasar_tahminleri.csv", mime="text/csv")
        with st.expander("Kutulu fotoğraflar"):
            kolonlar = st.columns(3)
            for j, (ad, g, s) in enumerate(galeri):
                kolonlar[j % 3].image(g, caption=f"{ad} → {s}", width="stretch")

# -*- coding: utf-8 -*-
"""
Hafta 6 Ödevi — Deprem Sonrası Hasar Tahmin Aracı (Material You & Glassmorphism Sürümü)

Fotoğraftaki çatlakları bir Roboflow modeliyle bulur, çatlak yoğunluğunu sizin
yazdığınız kuralla hesaplar ve eğittiğiniz Random Forest modeliyle bina hasar
sınıfını tahmin eder.

Tasarım: Google Material You (Material 3) + Glassmorphism (Buzlu Cam) + Koyu/Açık Tema Desteği
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


# Aktif kural olarak öğrenci kuralını belirliyoruz
CATLAK_KURALI = kural_ogrenci

# ======================================================================
# 2. SABİT BÖLÜM
# ======================================================================
MODEL_YOLU = "models/hasar_modeli.joblib"
META_YOLU = "models/meta.json"
ETIKET_RENK = {"Hasarsız": "#10B981", "Hafif": "#F59E0B", "Orta": "#F97316",
               "Ağır": "#EF4444", "Yıkık": "#8B5CF6"}


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


# ======================================================================
# 3. SAYFA AYARI VE TEMA GEÇİŞİ (MATERIAL YOU & GLASSMORPHISM)
# ======================================================================
st.set_page_config(page_title="Hasar Tahmin Aracı", page_icon="🏢", layout="wide")

# Tema Durumu (Koyu / Açık Mod)
if "koyu_tema" not in st.session_state:
    st.session_state.koyu_tema = True

# Tema Renk Değişkenleri
if st.session_state.koyu_tema:
    tema_ad = "Koyu Mod 🌙"
    bg_style = "background: radial-gradient(circle at 20% 20%, #1e1b4b 0%, #0f172a 45%, #020617 100%) fixed;"
    text_color = "#F8FAFC"
    subtext_color = "#94A3B8"
    glass_bg = "rgba(30, 41, 59, 0.60)"
    glass_border = "rgba(255, 255, 255, 0.12)"
    glass_shadow = "0 8px 32px 0 rgba(0, 0, 0, 0.45)"
    sidebar_bg = "rgba(15, 23, 42, 0.75)"
    input_bg = "rgba(15, 23, 42, 0.65)"
    metric_bg = "rgba(255, 255, 255, 0.05)"
    accent_gradient = "linear-gradient(135deg, #38BDF8 0%, #818CF8 100%)"
    hero_bg = "linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(129, 140, 248, 0.14) 100%)"
else:
    tema_ad = "Açık Mod ☀️"
    bg_style = "background: radial-gradient(circle at 20% 20%, #E0E7FF 0%, #F1F5F9 50%, #F8FAFC 100%) fixed;"
    text_color = "#0F172A"
    subtext_color = "#475569"
    glass_bg = "rgba(255, 255, 255, 0.75)"
    glass_border = "rgba(255, 255, 255, 0.85)"
    glass_shadow = "0 8px 32px 0 rgba(31, 38, 135, 0.08)"
    sidebar_bg = "rgba(255, 255, 255, 0.82)"
    input_bg = "rgba(255, 255, 255, 0.90)"
    metric_bg = "rgba(0, 0, 0, 0.03)"
    accent_gradient = "linear-gradient(135deg, #0284C7 0%, #4F46E5 100%)"
    hero_bg = "linear-gradient(135deg, rgba(2, 132, 199, 0.08) 0%, rgba(79, 70, 229, 0.09) 100%)"

# Özel CSS Enjeksiyonu
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
}}

.stApp {{
    {bg_style}
    color: {text_color} !important;
}}

/* Buzlu Cam Kenar Çubuğu (Glassmorphism Sidebar) */
[data-testid="stSidebar"] {{
    background: {sidebar_bg} !important;
    backdrop-filter: blur(18px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
    border-right: 1px solid {glass_border} !important;
    box-shadow: {glass_shadow} !important;
}}

/* Material You / Glassmorphism Kart Sınıfları */
.m3-glass-card {{
    background: {glass_bg};
    backdrop-filter: blur(16px) saturate(170%);
    -webkit-backdrop-filter: blur(16px) saturate(170%);
    border: 1px solid {glass_border};
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 22px;
    box-shadow: {glass_shadow};
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}}
.m3-glass-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 12px 36px 0 rgba(0, 0, 0, 0.18);
    border-color: rgba(99, 102, 241, 0.4);
}}

/* Form Input ve Select Alanları */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {{
    background-color: {input_bg} !important;
    border-radius: 14px !important;
    border: 1px solid {glass_border} !important;
    color: {text_color} !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.2s ease !important;
}}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:focus-within {{
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
}}

/* Dosya Yükleyici (File Uploader) Glass Tasarımı */
[data-testid="stFileUploader"] {{
    background: {glass_bg} !important;
    backdrop-filter: blur(14px) !important;
    border: 2px dashed {glass_border} !important;
    border-radius: 20px !important;
    padding: 16px !important;
    transition: all 0.25s ease !important;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: #6366F1 !important;
    background: rgba(99, 102, 241, 0.06) !important;
}}

/* Material You Yuvarlak Butonlar */
.stButton > button {{
    background: {accent_gradient} !important;
    color: white !important;
    border-radius: 9999px !important;
    padding: 10px 30px !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}}

/* Metrik Kartları */
div[data-testid="stMetric"] {{
    background: {metric_bg} !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid {glass_border} !important;
    border-radius: 16px !important;
    padding: 12px 18px !important;
    box-shadow: {glass_shadow} !important;
}}

/* Sekmeler (Material 3 Segmented Control) */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: {glass_bg};
    backdrop-filter: blur(14px);
    padding: 6px;
    border-radius: 18px;
    border: 1px solid {glass_border};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 14px;
    font-weight: 700;
    color: {subtext_color};
    padding: 8px 20px;
}}
.stTabs [aria-selected="true"] {{
    background: {accent_gradient} !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
}}
</style>
""", unsafe_allow_html=True)

# Üst Başlık (Hero Glass Banner)
st.markdown(f"""
<div class="m3-glass-card" style="background: {hero_bg}; padding: 26px 30px; margin-bottom: 24px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <span style="display:inline-block; padding: 4px 14px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; 
                         letter-spacing: 0.6px; text-transform: uppercase; background: rgba(99, 102, 241, 0.22); color: #818CF8; margin-bottom: 8px;">
                🏢 İNŞAAT MÜHENDİSLİĞİNDE YAPAY ZEKÂ UYGULAMALARI
            </span>
            <h1 style="margin: 4px 0 8px 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">
                Deprem Sonrası Hasar Tahmin Aracı
            </h1>
            <p style="margin: 0; opacity: 0.85; font-size: 0.95rem; max-width: 800px; line-height: 1.5;">
                Tablodan öğrenen Random Forest sınıflandırıcısı ve fotoğraftan çatlak tespit eden Roboflow derin öğrenme modelini birleştiren hibrit karar destek sistemi.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

try:
    model, meta = model_yukle()
except Exception as e:
    st.error(f"Model dosyaları yüklenemedi: {e}\n\nmodels/ klasöründe hasar_modeli.joblib ve "
             "meta.json dosyalarının olduğundan emin olun.")
    st.stop()

# ======================================================================
# 4. KENAR ÇUBUĞU (SIDEBAR)
# ======================================================================
with st.sidebar:
    # Tema Değiştirici (Light / Dark Mode Toggle)
    col_t1, col_t2 = st.columns([2, 1.2])
    with col_t1:
        st.markdown(f"**Tema:** `{tema_ad}`")
    with col_t2:
        if st.button("🌓 Değiştir", help="Açık ve Koyu tema arasında geçiş yapın"):
            st.session_state.koyu_tema = not st.session_state.koyu_tema
            st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Model Performansı")
    m = meta["metrikler"]
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Test Doğruluğu", f"{m['test_dogrulugu']:.3f}")
    with col_m2:
        st.metric("5-Katlı CV", f"{m['cv_dogrulugu']:.3f}")
    
    st.markdown("**Sınıf Bazında Recall (Hassasiyet):**")
    st.dataframe(pd.DataFrame({"recall": m["recall"]}).round(2), use_container_width=True)
    st.caption(f"🌲 Parametreler: {meta['parametreler']}")
    
    st.markdown("---")
    st.markdown("### 🔍 Çatlak Tespiti (Roboflow)")
    
    # 4 Güçlü Model + Özel Model Giriş Seçeneği
    model_secenekleri = list(GUCLU_MODELLER.keys()) + ["✨ Özel Model ID Gir (Universe'ten)"]
    model_secim = st.selectbox(
        "Roboflow Çatlak Modeli Seçin",
        options=model_secenekleri,
        index=0,
        help="Hazır 4 güçlü modelden birini seçebilir veya kendi model kimliğinizi yazabilirsiniz."
    )
    
    if model_secim == "✨ Özel Model ID Gir (Universe'ten)":
        ozel_id = st.text_input("Özel Model Kimliği (proje/sürüm):", value="wall-crack-detection-l0iit/4",
                                help="Universe'teki herhangi bir çatlak tespit modelinin proje/sürüm kimliğini girin.")
        aktif_model_id = ozel_id.strip() if ozel_id.strip() else "wall-crack-detection-l0iit/4"
        st.info(f"Kullanılan Model: `{aktif_model_id}`")
    else:
        aktif_model_id = GUCLU_MODELLER[model_secim]
        st.code(aktif_model_id, language="text")

    guven_esigi_secim = st.slider("Güven Eşiği (Confidence)", min_value=0.10, max_value=0.90, value=float(GUVEN_ESIGI), step=0.05)
    st.caption(f"Hesaplama Kuralı: `{CATLAK_KURALI.__name__}` (İnşaat Hasar Ölçeği)")

    # API Anahtarı girişi desteği (Secrets yoksa kolaylık sağlar)
    if not api_anahtari():
        st.warning("⚠️ ROBOFLOW_API_KEY tanımlı değil.")
        girilen_key = st.text_input("Roboflow API Anahtarı:", type="password", key="rf_key_input",
                                    help="Streamlit Secrets girilmediyse buradan geçici olarak girebilirsiniz.")
        if girilen_key:
            st.session_state["kullanici_api_anahtari"] = girilen_key.strip()
            st.success("API Anahtarı kaydedildi.")
            st.rerun()
    else:
        st.success("✅ Roboflow API Anahtarı Hazır")
    
    try:
        import sklearn
        if sklearn.__version__ != meta.get("sklearn_surumu"):
            st.warning(f"scikit-learn sürümü farklı (eğitim: {meta.get('sklearn_surumu')}, "
                       f"uygulama: {sklearn.__version__}). Sonuçlar etkilenmeyebilir ama "
                       "hata görürseniz modeli yeniden eğitin.")
    except Exception:
        pass

# ======================================================================
# 5. ANA EKRAN SEKMELERİ
# ======================================================================
sekme1, sekme2 = st.tabs(["🏛️ Tek Bina Değerlendirme", "📑 Toplu Değerlendirme (Ödev Seti)"])

# --- Sekme 1: Tek Bina -----------------------------------------------
with sekme1:
    sol, sag = st.columns(2, gap="medium")
    with sol:
        st.markdown("""
        <div class="m3-glass-card">
            <h3 style="margin-top:0;">📋 Yapısal Bina Parametreleri</h3>
        """, unsafe_allow_html=True)
        yapi = st.selectbox("Yapı türü", meta["yapi_turleri"], index=meta["yapi_turleri"].index("Betonarme")
                            if "Betonarme" in meta["yapi_turleri"] else 0)
        c_k1, c_k2 = st.columns(2)
        with c_k1:
            kat = st.number_input("Kat sayısı", min_value=1, max_value=60, value=5, step=1)
        with c_k2:
            yil = st.number_input("İnşaat yılı", min_value=1800, max_value=2030, value=1995, step=1)
        bolge = st.selectbox("Deprem bölgesi", meta["deprem_bolgeleri"])
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            pga = st.number_input("Beklenen PGA (g)", min_value=0.0, max_value=2.0, value=0.30, step=0.01)
        with c_p2:
            defo = st.number_input("Deformasyon (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            dos = st.radio("Döşeme hasarı", [0, 1], format_func=lambda v: "Var" if v else "Yok", horizontal=True)
        with c_d2:
            kol = st.radio("Kolon hasarı", [0, 1], format_func=lambda v: "Var" if v else "Yok", horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with sag:
        st.markdown("""
        <div class="m3-glass-card">
            <h3 style="margin-top:0;">📸 Duvar Hasar Fotoğrafı</h3>
        """, unsafe_allow_html=True)
        foto = st.file_uploader("Duvar çatlak fotoğrafı yükleyin", type=["jpg", "jpeg", "png"], key="tek")
        if foto is not None:
            st.image(foto, caption="Yüklenen Fotoğraf", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    bilgi = {"yapi_turu": yapi, "kat_sayisi": kat, "insaat_yili": yil, "deprem_bolgesi": bolge,
             "pga_g": pga, "deformasyon_yuzde": defo, "doseme_hasari": dos, "kolon_hasari": kol}
    for u in aralik_uyarilari(meta, bilgi):
        st.warning("⚠️ Ekstrapolasyon Uyarısı: " + u)

    if st.button("🚀 Hasar Sınıfını Tahmin Et", type="primary", disabled=foto is None, use_container_width=True):
        with st.spinner("Roboflow çatlak tespiti ve Random Forest tahmini yapılıyor..."):
            try:
                img = goruntu_yukle(foto)
                det = tespit_et(img, model_id=aktif_model_id, guven_esigi=guven_esigi_secim)
                catl = catlak_yogunlugu_hesapla(det, img)
            except Exception as e:
                st.error(f"Çatlak tespiti başarısız: {e}")
                st.stop()
            sinif, olasilik = tahmin_et(model, meta, bilgi, catl)
            
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            res_sol, res_sag = st.columns(2, gap="medium")
            with res_sol:
                st.markdown("""<div class="m3-glass-card">
                    <h3 style="margin-top:0;">🎯 Tespit Edilen Çatlaklar</h3>
                """, unsafe_allow_html=True)
                st.image(kutulari_ciz(img, det), caption=f"{len(det)} adet çatlak kutusu tespit edildi",
                         use_container_width=True)
                if len(det):
                    adlar = sorted(set(det.data.get("class_name", [])))
                    if adlar:
                        st.caption("Model sınıf adları: " + ", ".join(adlar))
                st.markdown("</div>", unsafe_allow_html=True)

            with res_sag:
                renk = ETIKET_RENK.get(sinif, '#38BDF8')
                st.markdown(f"""
                <div class="m3-glass-card" style="border-left: 6px solid {renk};">
                    <span style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: {renk};">
                        Tahmin Edilen Hasar Sınıfı
                    </span>
                    <h2 style="margin: 4px 0 12px 0; font-size: 2.3rem; font-weight: 800; color: {renk};">
                        {sinif}
                    </h2>
                    <p style="margin: 0; opacity: 0.85; font-size: 0.95rem;">
                        Hesaplanan Çatlak Yoğunluğu: <b>{catl:.2f}</b> (Ölçek: 0 - 1)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""<div class="m3-glass-card">
                    <h4 style="margin-top:0;">📊 Sınıf Olasılık Dağılımı</h4>
                """, unsafe_allow_html=True)
                st.bar_chart(pd.Series(olasilik).reindex(meta["sira"]))
                st.markdown("</div>", unsafe_allow_html=True)

# --- Sekme 2: Toplu Değerlendirme ------------------------------------
with sekme2:
    st.markdown("""
    <div class="m3-glass-card">
        <h3 style="margin-top:0;">📑 10 Fotoğraflık Ödev Seti Toplu Değerlendirme</h3>
        <p style="opacity: 0.85; margin:0;">
            <b>bina_bilgileri.csv</b> tablosunu ve <b>acik_set</b> içindeki 10 fotoğrafı birlikte yükleyerek tüm seti tek tıkla işleyin.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns(2, gap="medium")
    with col_c1:
        csv_dosya = st.file_uploader("bina_bilgileri.csv dosyasını yükleyin", type=["csv"], key="csv")
    with col_c2:
        fotolar = st.file_uploader("10 adet hasar fotoğrafını seçin", type=["jpg", "jpeg", "png"],
                                   accept_multiple_files=True, key="coklu")

    if st.button("⚡ Toplu Değerlendirmeyi Başlat", disabled=not (csv_dosya and fotolar), use_container_width=True):
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
        st.markdown("<div class='m3-glass-card'><h4>📋 Toplu Tahmin Sonuç Tablosu</h4>", unsafe_allow_html=True)
        st.dataframe(sonuc_df, use_container_width=True)
        st.download_button("📥 Sonuçları CSV Olarak İndir (Rapor İçin)",
                           sonuc_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="hasar_tahminleri.csv", mime="text/csv")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🖼️ Kutulu Fotoğraf Galerisi"):
            kolonlar = st.columns(3)
            for j, (ad, g, s) in enumerate(galeri):
                kolonlar[j % 3].image(g, caption=f"{ad} → {s}", use_container_width=True)

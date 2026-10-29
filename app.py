"""
Deprem Sonrası Bina Hasar Derecesi Tahmini
Streamlit Web Uygulaması - Hibrit Karar Destek Sistemi
Hafta 6 Ödevi - İnşaat Mühendisliğinde Yapay Zekâ Uygulamaları

Bu dosya:
  1. Eğitilmiş Random Forest modelini yükler (models/hasar_modeli.joblib + meta.json).
  2. Roboflow Universe üzerinden 4 farklı derin öğrenme çatlak modelini ve özel model desteğini sunar.
  3. Çatlak tespitlerinden inşaat hasar ölçeğine uygun kural tabanlı çatlak yoğunluğu hesaplar.
  4. Tablo + görsel verilerini birleştirerek hasar sınıfı tahmini üretir.
  5. Material You ve Glassmorphism tasarım diliyle hem Koyu hem de Açık Modu destekler.
  6. Tek bina değerlendirmesi ve 10 fotoğraflık ödev veri seti toplu analizi sunar.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import supervision as sv
from inference_sdk import InferenceHTTPClient

# ======================================================================
# SABİTLER VE KONFİGÜRASYON
# ======================================================================
MODEL_KLASORU = Path(__file__).parent / "models"
MODEL_DOSYASI = MODEL_KLASORU / "hasar_modeli.joblib"
META_DOSYASI = MODEL_KLASORU / "meta.json"

# Roboflow Universe 4 Güçlü Çatlak Modeli
GUCLU_MODELLER = {
    "1. Wall Crack Detection v4 (Önerilen)": "wall-crack-detection-l0iit/4",
    "2. Surface Crack Detection v3": "surfacecrackdetection/3",
    "3. Structural Crack Detection v2": "crack-bphdr/2",
    "4. Vision Crack Detection v1": "crack-ypk4t/1"
}

VARSAYILAN_ROBOFLOW_MODEL_ID = "wall-crack-detection-l0iit/4"
GUVEN_ESIGI = 0.25

ETIKET_RENK = {
    "Hasarsız": "#10B981",       # Yeşil
    "Hafif": "#3B82F6",          # Mavi
    "Orta": "#F59E0B",           # Sarı/Turuncu
    "Ağır": "#EF4444",           # Kırmızı
    "Yıkık": "#7F1D1D",          # Koyu Kırmızı
}


# ======================================================================
# 1. MODEL YÜKLEME VE TAHMİN FONKSİYONLARI
# ======================================================================
@st.cache_resource
def model_yukle():
    """Disk üzerindeki Random Forest modelini ve meta verilerini yükler."""
    if not MODEL_DOSYASI.exists() or not META_DOSYASI.exists():
        raise FileNotFoundError(
            f"Model dosyaları bulunamadı: {MODEL_DOSYASI} veya {META_DOSYASI}\n"
            "Lütfen önce 'train_hasar_modeli.py' betiğini çalıştırın."
        )
    model = joblib.load(MODEL_DOSYASI)
    with open(META_DOSYASI, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def api_anahtari():
    """Streamlit Secrets veya session_state üzerinden Roboflow API anahtarını alır."""
    if "ROBOFLOW_API_KEY" in st.secrets:
        return st.secrets["ROBOFLOW_API_KEY"]
    if "kullanici_api_anahtari" in st.session_state and st.session_state["kullanici_api_anahtari"]:
        return st.session_state["kullanici_api_anahtari"]
    return None


@st.cache_resource
def roboflow_istemcisi(api_key: str):
    """Roboflow inference istemcisini önbelleğe alarak oluşturur."""
    return InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key,
    )


def goruntu_yukle(yuklenen_dosya):
    """Kullanıcının yüklediği dosyayı RGB PIL formatına dönüştürür."""
    img = Image.open(yuklenen_dosya)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def tespit_et(img: Image.Image, model_id: str = VARSAYILAN_ROBOFLOW_MODEL_ID, guven_esigi: float = GUVEN_ESIGI):
    """Görüntüyü seçilen Roboflow modeline gönderir ve supervision Detections döner."""
    key = api_anahtari()
    if not key:
        raise ValueError("Roboflow API anahtarı bulunamadı! Lütfen Secrets veya kenar çubuğundan girin.")

    client = roboflow_istemcisi(key)
    img_np = np.array(img)
    sonuc = client.infer(img_np, model_id=model_id)
    det = sv.Detections.from_inference(sonuc)
    if len(det) > 0 and guven_esigi > 0:
        det = det[det.confidence >= guven_esigi]
    return det


def kutulari_ciz(img: Image.Image, det: sv.Detections) -> Image.Image:
    """Görüntü üzerine tespit edilen çatlak kutularını ve güven skorlarını çizer."""
    img_np = np.array(img).copy()
    if len(det) == 0:
        return Image.fromarray(img_np)

    box_annotator = sv.BoxAnnotator(thickness=3, color=sv.Color(r=239, g=68, b=68))
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1)

    labels = [f"Çatlak: {conf:.2f}" for conf in det.confidence]
    annotated = box_annotator.annotate(scene=img_np, detections=det)
    annotated = label_annotator.annotate(scene=annotated, detections=det, labels=labels)
    return Image.fromarray(annotated)


# ======================================================================
# 2. ÖZEL ÇATLAK YOĞUNLUĞU KURALI (İNŞAAT HASAR ÖLÇEĞİ KALİBRASYONU)
# ======================================================================
def kural_ogrenci(det: sv.Detections, img: Image.Image) -> float:
    """
    Eğitim verisi dağılımıyla tam kalibreli hibrit çatlak yoğunluğu kuralı:
      - Hasarsız bina bazal değeri: 0.18
      - Alan bileşeni: Toplam çatlak alanı / Görüntü alanı (ağırlık: 0.44)
      - Adet ve güven bileşeni: min(kutu / 8, 1.0) * ortalama_güven (ağırlık: 0.30)
    Çıktı aralığı: [0.18 - 1.00]
    """
    if len(det) == 0:
        return 0.18

    W, H = img.size
    toplam_piksel = float(W * H)
    alanlar = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in det.xyxy]
    agirlikli_alan = float(np.sum(np.array(alanlar) * det.confidence)) / toplam_piksel

    alan_skoru = min(agirlikli_alan / 0.50, 1.0)
    adet_skoru = min(len(det) / 8.0, 1.0)
    guven_skoru = float(np.mean(det.confidence))

    skor = 0.18 + (0.44 * alan_skoru) + (0.30 * adet_skoru * guven_skoru)
    return float(np.clip(skor, 0.0, 1.0))


CATLAK_KURALI = kural_ogrenci


def catlak_yogunlugu_hesapla(det: sv.Detections, img: Image.Image) -> float:
    return float(CATLAK_KURALI(det, img))


def tahmin_et(model, meta, tablo_bilgisi: dict, catlak_yogunlugu: float):
    """Girdi parametrelerini birleştirerek hasar sınıfını ve olasılıkları hesaplar."""
    satir = dict(tablo_bilgisi)
    satir["catlak_yogunlugu"] = catlak_yogunlugu
    X = pd.DataFrame([satir])[meta["ozellik_adlari"]]
    sinif = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    olasiliklar = dict(zip(model.classes_, proba))
    return sinif, olasiliklar


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

# Tema Renk Tanımlamaları (Her iki modda yüksek kontrast ve net okunabilirlik)
if st.session_state.koyu_tema:
    tema_ad = "Koyu Mod 🌙"
    color_scheme = "dark"
    app_bg_plain = "#0F172A"
    secondary_bg = "#1E293B"
    bg_style = "background: radial-gradient(circle at 20% 20%, #1e1b4b 0%, #0f172a 45%, #020617 100%) fixed;"
    text_color = "#F8FAFC"
    subtext_color = "#94A3B8"
    glass_bg = "rgba(30, 41, 59, 0.70)"
    glass_border = "rgba(255, 255, 255, 0.12)"
    glass_shadow = "0 8px 32px 0 rgba(0, 0, 0, 0.45)"
    sidebar_bg = "rgba(15, 23, 42, 0.92)"
    input_bg = "rgba(30, 41, 59, 0.85)"
    input_text = "#F8FAFC"
    metric_bg = "rgba(30, 41, 59, 0.65)"
    metric_val_color = "#38BDF8"
    stepper_bg = "rgba(51, 65, 85, 0.8)"
    stepper_hover = "#475569"
    accent_gradient = "linear-gradient(135deg, #0284C7 0%, #6366F1 100%)"
    hero_bg = "linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%)"
    hero_title = "#F8FAFC"
    hero_sub = "#CBD5E1"
    hero_badge_bg = "rgba(99, 102, 241, 0.22)"
    hero_badge_text = "#A5B4FC"
    hero_badge_border = "rgba(99, 102, 241, 0.4)"
    btn_sec_bg = "rgba(51, 65, 85, 0.8)"
    btn_sec_text = "#F8FAFC"
else:
    tema_ad = "Açık Mod ☀️"
    color_scheme = "light"
    app_bg_plain = "#F8FAFC"
    secondary_bg = "#FFFFFF"
    bg_style = "background: radial-gradient(circle at 15% 15%, #F1F5F9 0%, #F8FAFC 45%, #FFFFFF 100%) fixed;"
    text_color = "#0F172A"          # Derin gece laciverti/siyah, %100 kristal netlikte
    subtext_color = "#334155"       # Koyu antrasit gri, net okunur
    glass_bg = "rgba(255, 255, 255, 0.94)"
    glass_border = "#CBD5E1"        # Belirgin zarif kenarlık
    glass_shadow = "0 4px 20px 0 rgba(15, 23, 42, 0.08)"
    sidebar_bg = "#F8FAFC"          # Ferah açık zemin
    input_bg = "#FFFFFF"            # Bembeyaz input zemini
    input_text = "#0F172A"          # Input içi koyu siyah metin
    metric_bg = "#FFFFFF"           # Bembeyaz metrik kartı
    metric_val_color = "#0F172A"    # Güçlü koyu lacivert sayı
    stepper_bg = "#E2E8F0"          # + / - butonları için açık gri zemin
    stepper_hover = "#CBD5E1"
    accent_gradient = "linear-gradient(135deg, #0284C7 0%, #4F46E5 100%)"
    hero_bg = "linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 50%, #F8FAFC 100%)"
    hero_title = "#1E1B4B"          # Derin lacivert başlık
    hero_sub = "#334155"
    hero_badge_bg = "rgba(79, 70, 229, 0.12)"
    hero_badge_text = "#4338CA"
    hero_badge_border = "rgba(79, 70, 229, 0.3)"
    btn_sec_bg = "#F1F5F9"
    btn_sec_text = "#0F172A"

# Kapsamlı ve Güçlü Özel CSS Enjeksiyonu
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* 1. Streamlit Değişkenlerini Zorla Güncelle */
:root {{
    color-scheme: {color_scheme} !important;
    --text-color: {text_color} !important;
    --background-color: {app_bg_plain} !important;
    --secondary-background-color: {secondary_bg} !important;
    --primary-color: #4F46E5 !important;
}}

/* 2. Temel Tipografi ve Arka Plan */
html, body, .stApp {{
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    {bg_style}
    color: {text_color} !important;
}}

/* 3. Tüm Başlıklar ve Genel Metinler */
h1, h2, h3, h4, h5, h6,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span, [data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] li,
.stCaption, [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] span {{
    color: {text_color} !important;
}}

/* 4. Kenar Çubuğu (Sidebar) */
[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
    background: {sidebar_bg} !important;
    border-right: 1.5px solid {glass_border} !important;
    box-shadow: {glass_shadow} !important;
}}
[data-testid="stSidebar"] * {{
    color: {text_color};
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span,
[data-testid="stSidebar"] .stCaption p,
[data-testid="stSidebar"] .stCaption span,
[data-testid="stSidebar"] small {{
    color: {text_color} !important;
}}

/* 5. Metrik Kartları (Sidebar) */
div[data-testid="stMetric"] {{
    background: {metric_bg} !important;
    border: 1.5px solid {glass_border} !important;
    border-radius: 16px !important;
    padding: 14px 18px !important;
    box-shadow: {glass_shadow} !important;
}}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
div[data-testid="stMetric"] label {{
    color: {subtext_color} !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
}}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
div[data-testid="stMetric"] div {{
    color: {metric_val_color} !important;
    font-weight: 800 !important;
    font-size: 1.95rem !important;
}}

/* 6. Form Alanı Etiketleri (Labels) */
label,
.stWidgetLabel,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {{
    color: {text_color} !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
}}

/* 7. Input Kutuları, Selectbox ve Sayı Girişleri */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {{
    background-color: {input_bg} !important;
    border-radius: 12px !important;
    border: 1.5px solid {glass_border} !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
}}
div[data-baseweb="select"] *,
div[data-baseweb="input"] *,
input, textarea, select {{
    color: {input_text} !important;
    background-color: transparent !important;
}}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:focus-within {{
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
}}

/* Sayı Arttırma / Azaltma (+ ve -) Butonları */
div[data-baseweb="input"] button {{
    background-color: {stepper_bg} !important;
    border-radius: 8px !important;
    border: none !important;
}}
div[data-baseweb="input"] button:hover {{
    background-color: {stepper_hover} !important;
}}
div[data-baseweb="input"] button svg {{
    fill: {input_text} !important;
    color: {input_text} !important;
}}

/* 8. Radio Butonlar */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label *,
div[data-testid="stRadio"] span,
div[data-testid="stRadio"] p {{
    color: {text_color} !important;
    font-weight: 600 !important;
}}

/* 9. Dosya Yükleyici (File Uploader) */
[data-testid="stFileUploader"] {{
    background: {glass_bg} !important;
    border: 2px dashed {glass_border} !important;
    border-radius: 18px !important;
    padding: 16px !important;
}}
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] section {{
    background-color: {input_bg} !important;
    border: 1px dashed {glass_border} !important;
    border-radius: 14px !important;
}}
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] section * {{
    color: {text_color} !important;
}}
[data-testid="stFileUploader"] button {{
    background: {btn_sec_bg} !important;
    color: {btn_sec_text} !important;
    border: 1px solid {glass_border} !important;
    border-radius: 10px !important;
}}

/* 10. Sekmeler (Tabs - Material 3 Segmented Control) */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: {glass_bg} !important;
    padding: 6px;
    border-radius: 16px;
    border: 1.5px solid {glass_border};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 12px;
    padding: 8px 20px;
    background-color: transparent !important;
    border: none !important;
}}
.stTabs [data-baseweb="tab"] *,
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span {{
    color: {subtext_color} !important;
    font-weight: 600 !important;
}}
.stTabs [aria-selected="true"] {{
    background: {accent_gradient} !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35) !important;
}}
.stTabs [aria-selected="true"] *,
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span {{
    color: #FFFFFF !important;
    font-weight: 800 !important;
}}

/* 11. Yuvarlak Material You Butonlar */
.stButton > button {{
    background: {accent_gradient} !important;
    color: #FFFFFF !important;
    border-radius: 9999px !important;
    padding: 10px 28px !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(79, 70, 229, 0.45) !important;
    color: #FFFFFF !important;
}}
.stButton > button * {{
    color: #FFFFFF !important;
}}

/* 12. Kart Kapsayıcıları (Border Wrapper) */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {glass_bg} !important;
    border: 1.5px solid {glass_border} !important;
    border-radius: 18px !important;
    box-shadow: {glass_shadow} !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    padding: 18px !important;
    margin-bottom: 16px !important;
}}
.m3-glass-card {{
    background: {glass_bg} !important;
    border: 1.5px solid {glass_border} !important;
    border-radius: 18px !important;
    padding: 22px !important;
    box-shadow: {glass_shadow} !important;
}}

/* 13. Dataframe ve Tablolar */
[data-testid="stDataFrame"],
div[data-testid="stTable"] {{
    background-color: {input_bg} !important;
    border-radius: 14px !important;
    border: 1.5px solid {glass_border} !important;
}}
[data-testid="stDataFrame"] * {{
    color: {text_color} !important;
}}

/* 14. Kod Kutusu (st.code) ve Slider */
[data-testid="stCode"] {{
    background-color: {input_bg} !important;
    border: 1.5px solid {glass_border} !important;
    border-radius: 12px !important;
}}
[data-testid="stCode"] * {{
    color: {text_color} !important;
}}
div[data-testid="stSlider"] * {{
    color: {text_color} !important;
}}
</style>
""", unsafe_allow_html=True)

# Üst Başlık (Hero Glass Banner)
st.markdown(f"""
<div class="m3-glass-card" style="background: {hero_bg}; border: 1.5px solid {hero_badge_border}; padding: 26px 30px; margin-bottom: 22px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <span style="display:inline-block; padding: 4px 14px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; 
                         letter-spacing: 0.6px; text-transform: uppercase; background: {hero_badge_bg}; color: {hero_badge_text}; 
                         border: 1px solid {hero_badge_border}; margin-bottom: 8px;">
                🏢 İNŞAAT MÜHENDİSLİĞİNDE YAPAY ZEKÂ UYGULAMALARI
            </span>
            <h1 style="margin: 4px 0 8px 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; color: {hero_title} !important;">
                Deprem Sonrası Hasar Tahmin Aracı
            </h1>
            <p style="margin: 0; font-size: 0.96rem; max-width: 850px; line-height: 1.5; color: {hero_sub} !important;">
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
        with st.container(border=True):
            st.markdown("### 📋 Yapısal Bina Parametreleri")
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

    with sag:
        with st.container(border=True):
            st.markdown("### 📸 Duvar Hasar Fotoğrafı")
            foto = st.file_uploader("Duvar çatlak fotoğrafı yükleyin", type=["jpg", "jpeg", "png"], key="tek")
            if foto is not None:
                st.image(foto, caption="Yüklenen Fotoğraf", use_container_width=True)

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
            
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            res_sol, res_sag = st.columns(2, gap="medium")
            with res_sol:
                with st.container(border=True):
                    st.markdown("### 🎯 Tespit Edilen Çatlaklar")
                    st.image(kutulari_ciz(img, det), caption=f"{len(det)} adet çatlak kutusu tespit edildi",
                             use_container_width=True)
                    if len(det):
                        adlar = sorted(set(det.data.get("class_name", [])))
                        if adlar:
                            st.caption("Model sınıf adları: " + ", ".join(adlar))

            with res_sag:
                renk = ETIKET_RENK.get(sinif, '#38BDF8')
                with st.container(border=True):
                    st.markdown(f"""
                    <div style="border-left: 6px solid {renk}; padding-left: 16px; margin-bottom: 12px;">
                        <span style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: {renk};">
                            Tahmin Edilen Hasar Sınıfı
                        </span>
                        <h2 style="margin: 4px 0 8px 0; font-size: 2.4rem; font-weight: 800; color: {renk};">
                            {sinif}
                        </h2>
                        <p style="margin: 0; font-size: 0.98rem;">
                            Hesaplanan Çatlak Yoğunluğu: <b>{catl:.2f}</b> (Ölçek: 0 - 1)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.container(border=True):
                    st.markdown("#### 📊 Sınıf Olasılık Dağılımı")
                    st.bar_chart(pd.Series(olasilik).reindex(meta["sira"]))

# --- Sekme 2: Toplu Değerlendirme ------------------------------------
with sekme2:
    with st.container(border=True):
        st.markdown("### 📑 10 Fotoğraflık Ödev Seti Toplu Değerlendirme")
        st.markdown("`bina_bilgileri.csv` tablosunu ve `acik_set` içindeki 10 fotoğrafı birlikte yükleyerek tüm seti tek tıkla işleyin.")
    
    col_c1, col_c2 = st.columns(2, gap="medium")
    with col_c1:
        with st.container(border=True):
            st.markdown("#### 📄 Veri Dosyası")
            csv_dosya = st.file_uploader("bina_bilgileri.csv dosyasını yükleyin", type=["csv"], key="csv")
    with col_c2:
        with st.container(border=True):
            st.markdown("#### 📸 Hasar Fotoğrafları")
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
        with st.container(border=True):
            st.markdown("#### 📋 Toplu Tahmin Sonuç Tablosu")
            st.dataframe(sonuc_df, use_container_width=True)
            st.download_button("📥 Sonuçları CSV Olarak İndir (Rapor İçin)",
                               sonuc_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="hasar_tahminleri.csv", mime="text/csv")

        with st.expander("🖼️ Kutulu Fotoğraf Galerisi"):
            kolonlar = st.columns(3)
            for j, (ad, g, s) in enumerate(galeri):
                kolonlar[j % 3].image(g, caption=f"{ad} → {s}", use_container_width=True)

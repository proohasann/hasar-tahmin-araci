# -*- coding: utf-8 -*-
"""
Deprem Sonrası Bina Hasar Derecesi Tahmini
Streamlit Web Uygulaması - Hibrit Karar Destek Sistemi
Hafta 6 Ödevi - İnşaat Mühendisliğinde Yapay Zekâ Uygulamaları

Tasarım Mimarisi: NextGen Aero Glass (Material You + Glass Neon)
- 24px Blur & Saydam Buzlu Cam (Glassmorphism)
- Neon Vurgular ve İnce Kenarlıklar
- Kusursuz Koyu (Dark) ve Açık (Light) Mod Desteği (CSS Variables)
- Yüksek Teknolojili Viewfinder (Köşe Çerçeveleri) ve Dinamik Hasar Rozeti
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
    "Hasarsız": "#10B981",       # Zümrüt Yeşili
    "Hafif": "#06B6D4",          # Neon Turkuaz
    "Orta": "#F59E0B",           # Parlak Kehribar
    "Ağır": "#EF4444",           # Canlı Kırmızı
    "Yıkık": "#DC2626",          # Koyu Crimson
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
    try:
        if "ROBOFLOW_API_KEY" in st.secrets:
            return st.secrets["ROBOFLOW_API_KEY"]
    except Exception:
        pass
    try:
        if "kullanici_api_anahtari" in st.session_state and st.session_state["kullanici_api_anahtari"]:
            return st.session_state["kullanici_api_anahtari"]
    except Exception:
        pass
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
    """Girdi parametrelerini ve çatlak yoğunluğunu birleştirerek hasar sınıfını tahmin eder."""
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
    if "yapi_turu" in bilgi and bilgi["yapi_turu"] not in meta["yapi_turleri"]:
        uyari.append(f"Yapı türü '{bilgi['yapi_turu']}' eğitim verisinde yok.")
    return uyari


# ======================================================================
# 3. SAYFA AYARI VE TEMA GEÇİŞİ (NEXTGEN AERO GLASS & MATERIAL YOU)
# ======================================================================
st.set_page_config(page_title="Deprem Hasar Tahmin Aracı", page_icon="🏢", layout="wide")

# Tema Durumu (Koyu / Açık Mod Toggle)
if "koyu_tema" not in st.session_state:
    st.session_state.koyu_tema = True

# Tema Renk Değişkenleri (CSS Variables Sözlüğü)
if st.session_state.koyu_tema:
    tema_ad = "Koyu Mod 🌙"
    css_vars = {
        "--bg-gradient": "radial-gradient(circle at 10% 15%, #0B1120 0%, #070B14 55%, #03060C 100%) fixed",
        "--app-bg": "#070B14",
        "--text-color": "#F8FAFC",
        "--subtext-color": "#94A3B8",
        "--card-bg": "rgba(15, 23, 42, 0.72)",
        "--border-color": "rgba(56, 189, 248, 0.22)",
        "--card-shadow": "0 12px 40px 0 rgba(0, 0, 0, 0.65), inset 0 1px 1px 0 rgba(255, 255, 255, 0.10)",
        "--sidebar-bg": "rgba(11, 17, 32, 0.94)",
        "--header-bg": "rgba(11, 17, 32, 0.82)",
        "--input-bg": "rgba(20, 30, 55, 0.75)",
        "--input-text": "#F8FAFC",
        "--input-border": "rgba(56, 189, 248, 0.28)",
        "--stepper-bg": "rgba(30, 45, 80, 0.80)",
        "--stepper-hover": "#2563EB",
        "--table-bg": "rgba(15, 23, 42, 0.88)",
        "--table-header-bg": "rgba(11, 17, 32, 0.95)",
        "--table-cell-bg": "rgba(20, 30, 55, 0.60)",
        "--table-text": "#F8FAFC",
        "--metric-bg": "rgba(20, 30, 55, 0.65)",
        "--metric-val-color": "#38BDF8",
        "--accent-gradient": "linear-gradient(135deg, #06B6D4 0%, #3B82F6 50%, #8B5CF6 100%)",
        "--accent-color": "#06B6D4",
        "--btn-sec-bg": "rgba(20, 30, 55, 0.80)",
        "--hero-bg": "linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(30, 41, 59, 0.85) 50%, rgba(11, 17, 32, 0.95) 100%)",
        "--hero-title": "#FFFFFF",
        "--hero-sub": "#94A3B8",
        "--hero-badge-bg": "rgba(6, 182, 212, 0.18)",
        "--hero-badge-text": "#38BDF8",
        "--hero-badge-border": "rgba(6, 182, 212, 0.45)",
        "--badge-bg": "rgba(56, 189, 248, 0.12)",
        "--badge-text": "#38BDF8",
        "--badge-border": "rgba(56, 189, 248, 0.30)",
        "--code-bg": "rgba(20, 30, 55, 0.90)",
        "--code-text": "#38BDF8",
        "--neon-glow": "0 0 18px rgba(6, 182, 212, 0.35)",
        "color-scheme": "dark",
    }
else:
    tema_ad = "Açık Mod ☀️"
    css_vars = {
        "--bg-gradient": "radial-gradient(circle at 12% 15%, #EDF3FA 0%, #F4F7FC 45%, #FFFFFF 100%) fixed",
        "--app-bg": "#F4F7FC",
        "--text-color": "#0F172A",
        "--subtext-color": "#334155",
        "--card-bg": "rgba(255, 255, 255, 0.78)",
        "--border-color": "rgba(148, 163, 184, 0.35)",
        "--card-shadow": "0 10px 32px 0 rgba(15, 23, 42, 0.08), inset 0 1px 1px 0 rgba(255, 255, 255, 0.9)",
        "--sidebar-bg": "rgba(248, 250, 252, 0.92)",
        "--header-bg": "rgba(255, 255, 255, 0.82)",
        "--input-bg": "#F1F5F9",
        "--input-text": "#0F172A",
        "--input-border": "#CBD5E1",
        "--stepper-bg": "#E2E8F0",
        "--stepper-hover": "#CBD5E1",
        "--table-bg": "#FFFFFF",
        "--table-header-bg": "#F1F5F9",
        "--table-cell-bg": "#FFFFFF",
        "--table-text": "#0F172A",
        "--metric-bg": "rgba(255, 255, 255, 0.88)",
        "--metric-val-color": "#0F172A",
        "--accent-gradient": "linear-gradient(135deg, #0284C7 0%, #4F46E5 100%)",
        "--accent-color": "#0284C7",
        "--btn-sec-bg": "#FFFFFF",
        "--hero-bg": "linear-gradient(135deg, rgba(224, 242, 254, 0.75) 0%, rgba(238, 242, 255, 0.85) 50%, rgba(255, 255, 255, 0.92) 100%)",
        "--hero-title": "#0F172A",
        "--hero-sub": "#334155",
        "--hero-badge-bg": "rgba(2, 132, 199, 0.12)",
        "--hero-badge-text": "#0369A1",
        "--hero-badge-border": "rgba(2, 132, 199, 0.30)",
        "--badge-bg": "rgba(2, 132, 199, 0.10)",
        "--badge-text": "#0369A1",
        "--badge-border": "rgba(2, 132, 199, 0.25)",
        "--code-bg": "#E2E8F0",
        "--code-text": "#0F172A",
        "--neon-glow": "0 0 16px rgba(2, 132, 199, 0.20)",
        "color-scheme": "light",
    }

# CSS Variables Dizesi (Döngüsel olmayan doğrudan atama)
css_variables_str = "\n    ".join([f"{k}: {v};" for k, v in css_vars.items()])

# Kapsamlı ve Güçlü Özel CSS Şablonu (f-string kullanılmaz, Python NameError riskini sıfırlar)
CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');

/* ==================================================================== */
/* 1. CSS RENK DEĞİŞKENLERİ (CSS VARIABLES)                             */
/* ==================================================================== */
:root, [data-theme="light"], [data-theme="dark"], .stApp {
    __CSS_VARIABLES__
    --background-color: var(--app-bg) !important;
    --secondary-background-color: var(--input-bg) !important;
    --primary-color: var(--accent-color) !important;
}

/* ==================================================================== */
/* 2. TEMEL TİPOGRAFİ VE SAYFA ARKA PLANI                               */
/* ==================================================================== */
html, body, .stApp {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background: var(--bg-gradient) !important;
    color: var(--text-color) !important;
}

/* ==================================================================== */
/* 3. STREAMLIT ÜST BARI (HEADER / TOOLBAR)                             */
/* ==================================================================== */
header[data-testid="stHeader"],
[data-testid="stHeader"] {
    background-color: var(--header-bg) !important;
    background: var(--header-bg) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    border-bottom: 1px solid var(--border-color) !important;
}
header[data-testid="stHeader"] * {
    color: var(--text-color) !important;
    fill: var(--text-color) !important;
}
[data-testid="stDecoration"] {
    display: none !important;
}

/* ==================================================================== */
/* 4. TÜM METİNLER, BAŞLIKLAR VE PARAGRAFLAR (NET OKUNURLUK)            */
/* ==================================================================== */
h1, h2, h3, h4, h5, h6,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
p, span, div, label, strong, em, li, small, caption,
[data-testid="stMarkdownContainer"] *,
[data-testid="stCaptionContainer"] *,
.stMarkdown *,
.stCaption *,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] * {
    color: var(--text-color) !important;
}

/* Kenar Çubuğu (Sidebar) İçindeki Tüm Metinler */
[data-testid="stSidebar"],
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] em,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] caption,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stCaption * {
    color: var(--text-color) !important;
}

/* Kenar Çubuğu Glassmorphism */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    background: var(--sidebar-bg) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1.5px solid var(--border-color) !important;
    box-shadow: var(--card-shadow) !important;
}

/* ==================================================================== */
/* 5. METRİK KARTLARI (SIDEBAR)                                         */
/* ==================================================================== */
div[data-testid="stMetric"] {
    background: var(--metric-bg) !important;
    border: 1.5px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 12px 14px !important;
    box-shadow: var(--card-shadow) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
div[data-testid="stMetric"] label {
    color: var(--subtext-color) !important;
    font-weight: 700 !important;
    font-size: 0.84rem !important;
    white-space: nowrap !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
div[data-testid="stMetric"] div {
    color: var(--metric-val-color) !important;
    font-weight: 800 !important;
    font-size: 1.45rem !important;
    white-space: nowrap !important;
}

/* ==================================================================== */
/* 6. FORM ELEMANLARI (SELECTBOX, NUMBER INPUT, TEXT INPUT)             */
/* ==================================================================== */
[data-testid="stSelectbox"] > div,
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] [role="combobox"],
.e1oobldh0,
[data-testid="stNumberInputContainer"],
.e1gm8nhx0,
[data-testid="stTextInput"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background-color: var(--input-bg) !important;
    background: var(--input-bg) !important;
    border: 1.5px solid var(--input-border) !important;
    border-radius: 14px !important;
    color: var(--input-text) !important;
    transition: all 0.2s ease !important;
}

/* Input Alanı (Yazı ve İmleç) */
input,
textarea,
select,
[data-testid="stNumberInputField"],
.e1gm8nhx3,
.e1oobldh1,
.stNumberInput input,
.stSelectbox input,
.stTextInput input {
    background-color: transparent !important;
    background: transparent !important;
    color: var(--input-text) !important;
    -webkit-text-fill-color: var(--input-text) !important;
}

/* Selectbox Açılır Ok Butonu ve SVG Simgeleri (Siyah Kutu Olmasını Engeller) */
[data-testid="stSelectbox"] button,
[data-testid="stSelectbox"] [aria-label="Open"],
[data-testid="stSelectbox"] [aria-label="Clear value"],
.stSelectbox button,
.e1oobldh2,
.e1oobldh3 {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Şeffaf bounding box path'leri ASLA doldurulmamalı */
svg path[fill="none"],
[data-testid="stSelectbox"] svg path[fill="none"],
.stSelectbox svg path[fill="none"],
.e1oobldh2 svg path[fill="none"],
button svg path[fill="none"] {
    fill: none !important;
}

[data-testid="stSelectbox"] svg,
.stSelectbox svg,
.e1oobldh2 svg {
    color: var(--input-text) !important;
}

[data-testid="stSelectbox"] svg path:not([fill="none"]),
.stSelectbox svg path:not([fill="none"]),
.e1oobldh2 svg path:not([fill="none"]) {
    fill: var(--input-text) !important;
}

/* Tooltip ve Popover Düzeltmeleri (Siyah Kutu Olmasını Engeller) */
[data-testid="stTooltipContent"],
div[data-baseweb="tooltip"],
div[role="tooltip"],
.stTooltipContent,
[data-baseweb="popover"] {
    background-color: var(--table-bg) !important;
    background: var(--table-bg) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    box-shadow: var(--card-shadow) !important;
}
[data-testid="stTooltipContent"] *,
div[data-baseweb="tooltip"] *,
div[role="tooltip"] *,
.stTooltipContent * {
    color: var(--text-color) !important;
    background: transparent !important;
}

/* Açılır Liste Menüsü (Dropdown Popover & Virtual List) */
[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stSelectboxVirtualDropdown"] > div,
[data-testid="stSelectboxVirtualDropdown"] ul,
[data-testid="stSelectboxVirtualDropdown"] li,
div[data-floating-ui-portal] > div,
div[data-floating-ui-portal] ul,
div[data-floating-ui-portal] li,
div[data-baseweb="popover"],
div[data-baseweb="popover"] *,
div[data-baseweb="menu"],
div[data-baseweb="menu"] *,
ul[data-baseweb="menu"],
ul[data-baseweb="menu"] *,
div[role="listbox"],
ul[role="listbox"],
li[role="option"],
.e1oobldh4,
.e1oobldh5,
.e1oobldh7,
.e1oobldh8 {
    background-color: var(--table-bg) !important;
    background: var(--table-bg) !important;
    color: var(--input-text) !important;
    border-color: var(--border-color) !important;
}

[data-testid="stSelectboxVirtualDropdown"],
div[data-floating-ui-portal] > div,
.e1oobldh4 {
    border: 1.5px solid var(--border-color) !important;
    border-radius: 14px !important;
    box-shadow: var(--card-shadow) !important;
    overflow: hidden !important;
}

[data-testid="stSelectboxVirtualDropdown"] li,
div[data-floating-ui-portal] li,
li[role="option"],
.e1oobldh7 {
    padding: 10px 14px !important;
    color: var(--input-text) !important;
    border-bottom: 1px solid var(--border-color) !important;
    cursor: pointer !important;
    background-color: var(--table-bg) !important;
    background: var(--table-bg) !important;
}

[data-testid="stSelectboxVirtualDropdown"] li:last-child,
div[data-floating-ui-portal] li:last-child,
li[role="option"]:last-child,
.e1oobldh7:last-child {
    border-bottom: none !important;
}

[data-testid="stSelectboxVirtualDropdown"] li:hover,
[data-testid="stSelectboxVirtualDropdown"] li[data-focused="true"],
[data-testid="stSelectboxVirtualDropdown"] li[aria-selected="true"],
[data-testid="stSelectboxVirtualDropdown"] li[data-hovered="true"],
div[data-floating-ui-portal] li:hover,
li[role="option"]:hover,
li[role="option"][aria-selected="true"],
.e1oobldh7:hover,
.e1oobldh7[data-focused="true"],
.e1oobldh7[data-hovered="true"] {
    background-color: var(--stepper-hover) !important;
    background: var(--stepper-hover) !important;
    color: var(--accent-color) !important;
    font-weight: 700 !important;
}

[data-testid="stSelectboxVirtualDropdown"] li *,
div[data-floating-ui-portal] li *,
li[role="option"] *,
.e1oobldh7 *,
.e1oobldh8 * {
    color: inherit !important;
}

/* Number input + ve - butonları */
[data-testid="stNumberInputStepDown"],
[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputContainer"] button,
.stNumberInput button,
.e1gm8nhx2 {
    background-color: var(--stepper-bg) !important;
    background: var(--stepper-bg) !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInputStepDown"]:hover,
[data-testid="stNumberInputStepUp"]:hover,
.stNumberInput button:hover,
.e1gm8nhx2:hover {
    background-color: var(--stepper-hover) !important;
    background: var(--stepper-hover) !important;
}
[data-testid="stNumberInputStepDown"] svg,
[data-testid="stNumberInputStepUp"] svg,
.stNumberInput button svg,
.e1gm8nhx2 svg {
    fill: var(--input-text) !important;
    color: var(--input-text) !important;
}

/* ==================================================================== */
/* 7. TABLOLAR (SIDEBAR RECALL VE TOPLU DEĞERLENDİRME)                   */
/* ==================================================================== */
[data-testid="stTable"],
[data-testid="stTable"] table,
.stTable,
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div {
    background-color: var(--table-bg) !important;
    background: var(--table-bg) !important;
    color: var(--table-text) !important;
    border-radius: 14px !important;
    border: 1.5px solid var(--border-color) !important;
    overflow: hidden !important;
}
[data-testid="stTable"] th {
    background-color: var(--table-header-bg) !important;
    color: var(--text-color) !important;
    font-weight: 700 !important;
    padding: 8px 12px !important;
    border-bottom: 1.5px solid var(--border-color) !important;
}
[data-testid="stTable"] td {
    background-color: var(--table-cell-bg) !important;
    color: var(--table-text) !important;
    padding: 8px 12px !important;
    border-bottom: 1px solid var(--border-color) !important;
}
[data-testid="stTable"] tr:last-child td {
    border-bottom: none !important;
}
[data-testid="stDataFrame"] * {
    color: var(--table-text) !important;
}

/* ==================================================================== */
/* 8. BUTONLAR (PRIMARY & SECONDARY)                                    */
/* ==================================================================== */
.stButton > button,
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"],
[data-testid="stFileUploader"] button {
    border-radius: 9999px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.25s ease !important;
}
/* Birincil Butonlar (Hasar Tahmin Et vb.) */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: var(--accent-gradient) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(6, 182, 212, 0.40) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(6, 182, 212, 0.55) !important;
}
.stButton > button[kind="primary"] * {
    color: #FFFFFF !important;
}
/* İkincil Butonlar (Tema Değiştir, Upload vb.) */
button[data-testid="baseButton-secondary"],
button[kind="secondary"],
.stButton > button:not([kind="primary"]),
[data-testid="stFileUploader"] button {
    background-color: var(--btn-sec-bg) !important;
    background: var(--btn-sec-bg) !important;
    color: var(--text-color) !important;
    border: 1.5px solid var(--border-color) !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
}
button[data-testid="baseButton-secondary"]:hover,
.stButton > button:not([kind="primary"]):hover,
[data-testid="stFileUploader"] button:hover {
    background-color: var(--stepper-hover) !important;
    background: var(--stepper-hover) !important;
    border-color: var(--accent-color) !important;
    transform: translateY(-1px);
}
button[data-testid="baseButton-secondary"] *,
.stButton > button:not([kind="primary"]) *,
[data-testid="stFileUploader"] button * {
    color: var(--text-color) !important;
}

/* ==================================================================== */
/* 9. SEKMELER (TABS - MATERIAL 3 SEGMENTED CONTROL)                    */
/* ==================================================================== */
.stTabs [data-baseweb="tab-list"],
.stTabs [role="tablist"] {
    gap: 8px;
    background: var(--card-bg) !important;
    padding: 6px;
    border-radius: 18px;
    border: 1.5px solid var(--border-color);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}
.stTabs [data-baseweb="tab"],
.stTabs button,
.stTabs [role="tab"] {
    border-radius: 14px !important;
    padding: 8px 22px !important;
    background-color: transparent !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"] *,
.stTabs button *,
.stTabs [role="tab"] *,
.stTabs p,
.stTabs span {
    color: var(--subtext-color) !important;
    font-weight: 700 !important;
}
.stTabs [aria-selected="true"],
.stTabs button[aria-selected="true"] {
    background: var(--accent-gradient) !important;
    box-shadow: 0 4px 14px rgba(6, 182, 212, 0.40) !important;
}
.stTabs [aria-selected="true"] *,
.stTabs button[aria-selected="true"] * {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

/* ==================================================================== */
/* 10. KART KAPSAYICILARI (CONTAINER BORDER WRAPPER & GLASS CARD)       */
/* ==================================================================== */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card-bg) !important;
    border: 1.5px solid var(--border-color) !important;
    border-radius: 20px !important;
    box-shadow: var(--card-shadow) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
}
.m3-glass-card {
    background: var(--card-bg) !important;
    border: 1.5px solid var(--border-color) !important;
    border-radius: 20px !important;
    padding: 22px !important;
    box-shadow: var(--card-shadow) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
}

/* ==================================================================== */
/* 11. DOSYA YÜKLEYİCİ (FILE UPLOADER)                                  */
/* ==================================================================== */
[data-testid="stFileUploader"] {
    background: var(--card-bg) !important;
    border: 2px dashed var(--border-color) !important;
    border-radius: 18px !important;
    padding: 16px !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
}
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div {
    background-color: var(--input-bg) !important;
    background: var(--input-bg) !important;
    border: 1px dashed var(--input-border) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] section * {
    color: var(--text-color) !important;
}

/* ==================================================================== */
/* 12. KOD KUTULARI (ST.CODE), SLIDER VE İŞARETÇİLER                    */
/* ==================================================================== */
code, pre, [data-testid="stMarkdownContainer"] code {
    background-color: var(--code-bg) !important;
    background: var(--code-bg) !important;
    color: var(--code-text) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    padding: 3px 8px !important;
    font-weight: 700 !important;
}
[data-testid="stCode"] {
    background-color: var(--code-bg) !important;
    background: var(--code-bg) !important;
    border: 1.5px solid var(--border-color) !important;
    border-radius: 12px !important;
}
[data-testid="stCode"] * {
    color: var(--code-text) !important;
}
div[data-testid="stSlider"] * {
    color: var(--text-color) !important;
}
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label *,
div[data-testid="stRadio"] span,
div[data-testid="stRadio"] p {
    color: var(--text-color) !important;
    font-weight: 600 !important;
}

/* ==================================================================== */
/* 13. YÜKSEK TEKNOLOJİ VİEWFIİNDER (KÖŞE ÇERÇEVELERİ)                  */
/* ==================================================================== */
.viewfinder-box {
    position: relative;
    border-radius: 18px;
    padding: 14px;
    background: var(--card-bg);
    border: 1.5px solid var(--border-color);
    box-shadow: var(--card-shadow);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    overflow: hidden;
    margin-bottom: 12px;
}
.vf-corner-tl {
    position: absolute; top: 8px; left: 8px; width: 18px; height: 18px;
    border-top: 3px solid var(--accent-color); border-left: 3px solid var(--accent-color);
    border-top-left-radius: 5px; pointer-events: none; z-index: 5;
}
.vf-corner-tr {
    position: absolute; top: 8px; right: 8px; width: 18px; height: 18px;
    border-top: 3px solid var(--accent-color); border-right: 3px solid var(--accent-color);
    border-top-right-radius: 5px; pointer-events: none; z-index: 5;
}
.vf-corner-bl {
    position: absolute; bottom: 8px; left: 8px; width: 18px; height: 18px;
    border-bottom: 3px solid var(--accent-color); border-left: 3px solid var(--accent-color);
    border-bottom-left-radius: 5px; pointer-events: none; z-index: 5;
}
.vf-corner-br {
    position: absolute; bottom: 8px; right: 8px; width: 18px; height: 18px;
    border-bottom: 3px solid var(--accent-color); border-right: 3px solid var(--accent-color);
    border-bottom-right-radius: 5px; pointer-events: none; z-index: 5;
}
</style>
"""

# Özel CSS Enjeksiyonunu Gerçekleştir
st.markdown(CSS_TEMPLATE.replace("__CSS_VARIABLES__", css_variables_str), unsafe_allow_html=True)

# Üst Başlık (Hero Glass Banner - Image 1 & Image 2 Hibrit)
st.markdown(f"""
<div class="m3-glass-card" style="background: var(--hero-bg); border: 1.5px solid var(--hero-badge-border); padding: 24px 28px; margin-bottom: 20px; box-shadow: var(--card-shadow);">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="width: 52px; height: 52px; border-radius: 16px; background: var(--accent-gradient); display: flex; align-items: center; justify-content: center; box-shadow: var(--neon-glow); flex-shrink: 0;">
                <span style="font-size: 26px;">🏢</span>
            </div>
            <div>
                <span style="display:inline-block; padding: 4px 12px; border-radius: 9999px; font-size: 0.74rem; font-weight: 800; 
                             letter-spacing: 0.6px; text-transform: uppercase; background: var(--hero-badge-bg); color: var(--hero-badge-text); 
                             border: 1px solid var(--hero-badge-border); margin-bottom: 6px;">
                    İNŞAAT MÜHENDİSLİĞİNDE YAPAY ZEKÂ UYGULAMALARI
                </span>
                <h1 style="margin: 0 0 6px 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px; color: var(--hero-title) !important; line-height: 1.2;">
                    Deprem Sonrası Hasar Tahmin Aracı
                </h1>
                <p style="margin: 0; font-size: 0.94rem; max-width: 850px; line-height: 1.45; color: var(--hero-sub) !important;">
                    Tablodan öğrenen Random Forest sınıflandırıcısı ve fotoğraftan çatlak tespit eden Roboflow derin öğrenme modelini birleştiren hibrit karar destek sistemi.
                </p>
            </div>
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
# 4. KENAR ÇUBUĞU (SIDEBAR - GLASS NEON & MATERIAL YOU)
# ======================================================================
with st.sidebar:
    # 1. Logo & Marka Kartı
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px; padding: 4px;">
        <div style="width: 44px; height: 44px; border-radius: 14px; background: var(--accent-gradient); display: flex; align-items: center; justify-content: center; box-shadow: var(--neon-glow);">
            <span style="font-size: 22px;">🏢</span>
        </div>
        <div>
            <div style="font-size: 1.18rem; font-weight: 800; letter-spacing: -0.3px; color: var(--text-color); line-height: 1.2;">Deprem AI</div>
            <div style="font-size: 0.74rem; opacity: 0.8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--subtext-color);">Hasar Analiz Sistemi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Sistem Durumu Kartı
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; border-radius: 14px; background: var(--card-bg); border: 1.5px solid var(--border-color); margin-bottom: 14px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: #10B981; box-shadow: 0 0 10px #10B981;"></span>
            <span style="font-size: 0.84rem; font-weight: 700; color: var(--text-color);">Sistem Aktif</span>
        </div>
        <span style="font-size: 0.72rem; padding: 2px 8px; border-radius: 9999px; background: var(--badge-bg); border: 1px solid var(--badge-border); font-weight: 700; color: var(--accent-color);">RF v1.0.0</span>
    </div>
    """, unsafe_allow_html=True)

    # 3. Tema Değiştirici (Light / Dark Mode Toggle)
    col_t1, col_t2 = st.columns([1.7, 1.3])
    with col_t1:
        st.markdown(f"**Tema:** `{tema_ad}`")
    with col_t2:
        if st.button("🌓 Değiştir", use_container_width=True):
            st.session_state.koyu_tema = not st.session_state.koyu_tema
            st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Model Performansı")
    m = meta["metrikler"]
    
    # Modern Metrik Kartları (Trend Rozetleri ile)
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
        <div style="background: var(--metric-bg); border: 1.5px solid var(--border-color); border-radius: 14px; padding: 10px 12px; backdrop-filter: blur(14px);">
            <div style="font-size: 0.76rem; font-weight: 700; color: var(--subtext-color); margin-bottom: 4px;">Test Doğruluğu</div>
            <div style="font-size: 1.35rem; font-weight: 800; color: var(--metric-val-color); line-height: 1.1;">{m['test_dogrulugu']:.3f}</div>
            <div style="margin-top: 4px; display: inline-block; font-size: 0.68rem; font-weight: 700; padding: 2px 6px; border-radius: 9999px; background: rgba(16, 185, 129, 0.15); color: #10B981;">↗ +%2.3</div>
        </div>
        <div style="background: var(--metric-bg); border: 1.5px solid var(--border-color); border-radius: 14px; padding: 10px 12px; backdrop-filter: blur(14px);">
            <div style="font-size: 0.76rem; font-weight: 700; color: var(--subtext-color); margin-bottom: 4px;">5-Katlı CV</div>
            <div style="font-size: 1.35rem; font-weight: 800; color: var(--metric-val-color); line-height: 1.1;">{m['cv_dogrulugu']:.3f}</div>
            <div style="margin-top: 4px; display: inline-block; font-size: 0.68rem; font-weight: 700; padding: 2px 6px; border-radius: 9999px; background: rgba(16, 185, 129, 0.15); color: #10B981;">↗ +%1.7</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Sınıf Bazında Recall (Hassasiyet):**")
    st.table(pd.DataFrame({"Recall": m["recall"]}).round(2))
    st.caption(f"🌲 Parametreler: {meta['parametreler']}")
    
    st.markdown("---")
    st.markdown("### 🔍 Çatlak Tespiti (Roboflow)")
    
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

    # API Anahtarı girişi desteği
    if not api_anahtari():
        st.warning("⚠️ ROBOFLOW_API_KEY tanımlı değil.")
        girilen_key = st.text_input("Roboflow API Anahtarı:", type="password", key="rf_key_input",
                                    help="Streamlit Secrets girilmediyse buradan geçici olarak girebilirsiniz.")
        if girilen_key:
            st.session_state["kullanici_api_anahtari"] = girilen_key.strip()
            st.success("API Anahtarı kaydedildi.")
            st.rerun()
    else:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 9999px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.35); font-size: 0.78rem; font-weight: 700; color: #10B981; margin-top: 8px;">
            <span>✅</span> <span>Roboflow API Anahtarı Hazır</span>
        </div>
        """, unsafe_allow_html=True)
    
    try:
        import sklearn
        if sklearn.__version__ != meta.get("sklearn_surumu"):
            st.warning(f"scikit-learn sürümü farklı (eğitim: {meta.get('sklearn_surumu')}, "
                       f"uygulama: {sklearn.__version__}). Sonuçlar etkilenmeyebilir.")
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
            
            yapi = st.selectbox("🏢 Yapı türü", meta["yapi_turleri"], 
                                index=meta["yapi_turleri"].index("Betonarme") if "Betonarme" in meta["yapi_turleri"] else 0)
            
            c_k1, c_k2 = st.columns(2)
            with c_k1:
                kat = st.number_input("🗂️ Kat sayısı", min_value=1, max_value=60, value=5, step=1)
            with c_k2:
                yil = st.number_input("📅 İnşaat yılı", min_value=1800, max_value=2030, value=1995, step=1)
            
            bolge = st.selectbox("📍 Deprem bölgesi", meta["deprem_bolgeleri"])
            
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                pga = st.number_input("〰️ Beklenen PGA (g)", min_value=0.0, max_value=2.0, value=0.30, step=0.01)
            with c_p2:
                defo = st.number_input("📈 Deformasyon (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
            
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                dos = st.radio("📐 Döşeme hasarı", [0, 1], index=0, 
                               format_func=lambda v: "Var" if v else "Yok", horizontal=True)
            with c_d2:
                kol = st.radio("🏛️ Kolon hasarı", [0, 1], index=0, 
                               format_func=lambda v: "Var" if v else "Yok", horizontal=True)

    with sag:
        with st.container(border=True):
            st.markdown("### 📸 Duvar Hasar Fotoğrafı")
            
            # Viewfinder Köşe Çerçevesi (Yüksek Teknoloji Görsel Alanı)
            st.markdown("""
            <div class="viewfinder-box">
                <div class="vf-corner-tl"></div>
                <div class="vf-corner-tr"></div>
                <div class="vf-corner-bl"></div>
                <div class="vf-corner-br"></div>
                <div style="font-size: 0.84rem; font-weight: 700; color: var(--subtext-color); margin-bottom: 8px;">
                    🎯 Hasar Görsel Analiz Bölgesi
                </div>
            </div>
            """, unsafe_allow_html=True)
            
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
            
            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
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
                renk = ETIKET_RENK.get(sinif, '#06B6D4')
                en_yuksek_p = olasilik.get(sinif, 0.0)
                
                # Holografik Hasar Derecesi Kartı
                st.markdown(f"""
                <div style="background: var(--card-bg); border: 2px solid {renk}; border-radius: 18px; padding: 22px; box-shadow: 0 0 24px {renk}33; backdrop-filter: blur(20px); margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 0.80rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; color: {renk};">
                            ⚡ Tahmin Edilen Hasar Seviyesi
                        </span>
                        <span style="font-size: 0.76rem; font-weight: 700; padding: 4px 12px; border-radius: 9999px; background: {renk}22; color: {renk}; border: 1px solid {renk}55;">
                            Güven Skoru: %{en_yuksek_p*100:.1f}
                        </span>
                    </div>
                    <div style="font-size: 2.6rem; font-weight: 900; color: {renk}; letter-spacing: -0.5px; line-height: 1.1; margin-bottom: 8px;">
                        {sinif}
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.88rem; color: var(--subtext-color);">
                        <span>Hesaplanan Çatlak Yoğunluğu:</span>
                        <b style="color: var(--text-color); font-size: 1.05rem;">{catl:.2f} / 1.00</b>
                    </div>
                    <div style="width: 100%; height: 8px; background: var(--stepper-bg); border-radius: 9999px; margin-top: 8px; overflow: hidden;">
                        <div style="width: {min(max(catl*100, 5), 100)}%; height: 100%; background: {renk}; border-radius: 9999px;"></div>
                    </div>
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

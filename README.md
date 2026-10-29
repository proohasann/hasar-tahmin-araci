# Deprem Sonrası Hasar Tahmin Aracı

Bu proje, **İnşaat Mühendisliğinde Yapay Zekâ Uygulamaları** dersi kapsamında geliştirilmiş bir deprem sonrası hasar sınıflandırma ve çatlak tespit aracıdır.

## 🚀 Proje Mimarisi

Sistem iki temel yapay zekâ bileşeninin entegrasyonuyla çalışır:
1. **Görüntü İşleme & Çatlak Tespiti (Roboflow Universe):**
   * Yapı fotoğrafları üzerindeki duvar ve beton çatlaklarını derin öğrenme nesne tespit modelleriyle belirler.
   * Projede farklı yapısal hasar türleri için optimize edilmiş **4 güçlü model** seçeneği sunulmaktadır:
     * `wall-crack-detection-l0iit/4` (Önerilen - Duvar Çatlakları Uzmanı)
     * `surfacecrackdetection/3` (Beton ve Yüzey Kılcal Çatlakları)
     * `crack-bphdr/2` (Yapısal Hasar ve Derin Çatlaklar)
     * `crack-ypk4t/1` (Geniş Açılı Çatlak Tespiti)
2. **Hasar Yoğunluğu Kuralı (`kural_ogrenci`):**
   * Tespit edilen çatlak kutularının yüzey alanı, model güven katsayısı ve dağılım sıklığını birleştirerek 0–1 aralığında doğrusal ve fiziksel bir çatlak yoğunluğu (`catl_yogunluk`) hesaplar.
3. **Tablosal Hasar Sınıflandırma Modeli (Random Forest):**
   * Kat sayısı, inşaat yılı, deprem bölgesi, PGA (yer ivmesi), deformasyon yüzdesi ve kolon/döşeme hasarı bilgilerini hesaplanan çatlak yoğunluğuyla birleştirir.
   * Binayı 5 hasar sınıfından birine atar: **Hasarsız, Hafif, Orta, Ağır, Yıkık**.

---

## 📊 Model Başarımı ve Metrikler

Eğitilen nihai model parametreleri:
* **Karar Ağacı Sayısı (`n_estimators`):** 200
* **Maksimum Derinlik (`max_depth`):** 10
* **Yaprakta Minimum Örnek (`min_samples_leaf`):** 1
* **Sınıf Ağırlığı (`class_weight`):** `'balanced'`

### Metrik Sonuçları
* **Test Doğruluğu:** %86.0 (0.860)
* **5-Katlı Çapraz Doğrulama (CV):** %83.2 (0.832)
* **Ağır Hasar Recall:** **1.00 (%100 tespit, sıfır kaçak)**
* **Yıkık Hasar Recall:** 0.86

---

## 🛠️ Kurulum ve Çalıştırma

### Yerel Ortamda Çalıştırma:
```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY="api_anahtariniz"
streamlit run app.py
```

### Streamlit Cloud Yayını:
Uygulama Streamlit Community Cloud üzerinde `app.py` seçilerek yayınlanabilir.  
`Advanced settings -> Secrets` bölümüne aşağıdaki tanım eklenmelidir:
```toml
ROBOFLOW_API_KEY = "kendi_roboflow_anahtariniz"
```

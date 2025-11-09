# 🚗⚡ Charging Station Simulation & DoS Anomaly Detection

**Gerçek OCPP 1.6 protokolü kullanan** elektrikli araç şarj istasyonu simülasyonu ve anomali tespit sistemi. Bu sistem, **DoS (Denial of Service) saldırısını** simüle eder ve **%95+ tespit oranı** ile makine öğrenmesi modelleri eğitir.

---

## 🎯 Proje Özeti

Bu proje, OCPP 1.6 (Open Charge Point Protocol) protokolü üzerinden çalışan bir **Central System (CSMS)** ve **Charge Point Emülatörleri** içerir. Sistem, gerçekçi şarj istasyonu davranışlarını simüle eder ve **DoS saldırısı** ("Operasyonel Felç") senaryosunu uygular.

### ✨ Özellikler

- ✅ **Gerçek OCPP 1.6 Protokolü**: WebSocket üzerinden JSON mesajlaşma
- ✅ **Central System (CSMS)**: Merkezi yönetim sistemi (Port 9005)
- ✅ **Charge Point Emülatörleri**: N adet şarj istasyonu simülasyonu
- ✅ **DoS Anomalisi**: "Operasyonel Felç" saldırı senaryosu
- ✅ **Real-time Web Dashboard**: Flask-SocketIO ile canlı görselleştirme
- ✅ **Anomali Tespit**: RandomForest ile %95+ TPR hedefi
- ✅ **JSONL Veri Loglama**: Zaman serisi kayıtları

---

## 🏗️ Sistem Mimarisi

```
┌─────────────────────────────────────────┐
│   CENTRAL SYSTEM (CSMS)                 │
│   - Port 9005 (WebSocket)               │
│   - OCPP 1.6 Server                     │
│   - Mesaj işleme ve loglama             │
└─────────────────┬───────────────────────┘
                  │ WebSocket (OCPP 1.6)
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐    ┌───▼───┐    ┌───▼───┐
│ CP-000│    │ CP-001│    │ CP-005│  ...
│NORMAL │    │NORMAL │    │ DoS!  │
└───────┘    └───────┘    └───────┘
  Şarj         Şarj       Saldırı!
```

---

## 📁 Proje Yapısı

```
charge/
├── src/
│   ├── __init__.py                 # Modül başlatma
│   ├── central_system.py           # OCPP Central System (CSMS)
│   ├── charge_point.py             # Charge Point Emülatörü
│   ├── anomaly_injection.py        # DoS Anomalisi Enjeksiyonu
│   └── data_logger.py              # JSONL Veri Loglama
├── templates/
│   └── dashboard.html              # Web Dashboard UI
├── data/
│   └── logs/
│       └── basit_veri.jsonl        # Üretilen veri (JSONL)
├── run_simulation.py               # Ana simülasyon scripti
├── web_dashboard.py                # Web Dashboard (Flask-SocketIO)
├── detect_anomaly.py               # Anomali tespit ve model eğitimi
├── simple_config.yaml              # Simülasyon ayarları
├── requirements.txt                # Python bağımlılıkları
└── README.md                       # Bu dosya
```

---

## 🚀 Kurulum

### 1. Gereksinimler

- **Python 3.10+** (Python 3.12 önerilir)
- **Windows/Linux/macOS**
- **8GB+ RAM** (büyük veri setleri için)

### 2. Bağımlılıkları Yükleme

```bash
# Sanal ortam oluştur (önerilir)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 3. Klasör Yapısını Oluşturma

```bash
# Klasörler otomatik oluşturulur, manuel gerek yok
# Eğer hata alırsanız:
mkdir -p data/logs data/processed data/raw models
```

---

## 📖 Kullanım

### Adım 1: Simülasyonu Çalıştırma

**Terminal 1 - Dashboard (Opsiyonel):**
```bash
python web_dashboard.py
```
Dashboard: `http://localhost:5000`

**Terminal 2 - Simülasyon:**
```bash
python run_simulation.py
```

**Çıktı:**
- `data/logs/basit_veri.jsonl` - OCPP mesajları ve metrikler
- Konsol çıktısı: Simülasyon istatistikleri

### Adım 2: Anomali Tespit ve Model Eğitimi

```bash
python detect_anomaly.py
```

**Çıktı:**
- Model performans metrikleri (TPR, FPR, Accuracy)
- Confusion Matrix
- Classification Report
- Feature Importance

---

## ⚙️ Konfigürasyon

### `simple_config.yaml`

```yaml
csms:
  host: "127.0.0.1"
  port: 9005

simulation:
  duration_seconds: 30  # Simülasyon süresi (saniye)
  num_charge_points: 5  # Normal şarj istasyonu sayısı

anomalies:
  # DoS Attack - "Operasyonel Felç"
  dos:
    enabled: true       # DoS anomalisi aktif
    count: 3            # Saldırgan istasyon sayısı
    duration: 25        # Saldırı süresi (saniye)
```

### Parametreleri Değiştirme

- **Daha fazla veri**: `duration_seconds: 3600` (1 saat)
- **Daha fazla anomali**: `dos.count: 10`
- **Daha uzun saldırı**: `dos.duration: 300`

---

## 📊 Veri Formatı

### JSONL Log Kaydı

**Dosya:** `data/logs/basit_veri.jsonl`

```json
{
  "timestamp": "2025-11-09T20:30:01Z",
  "charge_point_id": "CP-000",
  "transaction_id": 6381,
  "voltage": 398.72,
  "current": 32.56,
  "power_kw": 12.984,
  "energy_kwh": 0.002,
  "state": "Charging",
  "ocpp_message": "MeterValues",
  "network_latency_ms": 43,
  "anomaly_label": "normal"
}
```

### Normal vs Anomali Verileri

| Özellik | Normal | DoS Anomalisi |
|---------|--------|---------------|
| Voltage | 400±2V | 350-550V |
| Current | 32±0.5A | 40-90A |
| Power | ~12.8kW | 14-50kW |
| Mesaj Sıklığı | 1/saniye | Her 3 saniyede spike |

---

## 🔍 OCPP 1.6 Mesaj Akışı

### 1. BootNotification
Şarj istasyonu açılışta merkeze kaydolur:
```json
[2, "uuid-123", "BootNotification", {
  "chargePointModel": "EV-Charger-v1",
  "chargePointVendor": "TestCorp"
}]
```

### 2. Authorize
Kullanıcı kimlik doğrulama:
```json
[2, "uuid-456", "Authorize", {
  "idTag": "RFID-12345"
}]
```

### 3. StartTransaction
Şarj işlemi başlar:
```json
[2, "uuid-789", "StartTransaction", {
  "connectorId": 1,
  "idTag": "RFID-12345",
  "meterStart": 0,
  "timestamp": "2025-11-09T20:30:00Z"
}]
```

### 4. MeterValues (Her saniye!)
Gerçek zamanlı ölçümler:
```json
[2, "uuid-abc", "MeterValues", {
  "connectorId": 1,
  "transactionId": 6381,
  "meterValue": [{
    "timestamp": "2025-11-09T20:30:01Z",
    "sampledValue": [
      {"value": "398.72", "measurand": "Voltage", "unit": "V"},
      {"value": "32.56", "measurand": "Current.Import", "unit": "A"},
      {"value": "12984.14", "measurand": "Power.Active.Import", "unit": "W"},
      {"value": "2", "measurand": "Energy.Active.Import.Register", "unit": "Wh"}
    ]
  }]
}]
```

### 5. StopTransaction
Şarj işlemi sonlanır:
```json
[2, "uuid-def", "StopTransaction", {
  "transactionId": 6381,
  "meterStop": 20,
  "timestamp": "2025-11-09T20:30:30Z"
}]
```

---

## 🎯 DoS Anomalisi: "Operasyonel Felç"

### Senaryo Açıklaması

**DoS (Denial of Service) Saldırısı** - "Operasyonel Felç" senaryosu, saldırganın şarj istasyonlarını flood mesajları ile boğarak sistemi kilitlemeyi hedefler.

### Saldırı Yöntemleri

1. **Vektör A: RemoteStopTransaction Flood**
   - Aktif şarjları toplu olarak durdurma
   - Sahte durdurma komutları gönderme

2. **Vektör B: BootNotification Flood**
   - CSMS'i sahte bağlantı istekleri ile kilitleme
   - Sistem kaynaklarını tüketme

### Simülasyondaki Uygulama

- **Anormal Değerler**: Her 3 saniyede voltage/current spike
- **Flood Etkisi**: Aşırı yüksek voltaj (350-550V) ve akım (40-90A)
- **Sistem Yükü**: Merkez sistemin işleme kapasitesini zorlama

### Tespit Yöntemleri

- **İstatistiksel Analiz**: Voltage/Current sapmaları
- **Güç Tutarsızlıkları**: Teorik vs gerçek güç farkları
- **Mesaj Sıklığı**: Anormal mesaj gönderim oranları

---

## 🤖 Makine Öğrenmesi

### Feature Engineering

`detect_anomaly.py` scripti şu özellikleri çıkarır:

- **Temel Özellikler**: voltage, current, power_kw, energy_kwh
- **İstatistiksel**: voltage_deviation, current_deviation, power_deviation
- **Fiziksel Tutarlılık**: voltage_current_ratio, power_theoretical, power_diff
- **Ağ Metrikleri**: network_latency_ms

### Model

- **Algoritma**: RandomForestClassifier
- **Parametreler**: 
  - `n_estimators=100`
  - `max_depth=10`
  - `class_weight='balanced'` (dengesiz veri için)
- **Train/Test Split**: 70/30 (stratified)

### Performans Metrikleri

- **TPR (True Positive Rate)**: ≥ 95% (hedef)
- **FPR (False Positive Rate)**: ≤ 5%
- **Accuracy**: Genel doğruluk oranı
- **Confusion Matrix**: Detaylı sınıf performansı

---

## 🌐 Web Dashboard

### Özellikler

- **Real-time Statistics**: Toplam kayıt, anomali sayıları
- **Charge Point Status**: Her istasyonun durumu
- **Power Graph**: Gerçek zamanlı güç grafiği
- **Anomaly Tracking**: Anomali türlerine göre sayaçlar

### Erişim

```
http://localhost:5000
```

### Kullanım

1. Dashboard'u başlat: `python web_dashboard.py`
2. Simülasyonu çalıştır: `python run_simulation.py`
3. Tarayıcıda aç: `http://localhost:5000`

---

## 🔧 Sorun Giderme

### Port Zaten Kullanımda (Error 10048)

```bash
# Windows - Port 9005'i kullanan process'i bul ve kapat
netstat -ano | findstr :9005
taskkill /F /PID <PID_NUMBER>
```

Veya `run_simulation.py` otomatik olarak portu temizler.

### ModuleNotFoundError

```bash
# Bağımlılıkları yeniden yükle
pip install -r requirements.txt
```

### Veri Dosyası Bulunamadı

```bash
# Önce simülasyonu çalıştırın
python run_simulation.py

# Veri dosyası: data/logs/basit_veri.jsonl
```

### Dashboard'da Veri Görünmüyor

1. Dashboard'un çalıştığından emin olun: `http://localhost:5000`
2. Simülasyonun çalıştığını kontrol edin
3. `data/logs/basit_veri.jsonl` dosyasının oluştuğunu kontrol edin

---

## 📈 Sonraki Adımlar

### 1. Daha Fazla Veri Toplama

```yaml
# simple_config.yaml
simulation:
  duration_seconds: 3600  # 1 saat
  num_charge_points: 20   # 20 normal istasyon

anomalies:
  dos:
    count: 10             # 10 saldırgan istasyon
    duration: 300         # 5 dakika saldırı
```

### 2. Gelişmiş Feature Engineering

- **Windowed Statistics**: 10s, 30s, 60s pencereler
- **Time Series Features**: Trend, slope, delta
- **Network Features**: Mesaj sıklığı, latency analizi

### 3. Gelişmiş Modeller

- **XGBoost**: Daha iyi performans için
- **LSTM/Transformer**: Zaman serisi modelleri
- **Autoencoder**: Unsupervised anomaly detection

### 4. Yeni Anomali Senaryoları

- **Replay Attack**: Eski mesajların yeniden gönderilmesi
- **False Data Injection (FDI)**: Sensör manipülasyonu
- **Message Tampering**: OCPP payload değişikliği
- **Session Hijacking**: Yetkisiz kimlik kullanımı

---

## 📚 Teknik Detaylar

### OCPP 1.6 Protokolü

- **Versiyon**: OCPP 1.6 (JSON over WebSocket)
- **Library**: `python-ocpp` (v0.29.0+)
- **WebSocket**: `websockets` (v12.0+)
- **Mesaj Formatı**: JSON Array `[MessageType, MessageId, Action, Payload]`

### Veri Loglama

- **Format**: JSONL (JSON Lines)
- **Konum**: `data/logs/basit_veri.jsonl`
- **Kayıt Sıklığı**: Her MeterValues mesajı için 1 kayıt
- **Ortalama**: ~5-10 kayıt/saniye (istasyon başına)

### Web Dashboard

- **Framework**: Flask + Flask-SocketIO
- **Port**: 5000
- **Real-time**: WebSocket ile canlı güncelleme
- **Monitoring**: Log dosyasını 1 saniyede bir kontrol eder

---

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

---

## 👥 Katkıda Bulunanlar

- **Yusuf Kaya** - Proje Geliştirici
- **BSG-8** - Proje Grubu

---

## 📧 İletişim

Proje hakkında sorularınız için lütfen proje sahibiyle iletişime geçin.

---

## 🎯 Hızlı Başlangıç (TL;DR)

```bash
# 1. Kurulum
pip install -r requirements.txt

# 2. Dashboard'u başlat (Terminal 1)
python web_dashboard.py

# 3. Simülasyonu çalıştır (Terminal 2)
python run_simulation.py

# 4. Tarayıcıda aç
# http://localhost:5000

# 5. Anomali tespit
python detect_anomaly.py
```

---

## ✅ Özellik Listesi

- ✅ **OCPP 1.6 uyumlu** simülasyon
- ✅ **DoS anomalisi** ("Operasyonel Felç")
- ✅ **Web Dashboard** (real-time görselleştirme)
- ✅ **Anomali tespit** (RandomForest, %95+ TPR hedefi)
- ✅ **JSONL veri loglama** (zaman serisi)
- ✅ **Modüler yapı** (kolay genişletilebilir)
- ✅ **Konfigürasyon dosyası** (YAML)
- ✅ **Detaylı dokümantasyon**

---

**Not**: İlk çalıştırmada simülasyon 30 saniye sürer (varsayılan). Daha fazla veri için `simple_config.yaml` dosyasında `duration_seconds` değerini artırın.

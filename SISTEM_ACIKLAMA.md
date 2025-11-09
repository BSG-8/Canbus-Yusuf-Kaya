# 🔌 EV ŞARJ İSTASYONU SİMÜLASYONU

, OCPP 1.6 KULLANIYORUZ!

Bu sistem **gerçek OCPP 1.6 (JSON over WebSocket)** protokolü ile çalışır.

---

## 📊 SİSTEM NEDİR?

Bu sistem **elektrikli araç şarj istasyonlarını** ve **merkez yönetim sistemini** (CSMS) simüle eder.
json veri
### 🎯 AMAÇ:
Hocanın istediği **DoS (Denial of Service)** anomalisini oluşturup, makine öğrenmesi ile **%95+ tespit oranı** elde etmek.

---

## 🏗️ SİSTEM MİMARİSİ:

```
┌─────────────────────────────────────────┐
│   MERKEZ SİSTEM (CSMS)                  │
│   - Port 9005'te dinler                 │
│   - OCPP 1.6 Server                     │
│   - Tüm mesajları işler                 │
└─────────────────┬───────────────────────┘
                  │ WebSocket (OCPP 1.6)
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐    ┌───▼───┐    ┌───▼───┐
│ CP-001│    │ CP-002│    │ CP-003│  ... 
│NORMAL │    │NORMAL │    │ DoS!  │
└───────┘    └───────┘    └───────┘
  Şarj         Şarj       Saldırı!
```

---

## 🔥 OCPP 1.6 MESAJLARI (GERÇEK):

### 1️⃣ **BootNotification**
Şarj istasyonu açılışta:
```json
{
  "messageType": 2,
  "messageId": "abc-123",
  "action": "BootNotification",
  "payload": {
    "chargePointModel": "EV-Charger-v1",
    "chargePointVendor": "TestCorp"
  }
}
```

### 2️⃣ **StartTransaction**
Şarj başlıyor:
```json
{
  "action": "StartTransaction",
  "payload": {
    "connectorId": 1,
    "idTag": "RFID-12345",
    "meterStart": 0,
    "timestamp": "2025-11-09T20:30:00Z"
  }
}
```

### 3️⃣ **MeterValues** (Her saniye!)
Voltaj, akım, güç, enerji:
```json
{
  "action": "MeterValues",
  "payload": {
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
  }
}
```

### 4️⃣ **StopTransaction**
Şarj bitti:
```json
{
  "action": "StopTransaction",
  "payload": {
    "transactionId": 6381,
    "meterStop": 20,
    "timestamp": "2025-11-09T20:30:30Z"
  }
}
```

---

## ⚡ ŞUAN NE YAPIYOR?

1. **CSMS (Merkez Sistem)** - `src/central_system.py`
   - Port 9005'te OCPP Server açıyor
   - Gelen mesajları işliyor
   - Veriyi JSONL'ye kaydediyor

2. **Charge Points (Şarj İstasyonları)** - `src/charge_point.py`
   - CSMS'e WebSocket ile bağlanıyor
   - OCPP 1.6 mesajları gönderiyor
   - Her saniye MeterValues gönderiyor

3. **DoS Anomalisi** - `src/anomaly_injection.py`
   - 3 istasyon **flood** yapıyor
   - Her 3 saniyede anormal değerler
   - Voltaj: 350-550V (normal: 400±2V)
   - Akım: 40-90A (normal: 32±0.5A)

4. **Veri Logger** - `src/data_logger.py`
   - Her MeterValues → JSONL kayıt
   - `data/logs/basit_veri.jsonl`

---

## 📁 DOSYA YAPISI:

```
charge/
├── src/
│   ├── central_system.py       ✅ OCPP 1.6 Server (CSMS)
│   ├── charge_point.py          ✅ OCPP 1.6 Client (Şarj istasyonu)
│   ├── anomaly_injection.py     ✅ DoS anomalisi
│   └── data_logger.py           ✅ JSONL kayıt
├── run_simulation.py            ✅ Ana program
├── simple_config.yaml           ✅ Ayarlar
├── web_dashboard.py             ✅ Arayüz
└── data/logs/basit_veri.jsonl   ✅ Üretilen veri
```

---

## 🚀 NASIL ÇALIŞTIRILIIR?

```bash
# 1. Dashboard'u aç (ayrı terminal)
python web_dashboard.py

# 2. Simülasyonu çalıştır
python run_simulation.py

# 3. Tarayıcıda aç
http://localhost:5000
```

---

## 📊 VERİ ÇıKTıSı:

**data/logs/basit_veri.jsonl** içinde:
```json
{"timestamp": "2025-11-09T20:30:01Z", "charge_point_id": "CP-000", "transaction_id": 6381, "voltage": 398.72, "current": 32.56, "power_kw": 12.984, "energy_kwh": 0.002, "state": "Charging", "ocpp_message": "MeterValues", "network_latency_ms": 43, "anomaly_label": "normal"}
{"timestamp": "2025-11-09T20:30:02Z", "charge_point_id": "CP-003", "transaction_id": 9353, "voltage": 487.33, "current": 76.21, "power_kw": 37.14, "energy_kwh": 0.002, "state": "Charging", "ocpp_message": "MeterValues", "network_latency_ms": 21, "anomaly_label": "voltage_anomaly"}
```

- **Normal kayıtlar**: voltage=400±2, current=32±0.5
- **DoS anomali**: voltage=350-550, current=40-90
- **Tespit hedefi**: %95+ doğruluk

---

## 🎯 SONRAKI ADIMLAR:

1. ✅ **Veri toplama** (ŞUAN BURADA!)
   - 5 normal + 3 DoS istasyon
   - 30 saniye simülasyon
   - ~150 kayıt

2. ⏭️ **Feature extraction**
   - Windowed mean/std
   - Delta values
   - Time series features

3. ⏭️ **Model training**
   - XGBoost / RandomForest
   - LSTM / Transformer
   - Target: %95+ TPR

4. ⏭️ **Evaluation**
   - Confusion matrix
   - ROC-AUC
   - Precision/Recall

---

## ✅ ÖZET:

**EVET, TAM OCPP 1.6 KULLANIYORUZ!**
- Gerçek OCPP mesajları
- WebSocket üzerinden
- JSON formatında
- Merkez sistem + Şarj istasyonları
- DoS anomalisi aktif
- Veri üretiliyor! 🎉


"""
Central System (CSMS) - OCPP Server
Tüm charge point'lerden gelen mesajları işler
"""

import random
from datetime import datetime
from ocpp.v16 import ChargePoint as CP, call_result
from ocpp.routing import on
from ocpp.v16.enums import RegistrationStatus, AuthorizationStatus

from .data_logger import veri_kaydet


class CentralSystem(CP):
    """OCPP Central System"""
    
    @on('BootNotification')
    def boot(self, charge_point_model, charge_point_vendor, **kwargs):
        return call_result.BootNotification(
            current_time=datetime.utcnow().isoformat(),
            interval=300,
            status=RegistrationStatus.accepted
        )
    
    @on('Heartbeat')
    def heartbeat(self, **kwargs):
        return call_result.Heartbeat(current_time=datetime.utcnow().isoformat())
    
    @on('Authorize')
    def authorize(self, id_tag, **kwargs):
        return call_result.Authorize(id_tag_info={'status': AuthorizationStatus.accepted})
    
    @on('StartTransaction')
    def start_tx(self, connector_id, id_tag, meter_start, timestamp, **kwargs):
        trans_id = random.randint(1000, 9999)
        return call_result.StartTransaction(
            transaction_id=trans_id,
            id_tag_info={'status': AuthorizationStatus.accepted}
        )
    
    @on('MeterValues')
    def meter(self, connector_id, meter_value, transaction_id=None, **kwargs):
        """Process and log meter values"""
        voltaj, akim, guc, enerji = 400, 32, 12.8, 0
        
        for mv in meter_value:
            timestamp = mv.get('timestamp')
            for sample in mv.get('sampled_value', []):
                measurand_str = str(sample.get('measurand', '')).lower()
                value = float(sample.get('value', 0))
                
                if 'voltage' in measurand_str:
                    voltaj = value
                elif 'current' in measurand_str:
                    akim = value
                elif 'power' in measurand_str:
                    guc = value / 1000
                elif 'energy' in measurand_str:
                    enerji = value / 1000
            
            # Basit anomali tespiti
            anomaly = "normal"
            if voltaj > 450 or voltaj < 350:
                anomaly = "voltage_anomaly"
            elif akim > 50 or akim < 5:
                anomaly = "current_anomaly"
            elif guc > 25 or guc < 1:
                anomaly = "power_anomaly"
            
            # Veriyi kaydet
            veri_kaydet({
                "timestamp": timestamp,
                "charge_point_id": self.id,
                "transaction_id": transaction_id,
                "voltage": voltaj,
                "current": akim,
                "power_kw": guc,
                "energy_kwh": enerji,
                "state": "Charging",
                "ocpp_message": "MeterValues",
                "network_latency_ms": random.randint(20, 50),
                "anomaly_label": anomaly
            })
        
        return call_result.MeterValues()
    
    @on('StopTransaction')
    def stop_tx(self, meter_stop, timestamp, transaction_id, **kwargs):
        return call_result.StopTransaction(id_tag_info={'status': 'Accepted'})
    
    @on('StatusNotification')
    def status(self, connector_id, error_code, status, **kwargs):
        return call_result.StatusNotification()


async def connection_handler(websocket):
    """Handle new charge point connection"""
    path = websocket.request.path
    cp_id = path.strip('/')
    
    cs = CentralSystem(cp_id, websocket)
    try:
        await cs.start()
    except Exception:
        # Bağlantı kapandı - normal
        pass

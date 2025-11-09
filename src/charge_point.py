"""
Charge Point Emulator - OCPP Client
Şarj istasyonunu simüle eder
"""

import asyncio
import random
from datetime import datetime
from ocpp.v16 import ChargePoint as CP, call
from ocpp.v16.enums import ChargePointStatus, Measurand


class ChargePointEmulator(CP):
    """OCPP Charge Point Emulator"""
    
    async def send_boot(self):
        """Boot notification gönder"""
        req = call.BootNotification(
            charge_point_model="EV-Charger-v1",
            charge_point_vendor="TestCorp"
        )
        return await self.call(req)
    
    async def send_status(self, status):
        """Status notification gönder"""
        req = call.StatusNotification(
            connector_id=1,
            error_code='NoError',
            status=status
        )
        return await self.call(req)
    
    async def charge_session(self, duration_seconds=30, anomaly_injector=None):
        """
        Şarj seansı
        
        Args:
            duration_seconds: Şarj süresi (saniye)
            anomaly_injector: Anomali enjekte edecek fonksiyon (optional)
        """
        # Authorize
        req = call.Authorize(id_tag='RFID-12345')
        await self.call(req)
        
        # Start transaction
        req = call.StartTransaction(
            connector_id=1,
            id_tag='RFID-12345',
            meter_start=0,
            timestamp=datetime.utcnow().isoformat()
        )
        resp = await self.call(req)
        trans_id = resp.transaction_id
        
        # Charging loop
        energy = 0.0
        for second in range(duration_seconds):
            energy += 0.002  # 7.2 kW
            
            # Normal değerler
            voltage = 400 + random.gauss(0, 2)
            current = 32 + random.gauss(0, 0.5)
            power = (voltage * current) / 1000
            
            # Anomali enjeksiyonu (eğer varsa)
            if anomaly_injector:
                voltage, current, power, energy = anomaly_injector(
                    second, voltage, current, power, energy
                )
            
            # MeterValues gönder
            await self._send_meter_values(trans_id, voltage, current, power, energy)
            await asyncio.sleep(1)
        
        # Stop transaction
        req = call.StopTransaction(
            meter_stop=int(energy * 1000),
            timestamp=datetime.utcnow().isoformat(),
            transaction_id=trans_id
        )
        await self.call(req)
    
    async def _send_meter_values(self, trans_id, voltage, current, power, energy):
        """MeterValues mesajı gönder"""
        req = call.MeterValues(
            connector_id=1,
            transaction_id=trans_id,
            meter_value=[{
                'timestamp': datetime.utcnow().isoformat(),
                'sampled_value': [
                    {'value': str(voltage), 'measurand': Measurand.voltage, 'unit': 'V'},
                    {'value': str(current), 'measurand': Measurand.current_import, 'unit': 'A'},
                    {'value': str(power * 1000), 'measurand': Measurand.power_active_import, 'unit': 'W'},
                    {'value': str(int(energy * 1000)), 'measurand': Measurand.energy_active_import_register, 'unit': 'Wh'}
                ]
            }]
        )
        await self.call(req)


async def run_charge_point(cp_id, host, port, duration=30, anomaly_type=None, delay=0):
    """
    Bir charge point'i çalıştır
    
    Args:
        cp_id: Charge point ID (örn: "CP-001")
        host: CSMS host
        port: CSMS port
        duration: Şarj süresi (saniye)
        anomaly_type: Anomali tipi (None=normal, "dos", "voltage_spike", vs.)
        delay: Başlatma gecikmesi (saniye)
    """
    import websockets
    from .anomaly_injection import get_anomaly_injector
    
    await asyncio.sleep(delay)
    
    async with websockets.connect(
        f'ws://{host}:{port}/{cp_id}',
        subprotocols=['ocpp1.6']
    ) as ws:
        cp = ChargePointEmulator(cp_id, ws)
        
        # Start message processing
        cp_task = asyncio.create_task(cp.start())
        
        # Boot & charge
        await cp.send_boot()
        await cp.send_status(ChargePointStatus.available)
        
        # Get anomaly injector (if any)
        anomaly_injector = get_anomaly_injector(anomaly_type, cp_id)
        
        # Charge session
        await cp.charge_session(duration, anomaly_injector)
        
        # Back to available
        await cp.send_status(ChargePointStatus.available)
        
        # Cleanup
        cp_task.cancel()
        try:
            await cp_task
        except asyncio.CancelledError:
            pass



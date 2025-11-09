"""
OCPP 1.6 Charge Point Emulator
Simulates EV charging station behavior with realistic metrics
"""

import asyncio
import logging
import random
from datetime import datetime
from typing import Optional, Dict
import websockets
from ocpp.v16 import call, ChargePoint as CP
from ocpp.v16.enums import ChargePointStatus, Measurand
# Unit and other enums may not be available in newer versions, use strings instead

logger = logging.getLogger(__name__)


class ChargePointEmulator(CP):
    """Emulates a single charge point with realistic behavior"""
    
    def __init__(self, charge_point_id: str, websocket, config: dict, 
                 anomaly_controller=None):
        super().__init__(charge_point_id, websocket)
        self.config = config
        self.anomaly_controller = anomaly_controller
        
        # State variables
        self.status = ChargePointStatus.available
        self.current_transaction_id: Optional[int] = None
        self.connector_id = 1
        self.id_tag = f"USER-{charge_point_id}"
        
        # Meter values (cumulative)
        self.energy_kwh = 0.0
        self.meter_value_wh = 0  # in Wh
        
        # Current measurements
        self.voltage = config["voltage_nominal"]
        self.current = 0.0  # Amps
        self.power_kw = 0.0
        
        # Timing
        self.heartbeat_interval = config.get("heartbeat_interval", 5)
        self.meter_interval = config.get("meter_values_interval", 1)
        
        # Tasks
        self.heartbeat_task = None
        self.meter_task = None
        self.transaction_task = None
        
        # Network simulation
        self.base_latency_ms = random.uniform(10, 50)
        
    async def simulate_network_latency(self):
        """Simulate realistic network latency"""
        latency = random.gauss(self.base_latency_ms, 5) / 1000
        await asyncio.sleep(max(0, latency))
    
    def calculate_realistic_values(self):
        """Calculate realistic voltage, current, power with noise"""
        if self.status == ChargePointStatus.charging:
            # Charging - realistic values
            base_voltage = self.config["voltage_nominal"]
            noise_std = self.config.get("voltage_noise_std", 2.0)
            
            self.voltage = random.gauss(base_voltage, noise_std)
            self.current = random.gauss(
                self.config.get("current_max", 32) * 0.9, 
                0.5
            )
            self.power_kw = (self.voltage * self.current) / 1000.0
            
            # Accumulate energy
            energy_delta = self.config.get("energy_rate", 0.002)  # kWh per second
            self.energy_kwh += energy_delta
            self.meter_value_wh = int(self.energy_kwh * 1000)
        else:
            # Not charging
            self.voltage = random.gauss(
                self.config["voltage_nominal"], 
                self.config.get("voltage_noise_std", 1.0)
            )
            self.current = 0.0
            self.power_kw = 0.0
    
    async def send_boot_notification(self):
        """Send BootNotification"""
        try:
            # Try new OCPP 2.x API first
            request = call.BootNotification(
                charging_station={
                    "model": "SimulatedEVSE-v1.0",
                    "vendor_name": "EVSimCorp"
                },
                reason="PowerUp"
            )
        except (AttributeError, TypeError):
            # Fallback to OCPP 1.6 API
            from ocpp.v16 import call as call_v16
            request = call_v16.BootNotification(
                charge_point_model="SimulatedEVSE-v1.0",
                charge_point_vendor="EVSimCorp"
            )
        
        logger.info(f"[{self.id}] → BootNotification")
        response = await self.call(request)
        logger.info(f"[{self.id}] ← BootNotification: {response.status}")
        
        if hasattr(response, 'interval'):
            self.heartbeat_interval = response.interval
            
    async def send_heartbeat(self):
        """Send periodic Heartbeat"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                try:
                    request = call.Heartbeat()
                except (AttributeError, TypeError):
                    from ocpp.v16 import call as call_v16
                    request = call_v16.Heartbeat()
                
                await self.call(request)
                logger.debug(f"[{self.id}] ♥ Heartbeat")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.id}] Heartbeat error: {e}")
                
    async def send_status_notification(self, status: ChargePointStatus, 
                                       error_code: str = "NoError"):
        """Send StatusNotification"""
        try:
            request = call.StatusNotification(
                connector_id=self.connector_id,
                error_code=error_code,
                status=status
            )
        except (AttributeError, TypeError):
            from ocpp.v16 import call as call_v16
            request = call_v16.StatusNotification(
                connector_id=self.connector_id,
                error_code=error_code,
                status=status
            )
        
        logger.info(f"[{self.id}] → StatusNotification: {status}")
        await self.call(request)
        
    async def start_transaction(self):
        """Start a charging transaction"""
        # Authorize first
        try:
            auth_request = call.Authorize(id_tag=self.id_tag)
        except (AttributeError, TypeError):
            from ocpp.v16 import call as call_v16
            auth_request = call_v16.Authorize(id_tag=self.id_tag)
        
        auth_response = await self.call(auth_request)
        
        if auth_response.id_tag_info["status"] != "Accepted":
            logger.warning(f"[{self.id}] Authorization failed")
            return False
        
        # Start transaction
        self.status = ChargePointStatus.preparing
        await self.send_status_notification(self.status)
        
        try:
            request = call.StartTransaction(
                connector_id=self.connector_id,
                id_tag=self.id_tag,
                meter_start=self.meter_value_wh,
                timestamp=datetime.utcnow().isoformat()
            )
        except (AttributeError, TypeError):
            from ocpp.v16 import call as call_v16
            request = call_v16.StartTransaction(
                connector_id=self.connector_id,
                id_tag=self.id_tag,
                meter_start=self.meter_value_wh,
                timestamp=datetime.utcnow().isoformat()
            )
        
        logger.info(f"[{self.id}] → StartTransaction")
        response = await self.call(request)
        
        self.current_transaction_id = response.transaction_id
        self.status = ChargePointStatus.charging
        await self.send_status_notification(self.status)
        
        logger.info(f"[{self.id}] ✓ Transaction started: {self.current_transaction_id}")
        return True
        
    async def stop_transaction(self, reason: str = "Local"):
        """Stop the current transaction"""
        if self.current_transaction_id is None:
            return
        
        self.status = ChargePointStatus.finishing
        await self.send_status_notification(self.status)
        
        try:
            request = call.StopTransaction(
                meter_stop=self.meter_value_wh,
                timestamp=datetime.utcnow().isoformat(),
                transaction_id=self.current_transaction_id,
                reason=reason
            )
        except (AttributeError, TypeError):
            from ocpp.v16 import call as call_v16
            request = call_v16.StopTransaction(
                meter_stop=self.meter_value_wh,
                timestamp=datetime.utcnow().isoformat(),
                transaction_id=self.current_transaction_id,
                reason=reason
            )
        
        logger.info(f"[{self.id}] → StopTransaction: {self.current_transaction_id}")
        await self.call(request)
        
        self.current_transaction_id = None
        self.status = ChargePointStatus.available
        await self.send_status_notification(self.status)
        
        logger.info(f"[{self.id}] ✓ Transaction stopped")
        
    async def send_meter_values(self):
        """Send periodic MeterValues during transaction"""
        while True:
            try:
                await asyncio.sleep(self.meter_interval)
                
                if self.current_transaction_id is None:
                    continue
                
                # Calculate current values
                self.calculate_realistic_values()
                
                # Check for anomaly injection
                voltage, current, power, energy = (
                    self.voltage, self.current, self.power_kw, self.energy_kwh
                )
                
                if self.anomaly_controller:
                    anomaly_data = await self.anomaly_controller.check_and_inject(
                        self.id, 
                        {
                            "voltage": voltage,
                            "current": current,
                            "power_kw": power,
                            "energy_kwh": energy
                        }
                    )
                    if anomaly_data:
                        voltage = anomaly_data.get("voltage", voltage)
                        current = anomaly_data.get("current", current)
                        power = anomaly_data.get("power_kw", power)
                        energy = anomaly_data.get("energy_kwh", energy)
                
                # Build MeterValues message
                meter_value = [
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "sampled_value": [
                            {
                                "value": str(voltage),
                                "context": "Sample.Periodic",
                                "measurand": Measurand.voltage,
                                "unit": "V"
                            },
                            {
                                "value": str(current),
                                "context": "Sample.Periodic",
                                "measurand": Measurand.current_import,
                                "unit": "A"
                            },
                            {
                                "value": str(power * 1000),  # in Watts
                                "context": "Sample.Periodic",
                                "measurand": Measurand.power_active_import,
                                "unit": "W"
                            },
                            {
                                "value": str(int(energy * 1000)),  # in Wh
                                "context": "Sample.Periodic",
                                "measurand": Measurand.energy_active_import_register,
                                "unit": "Wh"
                            }
                        ]
                    }
                ]
                
                try:
                    request = call.MeterValues(
                        connector_id=self.connector_id,
                        transaction_id=self.current_transaction_id,
                        meter_value=meter_value
                    )
                except (AttributeError, TypeError):
                    from ocpp.v16 import call as call_v16
                    request = call_v16.MeterValues(
                        connector_id=self.connector_id,
                        transaction_id=self.current_transaction_id,
                        meter_value=meter_value
                    )
                
                await self.call(request)
                logger.debug(f"[{self.id}] → MeterValues: {power:.2f}kW, {energy:.3f}kWh")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.id}] MeterValues error: {e}")
                
    async def simulate_charging_session(self, duration: int = 300):
        """Simulate a complete charging session"""
        logger.info(f"[{self.id}] Starting {duration}s charging session")
        
        success = await self.start_transaction()
        if not success:
            return
        
        # Charge for specified duration
        await asyncio.sleep(duration)
        
        await self.stop_transaction()
        
    async def run_simulation(self, session_duration: int = 300, 
                           idle_time: int = 60):
        """Run continuous charging simulation"""
        # Initial boot
        await self.send_boot_notification()
        await self.send_status_notification(ChargePointStatus.available)
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self.send_heartbeat())
        self.meter_task = asyncio.create_task(self.send_meter_values())
        
        try:
            while True:
                # Idle period
                await asyncio.sleep(idle_time)
                
                # Charging session
                await self.simulate_charging_session(session_duration)
                
        except asyncio.CancelledError:
            logger.info(f"[{self.id}] Simulation cancelled")
        finally:
            # Cleanup
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            if self.meter_task:
                self.meter_task.cancel()
                
            if self.current_transaction_id:
                await self.stop_transaction(reason="Other")


async def create_charge_point(cp_id: str, central_system_url: str, 
                              config: dict, anomaly_controller=None):
    """Factory function to create and connect a charge point"""
    try:
        async with websockets.connect(
            f"{central_system_url}/{cp_id}",
            subprotocols=["ocpp1.6"]
        ) as ws:
            cp = ChargePointEmulator(cp_id, ws, config, anomaly_controller)
            await cp.run_simulation(
                session_duration=random.randint(180, 600),
                idle_time=random.randint(30, 120)
            )
    except Exception as e:
        logger.error(f"[{cp_id}] Connection error: {e}")


if __name__ == "__main__":
    # Test single charge point
    logging.basicConfig(level=logging.INFO)
    
    config = {
        "voltage_nominal": 400.0,
        "voltage_noise_std": 2.0,
        "current_max": 32.0,
        "power_nominal": 7.4,
        "energy_rate": 0.002,
        "heartbeat_interval": 5,
        "meter_values_interval": 1
    }
    
    asyncio.run(
        create_charge_point(
            "CP-TEST-001", 
            "ws://localhost:9000",
            config
        )
    )


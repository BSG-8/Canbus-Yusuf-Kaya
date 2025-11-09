"""
Real-time Web Dashboard for EV Charging Simulation
Shows live statistics, anomalies, and charge point status
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import json
import threading
import time
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
import asyncio

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ev-charging-sim-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
simulation_stats = {
    "total_records": 0,
    "charge_points": {},
    "anomaly_counts": defaultdict(int),
    "records_per_second": 0,
    "start_time": None,
    "is_running": False,
    "recent_records": deque(maxlen=100)
}

LOG_FILE = Path("data/logs/basit_veri.jsonl")


def monitor_log_file():
    """Monitor log file and update stats"""
    last_position = 0
    last_count = 0
    
    while True:
        try:
            if LOG_FILE.exists():
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    # Seek to last position
                    f.seek(last_position)
                    
                    # Read new lines
                    new_lines = f.readlines()
                    last_position = f.tell()
                    
                    # Process new records
                    for line in new_lines:
                        if line.strip():
                            try:
                                record = json.loads(line)
                                process_record(record)
                            except json.JSONDecodeError:
                                continue
                    
                    # Calculate records per second
                    if simulation_stats["start_time"]:
                        elapsed = (datetime.now() - simulation_stats["start_time"]).total_seconds()
                        if elapsed > 0:
                            simulation_stats["records_per_second"] = simulation_stats["total_records"] / elapsed
                
                # Emit updates to connected clients
                if simulation_stats["total_records"] > last_count:
                    socketio.emit('stats_update', get_stats_json())
                    last_count = simulation_stats["total_records"]
            
            time.sleep(1)  # Check every second
            
        except Exception as e:
            print(f"Error monitoring log: {e}")
            time.sleep(5)


def process_record(record):
    """Process a single record and update stats"""
    simulation_stats["total_records"] += 1
    simulation_stats["is_running"] = True
    
    if not simulation_stats["start_time"]:
        simulation_stats["start_time"] = datetime.now()
    
    # Update charge point stats
    cp_id = record.get("charge_point_id", "Unknown")
    if cp_id not in simulation_stats["charge_points"]:
        simulation_stats["charge_points"][cp_id] = {
            "id": cp_id,
            "status": "Unknown",
            "voltage": 0,
            "current": 0,
            "power_kw": 0,
            "energy_kwh": 0,
            "last_update": None,
            "transaction_id": None
        }
    
    # Update values
    cp_data = simulation_stats["charge_points"][cp_id]
    cp_data["status"] = record.get("state", "Unknown")
    cp_data["voltage"] = record.get("voltage", 0) or 0
    cp_data["current"] = record.get("current", 0) or 0
    cp_data["power_kw"] = record.get("power_kw", 0) or 0
    cp_data["energy_kwh"] = record.get("energy_kwh", 0) or 0
    cp_data["last_update"] = record.get("timestamp")
    cp_data["transaction_id"] = record.get("transaction_id")
    
    # Anomaly tracking
    anomaly = record.get("anomaly_label", "normal")
    simulation_stats["anomaly_counts"][anomaly] += 1
    
    # Keep recent records
    simulation_stats["recent_records"].append({
        "timestamp": record.get("timestamp"),
        "cp_id": cp_id,
        "anomaly": anomaly,
        "power": cp_data["power_kw"]
    })


def get_stats_json():
    """Get stats as JSON for frontend"""
    return {
        "total_records": simulation_stats["total_records"],
        "charge_points": list(simulation_stats["charge_points"].values()),
        "anomaly_counts": dict(simulation_stats["anomaly_counts"]),
        "records_per_second": round(simulation_stats["records_per_second"], 2),
        "is_running": simulation_stats["is_running"],
        "uptime": (datetime.now() - simulation_stats["start_time"]).total_seconds() 
                  if simulation_stats["start_time"] else 0,
        "recent_records": list(simulation_stats["recent_records"])
    }


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/stats')
def get_stats():
    """API endpoint for stats"""
    return jsonify(get_stats_json())


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('stats_update', get_stats_json())


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnect"""
    print('Client disconnected')


def run_dashboard():
    """Run the dashboard server"""
    # Start log monitoring thread
    monitor_thread = threading.Thread(target=monitor_log_file, daemon=True)
    monitor_thread.start()
    
    print("\n" + "="*80)
    print("🌐 EV Charging Simulation Dashboard")
    print("="*80)
    print(f"📊 Dashboard URL: http://localhost:5000")
    print(f"📝 Monitoring: {LOG_FILE}")
    print("="*80 + "\n")
    
    # Run Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    run_dashboard()


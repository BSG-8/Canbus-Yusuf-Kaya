"""
Quick Test Script - Run a short simulation for testing
Duration: 5 minutes, 5 charge points
"""

import asyncio
import sys
import yaml
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_simulation import SimulationOrchestrator


async def quick_test():
    """Run quick 5-minute test"""
    
    # Load and modify config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Modify for quick test
    config["simulation"]["duration_seconds"] = 300  # 5 minutes
    config["simulation"]["charge_points"] = 5
    config["simulation"]["log_file"] = "data/logs/quick_test.jsonl"
    config["simulation"]["csv_output"] = "data/processed/quick_test.csv"
    
    # Save temporary config
    with open("config_test.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    
    print("="*80)
    print("🧪 QUICK TEST MODE")
    print("="*80)
    print("Duration: 5 minutes")
    print("Charge Points: 5")
    print("Output: data/logs/quick_test.jsonl")
    print("="*80 + "\n")
    
    # Run simulation
    orchestrator = SimulationOrchestrator("config_test.yaml")
    await orchestrator.setup()
    await orchestrator.run()
    
    print("\n" + "="*80)
    print("✓ Quick test complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Check logs: data/logs/quick_test.jsonl")
    print("  2. Generate features: python scripts/generate_features.py --input data/logs/quick_test.jsonl")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(quick_test())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")


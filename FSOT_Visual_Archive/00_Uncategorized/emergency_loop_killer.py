#!/usr/bin/env python3
"""
EMERGENCY LOOP KILLER
====================
This script will force-kill any Python processes that might be stuck in loops.
"""

import psutil
import os
import sys
import time
from datetime import datetime

print("🚨 EMERGENCY LOOP KILLER")
print("=" * 30)
print(f"Time: {datetime.now()}")
print()

# Find all Python processes
python_processes = []
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        proc_info = proc.as_dict(attrs=['pid', 'name', 'cmdline'])
        if proc_info['name'] and 'python' in proc_info['name'].lower():
            # Check if it's running your FSOT system
            cmdline = ' '.join(proc_info['cmdline'] or [])
            if any(keyword in cmdline.lower() for keyword in ['main.py', 'fsot', 'brain', 'neuromorphic']):
                python_processes.append((proc_info['pid'], cmdline))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

if python_processes:
    print("🔍 Found potentially stuck Python processes:")
    for pid, cmdline in python_processes:
        print(f"   PID {pid}: {cmdline[:80]}...")
    
    response = input("\n⚠️  Kill these processes? (y/N): ").lower().strip()
    
    if response == 'y':
        for pid, cmdline in python_processes:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                print(f"✅ Terminated PID {pid}")
                
                # Wait 3 seconds, then force kill if still running
                time.sleep(3)
                if proc.is_running():
                    proc.kill()
                    print(f"🔴 Force killed PID {pid}")
                    
            except Exception as e:
                print(f"❌ Could not kill PID {pid}: {e}")
        
        print("\n✅ Process cleanup completed")
    else:
        print("❌ No processes killed")
else:
    print("✅ No stuck Python processes found")

print(f"\nCompleted: {datetime.now()}")
print("🛡️ System should be safe now")

#!/usr/bin/env python3
"""
SIMPLE PROCESS KILLER
====================
Kill any stuck Python processes manually.
"""

import os
import sys

print("🚨 SIMPLE PROCESS KILLER")
print("Run this in a separate terminal:")
print()
print("# Windows:")
print("tasklist | findstr python")
print("taskkill /F /IM python.exe")
print()
print("# Or kill specific PID:")
print("taskkill /F /PID [PID_NUMBER]")
print()
print("# Alternative - Close all PowerShell/Terminal windows running Python")
print("# Then restart VS Code")
print()
print("✅ This will stop any endless loops immediately")

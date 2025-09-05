# FSOT Auto-Restart System for Pylance

## Overview

The FSOT 2.0 Hardwiring System now includes **automatic Pylance language server restart** functionality to resolve type checking issues that can occur when FSOT enforcement modifies class and function signatures.

## How It Works

### Automatic Restart Triggers

1. **Function Hardwiring**: Every time `@hardwire_fsot()` decorator is applied to a function
2. **Class Hardwiring**: Every time a class is hardwired for FSOT compliance  
3. **Threshold-Based**: Automatic restart after 5 accumulated changes
4. **Activation**: Automatic restart when `activate_fsot_hardwiring()` is called

### Restart Methods

The system uses multiple methods to ensure Pylance restarts:

1. **Workspace Settings Modification**: Updates `.vscode/settings.json` with timestamp
2. **Notification File**: Creates `.fsot_restart_needed` file as indicator
3. **Process Detection**: Finds VS Code processes (logging only, for safety)

## Usage

### Automatic Usage
```python
from fsot_hardwiring import hardwire_fsot

# This will automatically schedule a Pylance restart
@hardwire_fsot()
def my_function():
    pass
```

### Manual Restart
```python
from fsot_hardwiring import force_pylance_restart

# Force immediate restart
force_pylance_restart()
```

### Command Line
```bash
# From FSOT_Clean_System directory
python fsot_hardwiring.py --restart-pylance

# From project root
python restart_pylance.py
```

## Benefits

- ✅ **No More Manual Restarts**: Pylance automatically refreshes when FSOT makes changes
- ✅ **Type Safety Maintained**: Eliminates false type errors from dynamic modifications  
- ✅ **Seamless Integration**: Works transparently in VS Code environment
- ✅ **Fallback Support**: Provides notification files when direct restart fails

## Files Created

- `.fsot_restart_needed`: Notification file when restart is needed
- `.vscode/settings.json`: Updated with FSOT timestamps to trigger refresh

## Troubleshooting

If automatic restart doesn't work:

1. **Manual Restart**: `Ctrl+Shift+P` → `"Python: Restart Language Server"`
2. **Reload Window**: `Ctrl+Shift+P` → `"Developer: Reload Window"`
3. **Check Notification**: Look for `.fsot_restart_needed` file in project root

## Configuration

The restart system can be configured:

```python
# Change restart threshold (default: 5 changes)
vscode_integration.schedule_restart_if_needed(changes_threshold=3)

# Force immediate restart
vscode_integration.force_restart()
```

---

**The FSOT Auto-Restart System ensures seamless type checking while maintaining theoretical compliance!** 🌟

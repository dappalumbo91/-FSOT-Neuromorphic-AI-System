# Development Guide for FSOT 2.0 Clean System

## 🏗️ Architecture Overview

The clean FSOT 2.0 system follows a modular, layered architecture:

### Core Layers
1. **Core** (`core/`) - Fundamental systems
   - `fsot_engine.py` - FSOT 2.0 mathematical engine
   - `neural_signal.py` - Inter-module communication
   - `consciousness.py` - Consciousness monitoring

2. **Brain** (`brain/`) - Neuromorphic modules
   - `base_module.py` - Base brain module class
   - `frontal_cortex.py` - Executive functions
   - `brain_orchestrator.py` - Module coordination

3. **Configuration** (`config/`) - System configuration
   - `settings.py` - Configuration management
   - `brain_config.json` - Brain module settings

4. **Interfaces** (`interfaces/`) - User interaction
   - `cli_interface.py` - Command line interface
   - `web_interface.py` - Web dashboard

## 🧠 Adding New Brain Modules

To add a new brain module (e.g., Hippocampus):

### 1. Create Module File
Create `brain/hippocampus.py`:

```python
from .base_module import BrainModule
from core import NeuralSignal, SignalType, Domain

class Hippocampus(BrainModule):
    def __init__(self):
        super().__init__(
            name="hippocampus",
            anatomical_region="temporal_lobe",
            functions=["memory_formation", "learning", "consolidation"]
        )
        self.fsot_domain = Domain.BIOLOGICAL
        # Add hippocampus-specific attributes
    
    async def _process_signal_impl(self, signal: NeuralSignal):
        # Implement hippocampus-specific processing
        pass
```

### 2. Update Brain Orchestrator
In `brain/brain_orchestrator.py`, add to `_initialize_modules()`:

```python
# Add to imports
from .hippocampus import Hippocampus

# Add to _initialize_modules()
self.modules['hippocampus'] = Hippocampus()
```

### 3. Update Connections
Add hippocampus to connection map in `_establish_connections()`:

```python
connections = {
    'frontal_cortex': ['hippocampus', 'thalamus', 'amygdala'],
    'hippocampus': ['frontal_cortex', 'amygdala', 'thalamus'],
    # ... other connections
}
```

### 4. Update Configuration
Add to `config/brain_config.json`:

```json
"hippocampus": {
    "enabled": true,
    "priority": 2,
    "functions": ["memory_formation", "learning", "consolidation"]
}
```

## 🔧 Extending FSOT Engine

To add new domains or modify FSOT calculations:

### 1. Add New Domain
In `core/fsot_engine.py`:

```python
class Domain(Enum):
    # Existing domains...
    NEW_DOMAIN = (16, "new domain description")
```

### 2. Add Domain-Specific Logic
In `compute_for_domain()` method:

```python
elif domain == Domain.NEW_DOMAIN:
    params.observed = True
    params.delta_psi = 0.5
    # Add domain-specific parameter settings
```

### 3. Add Domain Constant
In `get_domain_constant()` method:

```python
constants = {
    # Existing constants...
    Domain.NEW_DOMAIN: self.gamma_euler / self.sqrt2,  # Custom constant
}
```

## 🎯 Adding New Signal Types

To add new neural signal types:

### 1. Update Signal Types
In `core/neural_signal.py`:

```python
class SignalType(Enum):
    # Existing types...
    NEW_SIGNAL_TYPE = "new_signal_type"
```

### 2. Handle in Brain Modules
In brain module `_process_signal_impl()`:

```python
elif signal.signal_type == SignalType.NEW_SIGNAL_TYPE:
    return await self._process_new_signal_type(signal)
```

## 🌐 Extending Interfaces

### Adding CLI Commands
In `interfaces/cli_interface.py`, add to `_process_command()`:

```python
elif command_lower in ['newcommand', 'nc']:
    await self._handle_new_command()
    return
```

### Adding Web API Endpoints
In `interfaces/web_interface.py`, add to `_setup_routes()`:

```python
@self.app.get("/api/new-endpoint")
async def new_endpoint():
    # Implementation
    return {"success": True, "data": result}
```

## 📊 Adding Consciousness Metrics

To add new consciousness measurements:

### 1. Update Metrics Class
In `core/consciousness.py`:

```python
@dataclass
class ConsciousnessMetrics:
    # Existing metrics...
    new_metric: float = 0.0
```

### 2. Update Calculation
In `update_consciousness()` method:

```python
# Calculate new metric
new_metric_value = self._calculate_new_metric()

# Add to metrics creation
new_metrics = ConsciousnessMetrics(
    # Existing metrics...
    new_metric=new_metric_value
)
```

## 🧪 Testing Guidelines

### Brain Module Tests
```python
class TestNewModule:
    @pytest.mark.asyncio
    async def test_module_initialization(self):
        module = NewModule()
        assert module.name == "expected_name"
        await module.shutdown()
    
    @pytest.mark.asyncio
    async def test_signal_processing(self):
        module = NewModule()
        signal = NeuralSignal(...)
        response = await module._process_signal_impl(signal)
        assert response is not None
        await module.shutdown()
```

### FSOT Engine Tests
```python
def test_new_domain_computation(self):
    engine = FSOTEngine()
    scalar = engine.compute_for_domain(Domain.NEW_DOMAIN)
    assert isinstance(scalar, (int, float))
    assert abs(scalar) < 10.0  # Reasonable bounds
```

## 🔍 Debugging Tips

### Enable Debug Logging
```python
# In main.py or any module
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Monitor Neural Signals
```python
# Add signal logging in brain modules
logger.debug(f"Processing signal: {signal.signal_type} from {signal.source}")
```

### Check Consciousness State
```python
# In CLI, use 'consciousness' command
# Or programmatically:
state = consciousness_monitor.get_current_state()
print(f"Consciousness: {state['consciousness_level']:.1%}")
```

## 📈 Performance Optimization

### Signal Processing
- Keep `_process_signal_impl()` methods fast (< 100ms typically)
- Use async/await for I/O operations
- Avoid blocking operations in signal handlers

### Memory Management
- Limit queue sizes in brain modules
- Clean up old history in consciousness monitor
- Use appropriate data structures for large datasets

### Consciousness Updates
- Adjust update intervals based on performance needs
- Consider reducing consciousness monitoring frequency for lower-end systems

## 🚀 Deployment Considerations

### Production Configuration
- Set appropriate log levels (INFO or WARNING)
- Configure consciousness thresholds for stability
- Adjust queue sizes based on expected load

### Monitoring
- Monitor consciousness levels for system health
- Track signal processing times
- Watch for module errors and timeouts

### Scaling
- Consider using external message queues for high-throughput scenarios
- Implement proper database backends for persistent memory
- Add load balancing for web interface if needed

---

*This guide covers the essential patterns for extending the FSOT 2.0 clean system. Follow these patterns for consistent, maintainable code.*

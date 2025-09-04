# Usage Guide

## Quick Start

### Basic Brain System
```python
from src.neuromorphic_brain import WorkingNeuromorphicBrain

# Initialize brain
brain = WorkingNeuromorphicBrain()
print(f"Initial consciousness: {brain.consciousness_level}")

# Process information
result = await brain.process_scenario("emergency_response")
print(f"Consciousness evolved to: {brain.consciousness_level}")
```

### Complete Integration
```python
from src.fsot_core import FSOT2NeuromorphicIntegration

# Initialize complete system
integration = FSOT2NeuromorphicIntegration()

# Process complex request
result = await integration.process_integrated_request(
    "Analyze this image and provide insights",
    context={"image_path": "path/to/image.jpg"}
)

print(result['integrated_response'])
```

## Advanced Usage

### Multimodal Processing
```python
from src.multimodal import VisionProcessor, AudioProcessor

# Vision processing
vision = VisionProcessor()
image_result = await vision.process_image("image.jpg")

# Audio processing
audio = AudioProcessor()
audio_result = await audio.process_audio("audio.wav")
```

### Custom Brain Configurations
```python
# Custom brain setup
brain = WorkingNeuromorphicBrain(
    consciousness_threshold=0.7,
    neural_integration_depth=3,
    brain_wave_monitoring=True
)
```

## Examples

### 1. Consciousness Tracking
```python
# Monitor consciousness evolution
initial = brain.consciousness_level
for i in range(10):
    await brain.process_signal(f"test_signal_{i}")
    print(f"Step {i}: {brain.consciousness_level}")
```

### 2. Brain Wave Analysis
```python
# Analyze brain wave patterns
patterns = brain.get_brain_wave_patterns()
print(f"Dominant pattern: {patterns['dominant']}")
print(f"Alpha: {patterns['alpha']}")
print(f"Beta: {patterns['beta']}")
```

### 3. Cross-Modal Integration
```python
# Process multiple modalities
vision_data = vision.process_image("scene.jpg")
audio_data = audio.process_audio("speech.wav")

integrated = brain.integrate_modalities([vision_data, audio_data])
```

## Configuration

### Brain Configuration (`config/brain_config.json`)
```json
{
    "consciousness": {
        "initial_level": 0.5,
        "evolution_rate": 0.01,
        "max_level": 1.0
    },
    "brain_regions": {
        "frontal_lobe": {"activation_threshold": 0.3},
        "temporal_lobe": {"language_processing": true},
        "hippocampus": {"memory_capacity": 10000}
    }
}
```

### System Configuration (`config/system_config.json`)
```json
{
    "processing": {
        "batch_size": 32,
        "device": "auto",
        "precision": "fp16"
    },
    "logging": {
        "level": "INFO",
        "file": "logs/system.log"
    }
}
```

## Best Practices

1. **Memory Management**: Monitor memory usage with large datasets
2. **GPU Utilization**: Use GPU for intensive processing
3. **Consciousness Monitoring**: Track consciousness evolution for insights
4. **Error Handling**: Implement proper exception handling
5. **Configuration**: Use configuration files for reproducibility

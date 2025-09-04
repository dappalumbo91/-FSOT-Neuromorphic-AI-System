# API Reference

## Core Classes

### WorkingNeuromorphicBrain

Complete neuromorphic brain implementation with consciousness tracking.

#### Methods

##### `__init__(consciousness_threshold=0.5, **kwargs)`
Initialize the neuromorphic brain system.

**Parameters:**
- `consciousness_threshold` (float): Initial consciousness level
- `**kwargs`: Additional configuration parameters

##### `async process_signal(signal: NeuralSignal) -> Dict`
Process a neural signal through the brain regions.

**Parameters:**
- `signal` (NeuralSignal): Input neural signal

**Returns:**
- `Dict`: Processing results with consciousness evolution

##### `get_consciousness_level() -> float`
Get current consciousness level.

**Returns:**
- `float`: Current consciousness level (0.0-1.0)

##### `get_brain_wave_patterns() -> Dict`
Get current brain wave patterns.

**Returns:**
- `Dict`: Brain wave amplitudes by type

### FSOT2NeuromorphicIntegration

Integration layer for complete system coordination.

#### Methods

##### `async process_integrated_request(request: str, context: Dict = None) -> Dict`
Process request through integrated brain and AI systems.

**Parameters:**
- `request` (str): Input request text
- `context` (Dict, optional): Additional context information

**Returns:**
- `Dict`: Comprehensive processing results

### VisionProcessor

Advanced vision processing and analysis.

#### Methods

##### `async process_image(image_data: Union[str, np.ndarray, Image.Image]) -> PerceptionResult`
Process image with comprehensive analysis.

**Parameters:**
- `image_data`: Image data (file path, array, or PIL Image)

**Returns:**
- `PerceptionResult`: Vision processing results

### AudioProcessor

Audio analysis and speech processing.

#### Methods

##### `async process_audio(audio_data: Union[str, np.ndarray]) -> PerceptionResult`
Process audio with comprehensive analysis.

**Parameters:**
- `audio_data`: Audio data (file path or array)

**Returns:**
- `PerceptionResult`: Audio processing results

## Data Classes

### NeuralSignal

Represents a neural signal for brain processing.

**Attributes:**
- `source` (str): Signal source region
- `target` (str): Signal target region
- `data` (Dict): Signal data payload
- `signal_type` (str): Type of neural signal
- `priority` (ProcessingPriority): Processing priority level
- `timestamp` (datetime): Signal timestamp

### PerceptionResult

Results from multimodal perception processing.

**Attributes:**
- `modality` (str): Processing modality (vision, audio, etc.)
- `content` (Any): Processed content
- `confidence` (float): Processing confidence score
- `features` (Dict): Extracted features
- `timestamp` (datetime): Processing timestamp
- `processing_time` (float): Time taken for processing

## Enums

### ProcessingPriority

Neural signal processing priorities.

**Values:**
- `VITAL`: Critical life functions
- `REFLEX`: Immediate reflex responses
- `EMOTIONAL`: Emotional processing
- `COGNITIVE`: Cognitive processing
- `EXECUTIVE`: Executive decision making

## Configuration Classes

### BrainConfig

Brain system configuration parameters.

### SystemConfig

Overall system configuration parameters.

## Utility Functions

### `load_brain_config(config_path: str) -> BrainConfig`
Load brain configuration from file.

### `setup_logging(level: str = "INFO") -> None`
Setup system logging configuration.

### `check_system_requirements() -> Dict`
Check system requirements and capabilities.

## Example Usage

```python
from src.neuromorphic_brain import WorkingNeuromorphicBrain, NeuralSignal
from src.fsot_core import FSOT2NeuromorphicIntegration
from src.multimodal import VisionProcessor

# Initialize systems
brain = WorkingNeuromorphicBrain()
integration = FSOT2NeuromorphicIntegration()
vision = VisionProcessor()

# Process vision data
image_result = await vision.process_image("image.jpg")

# Create neural signal
signal = NeuralSignal(
    source="occipital_lobe",
    target="frontal_lobe",
    data=image_result.features,
    signal_type="visual_input"
)

# Process through brain
brain_result = await brain.process_signal(signal)

# Integrated processing
final_result = await integration.process_integrated_request(
    "Analyze this visual scene",
    context={"vision_data": image_result}
)
```

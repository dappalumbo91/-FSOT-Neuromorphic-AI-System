#!/usr/bin/env python3
"""
Free Multi-Modal Processing System for Enhanced FSOT 2.0
========================================================

Free alternatives for vision, audio, and text processing without paid APIs:
- Vision: OpenCV for basic image processing
- Audio: SoundDevice/PyAudio for basic audio capture (optional)
- Text: Basic NLP with NLTK and spaCy-free alternatives
- Cross-Modal: Simple fusion mechanisms

Author: GitHub Copilot
"""

import cv2
import numpy as np
import json
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Union, cast
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
import logging

# Type helpers for numpy/opencv compatibility
def safe_mean(arr: Any) -> float:
    """Safely calculate mean with proper type casting"""
    return float(np.mean(np.asarray(arr, dtype=np.float64)))

def safe_std(arr: Any) -> float:
    """Safely calculate std with proper type casting"""
    return float(np.std(np.asarray(arr, dtype=np.float64)))

# Optional audio libraries
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Basic NLP libraries
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result from multi-modal processing"""
    modality: str
    content: Any
    features: Dict[str, Any]
    confidence: float
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class FreeVisionProcessor:
    """Free vision processing using OpenCV"""
    
    def __init__(self):
        self.logger = logging.getLogger('VisionProcessor')
        self.processing_history = []
        
    def process_image(self, image_path: Union[str, np.ndarray]) -> ProcessingResult:
        """Process image with OpenCV-based analysis"""
        start_time = time.time()
        
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
            else:
                image = image_path
            
            # Basic image analysis
            analysis = self._analyze_image_properties(image)
            
            # Object detection (basic)
            objects = self._detect_basic_objects(image)
            
            # Color analysis
            colors = self._analyze_colors(image)
            
            # Texture analysis
            texture = self._analyze_texture(image)
            
            features = {
                "image_properties": analysis,
                "detected_objects": objects,
                "color_analysis": colors,
                "texture_analysis": texture,
                "image_shape": image.shape
            }
            
            processing_time = time.time() - start_time
            confidence = self._calculate_confidence(features)
            
            result = ProcessingResult(
                modality="vision",
                content={"image_analysis": features},
                features=features,
                confidence=confidence,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={"opencv_version": cv2.__version__}
            )
            
            self.processing_history.append(result)
            self.logger.info(f"Processed image with confidence: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Image processing error: {e}")
            return ProcessingResult(
                modality="vision",
                content={"error": str(e)},
                features={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    def _analyze_image_properties(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze basic image properties"""
        height, width = image.shape[:2]
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate statistics with proper type casting
        gray_array = np.asarray(gray, dtype=np.float64)
        mean_brightness = float(np.mean(gray_array))
        std_brightness = float(np.std(gray_array))
        
        # Edge detection for complexity
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / float(height * width)
        
        return {
            "width": int(width),
            "height": int(height),
            "aspect_ratio": width / height,
            "mean_brightness": float(mean_brightness),
            "brightness_std": float(std_brightness),
            "edge_density": float(edge_density),
            "total_pixels": int(height * width)
        }
    
    def _detect_basic_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Basic object detection using OpenCV contours"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for i, contour in enumerate(contours[:10]):  # Limit to top 10
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area > 100:  # Filter small objects
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                objects.append({
                    "id": i,
                    "area": float(area),
                    "perimeter": float(perimeter),
                    "bounding_box": [int(x), int(y), int(w), int(h)],
                    "aspect_ratio": w / h if h > 0 else 0,
                    "type": "contour_object"
                })
        
        return objects
    
    def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze color properties of the image"""
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate dominant colors
        pixels = image.reshape(-1, 3)
        
        # Simple k-means clustering for dominant colors with proper initialization
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        pixels_float = pixels.astype(np.float32)
        
        # Initialize labels array for k-means
        labels = np.zeros((pixels_float.shape[0],), dtype=np.int32)
        
        try:
            _, labels, centers = cv2.kmeans(pixels_float, 5, labels, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            dominant_colors = centers.astype(int).tolist() if centers is not None else []
        except Exception:
            # Fallback if k-means fails
            dominant_colors = []
        
        # Calculate color statistics with safe functions
        return {
            "dominant_colors": dominant_colors,
            "mean_hue": safe_mean(hsv[:, :, 0]),
            "mean_saturation": safe_mean(hsv[:, :, 1]),
            "mean_value": safe_mean(hsv[:, :, 2]),
            "color_variance": safe_std(pixels) ** 2
        }
    
    def _analyze_texture(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze texture properties"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate local binary patterns (simplified)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Calculate texture complexity with safe functions
        gray_mean = safe_mean(gray)
        gray_std = safe_std(gray)
        texture_complexity = gray_std / gray_mean if gray_mean > 0 else 0.0
        
        return {
            "gradient_magnitude_mean": safe_mean(gradient_magnitude),
            "gradient_magnitude_std": safe_std(gradient_magnitude),
            "texture_complexity": texture_complexity
        }
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate processing confidence based on features"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on detected features
        if features.get("detected_objects"):
            confidence += 0.2
        
        if features.get("color_analysis", {}).get("color_variance", 0) > 1000:
            confidence += 0.1  # Rich color content
        
        if features.get("image_properties", {}).get("edge_density", 0) > 0.1:
            confidence += 0.1  # Good edge content
        
        return min(1.0, confidence)

class FreeAudioProcessor:
    """Free audio processing using basic audio libraries"""
    
    def __init__(self):
        self.logger = logging.getLogger('AudioProcessor')
        self.is_available = AUDIO_AVAILABLE
        self.processing_history = []
        
        if not self.is_available:
            self.logger.warning("Audio processing not available - install sounddevice: pip install sounddevice")
    
    def process_audio(self, audio_data: Union[str, np.ndarray], sample_rate: int = 44100) -> ProcessingResult:
        """Process audio with basic analysis"""
        start_time = time.time()
        
        if not self.is_available:
            return ProcessingResult(
                modality="audio",
                content={"error": "Audio processing not available"},
                features={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
        
        try:
            # Load audio data
            if isinstance(audio_data, str):
                # For now, just simulate loading from file
                audio = np.random.randn(sample_rate)  # Placeholder
                self.logger.warning("File loading not implemented - using simulated data")
            else:
                audio = audio_data
            
            # Basic audio analysis
            analysis = self._analyze_audio_properties(audio, sample_rate)
            
            features = {
                "audio_properties": analysis,
                "sample_rate": sample_rate,
                "duration": len(audio) / sample_rate
            }
            
            processing_time = time.time() - start_time
            confidence = self._calculate_audio_confidence(features)
            
            result = ProcessingResult(
                modality="audio",
                content={"audio_analysis": features},
                features=features,
                confidence=confidence,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={"audio_available": True}
            )
            
            self.processing_history.append(result)
            self.logger.info(f"Processed audio with confidence: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Audio processing error: {e}")
            return ProcessingResult(
                modality="audio",
                content={"error": str(e)},
                features={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    def _analyze_audio_properties(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze basic audio properties"""
        
        # Time domain analysis
        rms = np.sqrt(np.mean(audio**2))
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        
        # Simple frequency analysis using FFT
        fft = np.fft.fft(audio)
        frequencies = np.fft.fftfreq(len(audio), 1/sample_rate)
        magnitude = np.abs(fft)
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
        dominant_frequency = abs(frequencies[dominant_freq_idx])
        
        return {
            "rms_amplitude": float(rms),
            "zero_crossings": int(zero_crossings),
            "dominant_frequency": float(dominant_frequency),
            "spectral_centroid": float(np.sum(frequencies[:len(frequencies)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])),
            "audio_energy": float(np.sum(audio**2))
        }
    
    def _calculate_audio_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate audio processing confidence"""
        confidence = 0.5
        
        # Boost confidence based on signal properties
        if features.get("audio_properties", {}).get("rms_amplitude", 0) > 0.01:
            confidence += 0.2  # Good signal level
        
        if features.get("audio_properties", {}).get("dominant_frequency", 0) > 100:
            confidence += 0.1  # Reasonable frequency content
        
        return min(1.0, confidence)
    
    def record_audio(self, duration: float = 5.0, sample_rate: int = 44100) -> Optional[np.ndarray]:
        """Record audio from microphone"""
        if not self.is_available:
            self.logger.warning("Audio recording not available")
            return None
        
        try:
            self.logger.info(f"Recording {duration} seconds of audio...")
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()  # Wait for recording to complete
            
            return audio.flatten()
            
        except Exception as e:
            self.logger.error(f"Audio recording error: {e}")
            return None

class FreeTextProcessor:
    """Free text processing using basic NLP"""
    
    def __init__(self):
        self.logger = logging.getLogger('TextProcessor')
        self.is_available = NLTK_AVAILABLE
        self.processing_history = []
        
        # Download required NLTK data if available
        if self.is_available:
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
            except LookupError:
                self.logger.info("Downloading NLTK data...")
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                except:
                    self.logger.warning("Could not download NLTK data")
        else:
            self.logger.warning("NLTK not available - install with: pip install nltk")
    
    def process_text(self, text: str) -> ProcessingResult:
        """Process text with basic NLP analysis"""
        start_time = time.time()
        
        try:
            # Basic text analysis
            analysis = self._analyze_text_properties(text)
            
            # Sentiment analysis (basic)
            sentiment = self._basic_sentiment_analysis(text)
            
            # Keyword extraction
            keywords = self._extract_keywords(text)
            
            features = {
                "text_properties": analysis,
                "sentiment": sentiment,
                "keywords": keywords,
                "text_length": len(text)
            }
            
            processing_time = time.time() - start_time
            confidence = self._calculate_text_confidence(features)
            
            result = ProcessingResult(
                modality="text",
                content={"text_analysis": features},
                features=features,
                confidence=confidence,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={"nltk_available": self.is_available}
            )
            
            self.processing_history.append(result)
            self.logger.info(f"Processed text with confidence: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Text processing error: {e}")
            return ProcessingResult(
                modality="text",
                content={"error": str(e)},
                features={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    def _analyze_text_properties(self, text: str) -> Dict[str, Any]:
        """Analyze basic text properties"""
        
        # Basic tokenization
        if self.is_available:
            try:
                sentences = sent_tokenize(text)
                words = word_tokenize(text.lower())
            except:
                # Fallback to basic splitting
                sentences = text.split('.')
                words = text.lower().split()
        else:
            sentences = text.split('.')
            words = text.lower().split()
        
        # Calculate statistics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        
        # Character analysis
        char_count = len(text)
        alphabetic_chars = sum(1 for c in text if c.isalpha())
        numeric_chars = sum(1 for c in text if c.isdigit())
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "character_count": char_count,
            "avg_words_per_sentence": avg_words_per_sentence,
            "alphabetic_ratio": alphabetic_chars / char_count if char_count > 0 else 0,
            "numeric_ratio": numeric_chars / char_count if char_count > 0 else 0,
            "unique_words": len(set(words))
        }
    
    def _basic_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Basic sentiment analysis using word lists"""
        
        # Simple positive/negative word lists
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                         'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
                         'sad', 'angry', 'frustrated', 'disappointed', 'upset'}
        
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        else:
            sentiment_score = 0.0
        
        return {
            "sentiment_score": sentiment_score,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "sentiment_label": "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral"
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using basic frequency analysis"""
        
        if self.is_available:
            try:
                # Use NLTK for better processing
                words = word_tokenize(text.lower())
                stop_words = set(stopwords.words('english'))
                stemmer = PorterStemmer()
                
                # Filter words
                filtered_words = [
                    stemmer.stem(word) for word in words 
                    if word.isalpha() and word not in stop_words and len(word) > 2
                ]
            except:
                # Fallback
                filtered_words = self._basic_keyword_extraction(text)
        else:
            filtered_words = self._basic_keyword_extraction(text)
        
        # Count frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:10]]
    
    def _basic_keyword_extraction(self, text: str) -> List[str]:
        """Basic keyword extraction without NLTK"""
        basic_stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        words = text.lower().split()
        return [word for word in words if word.isalpha() and word not in basic_stop_words and len(word) > 2]
    
    def _calculate_text_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate text processing confidence"""
        confidence = 0.5
        
        # Boost confidence based on text properties
        text_props = features.get("text_properties", {})
        
        if text_props.get("word_count", 0) > 10:
            confidence += 0.2  # Sufficient text length
        
        if text_props.get("unique_words", 0) > 5:
            confidence += 0.1  # Good vocabulary diversity
        
        if features.get("keywords"):
            confidence += 0.1  # Keywords extracted
        
        return min(1.0, confidence)

class MultiModalFusion:
    """Simple multi-modal fusion without complex ML"""
    
    def __init__(self):
        self.logger = logging.getLogger('MultiModalFusion')
        self.fusion_history = []
        
    def fuse_modalities(self, 
                       vision_result: Optional[ProcessingResult] = None,
                       audio_result: Optional[ProcessingResult] = None,
                       text_result: Optional[ProcessingResult] = None) -> Dict[str, Any]:
        """Fuse multiple modality results"""
        
        start_time = time.time()
        
        fusion_result = {
            "fusion_timestamp": datetime.now().isoformat(),
            "modalities_processed": [],
            "combined_confidence": 0.0,
            "unified_features": {},
            "cross_modal_insights": [],
            "processing_summary": {}
        }
        
        results = [
            ("vision", vision_result),
            ("audio", audio_result), 
            ("text", text_result)
        ]
        
        total_confidence = 0.0
        valid_results = 0
        
        for modality, result in results:
            if result and result.confidence > 0:
                fusion_result["modalities_processed"].append(modality)
                fusion_result["unified_features"][modality] = result.features
                fusion_result["processing_summary"][modality] = {
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "timestamp": result.timestamp.isoformat()
                }
                
                total_confidence += result.confidence
                valid_results += 1
        
        # Calculate combined confidence
        if valid_results > 0:
            fusion_result["combined_confidence"] = total_confidence / valid_results
            
            # Generate cross-modal insights
            fusion_result["cross_modal_insights"] = self._generate_insights(fusion_result)
        
        fusion_result["fusion_processing_time"] = time.time() - start_time
        
        self.fusion_history.append(fusion_result)
        self.logger.info(f"Fused {valid_results} modalities with confidence: {fusion_result['combined_confidence']:.2f}")
        
        return fusion_result
    
    def _generate_insights(self, fusion_result: Dict[str, Any]) -> List[str]:
        """Generate simple cross-modal insights"""
        insights = []
        
        modalities = fusion_result["modalities_processed"]
        
        if "vision" in modalities and "text" in modalities:
            insights.append("Visual and textual information available for cross-reference")
        
        if "audio" in modalities and "text" in modalities:
            insights.append("Audio and text can be correlated for enhanced understanding")
        
        if len(modalities) >= 3:
            insights.append("Multi-modal context provides rich information for comprehensive analysis")
        
        # Check confidence levels
        confidences = [fusion_result["processing_summary"][mod]["confidence"] for mod in modalities]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        if avg_confidence > 0.8:
            insights.append("High confidence multi-modal processing achieved")
        elif avg_confidence > 0.5:
            insights.append("Moderate confidence multi-modal processing")
        else:
            insights.append("Low confidence processing - may need additional verification")
        
        return insights

class FreeMultiModalSystem:
    """Complete free multi-modal processing system"""
    
    def __init__(self):
        self.logger = logging.getLogger('MultiModalSystem')
        
        # Initialize processors
        self.vision_processor = FreeVisionProcessor()
        self.audio_processor = FreeAudioProcessor()
        self.text_processor = FreeTextProcessor()
        self.fusion_engine = MultiModalFusion()
        
        # System status
        self.is_initialized = True
        self.processing_count = 0
        
        self.logger.info("Free Multi-Modal System initialized")
    
    def process_input(self, 
                     image: Optional[Union[str, np.ndarray]] = None,
                     audio: Optional[Union[str, np.ndarray]] = None,
                     text: Optional[str] = None) -> Dict[str, Any]:
        """Process multi-modal input"""
        
        start_time = time.time()
        self.processing_count += 1
        
        self.logger.info(f"Processing multi-modal input #{self.processing_count}")
        
        # Process each modality
        vision_result = None
        audio_result = None  
        text_result = None
        
        if image is not None:
            self.logger.info("Processing vision input...")
            vision_result = self.vision_processor.process_image(image)
        
        if audio is not None:
            self.logger.info("Processing audio input...")
            audio_result = self.audio_processor.process_audio(audio)
        
        if text is not None:
            self.logger.info("Processing text input...")
            text_result = self.text_processor.process_text(text)
        
        # Fuse results
        self.logger.info("Fusing multi-modal results...")
        fusion_result = self.fusion_engine.fuse_modalities(
            vision_result=vision_result,
            audio_result=audio_result,
            text_result=text_result
        )
        
        # Compile final result
        total_time = time.time() - start_time
        
        final_result = {
            "session_id": self.processing_count,
            "total_processing_time": total_time,
            "individual_results": {
                "vision": asdict(vision_result) if vision_result else None,
                "audio": asdict(audio_result) if audio_result else None,
                "text": asdict(text_result) if text_result else None
            },
            "fusion_result": fusion_result,
            "system_status": self.get_system_status()
        }
        
        self.logger.info(f"Multi-modal processing complete in {total_time:.2f}s")
        
        return final_result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get multi-modal system status"""
        return {
            "initialized": self.is_initialized,
            "processing_count": self.processing_count,
            "vision_available": True,
            "audio_available": self.audio_processor.is_available,
            "text_available": self.text_processor.is_available,
            "opencv_version": cv2.__version__,
            "components": {
                "vision_processor": len(self.vision_processor.processing_history),
                "audio_processor": len(self.audio_processor.processing_history),
                "text_processor": len(self.text_processor.processing_history),
                "fusion_engine": len(self.fusion_engine.fusion_history)
            }
        }

# Global instance for easy access
_multimodal_system = None

def get_multimodal_system() -> FreeMultiModalSystem:
    """Get or create global multi-modal system instance"""
    global _multimodal_system
    if _multimodal_system is None:
        _multimodal_system = FreeMultiModalSystem()
    return _multimodal_system

if __name__ == "__main__":
    # Test the free multi-modal system
    print("🎭 Testing Free Multi-Modal Processing System")
    print("=" * 50)
    
    system = FreeMultiModalSystem()
    
    # Test text processing
    print("\n📝 Testing Text Processing...")
    text_input = "This is a great example of natural language processing. I love how AI can understand text!"
    
    # Test with just text
    result = system.process_input(text=text_input)
    print(f"   Text processing confidence: {result['individual_results']['text']['confidence']:.2f}")
    print(f"   Fusion confidence: {result['fusion_result']['combined_confidence']:.2f}")
    
    # Test vision processing with synthetic image
    print("\n🖼️ Testing Vision Processing...")
    synthetic_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    
    result = system.process_input(image=synthetic_image)
    print(f"   Vision processing confidence: {result['individual_results']['vision']['confidence']:.2f}")
    
    # Test multi-modal processing
    print("\n🎭 Testing Multi-Modal Processing...")
    result = system.process_input(
        image=synthetic_image,
        text="This image shows interesting patterns and colors."
    )
    
    print(f"   Combined processing:")
    print(f"   - Modalities: {result['fusion_result']['modalities_processed']}")
    print(f"   - Combined confidence: {result['fusion_result']['combined_confidence']:.2f}")
    print(f"   - Insights: {len(result['fusion_result']['cross_modal_insights'])}")
    
    # System status
    print("\n📊 System Status:")
    status = system.get_system_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for subkey, subvalue in value.items():
                print(f"     {subkey}: {subvalue}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Free Multi-Modal System test complete!")

"""
FSOT 2.0 Amygdala Brain Module
Safety Assessment and Emotional Processing
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .base_module import BrainModule
from core import NeuralSignal, SignalType, Priority

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat assessment levels"""
    SAFE = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

class EmotionalState(Enum):
    """Basic emotional states"""
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ANXIOUS = "anxious"
    ALERT = "alert"
    CALM = "calm"

@dataclass
class SafetyAssessment:
    """Safety assessment result"""
    threat_level: ThreatLevel
    confidence: float
    reasons: List[str]
    recommendations: List[str]
    timestamp: datetime

@dataclass
class EmotionalAssessment:
    """Emotional state assessment"""
    primary_emotion: EmotionalState
    intensity: float  # 0.0 to 1.0
    secondary_emotions: List[EmotionalState]
    context: str
    timestamp: datetime

class Amygdala(BrainModule):
    """
    Amygdala Brain Module - Safety and Emotional Processing
    
    Responsibilities:
    - Threat detection and safety assessment
    - Emotional state monitoring and processing
    - Risk evaluation and response recommendations
    - Safety filtering for system inputs/outputs
    """
    
    def __init__(self):
        super().__init__(
            name="amygdala",
            anatomical_region="limbic_system",
            functions=[
                "threat_detection",
                "safety_assessment",
                "emotional_processing",
                "risk_evaluation",
                "safety_filtering"
            ]
        )
        
        # Safety patterns and keywords
        self.threat_patterns = self._init_threat_patterns()
        self.safety_keywords = self._init_safety_keywords()
        self.emotional_keywords = self._init_emotional_keywords()
        
        # Assessment history
        self.safety_assessments: List[SafetyAssessment] = []
        self.emotional_assessments: List[EmotionalAssessment] = []
        
        # Thresholds
        self.threat_threshold = 0.7  # Threshold for flagging threats
        self.emotional_intensity_threshold = 0.6  # Threshold for emotional responses
        
        # Performance metrics
        self.assessments_performed = 0
        self.threats_detected = 0
        self.safety_blocks = 0
    
    def _init_threat_patterns(self) -> List[str]:
        """Initialize threat detection patterns"""
        return [
            r'\b(hack|exploit|vulnerability|attack|malware|virus)\b',
            r'\b(password|credential|login|secret|private key)\b',
            r'\b(harm|damage|destroy|delete|remove)\b',
            r'\b(illegal|unauthorized|forbidden|banned)\b',
            r'\b(suicide|self-harm|violence|weapon)\b'
        ]
    
    def _init_safety_keywords(self) -> Set[str]:
        """Initialize safety-related keywords"""
        return {
            'safe', 'secure', 'protect', 'privacy', 'confidential',
            'dangerous', 'risky', 'threat', 'warning', 'caution',
            'help', 'support', 'emergency', 'crisis', 'urgent'
        }
    
    def _init_emotional_keywords(self) -> Dict[EmotionalState, Set[str]]:
        """Initialize emotional keyword mappings"""
        return {
            EmotionalState.POSITIVE: {
                'happy', 'joy', 'excited', 'pleased', 'satisfied',
                'grateful', 'optimistic', 'hopeful', 'confident'
            },
            EmotionalState.NEGATIVE: {
                'sad', 'angry', 'frustrated', 'disappointed', 'upset',
                'worried', 'concerned', 'troubled', 'distressed'
            },
            EmotionalState.ANXIOUS: {
                'anxious', 'nervous', 'stressed', 'overwhelmed',
                'panic', 'fear', 'scared', 'worried', 'tense'
            },
            EmotionalState.ALERT: {
                'alert', 'focused', 'attentive', 'vigilant',
                'aware', 'ready', 'prepared', 'sharp'
            },
            EmotionalState.CALM: {
                'calm', 'peaceful', 'relaxed', 'serene',
                'tranquil', 'composed', 'balanced', 'stable'
            }
        }
    
    async def process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming neural signals"""
        try:
            if signal.signal_type == SignalType.SAFETY_CHECK:
                return await self._assess_safety(signal)
            elif signal.signal_type == SignalType.EMOTIONAL_ANALYSIS:
                return await self._assess_emotion(signal)
            elif signal.signal_type == SignalType.THREAT_DETECTION:
                return await self._detect_threats(signal)
            else:
                # Perform automatic safety screening on all signals
                safety_result = await self._screen_content(signal.data)
                if safety_result.threat_level.value >= ThreatLevel.MEDIUM.value:
                    return await self._create_safety_block_signal(signal, safety_result)
                return signal
                
        except Exception as e:
            logger.error(f"Error processing signal in amygdala: {e}")
            return None
    
    async def _process_signal_impl(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Module-specific signal processing implementation"""
        return await self.process_signal(signal)
    
    async def _assess_safety(self, signal: NeuralSignal) -> NeuralSignal:
        """Perform comprehensive safety assessment"""
        content = signal.data.get('content', '')
        context = signal.data.get('context', {})
        
        assessment = await self._perform_safety_assessment(content, context)
        self.safety_assessments.append(assessment)
        self.assessments_performed += 1
        
        if assessment.threat_level.value >= ThreatLevel.HIGH.value:
            self.threats_detected += 1
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.SAFETY_RESULT,
            data={
                'assessment': {
                    'threat_level': assessment.threat_level.name,
                    'confidence': assessment.confidence,
                    'reasons': assessment.reasons,
                    'recommendations': assessment.recommendations
                },
                'safe_to_proceed': assessment.threat_level.value < ThreatLevel.MEDIUM.value
            },
            priority=Priority.HIGH
        )
    
    async def _assess_emotion(self, signal: NeuralSignal) -> NeuralSignal:
        """Assess emotional content and state"""
        content = signal.data.get('content', '')
        context = signal.data.get('context', {})
        
        assessment = await self._perform_emotional_assessment(content, context)
        self.emotional_assessments.append(assessment)
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.EMOTIONAL_RESULT,
            data={
                'assessment': {
                    'primary_emotion': assessment.primary_emotion.value,
                    'intensity': assessment.intensity,
                    'secondary_emotions': [e.value for e in assessment.secondary_emotions],
                    'context': assessment.context
                },
                'emotional_response_needed': assessment.intensity > self.emotional_intensity_threshold
            },
            priority=Priority.HIGH
        )
    
    async def _detect_threats(self, signal: NeuralSignal) -> NeuralSignal:
        """Detect specific threats in content"""
        content = signal.data.get('content', '')
        
        threats = []
        for pattern in self.threat_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                threats.extend(matches)
        
        threat_level = ThreatLevel.SAFE
        if threats:
            threat_level = ThreatLevel.HIGH if len(threats) > 2 else ThreatLevel.MEDIUM
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.THREAT_RESULT,
            data={
                'threats_found': threats,
                'threat_level': threat_level.name,
                'threat_count': len(threats)
            },
            priority=Priority.HIGH if threats else Priority.LOW
        )
    
    async def _perform_safety_assessment(self, content: str, context: Dict[str, Any]) -> SafetyAssessment:
        """Perform detailed safety assessment"""
        reasons = []
        confidence = 0.5
        threat_level = ThreatLevel.SAFE
        recommendations = []
        
        # Check for threat patterns
        threat_score = 0
        for pattern in self.threat_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                threat_score += len(matches)  # Count each match
                reasons.append(f"Detected {len(matches)} threat pattern(s): {', '.join(matches)}")
        
        # Check for safety keywords
        content_lower = content.lower()
        safety_keywords_found = [kw for kw in self.safety_keywords if kw in content_lower]
        
        if 'dangerous' in safety_keywords_found or 'threat' in safety_keywords_found:
            threat_score += 1
            reasons.append("Contains danger-related keywords")
        
        # Determine threat level
        if threat_score >= 3:
            threat_level = ThreatLevel.CRITICAL
            confidence = 0.95
            recommendations.append("Critical threat detected")
            recommendations.append("Immediate security review required")
        elif threat_score >= 2:
            threat_level = ThreatLevel.HIGH
            confidence = 0.9
            recommendations.append("High-risk content detected")
            recommendations.append("Human review recommended")
        elif threat_score == 1:
            threat_level = ThreatLevel.MEDIUM
            confidence = 0.8
            recommendations.append("Moderate safety review needed")
        elif 'dangerous' in safety_keywords_found or 'threat' in safety_keywords_found:
            threat_level = ThreatLevel.LOW
            confidence = 0.7
            recommendations.append("Minor safety concern detected")
        else:
            threat_level = ThreatLevel.SAFE
            confidence = 0.9
            recommendations.append("Content appears safe")
        
        if not reasons:
            reasons.append("No specific threats detected")
        
        return SafetyAssessment(
            threat_level=threat_level,
            confidence=confidence,
            reasons=reasons,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    async def _perform_emotional_assessment(self, content: str, context: Dict[str, Any]) -> EmotionalAssessment:
        """Perform emotional content assessment"""
        content_lower = content.lower()
        
        # Score emotions based on keywords
        emotion_scores = {}
        for emotion, keywords in self.emotional_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                emotion_scores[emotion] = score / len(keywords)  # Normalize
        
        # Determine primary emotion
        primary_emotion = EmotionalState.NEUTRAL
        intensity = 0.0
        secondary_emotions = []
        
        if emotion_scores:
            # Primary emotion is the highest scoring
            primary_emotion = max(emotion_scores.keys(), key=lambda e: emotion_scores[e])
            intensity = min(1.0, emotion_scores[primary_emotion] * 2)  # Scale up
            
            # Secondary emotions are others with significant scores
            secondary_emotions = [
                emotion for emotion, score in emotion_scores.items()
                if emotion != primary_emotion and score > 0.1
            ]
        
        return EmotionalAssessment(
            primary_emotion=primary_emotion,
            intensity=intensity,
            secondary_emotions=secondary_emotions,
            context=context.get('situation', 'general'),
            timestamp=datetime.now()
        )
    
    async def _screen_content(self, data: Dict[str, Any]) -> SafetyAssessment:
        """Quick safety screening of content"""
        content = str(data.get('content', ''))
        if not content:
            return SafetyAssessment(
                threat_level=ThreatLevel.SAFE,
                confidence=1.0,
                reasons=["No content to assess"],
                recommendations=["Content is safe"],
                timestamp=datetime.now()
            )
        
        return await self._perform_safety_assessment(content, data.get('context', {}))
    
    async def _create_safety_block_signal(self, original_signal: NeuralSignal, assessment: SafetyAssessment) -> NeuralSignal:
        """Create a signal indicating content was blocked for safety"""
        self.safety_blocks += 1
        
        return NeuralSignal(
            source=self.name,
            target=original_signal.source,
            signal_type=SignalType.SAFETY_BLOCK,
            data={
                'blocked': True,
                'original_signal_type': original_signal.signal_type.name,
                'assessment': {
                    'threat_level': assessment.threat_level.name,
                    'confidence': assessment.confidence,
                    'reasons': assessment.reasons,
                    'recommendations': assessment.recommendations
                },
                'message': f"Content blocked due to {assessment.threat_level.name} threat level"
            },
            priority=Priority.HIGH
        )
    
    async def perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Clean old assessments (keep last 100)
        if len(self.safety_assessments) > 100:
            self.safety_assessments = self.safety_assessments[-100:]
        
        if len(self.emotional_assessments) > 100:
            self.emotional_assessments = self.emotional_assessments[-100:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get amygdala status"""
        base_status = super().get_status()
        
        # Recent assessments summary
        recent_safety = [a for a in self.safety_assessments if (datetime.now() - a.timestamp).seconds < 3600]
        recent_emotional = [a for a in self.emotional_assessments if (datetime.now() - a.timestamp).seconds < 3600]
        
        amygdala_status = {
            'total_assessments': self.assessments_performed,
            'threats_detected': self.threats_detected,
            'safety_blocks': self.safety_blocks,
            'recent_safety_assessments': len(recent_safety),
            'recent_emotional_assessments': len(recent_emotional),
            'threat_threshold': self.threat_threshold,
            'current_alert_level': self._get_current_alert_level()
        }
        
        base_status.update(amygdala_status)
        return base_status
    
    def _get_current_alert_level(self) -> str:
        """Determine current system alert level"""
        recent_threats = [
            a for a in self.safety_assessments[-10:]  # Last 10 assessments
            if a.threat_level.value >= ThreatLevel.MEDIUM.value
        ]
        
        if len(recent_threats) >= 3:
            return "HIGH"
        elif len(recent_threats) >= 1:
            return "MEDIUM"
        else:
            return "LOW"

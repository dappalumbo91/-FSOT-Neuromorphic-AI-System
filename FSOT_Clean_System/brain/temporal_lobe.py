"""
FSOT 2.0 Temporal Lobe Brain Module
Language Processing and Auditory Functions
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from .base_module import BrainModule
from core import NeuralSignal, SignalType, Priority

logger = logging.getLogger(__name__)

class LanguageMode(Enum):
    """Language processing modes"""
    COMPREHENSION = "comprehension"
    GENERATION = "generation"
    TRANSLATION = "translation"
    ANALYSIS = "analysis"
    LISTENING = "listening"

class AuditoryState(Enum):
    """Auditory processing states"""
    QUIET = "quiet"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"

@dataclass
class LanguagePattern:
    """Represents a recognized language pattern"""
    pattern_type: str
    content: str
    confidence: float
    context: str
    timestamp: datetime

@dataclass
class ConversationContext:
    """Conversation context tracking"""
    topic: str
    participants: List[str]
    mood: str
    complexity_level: int
    start_time: datetime
    last_interaction: datetime

class TemporalLobe(BrainModule):
    """
    Temporal Lobe Brain Module - Language and Auditory Processing
    
    Responsibilities:
    - Language comprehension and generation
    - Auditory processing and interpretation
    - Speech pattern recognition
    - Conversation management
    - Memory integration with language
    - Semantic understanding
    """
    
    def __init__(self):
        super().__init__(
            name="temporal_lobe",
            anatomical_region="temporal_cortex",
            functions=[
                "language_comprehension",
                "language_generation", 
                "auditory_processing",
                "speech_recognition",
                "conversation_management",
                "semantic_analysis",
                "memory_language_integration"
            ]
        )
        
        # Language processing state
        self.current_mode = LanguageMode.COMPREHENSION
        self.auditory_state = AuditoryState.QUIET
        
        # Language capabilities
        self.vocabulary_size = 50000
        self.language_models = ['english', 'technical', 'conversational']
        self.comprehension_accuracy = 0.92
        self.generation_fluency = 0.88
        
        # Conversation tracking
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.language_patterns: List[LanguagePattern] = []
        
        # Performance metrics
        self.words_processed = 0
        self.sentences_analyzed = 0
        self.conversations_managed = 0
        self.pattern_matches = 0
        
        # Language processing parameters
        self.semantic_depth = 3  # levels of semantic analysis
        self.context_window = 10  # sentences to maintain context
        self.response_formality = 0.7  # 0.0 casual to 1.0 formal
        
        # Initialize language components
        self._initialize_language_processors()
    
    def _initialize_language_processors(self):
        """Initialize language processing components"""
        # Common language patterns
        self.grammar_patterns = {
            'question': r'\b(what|when|where|why|how|who|which|is|are|do|does|did|can|could|would|will)\b',
            'command': r'\b(please|make|create|do|execute|run|start|stop|help)\b',
            'emotion': r'\b(happy|sad|excited|frustrated|angry|pleased|worried|confident)\b',
            'technical': r'\b(algorithm|function|variable|system|process|data|analysis)\b',
            'time': r'\b(now|today|tomorrow|yesterday|morning|evening|minute|hour|day|week)\b'
        }
        
        # Semantic categories
        self.semantic_categories = {
            'action': ['do', 'make', 'create', 'execute', 'perform', 'run'],
            'object': ['system', 'file', 'data', 'information', 'result'],
            'quality': ['good', 'bad', 'fast', 'slow', 'accurate', 'efficient'],
            'quantity': ['many', 'few', 'several', 'multiple', 'single', 'all'],
            'location': ['here', 'there', 'system', 'memory', 'storage', 'network']
        }
        
        # Conversation starters and responses
        self.conversation_templates = {
            'greeting': ['Hello', 'Hi', 'Good morning', 'How can I help?'],
            'acknowledgment': ['I understand', 'Got it', 'Understood', 'I see'],
            'clarification': ['Could you clarify?', 'What do you mean by?', 'Can you elaborate?'],
            'completion': ['Done', 'Completed', 'Finished', 'Ready']
        }
    
    async def _process_signal_impl(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming neural signals"""
        try:
            if signal.signal_type == SignalType.LANGUAGE_COMPREHENSION:
                return await self._comprehend_language(signal)
            elif signal.signal_type == SignalType.LANGUAGE_GENERATION:
                return await self._generate_language(signal)
            elif signal.signal_type == SignalType.AUDITORY_PROCESSING:
                return await self._process_auditory(signal)
            elif signal.signal_type == SignalType.CONVERSATION_MANAGEMENT:
                return await self._manage_conversation(signal)
            elif signal.signal_type == SignalType.SEMANTIC_ANALYSIS:
                return await self._analyze_semantics(signal)
            else:
                # Automatic language analysis for all text content
                return await self._analyze_language_content(signal)
                
        except Exception as e:
            logger.error(f"Error processing signal in temporal lobe: {e}")
            return None
    
    async def _comprehend_language(self, signal: NeuralSignal) -> NeuralSignal:
        """Comprehend and analyze language input"""
        text_data = signal.data.get('text', '')
        context = signal.data.get('context', {})
        
        # Perform comprehensive language analysis
        comprehension_result = await self._perform_language_comprehension(text_data, context)
        
        self.current_mode = LanguageMode.COMPREHENSION
        self.words_processed += len(text_data.split())
        self.sentences_analyzed += len(re.split(r'[.!?]+', text_data))
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.LANGUAGE_COMPREHENSION_RESULT,
            data={
                'comprehension_result': comprehension_result,
                'confidence': self.comprehension_accuracy,
                'processing_mode': self.current_mode.value,
                'semantic_depth': self.semantic_depth
            },
            priority=Priority.HIGH
        )
    
    async def _perform_language_comprehension(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed language comprehension"""
        if not text.strip():
            return {'error': 'No text to comprehend'}
        
        # Tokenize and analyze
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        
        # Pattern recognition
        patterns_found = {}
        for pattern_name, pattern_regex in self.grammar_patterns.items():
            matches = re.findall(pattern_regex, text, re.IGNORECASE)
            if matches:
                patterns_found[pattern_name] = matches
                self.pattern_matches += len(matches)
        
        # Semantic analysis
        semantic_analysis = await self._perform_semantic_analysis(' '.join(words), depth=2)
        
        # Intent detection
        intent = await self._detect_intent(text, patterns_found)
        
        # Context integration
        context_analysis = await self._integrate_context(text, context)
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'patterns_detected': patterns_found,
            'semantic_analysis': semantic_analysis,
            'intent': intent,
            'context_integration': context_analysis,
            'complexity_score': self._calculate_text_complexity(text),
            'comprehension_confidence': self.comprehension_accuracy
        }
    
    async def _generate_language(self, signal: NeuralSignal) -> NeuralSignal:
        """Generate language response"""
        generation_request = signal.data.get('generation', {})
        intent = generation_request.get('intent', 'respond')
        context = generation_request.get('context', {})
        style = generation_request.get('style', 'conversational')
        
        # Generate appropriate response
        generated_response = await self._perform_language_generation(intent, context, style)
        
        self.current_mode = LanguageMode.GENERATION
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.LANGUAGE_GENERATION_RESULT,
            data={
                'generated_text': generated_response,
                'generation_style': style,
                'fluency_score': self.generation_fluency,
                'word_count': len(generated_response.split())
            },
            priority=Priority.HIGH
        )
    
    async def _perform_language_generation(self, intent: str, context: Dict[str, Any], style: str) -> str:
        """Generate language based on intent and context"""
        # Select appropriate template based on intent
        if intent in self.conversation_templates:
            base_response = self.conversation_templates[intent][0]  # Use first template
        else:
            base_response = "I understand your request"
        
        # Enhance based on context
        if context.get('technical', False):
            base_response += " from a technical perspective"
        
        # Adjust formality
        if self.response_formality > 0.8:
            base_response = base_response.replace("I'm", "I am").replace("can't", "cannot")
        
        # Add context-specific content
        topic = context.get('topic', '')
        if topic:
            base_response += f" regarding {topic}"
        
        return base_response + "."
    
    async def _process_auditory(self, signal: NeuralSignal) -> NeuralSignal:
        """Process auditory input"""
        audio_data = signal.data.get('audio', {})
        audio_type = audio_data.get('type', 'speech')
        content = audio_data.get('content', '')
        
        # Process based on audio type
        processing_result = await self._analyze_auditory_content(audio_type, content)
        
        self.auditory_state = AuditoryState.PROCESSING
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.AUDITORY_PROCESSING_RESULT,
            data={
                'auditory_analysis': processing_result,
                'processing_state': self.auditory_state.value,
                'audio_type': audio_type
            },
            priority=Priority.HIGH
        )
    
    async def _analyze_auditory_content(self, audio_type: str, content: str) -> Dict[str, Any]:
        """Analyze auditory content"""
        if audio_type == 'speech':
            # Speech recognition and analysis
            return {
                'recognized_text': content,
                'confidence': 0.85,
                'speaker_characteristics': {'tone': 'neutral', 'pace': 'normal'},
                'speech_patterns': await self._detect_speech_patterns(content)
            }
        elif audio_type == 'sound':
            # Sound pattern recognition
            return {
                'sound_type': 'environmental',
                'classification': 'system_notification',
                'intensity': 0.6
            }
        else:
            return {'type': 'unknown', 'confidence': 0.1}
    
    async def _detect_speech_patterns(self, text: str) -> Dict[str, Any]:
        """Detect patterns in speech"""
        patterns = {}
        
        # Detect question patterns
        if re.search(r'\?', text):
            patterns['question'] = True
        
        # Detect emotional indicators
        emotion_words = re.findall(self.grammar_patterns['emotion'], text, re.IGNORECASE)
        if emotion_words:
            patterns['emotional_content'] = emotion_words
        
        # Detect technical language
        tech_words = re.findall(self.grammar_patterns['technical'], text, re.IGNORECASE)
        if tech_words:
            patterns['technical_content'] = tech_words
        
        return patterns
    
    async def _manage_conversation(self, signal: NeuralSignal) -> NeuralSignal:
        """Manage conversation context and flow"""
        conversation_data = signal.data.get('conversation', {})
        conversation_id = conversation_data.get('id', 'default')
        action = conversation_data.get('action', 'continue')
        content = conversation_data.get('content', '')
        
        # Manage conversation based on action
        if action == 'start':
            result = await self._start_conversation(conversation_id, conversation_data)
        elif action == 'continue':
            result = await self._continue_conversation(conversation_id, content)
        elif action == 'end':
            result = await self._end_conversation(conversation_id)
        else:
            result = {'error': f'Unknown conversation action: {action}'}
        
        self.conversations_managed += 1
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.CONVERSATION_MANAGEMENT_RESULT,
            data={
                'conversation_result': result,
                'conversation_id': conversation_id,
                'active_conversations': len(self.active_conversations)
            },
            priority=Priority.NORMAL
        )
    
    async def _start_conversation(self, conv_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new conversation"""
        self.active_conversations[conv_id] = ConversationContext(
            topic=data.get('topic', 'general'),
            participants=data.get('participants', ['user']),
            mood=data.get('mood', 'neutral'),
            complexity_level=data.get('complexity', 1),
            start_time=datetime.now(),
            last_interaction=datetime.now()
        )
        
        return {
            'status': 'conversation_started',
            'conversation_id': conv_id,
            'initial_response': 'Hello! How can I help you today?'
        }
    
    async def _continue_conversation(self, conv_id: str, content: str) -> Dict[str, Any]:
        """Continue an existing conversation"""
        if conv_id not in self.active_conversations:
            return {'error': 'Conversation not found'}
        
        conversation = self.active_conversations[conv_id]
        conversation.last_interaction = datetime.now()
        
        # Analyze content for conversation flow
        analysis = await self._analyze_conversation_content(content, conversation)
        
        return {
            'status': 'conversation_continued',
            'content_analysis': analysis,
            'conversation_duration': (datetime.now() - conversation.start_time).total_seconds(),
            'topic': conversation.topic
        }
    
    async def _end_conversation(self, conv_id: str) -> Dict[str, Any]:
        """End a conversation"""
        if conv_id in self.active_conversations:
            conversation = self.active_conversations.pop(conv_id)
            duration = (datetime.now() - conversation.start_time).total_seconds()
            
            return {
                'status': 'conversation_ended',
                'duration': duration,
                'topic': conversation.topic,
                'final_response': 'Thank you for the conversation!'
            }
        else:
            return {'error': 'Conversation not found'}
    
    async def _analyze_conversation_content(self, content: str, context: ConversationContext) -> Dict[str, Any]:
        """Analyze content within conversation context"""
        # Detect topic shifts
        topic_shift = await self._detect_topic_shift(content, context.topic)
        
        # Assess complexity
        complexity = self._calculate_text_complexity(content)
        
        # Detect mood changes
        mood = await self._detect_mood(content)
        
        return {
            'topic_shift': topic_shift,
            'complexity_level': complexity,
            'detected_mood': mood,
            'requires_clarification': complexity > context.complexity_level + 1
        }
    
    async def _analyze_semantics(self, signal: NeuralSignal) -> NeuralSignal:
        """Perform deep semantic analysis"""
        text = signal.data.get('text', '')
        depth = signal.data.get('depth', self.semantic_depth)
        
        semantic_result = await self._perform_semantic_analysis(text, depth)
        
        self.current_mode = LanguageMode.ANALYSIS
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.SEMANTIC_ANALYSIS_RESULT,
            data={
                'semantic_analysis': semantic_result,
                'analysis_depth': depth,
                'semantic_confidence': 0.87
            },
            priority=Priority.NORMAL
        )
    
    async def _perform_semantic_analysis(self, text: str, depth: int) -> Dict[str, Any]:
        """Perform semantic analysis at specified depth"""
        words = text.lower().split()
        
        # Level 1: Basic categorization
        categories = defaultdict(list)
        for word in words:
            for category, word_list in self.semantic_categories.items():
                if word in word_list:
                    categories[category].append(word)
        
        analysis = {
            'word_categories': dict(categories),
            'semantic_density': len(categories) / max(1, len(words))
        }
        
        # Level 2: Relationship analysis
        if depth >= 2:
            analysis['relationships'] = await self._analyze_word_relationships(words)
        
        # Level 3: Conceptual analysis
        if depth >= 3:
            analysis['concepts'] = await self._extract_concepts(text)
        
        return analysis
    
    async def _analyze_word_relationships(self, words: List[str]) -> Dict[str, Any]:
        """Analyze relationships between words"""
        relationships = []
        
        for i, word in enumerate(words):
            if i > 0:
                prev_word = words[i-1]
                relationship_type = self._classify_word_relationship(prev_word, word)
                if relationship_type:
                    relationships.append({
                        'word1': prev_word,
                        'word2': word,
                        'relationship': relationship_type
                    })
        
        return {'word_relationships': relationships}
    
    def _classify_word_relationship(self, word1: str, word2: str) -> Optional[str]:
        """Classify relationship between two words"""
        # Simple relationship classification
        if word1 in self.semantic_categories['action'] and word2 in self.semantic_categories['object']:
            return 'action_object'
        elif word1 in self.semantic_categories['quality'] and word2 in self.semantic_categories['object']:
            return 'quality_object'
        else:
            return None
    
    async def _extract_concepts(self, text: str) -> List[str]:
        """Extract high-level concepts from text"""
        concepts = []
        
        # Simple concept extraction based on key terms
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['system', 'computer', 'software']):
            concepts.append('technology')
        
        if any(word in text_lower for word in ['process', 'execute', 'run']):
            concepts.append('operation')
        
        if any(word in text_lower for word in ['data', 'information', 'analysis']):
            concepts.append('information_processing')
        
        return concepts
    
    async def _analyze_language_content(self, signal: NeuralSignal) -> NeuralSignal:
        """Analyze language content in any signal"""
        content = str(signal.data.get('content', ''))
        
        if len(content) > 10:  # Only analyze substantial content
            # Quick language analysis
            word_count = len(content.split())
            has_questions = '?' in content
            has_commands = any(word in content.lower() for word in ['please', 'make', 'do', 'create'])
            
            # Create language pattern
            if word_count > 5:
                pattern = LanguagePattern(
                    pattern_type='automatic_analysis',
                    content=content[:100],  # First 100 chars
                    confidence=0.7,
                    context=signal.signal_type.value,
                    timestamp=datetime.now()
                )
                self.language_patterns.append(pattern)
        
        # Return original signal (pass-through)
        return signal
    
    async def _detect_intent(self, text: str, patterns: Dict[str, List[str]]) -> str:
        """Detect user intent from text and patterns"""
        if 'question' in patterns:
            return 'question'
        elif 'command' in patterns:
            return 'command'
        elif any(word in text.lower() for word in ['help', 'assist', 'support']):
            return 'help_request'
        else:
            return 'statement'
    
    async def _integrate_context(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate text with provided context"""
        return {
            'context_relevance': 0.8,
            'context_type': context.get('type', 'general'),
            'context_integration_success': True
        }
    
    async def _detect_topic_shift(self, content: str, current_topic: str) -> bool:
        """Detect if content represents a topic shift"""
        # Simple topic shift detection
        return current_topic.lower() not in content.lower()
    
    async def _detect_mood(self, content: str) -> str:
        """Detect mood from content"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['happy', 'great', 'excellent', 'wonderful']):
            return 'positive'
        elif any(word in content_lower for word in ['sad', 'bad', 'terrible', 'awful']):
            return 'negative'
        elif any(word in content_lower for word in ['worried', 'concerned', 'anxious']):
            return 'concerned'
        else:
            return 'neutral'
    
    def _calculate_text_complexity(self, text: str) -> int:
        """Calculate text complexity score (1-5)"""
        words = text.split()
        sentences = len(re.split(r'[.!?]+', text))
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        
        complexity = 1
        if avg_word_length > 6:
            complexity += 1
        if len(words) / max(1, sentences) > 15:  # Long sentences
            complexity += 1
        if any(len(word) > 10 for word in words):  # Complex words
            complexity += 1
        
        return min(5, complexity)
    
    async def perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        current_time = datetime.now()
        
        # Clean old conversation contexts
        expired_conversations = [
            conv_id for conv_id, conv in self.active_conversations.items()
            if (current_time - conv.last_interaction).total_seconds() / 3600 > 2
        ]
        
        for conv_id in expired_conversations:
            del self.active_conversations[conv_id]
        
        # Clean old language patterns (keep last 100)
        if len(self.language_patterns) > 100:
            self.language_patterns = self.language_patterns[-100:]
        
        # Reset processing mode if idle
        if self.current_mode != LanguageMode.COMPREHENSION:
            self.current_mode = LanguageMode.COMPREHENSION
        
        if self.auditory_state != AuditoryState.QUIET:
            self.auditory_state = AuditoryState.QUIET
    
    def get_status(self) -> Dict[str, Any]:
        """Get temporal lobe status"""
        base_status = super().get_status()
        
        temporal_status = {
            'language_mode': self.current_mode.value,
            'auditory_state': self.auditory_state.value,
            'vocabulary_size': self.vocabulary_size,
            'comprehension_accuracy': self.comprehension_accuracy,
            'generation_fluency': self.generation_fluency,
            'words_processed': self.words_processed,
            'sentences_analyzed': self.sentences_analyzed,
            'conversations_managed': self.conversations_managed,
            'pattern_matches': self.pattern_matches,
            'active_conversations': len(self.active_conversations),
            'language_patterns_stored': len(self.language_patterns),
            'supported_languages': self.language_models
        }
        
        base_status.update(temporal_status)
        return base_status

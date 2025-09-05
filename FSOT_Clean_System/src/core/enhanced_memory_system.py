#!/usr/bin/env python3
"""
Enhanced Memory System for FSOT 2.0 Neuromorphic Brain
======================================================

Advanced brain-inspired memory architectures with multiple memory types:
- Working Memory: Active information processing
- Long-term Memory: Persistent knowledge storage  
- Episodic Memory: Experience and event storage
- Semantic Memory: Factual knowledge organization
- Procedural Memory: Skill and procedure storage
- Meta-Memory: Memory about memory processes

Author: GitHub Copilot
"""

import asyncio
import json
import sqlite3
import pickle
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memory in the enhanced system"""
    WORKING = "working"
    LONGTERM = "longterm"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    META = "meta"

@dataclass
class MemoryEntry:
    """Enhanced memory entry with metadata"""
    content: Any
    memory_type: MemoryType
    timestamp: datetime
    importance: float
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    associations: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    confidence: float = 1.0
    source: str = "unknown"
    
    def __post_init__(self):
        if self.associations is None:
            self.associations = []
        if self.tags is None:
            self.tags = []

class WorkingMemory:
    """Working memory for active information processing"""
    
    def __init__(self, capacity: int = 7):  # Miller's Rule: 7±2 items
        self.capacity = capacity
        self.items = {}
        self.attention_weights = {}
        self.decay_rate = 0.1
        self.lock = threading.Lock()
        
    def add_item(self, key: str, content: Any, importance: float = 0.5) -> bool:
        """Add item to working memory with capacity management"""
        with self.lock:
            # Remove least important item if at capacity
            if len(self.items) >= self.capacity:
                self._remove_least_important()
            
            entry = MemoryEntry(
                content=content,
                memory_type=MemoryType.WORKING,
                timestamp=datetime.now(),
                importance=importance,
                source="working_memory"
            )
            
            self.items[key] = entry
            self.attention_weights[key] = importance
            
            logger.debug(f"Added item to working memory: {key}")
            return True
    
    def get_item(self, key: str) -> Optional[Any]:
        """Retrieve item from working memory"""
        with self.lock:
            if key in self.items:
                entry = self.items[key]
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                
                # Strengthen attention weight
                self.attention_weights[key] = min(1.0, 
                    self.attention_weights[key] + 0.1)
                
                return entry.content
            return None
    
    def _remove_least_important(self):
        """Remove least important item from working memory"""
        if not self.items:
            return
            
        # Find item with lowest attention weight
        least_important = min(self.attention_weights.keys(), 
                            key=lambda k: self.attention_weights[k])
        
        del self.items[least_important]
        del self.attention_weights[least_important]
        
        logger.debug(f"Removed from working memory: {least_important}")
    
    def update_attention(self):
        """Update attention weights with decay"""
        with self.lock:
            for key in list(self.attention_weights.keys()):
                self.attention_weights[key] *= (1 - self.decay_rate)
                
                # Remove items with very low attention
                if self.attention_weights[key] < 0.01:
                    if key in self.items:
                        del self.items[key]
                    del self.attention_weights[key]
    
    def get_status(self) -> Dict[str, Any]:
        """Get working memory status"""
        with self.lock:
            return {
                "capacity": self.capacity,
                "current_items": len(self.items),
                "items": list(self.items.keys()),
                "attention_weights": dict(self.attention_weights)
            }

class LongTermMemory:
    """Long-term memory with consolidation and retrieval"""
    
    def __init__(self, db_path: str = "data/longterm_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize memory database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content BLOB,
                    memory_type TEXT,
                    timestamp TEXT,
                    importance REAL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    associations TEXT,
                    tags TEXT,
                    confidence REAL,
                    source TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
            ''')
    
    def store_memory(self, key: str, entry: MemoryEntry) -> bool:
        """Store memory in long-term storage"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Serialize complex content
                    content_blob = pickle.dumps(entry.content)
                    
                    conn.execute('''
                        INSERT OR REPLACE INTO memories 
                        (id, content, memory_type, timestamp, importance, 
                         access_count, last_accessed, associations, tags, 
                         confidence, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        key,
                        content_blob,
                        entry.memory_type.value,
                        entry.timestamp.isoformat(),
                        entry.importance,
                        entry.access_count,
                        entry.last_accessed.isoformat() if entry.last_accessed else None,
                        json.dumps(entry.associations),
                        json.dumps(entry.tags),
                        entry.confidence,
                        entry.source
                    ))
                    
                    logger.debug(f"Stored memory: {key}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error storing memory {key}: {e}")
            return False
    
    def retrieve_memory(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve memory from long-term storage"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('''
                        SELECT * FROM memories WHERE id = ?
                    ''', (key,))
                    
                    row = cursor.fetchone()
                    if row:
                        # Update access statistics
                        conn.execute('''
                            UPDATE memories 
                            SET access_count = access_count + 1,
                                last_accessed = ?
                            WHERE id = ?
                        ''', (datetime.now().isoformat(), key))
                        
                        # Reconstruct memory entry
                        return self._row_to_memory_entry(row)
                        
        except Exception as e:
            logger.error(f"Error retrieving memory {key}: {e}")
            
        return None
    
    def search_memories(self, 
                       memory_type: Optional[MemoryType] = None,
                       tags: Optional[List[str]] = None,
                       min_importance: float = 0.0,
                       limit: int = 10) -> List[Tuple[str, MemoryEntry]]:
        """Search memories by criteria"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    query = "SELECT * FROM memories WHERE importance >= ?"
                    params = [min_importance]
                    
                    if memory_type:
                        query += " AND memory_type = ?"
                        params.append(memory_type.value)
                    
                    query += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
                    params.append(limit)
                    
                    cursor = conn.execute(query, params)
                    results = []
                    
                    for row in cursor.fetchall():
                        memory_id = row[0]
                        entry = self._row_to_memory_entry(row)
                        
                        # Filter by tags if specified
                        if tags and entry.tags and not any(tag in entry.tags for tag in tags):
                            continue
                            
                        results.append((memory_id, entry))
                    
                    return results
                    
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def _row_to_memory_entry(self, row) -> MemoryEntry:
        """Convert database row to MemoryEntry"""
        content = pickle.loads(row[1])
        
        return MemoryEntry(
            content=content,
            memory_type=MemoryType(row[2]),
            timestamp=datetime.fromisoformat(row[3]),
            importance=row[4],
            access_count=row[5],
            last_accessed=datetime.fromisoformat(row[6]) if row[6] else None,
            associations=json.loads(row[7]) if row[7] else [],
            tags=json.loads(row[8]) if row[8] else [],
            confidence=row[9],
            source=row[10]
        )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Total memories
                    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                    
                    # By type
                    by_type = {}
                    for memory_type in MemoryType:
                        count = conn.execute(
                            "SELECT COUNT(*) FROM memories WHERE memory_type = ?",
                            (memory_type.value,)
                        ).fetchone()[0]
                        by_type[memory_type.value] = count
                    
                    # Most accessed
                    most_accessed = conn.execute('''
                        SELECT id, access_count FROM memories 
                        ORDER BY access_count DESC LIMIT 5
                    ''').fetchall()
                    
                    return {
                        "total_memories": total,
                        "by_type": by_type,
                        "most_accessed": most_accessed
                    }
                    
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}

class EpisodicMemory:
    """Episodic memory for experiences and events"""
    
    def __init__(self, db_path: str = "data/episodic_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize episodic memory database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    event_type TEXT,
                    context BLOB,
                    outcome BLOB,
                    timestamp TEXT,
                    duration REAL,
                    importance REAL,
                    emotional_valence REAL,
                    participants TEXT,
                    location TEXT,
                    success BOOLEAN,
                    lessons_learned TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_event_type ON episodes(event_type)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)
            ''')

    def record_episode(self, 
                      event_type: str,
                      context: Dict[str, Any],
                      outcome: Dict[str, Any],
                      importance: float = 0.5,
                      emotional_valence: float = 0.0,
                      participants: Optional[List[str]] = None,
                      location: str = "unknown",
                      success: bool = True,
                      lessons_learned: str = "") -> str:
        """Record an episodic memory"""
        
        episode_id = hashlib.md5(
            f"{event_type}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO episodes 
                        (id, event_type, context, outcome, timestamp, duration,
                         importance, emotional_valence, participants, location,
                         success, lessons_learned)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        episode_id,
                        event_type,
                        pickle.dumps(context),
                        pickle.dumps(outcome),
                        datetime.now().isoformat(),
                        0.0,  # Duration will be updated if needed
                        importance,
                        emotional_valence,
                        json.dumps(participants or []),
                        location,
                        success,
                        lessons_learned
                    ))
                    
                    logger.info(f"Recorded episode: {event_type}")
                    return episode_id
                    
        except Exception as e:
            logger.error(f"Error recording episode: {e}")
            return ""
    
    def retrieve_episodes(self, 
                         event_type: Optional[str] = None,
                         time_range: Optional[Tuple[datetime, datetime]] = None,
                         limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve episodes by criteria"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    query = "SELECT * FROM episodes"
                    params = []
                    conditions = []
                    
                    if event_type:
                        conditions.append("event_type = ?")
                        params.append(event_type)
                    
                    if time_range:
                        conditions.append("timestamp BETWEEN ? AND ?")
                        params.extend([t.isoformat() for t in time_range])
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                    
                    query += " ORDER BY timestamp DESC LIMIT ?"
                    params.append(limit)
                    
                    cursor = conn.execute(query, params)
                    episodes = []
                    
                    for row in cursor.fetchall():
                        episodes.append({
                            "id": row[0],
                            "event_type": row[1],
                            "context": pickle.loads(row[2]),
                            "outcome": pickle.loads(row[3]),
                            "timestamp": datetime.fromisoformat(row[4]),
                            "duration": row[5],
                            "importance": row[6],
                            "emotional_valence": row[7],
                            "participants": json.loads(row[8]),
                            "location": row[9],
                            "success": bool(row[10]),
                            "lessons_learned": row[11]
                        })
                    
                    return episodes
                    
        except Exception as e:
            logger.error(f"Error retrieving episodes: {e}")
            return []

class EnhancedMemorySystem:
    """Unified enhanced memory system coordinating all memory types"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory subsystems
        self.working_memory = WorkingMemory()
        self.longterm_memory = LongTermMemory(str(self.data_dir / "longterm_memory.db"))
        self.episodic_memory = EpisodicMemory(str(self.data_dir / "episodic_memory.db"))
        
        # Meta-memory tracking
        self.meta_memory = {}
        self.consolidation_threshold = 0.7
        
        # Start background processes
        self._start_background_tasks()
        
        logger.info("Enhanced Memory System initialized")
    
    def _start_background_tasks(self):
        """Start background memory management tasks"""
        
        def memory_maintenance():
            """Background memory maintenance"""
            while True:
                try:
                    # Update working memory attention
                    self.working_memory.update_attention()
                    
                    # Check for consolidation opportunities
                    self._check_consolidation()
                    
                    # Sleep for maintenance interval
                    threading.Event().wait(60)  # 1 minute intervals
                    
                except Exception as e:
                    logger.error(f"Memory maintenance error: {e}")
                    threading.Event().wait(60)
        
        # Start maintenance thread
        maintenance_thread = threading.Thread(target=memory_maintenance, daemon=True)
        maintenance_thread.start()
    
    def store(self, 
             key: str, 
             content: Any, 
             memory_type: MemoryType = MemoryType.WORKING,
             importance: float = 0.5,
             tags: Optional[List[str]] = None,
             source: str = "unknown") -> bool:
        """Store information in appropriate memory system"""
        
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(),
            importance=importance,
            tags=tags or [],
            source=source
        )
        
        if memory_type == MemoryType.WORKING:
            return self.working_memory.add_item(key, content, importance)
        else:
            return self.longterm_memory.store_memory(key, entry)
    
    def retrieve(self, key: str, prefer_recent: bool = True) -> Optional[Any]:
        """Retrieve information from memory system"""
        
        # Try working memory first
        content = self.working_memory.get_item(key)
        if content is not None:
            return content
        
        # Try long-term memory
        entry = self.longterm_memory.retrieve_memory(key)
        if entry:
            # Promote to working memory if important enough
            if entry.importance > 0.6:
                self.working_memory.add_item(key, entry.content, entry.importance)
            
            return entry.content
        
        return None
    
    def search(self, 
              query: str = "",
              memory_type: Optional[MemoryType] = None,
              tags: Optional[List[str]] = None,
              limit: int = 10) -> List[Tuple[str, Any]]:
        """Search across memory systems"""
        
        results = []
        
        # Search long-term memory
        memories = self.longterm_memory.search_memories(
            memory_type=memory_type,
            tags=tags,
            limit=limit
        )
        
        for memory_id, entry in memories:
            results.append((memory_id, entry.content))
        
        return results
    
    def record_experience(self, 
                         event_type: str,
                         context: Dict[str, Any],
                         outcome: Dict[str, Any],
                         success: bool = True,
                         importance: float = 0.5) -> str:
        """Record an experience in episodic memory"""
        
        episode_id = self.episodic_memory.record_episode(
            event_type=event_type,
            context=context,
            outcome=outcome,
            success=success,
            importance=importance
        )
        
        # Also store key outcomes in semantic memory if important
        if importance > self.consolidation_threshold:
            semantic_key = f"experience_{event_type}_{episode_id[:8]}"
            self.store(
                semantic_key,
                {
                    "event_type": event_type,
                    "outcome": outcome,
                    "success": success,
                    "lessons": context.get("lessons_learned", "")
                },
                memory_type=MemoryType.SEMANTIC,
                importance=importance,
                tags=["experience", event_type],
                source="episodic_consolidation"
            )
        
        return episode_id
    
    def _check_consolidation(self):
        """Check for memories that should be consolidated"""
        
        # Get working memory items that should be consolidated
        working_items = self.working_memory.get_status()
        
        for item_key in working_items["items"]:
            importance = working_items["attention_weights"].get(item_key, 0)
            
            if importance > self.consolidation_threshold:
                # Move to long-term memory
                content = self.working_memory.get_item(item_key)
                if content:
                    entry = MemoryEntry(
                        content=content,
                        memory_type=MemoryType.LONGTERM,
                        timestamp=datetime.now(),
                        importance=importance,
                        source="working_memory_consolidation"
                    )
                    
                    self.longterm_memory.store_memory(item_key, entry)
                    logger.debug(f"Consolidated memory: {item_key}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive memory system status"""
        
        return {
            "working_memory": self.working_memory.get_status(),
            "longterm_memory": self.longterm_memory.get_memory_stats(),
            "consolidation_threshold": self.consolidation_threshold,
            "total_episodes": len(self.episodic_memory.retrieve_episodes(limit=1000)),
            "data_directory": str(self.data_dir)
        }

# Global instance for easy access
_memory_system = None

def get_memory_system() -> EnhancedMemorySystem:
    """Get or create global memory system instance"""
    global _memory_system
    if _memory_system is None:
        _memory_system = EnhancedMemorySystem()
    return _memory_system

if __name__ == "__main__":
    # Test the enhanced memory system
    print("🧠 Testing Enhanced Memory System")
    print("=" * 40)
    
    memory = EnhancedMemorySystem()
    
    # Test working memory
    print("\n📝 Testing Working Memory...")
    memory.store("test_fact", "The sky is blue", MemoryType.WORKING, 0.8)
    retrieved = memory.retrieve("test_fact")
    print(f"   Stored and retrieved: {retrieved}")
    
    # Test long-term memory
    print("\n🏛️ Testing Long-term Memory...")
    memory.store("important_fact", "AI systems need memory", MemoryType.LONGTERM, 0.9, 
                tags=["AI", "memory", "important"])
    
    # Test episodic memory
    print("\n📚 Testing Episodic Memory...")
    episode_id = memory.record_experience(
        "learning_session",
        {"topic": "memory systems", "duration": 30},
        {"success": True, "knowledge_gained": "significant"},
        success=True,
        importance=0.8
    )
    print(f"   Recorded episode: {episode_id}")
    
    # Test search
    print("\n🔍 Testing Memory Search...")
    results = memory.search(tags=["AI"])
    print(f"   Found {len(results)} AI-related memories")
    
    # System status
    print("\n📊 Memory System Status:")
    status = memory.get_system_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Enhanced Memory System test complete!")

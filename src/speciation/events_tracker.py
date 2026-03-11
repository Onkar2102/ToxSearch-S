"""
events_tracker.py

Event audit trail tracking for speciation pipeline.
Tracks individual genome movements through the clustering and distribution process.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


class EventsTracker:
    """
    Tracks individual genome movements through the speciation pipeline.
    
    Provides audit trail for:
    - Clustering assignments (species_id assignment)
    - Capacity enforcement (archival events)
    - Species transitions (merges, extinctions)
    - Cluster 0 movements (outlier assignment, speciation from cluster 0)
    
    Events are logged with timestamps and details for post-hoc analysis.
    """
    
    def __init__(self, generation: int, logger=None):
        """
        Initialize events tracker for a generation.
        
        Args:
            generation: Current generation number
            logger: Optional logger instance
        """
        self.generation = generation
        self.events: List[Dict[str, Any]] = []
        self.logger = logger or get_logger("EventsTracker")
    
    def log(self, genome_id: str, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a genome event.
        
        Args:
            genome_id: Unique genome identifier
            event: Event type (e.g., "clustering_assigned", "capacity_archived", "species_merged")
            details: Optional event details (e.g., {"species_id": 1, "reason": "capacity"})
        """
        event_record = {
            "genome_id": genome_id,
            "event": event,
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.events.append(event_record)
        self.logger.debug(f"Genome {genome_id}: {event} (gen {self.generation})")
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save events tracker events to consolidated JSON file.
        
        Always saves to a single consolidated file (events_tracker.json) that contains
        all generations' events. This prevents file proliferation and makes analysis easier.
        
        Args:
            path: Optional path to save file. If None, uses default outputs_path / "events_tracker.json" (consolidated)
        """
        from utils import get_system_utils
        _, _, _, get_outputs_path, _, _, _ = get_system_utils()
        
        if path is None:
            outputs_path = get_outputs_path()
            path = str(outputs_path / "events_tracker.json")  # Always use consolidated file
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Always use consolidated format
            if path_obj.exists():
                # Load existing consolidated data
                with open(path_obj, 'r', encoding='utf-8') as f:
                    consolidated_data = json.load(f)
                
                # Ensure "generations" key exists
                if "generations" not in consolidated_data:
                    consolidated_data["generations"] = []
                
                # Check if this generation already exists (update it)
                gen_entry = None
                for gen in consolidated_data["generations"]:
                    if gen.get("generation") == self.generation:
                        gen_entry = gen
                        break
                
                if gen_entry:
                    # Update existing generation entry
                    gen_entry["total_events"] = len(self.events)
                    gen_entry["events"] = self.events
                else:
                    # Add new generation entry
                    consolidated_data["generations"].append({
                        "generation": self.generation,
                        "total_events": len(self.events),
                        "events": self.events
                    })
                
                # Update summary statistics
                total_events = sum(gen["total_events"] for gen in consolidated_data["generations"])
                consolidated_data["summary"] = {
                    "total_generations": len(consolidated_data["generations"]),
                    "total_events": total_events,
                    "last_updated": datetime.now().isoformat()
                }
                
                # Save consolidated data
                with open(path_obj, 'w', encoding='utf-8') as f:
                    json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Consolidated events tracker: added {len(self.events)} events for generation {self.generation} to {path}")
            else:
                # Create new consolidated file
                consolidated_data = {
                    "generations": [{
                        "generation": self.generation,
                        "total_events": len(self.events),
                        "events": self.events
                    }],
                    "summary": {
                        "total_generations": 1,
                        "total_events": len(self.events),
                        "last_updated": datetime.now().isoformat()
                    }
                }
                with open(path_obj, 'w', encoding='utf-8') as f:
                    json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Created consolidated events tracker with {len(self.events)} events for generation {self.generation} at {path}")
        except Exception as e:
            self.logger.error(f"Failed to save events tracker to {path}: {e}", exc_info=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of tracked events.
        
        Returns:
            Dictionary with event counts and statistics
        """
        event_counts = {}
        for event in self.events:
            event_type = event["event"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "generation": self.generation,
            "total_events": len(self.events),
            "unique_genomes": len(set(e["genome_id"] for e in self.events)),
            "event_counts": event_counts
        }

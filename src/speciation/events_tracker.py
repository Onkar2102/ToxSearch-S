

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


class EventsTracker:
    """Tracks individual genome movements through the speciation pipeline. Provides audit trail for: - Clustering assignments (species_id assignment) - Capacity enforcement (archival events) - Species transitions (merges, extinctions) - Cluster 0 movements (outlier assignment, speciation from cluster 0) Events are logged with timestamps and details for post-hoc analysis."""
    
    def __init__(self, generation: int, logger=None):
        
        self.generation = generation
        self.events: List[Dict[str, Any]] = []
        self.logger = logger or get_logger("EventsTracker")
    
    def log(self, genome_id: str, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        
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
        
        from utils import get_system_utils
        _, _, _, get_outputs_path, _, _, _ = get_system_utils()
        
        if path is None:
            outputs_path = get_outputs_path()
            path = str(outputs_path / "events_tracker.json")
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if path_obj.exists():
                with open(path_obj, 'r', encoding='utf-8') as f:
                    consolidated_data = json.load(f)
                
                if "generations" not in consolidated_data:
                    consolidated_data["generations"] = []
                
                gen_entry = None
                for gen in consolidated_data["generations"]:
                    if gen.get("generation") == self.generation:
                        gen_entry = gen
                        break
                
                if gen_entry:
                    gen_entry["total_events"] = len(self.events)
                    gen_entry["events"] = self.events
                else:
                    consolidated_data["generations"].append({
                        "generation": self.generation,
                        "total_events": len(self.events),
                        "events": self.events
                    })
                
                total_events = sum(gen["total_events"] for gen in consolidated_data["generations"])
                consolidated_data["summary"] = {
                    "total_generations": len(consolidated_data["generations"]),
                    "total_events": total_events,
                    "last_updated": datetime.now().isoformat()
                }
                
                with open(path_obj, 'w', encoding='utf-8') as f:
                    json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Consolidated events tracker: added {len(self.events)} events for generation {self.generation} to {path}")
            else:
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

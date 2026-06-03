
from dataclasses import dataclass
from typing import Dict, List, Optional

_ACTIVE_EVALUATOR: Optional[str] = None
_ACTIVE_NORTH_STAR: Optional[str] = None

GOOGLE_NORTH_STAR_CHOICES = [
    "toxicity",
    "severe_toxicity",
    "identity_attack",
    "insult",
    "profanity",
    "threat",
    "sexually_explicit",
    "flirtation",
]

GOOGLE_PHENOTYPE_SCORE_ORDER = [
    "toxicity",
    "threat",
    "profanity",
    "sexually_explicit",
    "identity_attack",
    "flirtation",
    "insult",
    "severe_toxicity",
]

OPENAI_NORTH_STAR_CHOICES = [
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "violence",
    "violence/graphic",
    "sexual",
    "sexual/minors",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
    "illicit",
    "illicit/violent",
]

OPENAI_PHENOTYPE_SCORE_ORDER = list(OPENAI_NORTH_STAR_CHOICES)

OPENAI_METRIC_ALIASES: Dict[str, str] = {
    "toxicity": "harassment",
    "severe_toxicity": "harassment/threatening",
    "threat": "violence",
    "identity_attack": "hate",
    "insult": "harassment",
    "profanity": "harassment",
    "sexually_explicit": "sexual",
    "flirtation": "sexual",
}


@dataclass(frozen=True)
class EvaluatorProfile:
    name: str
    backend_key: str
    default_north_star: str
    north_star_choices: tuple
    phenotype_score_order: tuple
    metric_aliases: tuple = ()

    def resolve_metric_alias(self, metric: str) -> str:
        aliases = dict(self.metric_aliases) if self.metric_aliases else {}
        return aliases.get(metric, metric)


GOOGLE_PROFILE = EvaluatorProfile(
    name="google",
    backend_key="google",
    default_north_star="toxicity",
    north_star_choices=tuple(GOOGLE_NORTH_STAR_CHOICES),
    phenotype_score_order=tuple(GOOGLE_PHENOTYPE_SCORE_ORDER),
)

OPENAI_PROFILE = EvaluatorProfile(
    name="openai",
    backend_key="openai",
    default_north_star="violence",
    north_star_choices=tuple(OPENAI_NORTH_STAR_CHOICES),
    phenotype_score_order=tuple(OPENAI_PHENOTYPE_SCORE_ORDER),
    metric_aliases=tuple(OPENAI_METRIC_ALIASES.items()),
)

_PROFILES: Dict[str, EvaluatorProfile] = {
    "google": GOOGLE_PROFILE,
    "perspective": GOOGLE_PROFILE,
    "openai": OPENAI_PROFILE,
    "omni": OPENAI_PROFILE,
}


def resolve_evaluator(name: Optional[str] = None) -> EvaluatorProfile:
    key = (name or "google").strip().lower()
    profile = _PROFILES.get(key)
    if profile is None:
        valid = sorted({p.name for p in _PROFILES.values()})
        raise ValueError(f"Unknown evaluator {name!r}; choose one of: {valid}")
    return profile


def set_active_evaluator(name: str) -> EvaluatorProfile:
    global _ACTIVE_EVALUATOR
    profile = resolve_evaluator(name)
    _ACTIVE_EVALUATOR = profile.name
    return profile


def get_active_evaluator() -> EvaluatorProfile:
    return resolve_evaluator(_ACTIVE_EVALUATOR or "google")


def set_active_north_star(metric: str) -> str:
    global _ACTIVE_NORTH_STAR
    _ACTIVE_NORTH_STAR = metric
    return metric


def get_active_north_star(fallback: Optional[str] = None) -> str:
    if _ACTIVE_NORTH_STAR:
        return _ACTIVE_NORTH_STAR
    if fallback:
        return fallback
    return get_active_evaluator().default_north_star


def validate_north_star(profile: EvaluatorProfile, metric: str) -> str:
    if metric not in profile.north_star_choices:
        choices = ", ".join(profile.north_star_choices)
        raise ValueError(
            f"Invalid --north-star-metric {metric!r} for evaluator {profile.name}; "
            f"valid choices: {choices}"
        )
    return metric


def moderation_methods_to_evaluator(methods: Optional[List[str]]) -> str:
    if not methods:
        return "google"
    lowered = [str(m).strip().lower() for m in methods]
    if "all" in lowered:
        return "google"
    if any(m in ("openai", "omni") for m in lowered):
        return "openai"
    if any(m in ("google", "perspective") for m in lowered):
        return "google"
    return "google"


def moderation_methods_to_backend_list(profile: EvaluatorProfile) -> List[str]:
    return [profile.backend_key]


__all__ = [
    "EvaluatorProfile",
    "GOOGLE_PROFILE",
    "OPENAI_PROFILE",
    "resolve_evaluator",
    "set_active_evaluator",
    "get_active_evaluator",
    "set_active_north_star",
    "get_active_north_star",
    "validate_north_star",
    "moderation_methods_to_evaluator",
    "moderation_methods_to_backend_list",
    "GOOGLE_NORTH_STAR_CHOICES",
    "OPENAI_NORTH_STAR_CHOICES",
]

"""
Claude integration for ORTHON narrative generation.

Transforms computed metrics into human insight.
"""

import json
from typing import Optional, List, Dict, Any

# Try to import anthropic, gracefully handle if not installed
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None


def get_client():
    """Get Anthropic client if available."""
    if not ANTHROPIC_AVAILABLE:
        return None
    try:
        return anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    except Exception:
        return None


def generate_analysis(results: dict, domain_hint: str = None) -> str:
    """
    Generate narrative analysis of computed results.

    Args:
        results: Dict with typology, groups, dynamics, mechanics
        domain_hint: Optional domain context ("hydraulic", "cardio", etc.)

    Returns:
        2-3 paragraph analysis
    """
    client = get_client()

    if client is None:
        return generate_fallback_analysis(results, domain_hint)

    system = """You are an expert signal analyst explaining findings from ORTHON,
a domain-agnostic signal analysis framework. Write like a knowledgeable colleague
explaining results — specific, direct, insightful. Reference actual signal names
and values. No hedging or filler."""

    prompt = f"""
Analyze these signal processing results:

DATASET: {results.get('metadata', {}).get('name', 'Unknown')}
- {results.get('metadata', {}).get('n_signals', 0)} signals
- {results.get('metadata', {}).get('n_samples', 0)} samples
{f"- Domain: {domain_hint}" if domain_hint else ""}

SIGNAL TYPOLOGY (behavioral characteristics):
{format_typology_summary(results.get('typology', []))}

GROUPS DETECTED ({results.get('groups', {}).get('n_clusters', 0)} clusters):
{format_groups_summary(results.get('groups', {}))}

SYSTEM DYNAMICS:
- Mean coherence: {results.get('dynamics', {}).get('mean_coherence', 0):.2f}
- Coherence range: {results.get('dynamics', {}).get('coherence_min', 0):.2f} - {results.get('dynamics', {}).get('coherence_max', 0):.2f}
- Transitions detected: {len(results.get('dynamics', {}).get('transitions', []))}
{format_transitions(results.get('dynamics', {}).get('transitions', []))}

CAUSAL STRUCTURE:
- Primary drivers: {', '.join(results.get('mechanics', {}).get('drivers', [])) or 'None identified'}
- Primary followers: {', '.join(results.get('mechanics', {}).get('followers', [])) or 'None identified'}
- Causal density: {results.get('mechanics', {}).get('causal_density', 0):.2f}
{format_top_causal_links(results.get('mechanics', {}))}

Write 2-3 paragraphs explaining:
1. What types of signals are present and how they naturally group
2. Any notable events, transitions, or stability patterns
3. The causal structure — what drives what

Be specific. Use signal names. Mention actual values where meaningful.
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return generate_fallback_analysis(results, domain_hint)


def generate_fallback_analysis(results: dict, domain_hint: str = None) -> str:
    """
    Generate analysis without Claude API (fallback).

    Creates a structured summary from the computed metrics.
    """
    metadata = results.get('metadata', {})
    typology = results.get('typology', [])
    groups = results.get('groups', {})
    dynamics = results.get('dynamics', {})
    mechanics = results.get('mechanics', {})

    name = metadata.get('name', 'your dataset')
    n_signals = metadata.get('n_signals', len(typology))
    n_samples = metadata.get('n_samples', 0)

    # Build narrative paragraphs
    paragraphs = []

    # Paragraph 1: Typology and groups
    n_clusters = groups.get('n_clusters', 0)
    if n_clusters > 1:
        para1 = f"Your {name} contains {n_signals} signals that naturally organize into {n_clusters} distinct behavioral groups. "
    else:
        para1 = f"Your {name} contains {n_signals} signals. "

    # Describe dominant traits
    trait_counts = {}
    for sig in typology:
        trait = sig.get('dominant_trait', 'unknown')
        trait_counts[trait] = trait_counts.get(trait, 0) + 1

    if trait_counts:
        top_traits = sorted(trait_counts.items(), key=lambda x: -x[1])[:3]
        trait_desc = ", ".join([f"{count} {trait}" for trait, count in top_traits])
        para1 += f"The signals show diverse characteristics: {trait_desc}."

    paragraphs.append(para1)

    # Paragraph 2: Dynamics
    mean_coh = dynamics.get('mean_coherence', 0.5)
    transitions = dynamics.get('transitions', [])

    if mean_coh > 0.7:
        para2 = "The system shows strong coherence — signals move together. "
    elif mean_coh < 0.4:
        para2 = "The system shows weak coherence — signals behave independently. "
    else:
        para2 = "The system shows moderate coherence with some coordination between signals. "

    if transitions:
        t = transitions[0]
        para2 += f"A significant transition occurred around sample {t.get('time', 'unknown')}, "
        para2 += f"where coherence shifted from {t.get('from_coherence', 0):.2f} to {t.get('to_coherence', 0):.2f}."
    else:
        para2 += "No major regime transitions were detected."

    paragraphs.append(para2)

    # Paragraph 3: Causality
    drivers = mechanics.get('drivers', [])
    followers = mechanics.get('followers', [])

    if drivers:
        para3 = f"Causal analysis identifies {drivers[0]} as a primary driver in the system"
        if len(drivers) > 1:
            para3 += f", along with {', '.join(drivers[1:])}. "
        else:
            para3 += ". "

        n_caused = count_caused(mechanics, drivers[0])
        if n_caused > 0:
            para3 += f"It Granger-causes {n_caused} other signals. "

        if followers:
            para3 += f"The main followers are {', '.join(followers[:3])}."
    else:
        para3 = "No strong causal drivers were identified — the signals may be responding to external factors rather than each other."

    paragraphs.append(para3)

    return "\n\n".join(paragraphs)


def chat_about_data(
    question: str,
    results: dict,
    history: list = None
) -> str:
    """
    Answer follow-up questions about the analyzed data.

    Args:
        question: User's question
        results: Computed analysis results
        history: Previous conversation turns

    Returns:
        Claude's response
    """
    client = get_client()

    if client is None:
        return generate_fallback_chat(question, results)

    system = f"""You are helping a user understand their signal analysis from ORTHON.

You have access to these computed results:

METADATA:
{json.dumps(results.get('metadata', {}), indent=2)}

TYPOLOGY (per-signal characteristics):
{json.dumps(results.get('typology', [])[:10], indent=2)}  # Limit for token count

GROUPS:
{json.dumps(results.get('groups', {}), indent=2)}

DYNAMICS:
{json.dumps(results.get('dynamics', {}), indent=2)}

MECHANICS (causality):
{json.dumps(results.get('mechanics', {}), indent=2)}

Answer questions about THIS SPECIFIC DATA. Be precise — reference signal names,
actual values, specific time points. Explain the math if asked, but lead with
practical insight. If you don't have enough information to answer, say so."""

    messages = history or []
    messages.append({"role": "user", "content": question})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system,
            messages=messages
        )
        return response.content[0].text
    except Exception:
        return generate_fallback_chat(question, results)


def generate_fallback_chat(question: str, results: dict) -> str:
    """Fallback chat response without Claude API."""

    question_lower = question.lower()

    # Simple keyword-based responses
    if 'driver' in question_lower or 'cause' in question_lower or 'granger' in question_lower:
        drivers = results.get('mechanics', {}).get('drivers', [])
        if drivers:
            return f"Based on Granger causality analysis, {drivers[0]} appears to be the primary driver in this system. It shows predictive power over other signals, meaning its past values help forecast their future behavior."
        return "No strong causal drivers were identified in this dataset. The signals may be responding to external factors rather than each other."

    if 'group' in question_lower or 'cluster' in question_lower:
        groups = results.get('groups', {})
        n = groups.get('n_clusters', 0)
        if n > 1:
            return f"The signals cluster into {n} distinct groups based on their behavioral characteristics. Signals in the same group share similar typology profiles — they tend to have similar memory, volatility, and periodicity patterns."
        return "The signals don't form distinct clusters — they share similar behavioral characteristics."

    if 'transition' in question_lower or 'change' in question_lower or 'coherence' in question_lower:
        transitions = results.get('dynamics', {}).get('transitions', [])
        if transitions:
            t = transitions[0]
            return f"A significant transition was detected around sample {t.get('time', 'unknown')}. The system coherence dropped from {t.get('from_coherence', 0):.2f} to {t.get('to_coherence', 0):.2f}, indicating the signals became less synchronized. This could indicate a regime change, fault condition, or external disturbance."
        return "No major transitions were detected. The system maintained relatively stable behavior throughout the observation period."

    if 'memory' in question_lower or 'hurst' in question_lower:
        return "Memory (measured by the Hurst exponent) indicates how much a signal 'remembers' its past. A high memory score (>0.7) means the signal is persistent — trends tend to continue. A low score (<0.3) means the signal is anti-persistent or mean-reverting."

    if 'volatility' in question_lower or 'garch' in question_lower:
        return "Volatility clustering (measured by GARCH) indicates whether periods of high variability tend to cluster together. High volatility scores suggest the signal shows bursts of activity followed by calm periods — common in systems with feedback loops or shock propagation."

    # Default response
    return "I can help you understand the analysis results. Try asking about:\n- Which signals drive the system (causality)\n- How the signals group together (clustering)\n- When transitions occurred (dynamics)\n- What the typology scores mean (memory, volatility, etc.)"


def generate_insight_cards(results: dict) -> List[Dict[str, Any]]:
    """
    Generate structured insight cards for UI display.

    Returns:
        List of insight cards with type, headline, detail, chart_type
    """
    cards = []

    # Groups card
    n_groups = results.get('groups', {}).get('n_clusters', 0)
    if n_groups > 1:
        cards.append({
            'type': 'groups',
            'icon': '📊',
            'headline': f'{n_groups} Signal Groups',
            'detail': describe_groups_brief(results.get('groups', {})),
            'chart_type': 'scatter',
            'link_to': 'Groups',
        })

    # Transition card
    transitions = results.get('dynamics', {}).get('transitions', [])
    if transitions:
        t = transitions[0]
        cards.append({
            'type': 'transition',
            'icon': '⚡',
            'headline': f"Transition at t={t.get('time', '?')}",
            'detail': f"Coherence: {t.get('from_coherence', 0):.2f} → {t.get('to_coherence', 0):.2f}",
            'chart_type': 'line',
            'link_to': 'Dynamics',
        })

    # Driver card
    drivers = results.get('mechanics', {}).get('drivers', [])
    if drivers:
        n_caused = count_caused(results.get('mechanics', {}), drivers[0])
        cards.append({
            'type': 'causality',
            'icon': '🎯',
            'headline': f"{drivers[0]} Drives System",
            'detail': f"Granger-causes {n_caused} signals" if n_caused else "Primary causal influence",
            'chart_type': 'network',
            'link_to': 'Mechanics',
        })

    # Anomaly card
    anomalies = find_anomalous_signals(results.get('typology', []))
    if anomalies:
        cards.append({
            'type': 'anomaly',
            'icon': '🚨',
            'headline': f"{anomalies[0]['signal']} is Unusual",
            'detail': anomalies[0]['reason'],
            'chart_type': 'radar',
            'link_to': 'Typology',
        })

    return cards[:3]  # Max 3 cards on discovery page


# -----------------------------------------------------------------------------
# Formatting Helpers
# -----------------------------------------------------------------------------

def format_typology_summary(typology: list) -> str:
    """Format typology for prompt."""
    if not typology:
        return "- No typology data available"

    lines = []
    for sig in typology[:10]:  # Limit for token count
        traits = []
        if sig.get('memory', 0.5) > 0.7:
            traits.append('persistent')
        elif sig.get('memory', 0.5) < 0.3:
            traits.append('forgetful')
        if sig.get('volatility', 0.5) > 0.7:
            traits.append('volatile')
        if sig.get('frequency', 0.5) > 0.7:
            traits.append('periodic')

        trait_str = ', '.join(traits) if traits else 'neutral'
        sig_id = sig.get('signal_id', 'unknown')
        lines.append(f"- {sig_id}: {trait_str}")

    if len(typology) > 10:
        lines.append(f"- ... and {len(typology) - 10} more signals")

    return '\n'.join(lines)


def format_groups_summary(groups: dict) -> str:
    """Format groups for prompt."""
    clusters = groups.get('clusters', [])
    if not clusters:
        return "- No distinct groups detected"

    lines = []
    for i, group in enumerate(clusters):
        members = group.get('members', [])
        member_str = ', '.join(members[:5])
        if len(members) > 5:
            member_str += f" +{len(members)-5} more"
        trait = group.get('dominant_trait', 'mixed')
        lines.append(f"- Group {i+1}: [{member_str}] — {trait}")

    return '\n'.join(lines)


def format_transitions(transitions: list) -> str:
    """Format transitions for prompt."""
    if not transitions:
        return "- No significant transitions detected"

    lines = []
    for t in transitions[:3]:
        lines.append(f"- t={t.get('time', '?')}: coherence {t.get('from_coherence', 0):.2f} → {t.get('to_coherence', 0):.2f}")

    return '\n'.join(lines)


def format_top_causal_links(mechanics: dict) -> str:
    """Format top causal links for prompt."""
    links = mechanics.get('top_links', [])[:5]
    if not links:
        return "- No strong causal links detected"

    lines = []
    for link in links:
        source = link.get('source', '?')
        target = link.get('target', '?')
        granger = link.get('granger_f', 0)
        te = link.get('transfer_entropy', 0)
        lines.append(f"- {source} → {target}: F={granger:.1f}, TE={te:.3f} bits")

    return '\n'.join(lines)


def describe_groups_brief(groups: dict) -> str:
    """One-line group description for card."""
    clusters = groups.get('clusters', [])
    if not clusters:
        return "No distinct clusters"
    cluster_sizes = [len(c.get('members', [])) for c in clusters]
    return f"Clusters of {', '.join(map(str, cluster_sizes))} signals"


def count_caused(mechanics: dict, driver: str) -> int:
    """Count how many signals a driver causes."""
    return sum(1 for link in mechanics.get('top_links', [])
               if link.get('source') == driver)


def find_anomalous_signals(typology: list) -> List[Dict]:
    """Find signals with unusual characteristics."""
    anomalies = []
    for sig in typology:
        distinctiveness = sig.get('distinctiveness', 0)
        if distinctiveness > 0.4:
            anomalies.append({
                'signal': sig.get('signal_id', 'unknown'),
                'reason': f"Extreme {sig.get('dominant_trait', 'characteristics')} (distinctiveness: {distinctiveness:.2f})",
                'distinctiveness': distinctiveness,
            })

    return sorted(anomalies, key=lambda x: -x.get('distinctiveness', 0))


def get_chat_suggestions(results: dict) -> List[str]:
    """Generate relevant chat suggestions based on results."""
    suggestions = []

    # Check for transitions
    transitions = results.get('dynamics', {}).get('transitions', [])
    if transitions:
        t = transitions[0]
        suggestions.append(f"Why did coherence drop at t={t.get('time', '?')}?")

    # Check for drivers
    drivers = results.get('mechanics', {}).get('drivers', [])
    if drivers:
        suggestions.append(f"What makes {drivers[0]} the primary driver?")

    # Check for groups
    n_groups = results.get('groups', {}).get('n_clusters', 0)
    if n_groups > 1:
        suggestions.append("What distinguishes the signal groups?")

    # Default suggestions
    suggestions.extend([
        "Which signals should I monitor for early warning?",
        "What do the typology scores mean?",
    ])

    return suggestions[:4]

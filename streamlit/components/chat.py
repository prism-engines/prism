"""
Chat interface component for ORTHON Discovery page.

Allows follow-up questions about analyzed data.
"""

import streamlit as st
from typing import Dict, List, Any, Callable

# Import Claude analyst
try:
    from utils.claude_analyst import chat_about_data, get_chat_suggestions
except ImportError:
    chat_about_data = None
    get_chat_suggestions = None


def render_chat_interface(results: dict, key_prefix: str = "chat"):
    """
    Render the chat interface for data questions.

    Args:
        results: Analysis results to chat about
        key_prefix: Prefix for session state keys
    """
    st.markdown("### 💬 Ask about this data")

    # Initialize chat history
    history_key = f"{key_prefix}_history"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    # Display chat history
    for i, msg in enumerate(st.session_state[history_key]):
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    # Show suggestions if no chat yet
    if not st.session_state[history_key]:
        render_suggestions(results, key_prefix)

    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        handle_chat_input(prompt, results, key_prefix)


def render_suggestions(results: dict, key_prefix: str):
    """Render clickable suggestion buttons."""
    st.caption("Try asking:")

    if get_chat_suggestions:
        suggestions = get_chat_suggestions(results)
    else:
        suggestions = [
            "Which signals should I monitor?",
            "What do the typology scores mean?",
            "Are there any concerning patterns?",
        ]

    # Render as buttons in columns
    cols = st.columns(min(len(suggestions), 2))
    for i, suggestion in enumerate(suggestions[:4]):
        col = cols[i % 2]
        with col:
            if st.button(
                suggestion,
                key=f"{key_prefix}_suggest_{i}",
                use_container_width=True
            ):
                handle_chat_input(suggestion, results, key_prefix)


def handle_chat_input(prompt: str, results: dict, key_prefix: str):
    """Handle chat input and get response."""
    history_key = f"{key_prefix}_history"

    # Add user message
    st.session_state[history_key].append({
        'role': 'user',
        'content': prompt
    })

    # Get response
    if chat_about_data:
        with st.spinner("Thinking..."):
            response = chat_about_data(
                prompt,
                results,
                st.session_state[history_key][:-1]
            )
    else:
        response = "Chat functionality requires the Anthropic API. Please set your ANTHROPIC_API_KEY environment variable."

    # Add assistant message
    st.session_state[history_key].append({
        'role': 'assistant',
        'content': response
    })

    st.rerun()


def clear_chat_history(key_prefix: str = "chat"):
    """Clear chat history."""
    history_key = f"{key_prefix}_history"
    if history_key in st.session_state:
        st.session_state[history_key] = []


def render_chat_history_compact(key_prefix: str = "chat"):
    """Render compact chat history for sidebar or footer."""
    history_key = f"{key_prefix}_history"
    history = st.session_state.get(history_key, [])

    if not history:
        return

    with st.expander(f"Chat History ({len(history)//2} exchanges)", expanded=False):
        for msg in history:
            role_icon = "🧑" if msg['role'] == 'user' else "🤖"
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            st.caption(f"{role_icon} {content}")


def get_chat_context(results: dict) -> str:
    """
    Generate context string for chat.
    Used when providing context to Claude.
    """
    metadata = results.get('metadata', {})
    typology = results.get('typology', [])
    groups = results.get('groups', {})
    dynamics = results.get('dynamics', {})
    mechanics = results.get('mechanics', {})

    context_parts = []

    # Dataset info
    context_parts.append(f"Dataset: {metadata.get('name', 'Unknown')}")
    context_parts.append(f"Signals: {metadata.get('n_signals', len(typology))}")
    context_parts.append(f"Samples: {metadata.get('n_samples', 0)}")

    # Key findings
    n_clusters = groups.get('n_clusters', 0)
    if n_clusters > 1:
        context_parts.append(f"Groups: {n_clusters} clusters detected")

    mean_coh = dynamics.get('mean_coherence', 0)
    context_parts.append(f"Mean coherence: {mean_coh:.2f}")

    drivers = mechanics.get('drivers', [])
    if drivers:
        context_parts.append(f"Primary driver: {drivers[0]}")

    return "\n".join(context_parts)

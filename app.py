"""Portfolio Commentary Generator - Streamlit Entry Point."""

import streamlit as st

st.set_page_config(
    page_title="Portfolio Commentary Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Render the main application."""
    st.title("📊 Portfolio Commentary Generator")
    st.markdown("---")

    st.info(
        "**Status: Phase 0 - Scaffolding Complete**\n\n"
        "This application is under development. "
        "See PHASE_STATUS.md for current progress."
    )

    st.markdown("### Coming Soon")
    st.markdown(
        """
        - 📁 Excel upload and parsing
        - 🔍 Automatic top/bottom 5 identification
        - ✍️ AI-generated draft commentary
        - ✏️ Review and editing interface
        - 📄 Word document export
        """
    )

    # Verify secrets are loadable (won't fail if missing, just shows warning)
    with st.sidebar:
        st.markdown("### Configuration Status")
        try:
            if st.secrets.get("OPENAI_API_KEY"):
                st.success("✓ API key configured")
            else:
                st.warning("⚠ API key not configured")
        except Exception:
            st.warning("⚠ Secrets not configured")

        st.markdown("---")
        st.markdown("*v0.1.0 - Scaffolding*")


if __name__ == "__main__":
    main()

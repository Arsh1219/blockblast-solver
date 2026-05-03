"""Block Blast Solver - Streamlit UI.

Upload a screenshot of an 8x8 Block Blast board with three pieces below it.
The app auto-detects the grid state and pieces, then shows a step-by-step
solution that maximises line clears.
"""

import io
import streamlit as st
from PIL import Image

import solver


st.set_page_config(page_title="Block Blast Solver", page_icon="🧱", layout="wide")

st.title("🧱 Block Blast Solver")
st.caption("Upload a screenshot. Get a step-by-step solution.")

with st.sidebar:
    st.header("Options")
    grid_size = st.selectbox("Grid size", options=[8, 9, 10], index=0,
                             help="Most Block Blast boards are 8x8.")
    show_debug = st.checkbox("Show detection details", value=False)

uploaded = st.file_uploader(
    "Upload a screenshot (.jpg / .png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("📱 Upload a screenshot to begin. The full board and all three pieces below it should be visible.")
    st.stop()


# Load image
try:
    pil = Image.open(uploaded).convert("RGB")
except Exception as e:
    st.error(f"Couldn't read that image: {e}")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Your screenshot")
    st.image(pil, use_container_width=True)

# Run analysis
with st.spinner("Detecting board and solving..."):
    try:
        analysis = solver.analyze_image(pil, grid_size=grid_size)
    except Exception as e:
        st.error(f"Couldn't parse the screenshot: {e}")
        st.info("Tips: make sure the full grid and all three pieces are in frame, "
                "and that the image isn't cropped or zoomed.")
        st.stop()

with col2:
    st.subheader("Solution")
    if analysis.solution is None:
        st.error("No valid placement found — these three pieces don't fit on this board.")
    else:
        sol = analysis.solution
        clears = sol.line_clears
        emoji = "🎯" if clears >= 2 else ("✨" if clears == 1 else "✅")
        st.success(f"{emoji} Found a solution with **{clears} line clear{'s' if clears != 1 else ''}**.")

        st.markdown("**Placement order (rows and columns are 0-indexed from top-left):**")
        for i, m in enumerate(sol.moves, start=1):
            st.markdown(f"{i}. Place **{m.piece.name}** at **row {m.row}, col {m.col}**")

        # Render preview
        preview = solver.render_solution(analysis.board, sol)
        st.image(preview, caption="Step-by-step solution", use_container_width=True)

        # Download button
        buf = io.BytesIO()
        preview.save(buf, format="PNG")
        st.download_button(
            "Download solution image",
            data=buf.getvalue(),
            file_name="blockblast_solution.png",
            mime="image/png",
        )

if show_debug:
    st.divider()
    st.subheader("Detection details")
    gb = analysis.bounds
    st.write(f"Grid bounds: top={gb.top}, bottom={gb.bottom}, "
             f"left={gb.left}, right={gb.right}, "
             f"cell ≈ {gb.cell_w:.1f} × {gb.cell_h:.1f} px")
    st.text("Detected board:")
    st.code("\n".join(" ".join("X" if c else "." for c in row) for row in analysis.board))
    st.text("Detected pieces:")
    for p in analysis.pieces:
        st.write(f"**{p.name}** — {p.height} rows × {p.width} cols, cells: {p.cells}")

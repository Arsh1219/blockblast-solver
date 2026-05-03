# 🧱 Block Blast Solver

Upload a screenshot of a Block Blast board (8×8 grid + 3 pieces below) and get a step-by-step solution that maximises line clears.

## Run locally

```bash
./deploy.sh local
```

(or `pip install -r requirements.txt && streamlit run app.py`)

The app opens at `http://localhost:8501`.

## Deploy

A `deploy.sh` helper covers the common targets:

```bash
./deploy.sh local        # local dev
./deploy.sh docker       # build + run the included Dockerfile
./deploy.sh streamlit    # walkthrough for Streamlit Community Cloud
./deploy.sh hf           # walkthrough for Hugging Face Spaces
```

### Streamlit Community Cloud (free public URL)

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **New app**.
3. Pick this repo, set the main file to `app.py`, and click **Deploy**.
4. You get a URL like `https://<you>-blockblast-solver.streamlit.app`. Pushes
   to the selected branch redeploy automatically.

### Hugging Face Spaces (also free)

1. Create a new Space, SDK = **Streamlit**, hardware = CPU basic.
2. Upload `app.py`, `solver.py`, `requirements.txt`, `README.md` (or push the
   repo to the Space's git remote).
3. Public URL is provided once the build finishes.

### Docker / any container host (Fly.io, Railway, Render, …)

The included `Dockerfile` exposes port 8501 and respects `$PORT`:

```bash
docker build -t blockblast-solver .
docker run --rm -p 8501:8501 blockblast-solver
```

## How it works

- `solver.detect_grid_bounds` — finds the playing grid by masking the dark blue background and orange bricks.
- `solver.read_grid_state` — samples each cell centre to determine filled vs empty.
- `solver.detect_pieces` — finds the three piece bounding boxes below the board, infers a common cell size, and reads each piece's shape.
- `solver.solve` — brute-force search over all piece orderings and placements, with line-clear mechanics applied between placements. Returns the placement that clears the most lines.
- `solver.render_solution` — produces an annotated step-by-step PNG.

## Limits / notes

- Pieces are not rotated (matches the standard Block Blast UX).
- Grid size defaults to 8×8; sidebar lets you switch to 9 or 10.
- Detection assumes the standard Block Blast colour palette (dark blue grid, orange bricks). Heavy theme changes may need threshold tuning in `solver.brick_mask` / `solver.grid_bg_mask`.

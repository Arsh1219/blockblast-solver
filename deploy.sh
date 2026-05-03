#!/usr/bin/env bash
# Deploy the Block Blast Solver Streamlit app.
#
# Modes:
#   ./deploy.sh local        - run the app locally on http://localhost:8501
#   ./deploy.sh docker       - build & run a Docker container on :8501
#   ./deploy.sh streamlit    - print step-by-step instructions for Streamlit
#                              Community Cloud (free, public URL)
#   ./deploy.sh hf           - print step-by-step instructions for Hugging
#                              Face Spaces (free, public URL)
#
# With no argument, prints usage.

set -euo pipefail

MODE="${1:-}"
PORT="${PORT:-8501}"
IMAGE_NAME="blockblast-solver"

repo_url() {
    git config --get remote.origin.url 2>/dev/null \
        | sed -E 's#git@github.com:#https://github.com/#; s#\.git$##'
}

case "$MODE" in
    local)
        if ! python3 -c "import streamlit" >/dev/null 2>&1; then
            echo ">> Installing dependencies..."
            python3 -m pip install -r requirements.txt
        fi
        echo ">> Starting Streamlit on http://localhost:${PORT}"
        exec streamlit run app.py --server.port="${PORT}" --server.address=0.0.0.0
        ;;

    docker)
        echo ">> Building Docker image: ${IMAGE_NAME}"
        docker build -t "${IMAGE_NAME}" .
        echo ">> Running container on http://localhost:${PORT}"
        exec docker run --rm -it -p "${PORT}:8501" -e "PORT=8501" "${IMAGE_NAME}"
        ;;

    streamlit)
        URL="$(repo_url || true)"
        cat <<EOF
Deploy to Streamlit Community Cloud (free, public URL)

  1. Push this branch to GitHub (already done if you ran 'git push').
  2. Open https://share.streamlit.io and sign in with GitHub.
  3. Click "New app".
  4. Repository:   ${URL:-<your GitHub repo URL>}
     Branch:       $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
     Main file:    app.py
  5. Click "Deploy". You'll get a public URL like
     https://<your-handle>-blockblast-solver.streamlit.app

  Pushes to the selected branch redeploy automatically.
EOF
        ;;

    hf)
        cat <<'EOF'
Deploy to Hugging Face Spaces (free, public URL)

  1. Go to https://huggingface.co/new-space
  2. Owner:    <your account>
     Name:     blockblast-solver
     SDK:      Streamlit
     Hardware: CPU basic (free)
     Visibility: Public
  3. Create the Space, then in "Files" upload:
       app.py, solver.py, requirements.txt, README.md
     Or, from the command line:
       git remote add space https://huggingface.co/spaces/<you>/blockblast-solver
       git push space HEAD:main
  4. Wait ~1 min for the build to finish. Public URL:
       https://huggingface.co/spaces/<you>/blockblast-solver
EOF
        ;;

    ""|-h|--help|help)
        cat <<EOF
Block Blast Solver - deploy helper

Usage: ./deploy.sh <mode>

Modes:
  local       Run locally on http://localhost:${PORT}
  docker      Build & run a Docker container on :${PORT}
  streamlit   Instructions for Streamlit Community Cloud (recommended)
  hf          Instructions for Hugging Face Spaces

Environment:
  PORT        Port to listen on (default 8501)
EOF
        ;;

    *)
        echo "Unknown mode: ${MODE}" >&2
        echo "Run './deploy.sh' for usage." >&2
        exit 1
        ;;
esac

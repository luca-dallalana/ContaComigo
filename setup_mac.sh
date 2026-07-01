#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step()  { echo -e "\n${GREEN}==>${NC} $1"; }
warn()  { echo -e "${YELLOW}AVISO:${NC} $1"; }
error() { echo -e "${RED}ERRO:${NC} $1"; exit 1; }

cd "$(dirname "$0")"

# Homebrew
step "A verificar Homebrew..."
if ! command -v brew &>/dev/null; then
    step "A instalar Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Python 3.13
step "A verificar Python..."
if ! command -v python3.13 &>/dev/null; then
    step "A instalar Python 3.13..."
    brew install python@3.13
fi
PYTHON=$(command -v python3.13)

# Docker
step "A verificar Docker..."
if ! command -v docker &>/dev/null; then
    step "A instalar Docker Desktop..."
    brew install --cask docker
    echo ""
    echo "Abre o Docker Desktop e espera que arranque. Carrega Enter quando estiver pronto."
    read -r
fi
if ! docker info &>/dev/null; then
    step "A iniciar Docker Desktop..."
    open -a Docker
    echo "A aguardar que o Docker arranque..."
    until docker info &>/dev/null; do sleep 2; done
fi

# Ollama
step "A verificar Ollama..."
if ! command -v ollama &>/dev/null; then
    step "A instalar Ollama..."
    brew install ollama
fi
if ! pgrep -x "ollama" &>/dev/null; then
    step "A iniciar Ollama..."
    ollama serve &>/dev/null &
    sleep 3
fi

# .env
step "A configurar o ambiente..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "Configura as tuas credenciais:"
    read -rp  "  Utilizador PostgreSQL (ex: irs): " pg_user
    read -rsp "  Password PostgreSQL: " pg_pass
    echo ""
    read -rsp "  Groq API Key (grátis em console.groq.com): " groq_key
    echo ""

    sed -i '' "s|POSTGRES_USER=.*|POSTGRES_USER=${pg_user}|"     .env
    sed -i '' "s|POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=${pg_pass}|" .env
    sed -i '' "s|GROQ_API_KEY=.*|GROQ_API_KEY=${groq_key}|"       .env
else
    warn ".env já existe — a saltar configuração."
fi

set -a; source .env; set +a

# Base de dados
step "A iniciar a base de dados..."
docker compose up -d
echo "A aguardar que a base de dados esteja pronta..."
until docker compose exec pgvector pg_isready -U "${POSTGRES_USER}" -d irs_assistant &>/dev/null; do
    sleep 2
done

# Dependências Python
step "A instalar dependências Python..."
"$PYTHON" -m venv .venv
.venv/bin/pip install -r requirements.txt -q

# Modelo de embedding
step "A descarregar o modelo de embedding..."
ollama pull nomic-embed-text

# Ingestão de documentos
step "A descarregar e indexar documentos (~2-3 min)..."
.venv/bin/python scripts/ingest.py

# Chunks curados
step "A inserir chunks curados..."
.venv/bin/python scripts/insert_curated_chunks.py

# Iniciar app
step "Tudo pronto!"
echo ""
echo "A abrir ContaComigo em http://localhost:8501"
echo ""
.venv/bin/streamlit run app/streamlit_app.py

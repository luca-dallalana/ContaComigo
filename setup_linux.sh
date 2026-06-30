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

if [ "$EUID" -eq 0 ]; then
    error "Não corras este script como root. Usa um utilizador normal com sudo."
fi

# Python 3.11+
step "A verificar Python..."
if ! python3 -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
    step "A instalar Python 3.11..."
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-venv python3-pip
fi
PYTHON=$(command -v python3.11 || command -v python3)

# Docker
step "A verificar Docker..."
if ! command -v docker &>/dev/null; then
    step "A instalar Docker..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    warn "Adicionado ao grupo docker. Faz logout/login para aplicar, ou corre: newgrp docker"
    newgrp docker
fi
if ! docker info &>/dev/null; then
    step "A iniciar Docker..."
    sudo systemctl start docker
    until docker info &>/dev/null; do sleep 2; done
fi

# Ollama
step "A verificar Ollama..."
if ! command -v ollama &>/dev/null; then
    step "A instalar Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
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

    sed -i "s|POSTGRES_USER=.*|POSTGRES_USER=${pg_user}|"         .env
    sed -i "s|POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=${pg_pass}|"  .env
    sed -i "s|GROQ_API_KEY=.*|GROQ_API_KEY=${groq_key}|"           .env
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
step "A descarregar o modelo de embedding (~1 GB)..."
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

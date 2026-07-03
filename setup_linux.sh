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

# Python 3.13
step "A verificar Python 3.13..."
command -v python3.13 &>/dev/null || error "Python 3.13 não encontrado. Instala em https://www.python.org/downloads/"
PYTHON=$(command -v python3.13)

# Docker
step "A verificar Docker..."
command -v docker &>/dev/null || error "Docker não encontrado. Instala em https://docs.docker.com/engine/install/"
if ! docker info &>/dev/null; then
    step "A iniciar Docker..."
    sudo systemctl start docker
    until docker info &>/dev/null; do sleep 2; done
fi

# Ollama
step "A verificar Ollama..."
command -v ollama &>/dev/null || error "Ollama não encontrado. Instala em https://ollama.com"
if ! pgrep -x "ollama" &>/dev/null; then
    step "A iniciar Ollama..."
    ollama serve &>/dev/null &
    echo "A aguardar que o Ollama arranque..."
    until ollama list &>/dev/null; do sleep 2; done
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

    python3 -c "
import re, sys
content = open('.env').read()
content = re.sub(r'^POSTGRES_USER=.*', 'POSTGRES_USER=' + sys.argv[1], content, flags=re.M)
content = re.sub(r'^POSTGRES_PASSWORD=.*', 'POSTGRES_PASSWORD=' + sys.argv[2], content, flags=re.M)
content = re.sub(r'^GROQ_API_KEY=.*', 'GROQ_API_KEY=' + sys.argv[3], content, flags=re.M)
open('.env', 'w').write(content)
" "$pg_user" "$pg_pass" "$groq_key"
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

# Inicializar base de dados
step "A criar tabelas e índices..."
.venv/bin/python scripts/init_db.py

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

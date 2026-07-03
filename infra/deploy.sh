#!/usr/bin/env bash
set -euo pipefail

# Usage: ./infra/deploy.sh <key-file> <key-name>
#   key-file  — path to the .pem private key file (e.g. ~/.ssh/my-key.pem)
#   key-name  — name of the key pair in AWS (e.g. my-key)
#
# Optional env vars:
#   GENERATION_MODEL  — Ollama model tag (default: qwen2.5:32b)
#   AWS_REGION        — override region (default: us-east-1)
#   INSTANCE_TYPE     — override instance type (default: g5.xlarge)

KEY_FILE="${1:?Usage: $0 <key-file> <key-name>}"
KEY_NAME="${2:?Usage: $0 <key-file> <key-name>}"
GENERATION_MODEL="${GENERATION_MODEL:-qwen2.5:32b}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
AWS_REGION="${AWS_REGION:-us-east-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MY_IP="$(curl -s ifconfig.me)"
MY_CIDR="${MY_IP}/32"

SSH_OPTS="-i $KEY_FILE -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=60"

echo "==================================================================="
echo " IRS Assistant — EC2 GPU Eval"
echo "  Instance type : $INSTANCE_TYPE"
echo "  Model         : $GENERATION_MODEL"
echo "  Region        : $AWS_REGION"
echo "  Your IP       : $MY_CIDR"
echo "==================================================================="
echo ""

# ── 1. Terraform provision ──────────────────────────────────────────────
echo ">>> [1/9] Provisioning EC2 instance..."
terraform -chdir="$SCRIPT_DIR" init -upgrade -input=false
terraform -chdir="$SCRIPT_DIR" apply -auto-approve -input=false \
  -var "key_name=$KEY_NAME" \
  -var "my_cidr=$MY_CIDR" \
  -var "generation_model=$GENERATION_MODEL" \
  -var "instance_type=$INSTANCE_TYPE" \
  -var "aws_region=$AWS_REGION"

IP="$(terraform -chdir="$SCRIPT_DIR" output -raw instance_ip)"
echo "    Instance IP: $IP"

# ── 2. Wait for SSH ─────────────────────────────────────────────────────
echo ""
echo ">>> [2/9] Waiting for SSH to become available..."
for i in $(seq 1 40); do
  if ssh $SSH_OPTS ubuntu@"$IP" true 2>/dev/null; then
    echo "    SSH ready."
    break
  fi
  echo "    Attempt $i/40, retrying in 10s..."
  sleep 10
done

# ── 3. Install runtime on EC2 ───────────────────────────────────────────
echo ""
echo ">>> [3/9] Installing Docker, PostgreSQL client, and Ollama..."
ssh $SSH_OPTS ubuntu@"$IP" bash <<'REMOTE'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -q
sudo apt-get install -y docker.io postgresql-client python3-pip python3-venv curl
sudo systemctl start docker

# pgvector postgres container
sudo docker run -d \
  --name pgvector \
  --restart unless-stopped \
  -e POSTGRES_USER=irs \
  -e POSTGRES_PASSWORD=irs \
  -e POSTGRES_DB=irs_assistant \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sudo sh
sudo systemctl enable ollama
sudo systemctl start ollama
echo "Runtime installation complete."
REMOTE

# ── 4. Push codebase ────────────────────────────────────────────────────
echo ""
echo ">>> [4/9] Pushing codebase to EC2..."
rsync -az --progress \
  --exclude='.venv' \
  --exclude='data/raw' \
  --exclude='.git' \
  --exclude='infra' \
  --exclude='eval/results*.json' \
  -e "ssh $SSH_OPTS" \
  "$PROJECT_ROOT/" \
  "ubuntu@$IP:~/irs-assistant/"

# ── 5. pg_dump local DB ─────────────────────────────────────────────────
echo ""
echo ">>> [5/9] Dumping local database..."
docker compose -f "$PROJECT_ROOT/docker-compose.yml" exec -T pgvector \
  pg_dump --no-owner --no-acl -U irs irs_assistant \
  > /tmp/irs_assistant_dump.sql
echo "    Dump size: $(du -sh /tmp/irs_assistant_dump.sql | cut -f1)"

echo "    Uploading dump to EC2..."
scp $SSH_OPTS /tmp/irs_assistant_dump.sql "ubuntu@$IP:/tmp/irs_assistant_dump.sql"

# ── 6. Restore DB on EC2 ────────────────────────────────────────────────
echo ""
echo ">>> [6/9] Restoring database on EC2..."
ssh $SSH_OPTS ubuntu@"$IP" bash <<'REMOTE'
set -euo pipefail
echo "  Waiting for postgres container to be healthy..."
for i in $(seq 1 30); do
  if PGPASSWORD=irs psql -h localhost -U irs -d irs_assistant -c '\q' 2>/dev/null; then
    echo "  Postgres ready."
    break
  fi
  sleep 3
done
PGPASSWORD=irs psql -h localhost -U irs -d irs_assistant < /tmp/irs_assistant_dump.sql
echo "  Restore complete. Chunk count:"
PGPASSWORD=irs psql -h localhost -U irs -d irs_assistant -c "SELECT COUNT(*) FROM chunks;" 2>/dev/null
REMOTE

# ── 7. Pull Ollama models ───────────────────────────────────────────────
echo ""
echo ">>> [7/9] Pulling Ollama models (nomic-embed-text + $GENERATION_MODEL)..."
echo "    This step can take 10-20 minutes depending on model size."
ssh $SSH_OPTS ubuntu@"$IP" "ollama pull nomic-embed-text:latest"
ssh $SSH_OPTS ubuntu@"$IP" "ollama pull $GENERATION_MODEL"

# ── 8. Write .env and install Python deps ──────────────────────────────
echo ""
echo ">>> [8/9] Writing .env and installing Python dependencies..."
ssh $SSH_OPTS ubuntu@"$IP" "cat > ~/irs-assistant/.env" <<EOF
POSTGRES_URL=postgresql://irs:irs@localhost:5432/irs_assistant
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text:latest
GENERATION_MODEL=$GENERATION_MODEL
TOP_K=12
VECTOR_WEIGHT=0.7
BM25_WEIGHT=0.3
EOF

ssh $SSH_OPTS ubuntu@"$IP" bash <<'REMOTE'
set -euo pipefail
cd ~/irs-assistant
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
echo "Python deps installed."
REMOTE

# ── 9. Run eval ─────────────────────────────────────────────────────────
echo ""
echo ">>> [9/9] Running evaluation harness..."
ssh $SSH_OPTS ubuntu@"$IP" bash <<'REMOTE'
set -euo pipefail
cd ~/irs-assistant
.venv/bin/python eval/run_eval.py --output eval/results_ec2.json 2>&1 | tee /tmp/eval.log
REMOTE

echo ""
echo ">>> Pulling results..."
scp $SSH_OPTS "ubuntu@$IP:~/irs-assistant/eval/results_ec2.json" \
  "$PROJECT_ROOT/eval/results_ec2.json"

echo ""
echo "==================================================================="
echo " Eval complete. Results saved to eval/results_ec2.json"
echo " Instance is still running at $IP (~\$1/hr)."
echo "==================================================================="
echo ""
read -rp "Destroy the instance now? [y/N]: " answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
  echo "Destroying instance..."
  terraform -chdir="$SCRIPT_DIR" destroy -auto-approve -input=false \
    -var "key_name=$KEY_NAME" \
    -var "my_cidr=$MY_CIDR" \
    -var "generation_model=$GENERATION_MODEL" \
    -var "instance_type=$INSTANCE_TYPE" \
    -var "aws_region=$AWS_REGION"
  echo "Instance destroyed."
else
  echo ""
  echo "To destroy later:"
  echo "  terraform -chdir=infra destroy -auto-approve \\"
  echo "    -var key_name=$KEY_NAME \\"
  echo "    -var my_cidr=$MY_CIDR \\"
  echo "    -var generation_model=$GENERATION_MODEL"
fi

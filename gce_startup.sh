#!/bin/bash
# GCE startup script for NBA Meta-Learner training

# Configuration from metadata
PROJECT_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
REPO_URL=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/repo_url" -H "Metadata-Flavor: Google")
BUCKET_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket_name" -H "Metadata-Flavor: Google")
SECRET_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/secret_id" -H "Metadata-Flavor: Google")

# Set defaults if metadata is missing
REPO_URL=${REPO_URL:-"https://github.com/tyriqmiles0529-pixel/meep.git"}
BUCKET_NAME=${BUCKET_NAME:-"nba-models"}
SECRET_ID=${SECRET_ID:-"kaggle-credentials"}

echo "=== NBA Meta-Learner GCE Setup ==="
echo "Project: $PROJECT_ID"
echo "Repo: $REPO_URL"
echo "Bucket: $BUCKET_NAME"
echo "Secret ID: $SECRET_ID"

# Update system packages
echo "Updating system packages..."
apt-get update -y
apt-get install -y python3-pip python3-venv git

# Create application directory
APP_DIR="/opt/nba-predictor"
mkdir -p $APP_DIR
cd $APP_DIR

# Clone repository
echo "Cloning repository..."
git clone $REPO_URL .

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional system dependencies for ML libraries
apt-get install -y build-essential

# Set up Google Cloud authentication
echo "Setting up Google Cloud authentication..."
# Use the service account attached to the instance
gcloud auth activate-service-account --quiet

# Download models from GCS
echo "Downloading models from GCS..."
mkdir -p /tmp/model_cache
gsutil -m cp -r gs://$BUCKET_NAME/models/*.pkl /tmp/model_cache/ 2>/dev/null || echo "No models found in bucket"
gsutil -m cp -r gs://$BUCKET_NAME/models/*.json /tmp/model_cache/ 2>/dev/null || echo "No metadata found in bucket"

# Create training script launcher
cat > /opt/nba-predictor/run_training.sh << 'EOF'
#!/bin/bash
cd /opt/nba-predictor
source venv/bin/activate

# Get instance metadata
PROJECT_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
BUCKET_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket_name" -H "Metadata-Flavor: Google")
SECRET_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/secret_id" -H "Metadata-Flavor: Google")
CONFIG_FILE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/config_file" -H "Metadata-Flavor: Google")

# Set defaults
CONFIG_FILE=${CONFIG_FILE:-"experiments/v4_full.yaml"}

echo "Starting training with config: $CONFIG_FILE"
echo "Project: $PROJECT_ID"
echo "Bucket: $BUCKET_NAME"

# Run training
python gce_train_meta_v4.py \
    --config "$CONFIG_FILE" \
    --project-id "$PROJECT_ID" \
    --bucket-name "$BUCKET_NAME" \
    --secret-id "$SECRET_ID"

# Upload logs to GCS when done
LOG_FILE="/tmp/training_$(date +%Y%m%d_%H%M%S).log"
gsutil cp /tmp/training.log gs://$BUCKET_NAME/logs/ 2>/dev/null || true
EOF

chmod +x /opt/nba-predictor/run_training.sh

# Create systemd service for auto-start
cat > /etc/systemd/system/nba-training.service << EOF
[Unit]
Description=NBA Meta-Learner Training
After=network.target

[Service]
Type=oneshot
User=root
WorkingDirectory=/opt/nba-predictor
ExecStart=/opt/nba-predictor/run_training.sh
StandardOutput=append:/tmp/training.log
StandardError=append:/tmp/training.log

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
systemctl daemon-reload
systemctl enable nba-training.service

# Create monitoring script
cat > /opt/nba-predictor/monitor.sh << 'EOF'
#!/bin/bash
echo "=== NBA Training Monitor ==="
echo "Service status:"
systemctl status nba-training.service --no-pager
echo ""
echo "Recent logs:"
tail -50 /tmp/training.log
echo ""
echo "GCS bucket contents:"
gsutil ls gs://$BUCKET_NAME/models/ | head -10
EOF

chmod +x /opt/nba-predictor/monitor.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run training manually:"
echo "  sudo /opt/nba-predictor/run_training.sh"
echo ""
echo "To monitor training:"
echo "  sudo /opt/nba-predictor/monitor.sh"
echo ""
echo "To check logs:"
echo "  tail -f /tmp/training.log"
echo ""
echo "Training will start automatically in 60 seconds..."

# Start training after 60 seconds
sleep 60
systemctl start nba-training.service

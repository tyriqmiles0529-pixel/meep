#!/bin/bash
# Create GCE instance for NBA Meta-Learner training - Production version with cost controls

# Configuration
PROJECT_ID="$1"
INSTANCE_NAME="$2"
ZONE="${3:-us-central1-a}"
REPO_URL="${4:-https://github.com/tyriqmiles0529-pixel/meep.git}"
BUCKET_NAME="$5"
SECRET_ID="${6:-kaggle-credentials}"
CONFIG_FILE="${7:-experiments/v4_full.yaml}"
PREEMPTIBLE="${8:-true}"

if [ -z "$PROJECT_ID" ] || [ -z "$INSTANCE_NAME" ] || [ -z "$BUCKET_NAME" ]; then
    echo "Usage: ./create_instance_prod.sh <project_id> <instance_name> [zone] [repo_url] <bucket_name> [secret_id] [config_file] [preemptible]"
    echo "Example: ./create_instance_prod.sh my-project nba-trainer-1 us-central1-a https://github.com/user/repo.git nba-models kaggle-credentials experiments/v4_full.yaml true"
    exit 1
fi

echo "Creating GCE instance for NBA training (Production)..."
echo "Project: $PROJECT_ID"
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Repo: $REPO_URL"
echo "Bucket: $BUCKET_NAME"
echo "Secret: $SECRET_ID"
echo "Config: $CONFIG_FILE"
echo "Preemptible: $PREEMPTIBLE"

# Set the project
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable monitoring.googleapis.com

# Create enhanced startup script with auto-shutdown
ENHANCED_STARTUP_SCRIPT=$(cat << 'EOF'
#!/bin/bash
# Enhanced startup script with auto-shutdown and monitoring

# Configuration from metadata
PROJECT_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
INSTANCE_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
REPO_URL=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/repo_url" -H "Metadata-Flavor: Google")
BUCKET_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket_name" -H "Metadata-Flavor: Google")
SECRET_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/secret_id" -H "Metadata-Flavor: Google")
IS_PREEMPTIBLE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/scheduling/preemptible" -H "Metadata-Flavor: Google")

# Set defaults
REPO_URL=${REPO_URL:-"https://github.com/tyriqmiles0529-pixel/meep.git"}
BUCKET_NAME=${BUCKET_NAME:-"nba-models"}
SECRET_ID=${SECRET_ID:-"kaggle-credentials"}

echo "=== NBA Meta-Learner GCE Setup (Production) ==="
echo "Project: $PROJECT_ID"
echo "Instance: $INSTANCE_NAME"
echo "Repo: $REPO_URL"
echo "Bucket: $BUCKET_NAME"
echo "Secret ID: $SECRET_ID"
echo "Preemptible: $IS_PREEMPTIBLE"

# Function to shutdown instance after training
shutdown_after_training() {
    echo "Training completed. Shutting down instance in 60 seconds..."
    sleep 60
    if [ "$IS_PREEMPTIBLE" = "true" ]; then
        echo "Preemptible instance - will be cleaned up automatically"
    else
        sudo shutdown -h now
    fi
}

# Function to handle preemption
handle_preemption() {
    echo "Instance is being preempted. Saving state..."
    # Upload any remaining logs
    gsutil cp /tmp/training.log gs://$BUCKET_NAME/logs/preemption_$(date +%Y%m%d_%H%M%S).log 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap handle_preemption SIGTERM

# Update system packages
echo "Updating system packages..."
apt-get update -y
apt-get install -y python3-pip python3-venv git

# Create application directory
APP_DIR="/opt/nba-predictor"
if [ -d "$APP_DIR" ]; then
    echo "Repository already exists, updating..."
    cd $APP_DIR
    git pull origin main
else
    mkdir -p $APP_DIR
    cd $APP_DIR
    git clone $REPO_URL .
fi

# Create virtual environment
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional system dependencies for ML libraries
apt-get install -y build-essential

# Set up Google Cloud authentication
echo "Setting up Google Cloud authentication..."
gcloud auth activate-service-account --quiet

# Download models from GCS
echo "Downloading models from GCS..."
mkdir -p /tmp/model_cache
gsutil -m cp -r gs://$BUCKET_NAME/models/*.pkl /tmp/model_cache/ 2>/dev/null || echo "No models found in bucket"
gsutil -m cp -r gs://$BUCKET_NAME/models/*.json /tmp/model_cache/ 2>/dev/null || echo "No metadata found in bucket"

# Create training script launcher with auto-shutdown
cat > /opt/nba-predictor/run_training.sh << 'EOL'
#!/bin/bash
cd /opt/nba-predictor
source venv/bin/activate

# Get instance metadata
PROJECT_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
INSTANCE_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
BUCKET_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket_name" -H "Metadata-Flavor: Google")
SECRET_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/secret_id" -H "Metadata-Flavor: Google")
CONFIG_FILE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/config_file" -H "Metadata-Flavor: Google")
IS_PREEMPTIBLE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/scheduling/preemptible" -H "Metadata-Flavor: Google")

# Set defaults
CONFIG_FILE=${CONFIG_FILE:-"experiments/v4_full.yaml"}

echo "Starting training with config: $CONFIG_FILE"
echo "Project: $PROJECT_ID"
echo "Instance: $INSTANCE_NAME"
echo "Bucket: $BUCKET_NAME"

# Run training with robust script
if [ -f "gce_train_meta_v4_robust.py" ]; then
    python gce_train_meta_v4_robust.py \
        --config "$CONFIG_FILE" \
        --project-id "$PROJECT_ID" \
        --bucket-name "$BUCKET_NAME" \
        --secret-id "$SECRET_ID"
    TRAINING_EXIT_CODE=$?
else
    python gce_train_meta_v4.py \
        --config "$CONFIG_FILE" \
        --project-id "$PROJECT_ID" \
        --bucket-name "$BUCKET_NAME" \
        --secret-id "$SECRET_ID"
    TRAINING_EXIT_CODE=$?
fi

# Upload final logs
LOG_FILE="/tmp/training_$(date +%Y%m%d_%H%M%S).log"
cp /tmp/training.log "$LOG_FILE"
gsutil cp "$LOG_FILE" gs://$BUCKET_NAME/logs/ 2>/dev/null || true

# Send custom metric to Cloud Monitoring
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
    # Send success metric
    gcloud monitoring metrics describe --format="value(type)" custom.googleapis.com/nba/training_success 2>/dev/null || \
    gcloud monitoring metrics create custom.googleapis.com/nba/training_success --display-name="NBA Training Success" --type="BOOL" --unit="1"
    echo "{\"value\": \"1\"}" | gcloud monitoring timeseries create --metric="custom.googleapis.com/nba/training_success" --resource-type="gce_instance" --resource-labels="instance_id=$INSTANCE_NAME,zone=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone -H Metadata-Flavor: Google)"
else
    echo "Training failed with exit code: $TRAINING_EXIT_CODE"
    # Send failure metric
    gcloud monitoring metrics describe --format="value(type)" custom.googleapis.com/nba/training_failure 2>/dev/null || \
    gcloud monitoring metrics create custom.googleapis.com/nba/training_failure --display-name="NBA Training Failure" --type="BOOL" --unit="1"
    echo "{\"value\": \"1\"}" | gcloud monitoring timeseries create --metric="custom.googleapis.com/nba/training_failure" --resource-type="gce_instance" --resource-labels="instance_id=$INSTANCE_NAME,zone=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone -H Metadata-Flavor: Google)"
fi

# Auto-shutdown if not preemptible
if [ "$IS_PREEMPTIBLE" != "true" ]; then
    echo "Shutting down instance after training completion..."
    sudo shutdown -h now
fi
EOL

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
TimeoutStartSec=1800

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
echo "Instance: $(curl -s http://metadata.google.internal/computeMetadata/v1/instance/name -H Metadata-Flavor: Google)"
echo "Zone: $(curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone -H Metadata-Flavor: Google)"
echo "Preemptible: $(curl -s http://metadata.google.internal/computeMetadata/v1/instance/scheduling/preemptible -H Metadata-Flavor: Google)"
echo ""
echo "Service status:"
systemctl status nba-training.service --no-pager
echo ""
echo "Recent logs:"
tail -50 /tmp/training.log
echo ""
echo "GCS bucket contents:"
gsutil ls gs://$BUCKET_NAME/models/ | head -10
echo ""
echo "Disk usage:"
df -h
echo ""
echo "Memory usage:"
free -h
EOF

chmod +x /opt/nba-predictor/monitor.sh

echo ""
echo "✅ Production setup complete!"
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
echo "To view logs in GCS:"
echo "  gsutil ls gs://$BUCKET_NAME/logs/"
echo ""
echo "Training will start automatically in 60 seconds..."

# Start training after 60 seconds
sleep 60
systemctl start nba-training.service
EOF
)

# Build instance creation command
INSTANCE_CMD=(
    gcloud compute instances create "$INSTANCE_NAME"
    --zone="$ZONE"
    --machine-type="n1-standard-16"
    --image-family="ubuntu-2004-lts"
    --image-project="ubuntu-os-cloud"
    --boot-disk-size="200GB"
    --boot-disk-type="pd-ssd"
    --scopes="cloud-platform"
    --metadata="repo_url=$REPO_URL,bucket_name=$BUCKET_NAME,secret_id=$SECRET_ID,config_file=$CONFIG_FILE,startup-script=$ENHANCED_STARTUP_SCRIPT"
    --tags="nba-training"
    --no-shielded-secure-boot
    --no-shielded-vtpm
    --no-shielded-integrity-monitoring
)

# Add preemptible flag if requested
if [ "$PREEMPTIBLE" = "true" ]; then
    INSTANCE_CMD+=(--preemptible)
    echo "Creating PREEMPTIBLE instance (80% cost savings)..."
else
    echo "Creating standard instance..."
fi

# Execute instance creation
"${INSTANCE_CMD[@]}"

# Create firewall rule to allow SSH (if not exists)
echo "Setting up firewall rules..."
if ! gcloud compute firewall-rules describe "allow-ssh-nba-training" --quiet 2>/dev/null; then
    gcloud compute firewall-rules create "allow-ssh-nba-training" \
        --allow tcp:22 \
        --source-ranges "0.0.0.0/0" \
        --target-tags "nba-training"
fi

# Set up budget alerts (optional)
echo ""
echo "💡 Cost Management Tips:"
echo "1. Set up budget alerts: https://console.cloud.google.com/billing"
echo "2. Monitor costs: gcloud billing budgets list"
echo "3. This instance will auto-shutdown after training completion"
if [ "$PREEMPTIBLE" = "true" ]; then
    echo "4. Using preemptible instance saves ~80% on compute costs"
fi

echo ""
echo "✅ Instance creation complete!"
echo ""
echo "Instance details:"
echo "  Name: $INSTANCE_NAME"
echo "  Zone: $ZONE"
echo "  Machine: n1-standard-16 (16 vCPUs, 60GB RAM)"
echo "  Disk: 200GB SSD"
echo "  Preemptible: $PREEMPTIBLE"
echo ""
echo "To connect to the instance:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To monitor training:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command 'sudo /opt/nba-predictor/monitor.sh'"
echo ""
echo "To view logs:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command 'tail -f /tmp/training.log'"
echo ""
echo "To stop the instance when done:"
if [ "$PREEMPTIBLE" = "true" ]; then
    echo "  Instance will auto-shutdown after training"
else
    echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
fi
echo ""
echo "To view Cloud Monitoring metrics:"
echo "  https://console.cloud.google.com/monitoring"

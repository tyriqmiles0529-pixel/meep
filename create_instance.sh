#!/bin/bash
# Create GCE instance for NBA Meta-Learner training

# Configuration
PROJECT_ID="$1"
INSTANCE_NAME="$2"
ZONE="${3:-us-central1-a}"
REPO_URL="${4:-https://github.com/tyriqmiles0529-pixel/meep.git}"
BUCKET_NAME="$5"
SECRET_ID="${6:-kaggle-credentials}"
CONFIG_FILE="${7:-experiments/v4_full.yaml}"

if [ -z "$PROJECT_ID" ] || [ -z "$INSTANCE_NAME" ] || [ -z "$BUCKET_NAME" ]; then
    echo "Usage: ./create_instance.sh <project_id> <instance_name> [zone] [repo_url] <bucket_name> [secret_id] [config_file]"
    echo "Example: ./create_instance.sh my-project nba-trainer-1 us-central1-a https://github.com/user/repo.git nba-models kaggle-credentials experiments/v4_full.yaml"
    exit 1
fi

echo "Creating GCE instance for NBA training..."
echo "Project: $PROJECT_ID"
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Repo: $REPO_URL"
echo "Bucket: $BUCKET_NAME"
echo "Secret: $SECRET_ID"
echo "Config: $CONFIG_FILE"

# Set the project
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Create instance with metadata
echo "Creating GCE instance..."
gcloud compute instances create "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --machine-type="n1-standard-16" \
    --image-family="ubuntu-2004-lts" \
    --image-project="ubuntu-os-cloud" \
    --boot-disk-size="200GB" \
    --boot-disk-type="pd-ssd" \
    --scopes="cloud-platform" \
    --metadata="repo_url=$REPO_URL,bucket_name=$BUCKET_NAME,secret_id=$SECRET_ID,config_file=$CONFIG_FILE,startup-script=$(cat gce_startup.sh)" \
    --tags="nba-training" \
    --no-shielded-secure-boot \
    --no-shielded-vtpm \
    --no-shielded-integrity-monitoring

# Create firewall rule to allow SSH (if not exists)
echo "Setting up firewall rules..."
if ! gcloud compute firewall-rules describe "allow-ssh-nba-training" --quiet 2>/dev/null; then
    gcloud compute firewall-rules create "allow-ssh-nba-training" \
        --allow tcp:22 \
        --source-ranges "0.0.0.0/0" \
        --target-tags "nba-training"
fi

echo ""
echo "✅ Instance creation complete!"
echo ""
echo "Instance details:"
echo "  Name: $INSTANCE_NAME"
echo "  Zone: $ZONE"
echo "  Machine: n1-standard-16 (16 vCPUs, 60GB RAM)"
echo "  Disk: 200GB SSD"
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
echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"

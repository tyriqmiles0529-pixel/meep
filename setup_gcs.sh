#!/bin/bash
# Setup Google Cloud Storage buckets for NBA training

# Configuration
PROJECT_ID="$1"
BUCKET_NAME="$2"
REGION="${3:-us-central1}"

if [ -z "$PROJECT_ID" ] || [ -z "$BUCKET_NAME" ]; then
    echo "Usage: ./setup_gcs.sh <project_id> <bucket_name> [region]"
    echo "Example: ./setup_gcs.sh my-project nba-models us-central1"
    exit 1
fi

echo "Setting up GCS infrastructure for NBA training..."
echo "Project: $PROJECT_ID"
echo "Bucket: $BUCKET_NAME"
echo "Region: $REGION"

# Set the project
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable storage.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable compute.googleapis.com

# Create the main bucket
echo "Creating GCS bucket: $BUCKET_NAME"
if gsutil ls -b gs://"$BUCKET_NAME" 2>/dev/null; then
    echo "Bucket already exists"
else
    gsutil mb -l "$REGION" gs://"$BUCKET_NAME"
    echo "Bucket created successfully"
fi

# Create folder structure
echo "Creating folder structure..."
gsutil -m cp -r /dev/null gs://"$BUCKET_NAME"/models/ 2>/dev/null || true
gsutil -m cp -r /dev/null gs://"$BUCKET_NAME"/data/ 2>/dev/null || true
gsutil -m cp -r /dev/null gs://"$BUCKET_NAME"/results/ 2>/dev/null || true
gsutil -m cp -r /dev/null gs://"$BUCKET_NAME"/logs/ 2>/dev/null || true

# Set bucket permissions (allow project members to access)
echo "Setting bucket permissions..."
gsutil iam ch "projectViewer:$PROJECT_ID:objectViewer" gs://"$BUCKET_NAME"
gsutil iam ch "projectEditor:$PROJECT_ID:objectAdmin" gs://"$BUCKET_NAME"

echo ""
echo "✅ GCS setup complete!"
echo "Bucket: gs://$BUCKET_NAME"
echo ""
echo "Next steps:"
echo "1. Upload your existing models to gs://$BUCKET_NAME/models/"
echo "2. Set up Kaggle credentials in Secret Manager:"
echo "   echo '{\"username\":\"YOUR_KAGGLE_USERNAME\",\"key\":\"YOUR_KAGGLE_KEY\"}' | gcloud secrets create kaggle-credentials --data-file=-"
echo "3. Create GCE instance with startup script"

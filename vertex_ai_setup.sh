#!/bin/bash
# Vertex AI Training Setup Script

# Configuration
PROJECT_ID="your-project-id"  # CHANGE THIS
REGION="us-central1"
BUCKET_NAME="nba-predictor-models"
IMAGE_NAME="nba-trainer"
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

echo "=========================================="
echo "🏀 Vertex AI Training Setup"
echo "=========================================="

# Step 1: Create Cloud Storage bucket
echo ""
echo "📦 Creating Cloud Storage bucket..."
gsutil mb -l ${REGION} gs://${BUCKET_NAME} 2>/dev/null || echo "  Bucket already exists"

# Step 2: Build and push Docker image
echo ""
echo "🐳 Building Docker image..."
docker build -f Dockerfile.vertexai -t ${IMAGE_URI} .

echo ""
echo "📤 Pushing image to Container Registry..."
docker push ${IMAGE_URI}

# Step 3: Create Kaggle secret in Secret Manager
echo ""
echo "🔐 Setting up Kaggle credentials in Secret Manager..."
echo "Please enter your Kaggle username:"
read KAGGLE_USERNAME
echo "Please enter your Kaggle API key:"
read -s KAGGLE_KEY

# Create secrets
echo -n "${KAGGLE_USERNAME}" | gcloud secrets create kaggle-username \
    --data-file=- \
    --replication-policy="automatic" 2>/dev/null || echo "  Secret already exists"

echo -n "${KAGGLE_KEY}" | gcloud secrets create kaggle-key \
    --data-file=- \
    --replication-policy="automatic" 2>/dev/null || echo "  Secret already exists"

echo ""
echo "✅ Setup complete!"
echo ""
echo "=========================================="
echo "📋 Next Steps:"
echo "=========================================="
echo "1. Go to Vertex AI > Training"
echo "2. Create new training job"
echo "3. Use custom container: ${IMAGE_URI}"
echo "4. Set output directory: gs://${BUCKET_NAME}/models/"
echo "5. Add arguments (see vertex_ai_config.txt)"
echo "=========================================="

#!/bin/bash
# Cleanup script for GCS storage to control costs

# Configuration
BUCKET_NAME="$1"
DAYS_TO_KEEP="${2:-30}"
PROJECT_ID="${3:-$(gcloud config get-value project)}"

if [ -z "$BUCKET_NAME" ]; then
    echo "Usage: ./cleanup_gcs.sh <bucket_name> [days_to_keep] [project_id]"
    echo "Example: ./cleanup_gcs.sh nba-models 30 my-project"
    echo ""
    echo "This script will:"
    echo "  - Remove logs older than $DAYS_TO_KEEP days"
    echo "  - Remove old checkpoints"
    echo "  - Show storage cost savings"
    exit 1
fi

echo "=== GCS Cleanup Script ==="
echo "Bucket: $BUCKET_NAME"
echo "Project: $PROJECT_ID"
echo "Days to keep logs: $DAYS_TO_KEEP"
echo ""

# Set project
gcloud config set project "$PROJECT_ID"

# Calculate cutoff date
CUTOFF_DATE=$(date -d "$DAYS_TO_KEEP days ago" +%Y%m%d)
echo "Removing files older than: $CUTOFF_DATE"
echo ""

# Function to format bytes
format_bytes() {
    local bytes=$1
    if [ $bytes -lt 1024 ]; then
        echo "${bytes}B"
    elif [ $bytes -lt 1048576 ]; then
        echo "$(( bytes / 1024 ))KB"
    elif [ $bytes -lt 1073741824 ]; then
        echo "$(( bytes / 1048576 ))MB"
    else
        echo "$(( bytes / 1073741824 ))GB"
    fi
}

# Clean up old logs
echo "🧹 Cleaning up old logs..."
LOGS_SIZE_BEFORE=0
LOGS_SIZE_AFTER=0
LOGS_DELETED=0

# Get size before cleanup
for file in $(gsutil ls gs://$BUCKET_NAME/logs/training_*.log 2>/dev/null || true); do
    size=$(gsutil du -s "$file" | awk '{print $1}')
    LOGS_SIZE_BEFORE=$((LOGS_SIZE_BEFORE + size))
done

# Delete old logs
for file in $(gsutil ls gs://$BUCKET_NAME/logs/training_*.log 2>/dev/null || true); do
    filename=$(basename "$file")
    if [[ $filename =~ training_([0-9]{8})_[0-9]{6}\.log ]]; then
        file_date="${BASH_REMATCH[1]}"
        if [ "$file_date" -lt "$CUTOFF_DATE" ]; then
            size=$(gsutil du -s "$file" | awk '{print $1}')
            gsutil rm "$file" 2>/dev/null && LOGS_DELETED=$((LOGS_DELETED + 1))
            LOGS_SIZE_AFTER=$((LOGS_SIZE_AFTER + size))
            echo "  Deleted: $filename ($(format_bytes $size))"
        fi
    fi
done

echo "  Logs deleted: $LOGS_DELETED"
echo "  Space freed: $(format_bytes $LOGS_SIZE_AFTER)"
echo ""

# Clean up old checkpoints
echo "🧹 Cleaning up old checkpoints..."
CHECKPOINTS_SIZE_BEFORE=0
CHECKPOINTS_SIZE_AFTER=0
CHECKPOINTS_DELETED=0

# Get size before cleanup
for file in $(gsutil ls gs://$BUCKET_NAME/checkpoints/*.pkl 2>/dev/null || true); do
    size=$(gsutil du -s "$file" | awk '{print $1}')
    CHECKPOINTS_SIZE_BEFORE=$((CHECKPOINTS_SIZE_BEFORE + size))
done

# Delete all checkpoints except the most recent
if [ $(gsutil ls gs://$BUCKET_NAME/checkpoints/*.pkl 2>/dev/null | wc -l) -gt 1 ]; then
    # Keep only the latest checkpoint
    LATEST_CHECKPOINT=$(gsutil ls -la gs://$BUCKET_NAME/checkpoints/*.pkl 2>/dev/null | sort -k6,7 | tail -1 | awk '{print $NF}')
    
    for file in $(gsutil ls gs://$BUCKET_NAME/checkpoints/*.pkl 2>/dev/null); do
        if [ "$file" != "$LATEST_CHECKPOINT" ]; then
            size=$(gsutil du -s "$file" | awk '{print $1}')
            gsutil rm "$file" 2>/dev/null && CHECKPOINTS_DELETED=$((CHECKPOINTS_DELETED + 1))
            CHECKPOINTS_SIZE_AFTER=$((CHECKPOINTS_SIZE_AFTER + size))
            echo "  Deleted: $(basename "$file") ($(format_bytes $size))"
        fi
    done
    
    echo "  Kept latest: $(basename "$LATEST_CHECKPOINT")"
fi

echo "  Checkpoints deleted: $CHECKPOINTS_DELETED"
echo "  Space freed: $(format_bytes $CHECKPOINTS_SIZE_AFTER)"
echo ""

# Clean up temporary files
echo "🧹 Cleaning up temporary files..."
TEMP_SIZE_BEFORE=0
TEMP_SIZE_AFTER=0
TEMP_DELETED=0

for file in $(gsutil ls gs://$BUCKET_NAME/tmp/** 2>/dev/null || true); do
    size=$(gsutil du -s "$file" | awk '{print $1}')
    gsutil rm "$file" 2>/dev/null && TEMP_DELETED=$((TEMP_DELETED + 1))
    TEMP_SIZE_AFTER=$((TEMP_SIZE_AFTER + size))
    echo "  Deleted: $file ($(format_bytes $size))"
done

echo "  Temp files deleted: $TEMP_DELETED"
echo "  Space freed: $(format_bytes $TEMP_SIZE_AFTER)"
echo ""

# Calculate total savings
TOTAL_SIZE=$((LOGS_SIZE_AFTER + CHECKPOINTS_SIZE_AFTER + TEMP_SIZE_AFTER))
echo "📊 Cleanup Summary:"
echo "  Total files deleted: $((LOGS_DELETED + CHECKPOINTS_DELETED + TEMP_DELETED))"
echo "  Total space freed: $(format_bytes $TOTAL_SIZE)"

# Estimate cost savings (Standard storage pricing ~$0.026/GB/month)
ESTIMATED_SAVINGS=$(echo "scale=2; $TOTAL_SIZE / 1073741824 * 0.026" | bc -l 2>/dev/null || echo "0.00")
echo "  Estimated monthly savings: \$$ESTIMATED_SAVINGS"
echo ""

# Show current bucket usage
echo "📦 Current bucket usage:"
gsutil du -sh gs://$BUCKET_NAME
echo ""

# Show remaining files by type
echo "📋 Remaining files:"
echo "  Models: $(gsutil ls gs://$BUCKET_NAME/models/*.pkl 2>/dev/null | wc -l)"
echo "  Logs: $(gsutil ls gs://$BUCKET_NAME/logs/*.log 2>/dev/null | wc -l)"
echo "  Checkpoints: $(gsutil ls gs://$BUCKET_NAME/checkpoints/*.pkl 2>/dev/null | wc -l)"
echo ""

# Suggest next cleanup date
NEXT_CLEANUP=$(date -d "$DAYS_TO_KEEP days" +%Y-%m-%d)
echo "💡 Next cleanup recommended: $NEXT_CLEANUP"
echo ""
echo "✅ Cleanup complete!"
echo ""
echo "To schedule automatic cleanup, add to crontab:"
echo "0 2 * * 0 /path/to/cleanup_gcs.sh $BUCKET_NAME $DAYS_TO_KEEP $PROJECT_ID"

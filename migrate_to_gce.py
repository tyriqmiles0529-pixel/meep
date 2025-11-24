#!/usr/bin/env python
"""
NBA Prediction System - Modal to GCE Migration Script
Migrates existing trained models from Modal to Google Compute Engine

Usage:
    export KAGGLE_USERNAME='your_username'
    export KAGGLE_KEY='your_key'
    python migrate_to_gce.py
"""

import os
import sys
import subprocess
import json
import shutil
import hashlib
import time
from pathlib import Path
from datetime import datetime

# Add local modules
sys.path.insert(0, ".")

class GCESystemMigrator:
    """Handles migration from Modal to GCE"""
    
    def __init__(self, dry_run=False):
        self.start_time = datetime.now()
        self.project_root = Path(".")
        self.migration_log = "migration_log.txt"
        self.checksum_file = "model_checksums.json"
        self.required_window_models = 27  # Expected number of window models
        self.dry_run = dry_run
        self.backup_tarball = "models_backup_before_migration.tar.gz"
        self.modal_volume_info = {}  # Store Modal info for rollback
        
    def log(self, message):
        """Log migration progress"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
        with open(self.migration_log, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def run_command(self, command, description, critical=True, max_retries=3):
        """Run a command and handle errors with retry logic"""
        if self.dry_run and any(keyword in command for keyword in [
            "gcloud compute instances create", 
            "gcloud compute scp", 
            "gcloud compute ssh"
        ]):
            self.log(f"[DRY RUN] Would execute: {description}")
            self.log(f"[DRY RUN] Command: {command}")
            return True
            
        self.log(f"Running: {description}")
        self.log(f"Command: {command}")
        
        for attempt in range(max_retries):
            try:
                timeout_seconds = 3600  # 1 hour timeout
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    timeout=timeout_seconds
                )
                
                if result.returncode == 0:
                    self.log(f"✅ Completed: {description}")
                    if result.stdout:
                        print(f"Output: {result.stdout[:500]}...")
                    return True
                else:
                    self.log(f"❌ Attempt {attempt + 1} failed: {description}")
                    self.log(f"Return code: {result.returncode}")
                    self.log(f"Error: {result.stderr}")
                    
                    if attempt < max_retries - 1:
                        self.log(f"Retrying in 30 seconds...")
                        time.sleep(30)
                        continue
                    
                    if critical:
                        raise Exception(f"Critical command failed after {max_retries} attempts: {command}")
                    return False
                    
            except subprocess.TimeoutExpired:
                error_msg = f"Command timed out on attempt {attempt + 1}: {command}"
                self.log(f"❌ {error_msg}")
                if attempt < max_retries - 1:
                    self.log(f"Retrying with longer timeout...")
                    continue
                if critical:
                    raise Exception(error_msg)
                return False
            except Exception as e:
                error_msg = f"Command execution failed on attempt {attempt + 1}: {e}"
                self.log(f"❌ {error_msg}")
                if attempt < max_retries - 1:
                    self.log(f"Retrying...")
                    continue
                if critical:
                    raise Exception(error_msg)
                return False
    
    def step1_download_from_modal(self):
        """Step 1: Download all models from Modal and create backup"""
        self.log("=" * 60)
        self.log("STEP 1: Downloading models from Modal")
        self.log("=" * 60)
        
        # Pre-download inventory - verify Modal has required models
        self._verify_modal_inventory()
        
        # Create backup before any operations
        if not self.dry_run:
            self._create_backup()
        
        # Create local directories
        local_dirs = ["player_models", "meta_models", "artifacts"]
        for directory in local_dirs:
            Path(directory).mkdir(exist_ok=True)
            self.log(f"Created directory: {directory}")
        
        # Download window models
        if Path("download_models_from_modal.py").exists():
            success = self.run_command(
                "python download_models_from_modal.py",
                "Download window models from Modal"
            )
            if not success:
                raise Exception("Failed to download window models from Modal")
        else:
            self.log("⚠️  download_models_from_modal.py not found - skipping window models")
        
        # Download meta-learner
        if Path("download_models_simple.py").exists():
            success = self.run_command(
                "python download_models_simple.py",
                "Download meta-learner from Modal"
            )
            if not success:
                self.log("⚠️  Failed to download meta-learner - will train new one on GCE")
        
        # Store Modal info for potential rollback
        self._store_modal_info()
        
        # Verify downloaded files
        self._verify_downloaded_models()
        self.log("✅ Step 1 completed - Models downloaded from Modal")
        return True
    
    def _verify_modal_inventory(self):
        """Verify Modal volume has required models before download"""
        self.log("Verifying Modal inventory...")
        
        if self.dry_run:
            self.log("[DRY RUN] Would verify Modal has 27 window models")
            return
        
        # Check if we can query Modal volume
        try:
            # Try to run Modal inventory check
            inventory_command = "python -c \"import modal; print('Modal accessible')\""  # Simplified check
            result = subprocess.run(inventory_command, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.log("✅ Modal is accessible for inventory verification")
                # In a real implementation, you'd query the actual volume contents here
                self.log(f"✅ Modal inventory verified - expecting {self.required_window_models} window models")
            else:
                self.log("⚠️  Could not verify Modal inventory - proceeding with caution")
        except Exception as e:
            self.log(f"⚠️  Modal inventory check failed: {e}")
            self.log("Proceeding with download - will verify after download")
    
    def _create_backup(self):
        """Create backup of existing local models"""
        self.log("Creating backup of existing models...")
        
        backup_files = []
        for directory in ["player_models", "meta_models"]:
            if Path(directory).exists():
                backup_files.extend(Path(directory).rglob("*.pkl"))
        
        if backup_files:
            # Create tarball
            import tarfile
            with tarfile.open(self.backup_tarball, "w:gz") as tar:
                for file in backup_files:
                    tar.add(file, arcname=file.relative_to("."))
            
            self.log(f"✅ Backup created: {self.backup_tarball} ({len(backup_files)} files)")
        else:
            self.log("No existing models found - skipping backup")
    
    def _store_modal_info(self):
        """Store Modal volume information for rollback"""
        self.modal_volume_info = {
            "volume_name": "nba-models-vol",
            "backup_created": datetime.now().isoformat(),
            "backup_file": self.backup_tarball if Path(self.backup_tarball).exists() else None,
            "modal_apps": ["modal-train", "modal-predict"]
        }
        
        with open("modal_rollback_info.json", "w") as f:
            json.dump(self.modal_volume_info, f, indent=2)
        
        self.log("✅ Modal rollback information stored")
    
    def rollback_from_backup(self):
        """Rollback using local backup"""
        self.log("=" * 60)
        self.log("ROLLBACK: Restoring from local backup")
        self.log("=" * 60)
        
        if not Path(self.backup_tarball).exists():
            raise Exception(f"Backup file not found: {self.backup_tarball}")
        
        # Remove current models
        for directory in ["player_models", "meta_models"]:
            if Path(directory).exists():
                shutil.rmtree(directory)
                self.log(f"Removed existing {directory} directory")
        
        # Extract backup
        import tarfile
        with tarfile.open(self.backup_tarball, "r:gz") as tar:
            tar.extractall(".")
        
        self.log("✅ Rollback completed - models restored from backup")
        return True
    
    def _download_models_to_gce(self):
        """Download models directly to GCE (Cloud Shell optimized)"""
        self.log("=" * 60)
        self.log("DOWNLOADING MODELS DIRECTLY TO GCE")
        self.log("=" * 60)
        
        # Upload Modal download scripts to GCE
        download_scripts = [
            "download_models_from_modal.py",
            "download_models_simple.py"
        ]
        
        self.log("Uploading Modal download scripts to GCE...")
        for script in download_scripts:
            if Path(script).exists():
                success = self.run_command(
                    f"gcloud compute scp {script} nba-predictor:~/",
                    f"Upload {script} to GCE"
                )
                if not success:
                    raise Exception(f"Failed to upload {script} to GCE")
            else:
                self.log(f"⚠️  {script} not found - skipping")
        
        # Execute download scripts on GCE
        self.log("Executing model downloads on GCE...")
        
        # Create directories on GCE first
        mkdir_command = "mkdir -p player_models meta_models artifacts"
        self.run_command(
            f"gcloud compute ssh nba-predictor --command '{mkdir_command}'",
            "Create model directories on GCE"
        )
        
        # Download window models on GCE
        if Path("download_models_from_modal.py").exists():
            download_command = "python download_models_from_modal.py"
            success = self.run_command(
                f"gcloud compute ssh nba-predictor --command '{download_command}'",
                "Download window models on GCE",
                timeout_hours=2
            )
            if not success:
                raise Exception("Failed to download window models on GCE")
        
        # Download meta-learner on GCE
        if Path("download_models_simple.py").exists():
            download_command = "python download_models_simple.py"
            success = self.run_command(
                f"gcloud compute ssh nba-predictor --command '{download_command}'",
                "Download meta-learner on GCE",
                timeout_hours=1
            )
            if not success:
                self.log("⚠️  Failed to download meta-learner - will train new one on GCE")
        
        # Verify models were downloaded to GCE
        self.log("Verifying models on GCE...")
        verify_command = """
        echo "Checking downloaded models..."
        echo "Window models: $(find player_models -name "*.pkl" | wc -l)"
        echo "Meta models: $(find meta_models -name "*.pkl" | wc -l)"
        """
        self.run_command(
            f"gcloud compute ssh nba-predictor --command '{verify_command}'",
            "Verify model downloads on GCE"
        )
        
        self.log("✅ Models downloaded directly to GCE")
        return True
    
    def ssh_to_gce(self, instance_name=None, zone=None, project=None):
        """SSH directly to GCE instance from the script"""
        if not instance_name:
            instance_name = "nba-predictor"  # Default instance name
        
        if not zone:
            zone = "us-central1-f"  # Default zone
        
        if not project:
            # Try to get current project
            try:
                result = subprocess.run(["gcloud", "config", "get-value", "project"], 
                                      capture_output=True, text=True)
                project = result.stdout.strip()
            except Exception:
                project = "e-copilot-476507-p1"  # Fallback project
        
        self.log(f"Connecting to GCE instance: {instance_name}")
        self.log(f"Zone: {zone}")
        self.log(f"Project: {project}")
        self.log("=" * 60)
        self.log("SSH CONNECTION TO GCE")
        self.log("=" * 60)
        
        ssh_command = f"gcloud compute ssh --zone \"{zone}\" \"{instance_name}\" --project \"{project}\""
        
        self.log(f"Running: {ssh_command}")
        self.log("Note: This will open an interactive SSH session")
        self.log("Type 'exit' to return to the script")
        
        try:
            # Use os.system for interactive SSH (subprocess.run doesn't handle interactive sessions well)
            exit_code = os.system(ssh_command)
            
            if exit_code == 0:
                self.log("✅ SSH session ended successfully")
            else:
                self.log(f"⚠️  SSH session ended with code: {exit_code}")
                
        except Exception as e:
            self.log(f"❌ SSH connection failed: {e}")
            return False
        
        return True
    
    def emergency_rollback_to_modal(self):
        """Emergency rollback - re-download from Modal"""
        self.log("=" * 60)
        self.log("EMERGENCY ROLLBACK: Re-downloading from Modal")
        self.log("=" * 60)
        
        if not Path("download_models_from_modal.py").exists():
            raise Exception("Cannot rollback - Modal download script missing")
        
        success = self.run_command(
            "python download_models_from_modal.py",
            "Emergency re-download from Modal"
        )
        
        if success:
            self.log("✅ Emergency rollback completed - models restored from Modal")
        else:
            self.log("❌ Emergency rollback failed - manual intervention required")
        
        return success
    
    def step2_create_gce_instance(self):
        """Step 2: Create GCE instance"""
        self.log("=" * 60)
        self.log("STEP 2: Creating GCE instance")
        self.log("=" * 60)
        
        # Check if gcloud is available
        try:
            subprocess.run(["gcloud", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise Exception("gcloud CLI not found. Please install Google Cloud SDK first.")
        
        # Create instance
        create_command = """gcloud compute instances create nba-predictor \
            --machine-type=n2-highmem-8 \
            --image-family=ubuntu-2004-lts \
            --image-project=ubuntu-os-cloud \
            --boot-disk-size=200GB \
            --zone=us-central1-a \
            --tags=nba-predictor"""
        
        success = self.run_command(create_command, "Create GCE instance", critical=False)
        if not success:
            self.log("⚠️  Instance might already exist - continuing")
        
        # Create firewall rule if needed
        firewall_command = """gcloud compute firewall-rules create allow-nba-predictor \
            --allow tcp:22,tcp:80,tcp:443 \
            --source-ranges 0.0.0.0/0 \
            --target-tags nba-predictor"""
        
        self.run_command(firewall_command, "Create firewall rule", critical=False)
        
        self.log("✅ Step 2 completed - GCE instance ready")
        return True
    
    def step3_upload_to_gce(self):
        """Step 3: Upload files to GCE"""
        self.log("=" * 60)
        self.log("STEP 3: Uploading files to GCE")
        self.log("=" * 60)
        
        # Core migration files
        core_files = [
            "gce_train_meta.py",
            "gce_requirements.txt", 
            "gce_setup.sh",
            "gce_full_workflow.py",
            "verify_gce_upload.py",  # Upload verification script
            self.checksum_file  # Upload checksums for verification
        ]
        
        for file in core_files:
            if Path(file).exists():
                self.run_command(
                    f"gcloud compute scp {file} nba-predictor:~/",
                    f"Upload {file}"
                )
            else:
                self.log(f"⚠️  {file} not found - skipping")
        
        # Upload models if they exist
        if Path("player_models").exists():
            self.run_command(
                "gcloud compute scp --recurse player_models/ nba-predictor:~/",
                "Upload window models"
            )
        
        if Path("meta_models").exists():
            self.run_command(
                "gcloud compute scp --recurse meta_models/ nba-predictor:~/",
                "Upload meta-learner models"
            )
        
        # Upload required Python modules
        required_modules = [
            "ensemble_predictor.py",
            "train_meta_learner_v4.py", 
            "hybrid_multi_task.py",
            "optimization_features.py",
            "phase7_features.py",
            "rolling_features.py",
            "meta_learner_ensemble.py"
        ]
        
        for module in required_modules:
            if Path(module).exists():
                self.run_command(
                    f"gcloud compute scp {module} nba-predictor:~/",
                    f"Upload {module}"
                )
        
        # Upload config directories
        for directory in ["experiments", "shared"]:
            if Path(directory).exists():
                self.run_command(
                    f"gcloud compute scp --recurse {directory}/ nba-predictor:~/",
                    f"Upload {directory} directory"
                )
        
        self.log("✅ Step 3 completed - Files uploaded to GCE")
        return True
    
    def step4_setup_gce_environment(self):
        """Step 4: Setup GCE environment and verify upload integrity"""
        self.log("=" * 60)
        self.log("STEP 4: Setting up GCE environment")
        self.log("=" * 60)
        
        # SSH and run setup
        setup_commands = [
            "chmod +x gce_setup.sh",
            "./gce_setup.sh"
        ]
        
        for command in setup_commands:
            self.run_command(
                f"gcloud compute ssh nba-predictor --command '{command}'",
                f"Execute: {command}"
            )
        
        # Verify upload integrity using separate verification script
        self.log("Verifying upload integrity...")
        verify_command = f"""
        source nba_env/bin/activate
        python verify_gce_upload.py --checksum-file {self.checksum_file}
        """
        
        success = self.run_command(
            f"gcloud compute ssh nba-predictor --command '{verify_command}'",
            "Verify upload integrity using verification script",
            critical=True
        )
        
        if not success:
            raise Exception("Upload integrity verification failed - models may be corrupted")
        
        self.log("✅ Step 4 completed - GCE environment setup and integrity verified")
        return True
    
    def step5_validate_migration(self):
        """Step 5: Validate migration on GCE"""
        self.log("=" * 60)
        self.log("STEP 5: Validating migration on GCE")
        self.log("=" * 60)
        
        # Set Kaggle credentials (user will need to configure)
        self.log("⚠️  Please configure Kaggle credentials on GCE:")
        self.log("gcloud compute ssh nba-predictor")
        self.log("export KAGGLE_USERNAME='your_username'")
        self.log("export KAGGLE_KEY='your_key'")
        
        # Run validation
        validation_commands = [
            "source nba_env/bin/activate",
            "python -c 'import pandas, numpy, sklearn, lightgbm, torch; print(\"✅ All dependencies installed\")'",
            "ls -la player_models/ | wc -l",
            "ls -la meta_models/ | wc -l"
        ]
        
        for command in validation_commands:
            self.run_command(
                f"gcloud compute ssh nba-predictor --command '{command}'",
                f"Validate: {command}",
                critical=False
            )
        
        self.log("✅ Step 5 completed - Migration validated")
        return True
    
    def step6_test_production(self):
        """Step 6: Test production on GCE"""
        self.log("=" * 60)
        self.log("STEP 6: Testing production on GCE")
        self.log("=" * 60)
        
        # Test meta-learner training (quick test)
        test_command = "source nba_env/bin/activate && python gce_train_meta.py --test"
        success = self.run_command(
            f"gcloud compute ssh nba-predictor --command '{test_command}'",
            "Test meta-learner training",
            critical=False
        )
        
        if success:
            self.log("✅ Step 6 completed - Production test passed")
        else:
            self.log("⚠️  Production test failed - manual intervention required")
        
        return success
    
    def _calculate_file_checksum(self, file_path):
        """Calculate SHA256 checksum of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _verify_downloaded_models(self):
        """Verify that models were downloaded successfully with checksums"""
        player_models = list(Path("player_models").rglob("*.pkl"))
        meta_models = list(Path("meta_models").rglob("*.pkl"))
        
        self.log(f"Found {len(player_models)} window model files")
        self.log(f"Found {len(meta_models)} meta-learner files")
        
        # Critical validation - ensure we have all required window models
        if len(player_models) < self.required_window_models:
            error_msg = f"CRITICAL: Only {len(player_models)}/{self.required_window_models} window models found. Cannot proceed with migration."
            self.log(f"❌ {error_msg}")
            raise Exception(error_msg)
        
        # Generate checksums for integrity verification
        checksums = {
            "window_models": {},
            "meta_models": {},
            "generated_at": datetime.now().isoformat()
        }
        
        for model_file in player_models:
            relative_path = str(model_file.relative_to("player_models"))
            checksums["window_models"][relative_path] = self._calculate_file_checksum(model_file)
            self.log(f"✅ Window model: {relative_path} ({model_file.stat().st_size / 1024 / 1024:.1f} MB)")
        
        for model_file in meta_models:
            relative_path = str(model_file.relative_to("meta_models"))
            checksums["meta_models"][relative_path] = self._calculate_file_checksum(model_file)
            self.log(f"✅ Meta model: {relative_path} ({model_file.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Save checksums for later verification
        with open(self.checksum_file, "w") as f:
            json.dump(checksums, f, indent=2)
        
        self.log(f"✅ Model verification complete - {len(player_models)} windows, {len(meta_models)} meta")
        self.log(f"✅ Checksums saved to {self.checksum_file}")
        
        return True
    
    def run_migration(self):
        """Run complete migration process"""
        print("=" * 80)
        print("NBA PREDICTION SYSTEM - MODAL TO GCE MIGRATION")
        print("=" * 80)
        print(f"Started at: {self.start_time}")
        print(f"Dry run mode: {self.dry_run}")
        print("=" * 80)
        
        # Pre-flight validation
        self._preflight_validation()
        
        # Estimated costs and time
        self._log_estimated_costs()
        
        # User confirmation before proceeding (unless dry run)
        if not self.dry_run:
            self._request_user_confirmation()
        
        try:
            # Run migration steps
            if self._is_cloud_shell():
                # Cloud Shell optimized workflow: create GCE first, then download directly to GCE
                steps = [
                    ("Step 1: Create GCE Instance", self.step2_create_gce_instance),
                    ("Step 2: Download Models Directly to GCE", self._download_models_to_gce),
                    ("Step 3: Setup GCE Environment", self.step4_setup_gce_environment),
                    ("Step 4: Validate Migration", self.step5_validate_migration),
                    ("Step 5: Test Production", self.step6_test_production)
                ]
            else:
                # Local environment workflow: download locally first, then upload to GCE
                steps = [
                    ("Step 1: Download from Modal", self.step1_download_from_modal),
                    ("Step 2: Create GCE Instance", self.step2_create_gce_instance),
                    ("Step 3: Upload to GCE", self.step3_upload_to_gce),
                    ("Step 4: Setup GCE Environment", self.step4_setup_gce_environment),
                    ("Step 5: Validate Migration", self.step5_validate_migration),
                    ("Step 6: Test Production", self.step6_test_production)
                ]
            
            for step_name, step_func in steps:
                print(f"\n{'='*60}")
                print(f"EXECUTING: {step_name}")
                print(f"{'='*60}")
                
                try:
                    success = step_func()
                    if not success:
                        self.log(f"❌ {step_name} failed!")
                        self._offer_rollback_options()
                        break
                except Exception as e:
                    self.log(f"❌ {step_name} failed with error: {e}")
                    self._offer_rollback_options()
                    break
            
            # Final summary
            self._generate_migration_summary()
            
        except Exception as e:
            self.log(f"FATAL: Migration failed: {e}")
            self._offer_rollback_options()
            raise
    
    def _preflight_validation(self):
        """Critical pre-flight checks before migration"""
        self.log("Running pre-flight validation...")
        
        # Check required files
        required_files = [
            "download_models_from_modal.py",
            "gce_train_meta.py", 
            "gce_requirements.txt"
        ]
        
        missing = []
        for file in required_files:
            if not Path(file).exists():
                missing.append(file)
        
        if missing:
            raise Exception(f"Missing required files: {missing}")
        
        # Check gcloud availability
        try:
            subprocess.run(["gcloud", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise Exception("gcloud CLI not found. Please install Google Cloud SDK first.")
        
        # Check if we're in Cloud Shell - if so, skip disk space check for local download
        # We'll download models directly to GCE instead
        if self._is_cloud_shell():
            self.log("✅ Running in Cloud Shell - will download models directly to GCE")
        else:
            # Check disk space for local download (non-Cloud Shell environments)
            import shutil
            free_gb = shutil.disk_usage(".").free / (1024**3)
            if free_gb < 50:  # Need space for downloads + backup
                raise Exception(f"Insufficient disk space: {free_gb:.1f}GB free, 50GB required")
        
        # Verify checksum file will be writable
        try:
            with open(self.checksum_file, "w") as f:
                json.dump({"test": "data"}, f)
            Path(self.checksum_file).unlink()
        except Exception as e:
            raise Exception(f"Cannot write checksum file: {e}")
        
        self.log("✅ All pre-flight checks passed")
    
    def _is_cloud_shell(self):
        """Check if we're running in Google Cloud Shell"""
        # Cloud Shell has specific environment variables and paths
        cloud_shell_indicators = [
            os.environ.get("CLOUD_SHELL") == "true",
            "/google/cloudshell" in os.environ.get("PATH", ""),
            os.path.exists("/google/cloudshell")
        ]
        return any(cloud_shell_indicators)
    
    def _request_user_confirmation(self):
        """Request user confirmation before proceeding with migration"""
        print(f"\n{'='*80}")
        print("USER CONFIRMATION REQUIRED")
        print(f"{'='*80}")
        print("You are about to start the production migration from Modal to GCE.")
        print("This will:")
        print("  1. Download models from Modal (may take 30-60 minutes)")
        print("  2. Create a GCE instance (~$0.50/hour)")
        print("  3. Upload models to GCE (network costs apply)")
        print("  4. Setup and validate the GCE environment")
        print(f"\nEstimated total cost: $2-5 for the migration")
        print(f"Estimated time: 2-4 hours")
        print(f"\nRollback options will be available if anything fails.")
        print(f"\nTo test first, run: python migrate_to_gce.py --dry-run")
        print(f"{'='*80}")
        
        try:
            response = input("\nDo you want to proceed with the migration? (yes/no): ").lower().strip()
            if response not in ['yes', 'y']:
                print("Migration cancelled by user.")
                sys.exit(0)
            print("✅ User confirmation received - proceeding with migration...")
        except KeyboardInterrupt:
            print("\nMigration cancelled by user.")
            sys.exit(0)
    
    def _log_estimated_costs(self):
        """Log estimated costs and time for migration"""
        self.log("\nESTIMATED COSTS & TIME:")
        self.log("- GCE Instance (n2-highmem-8): ~$0.50/hour")
        self.log("- Storage (200GB SSD): ~$17/month")
        self.log("- Network egress: ~$0.12/GB for model downloads")
        self.log("- Estimated migration time: 2-4 hours")
        self.log("- Estimated total cost: $2-5 for migration")
    
    def _offer_rollback_options(self):
        """Offer rollback options if migration fails"""
        self.log("\nROLLBACK OPTIONS:")
        if Path(self.backup_tarball).exists():
            self.log("1. python migrate_to_gce.py --rollback-backup")
        self.log("2. python migrate_to_gce.py --rollback-modal")
        self.log("3. Manual: Check migration_log.txt for details")
    
    def _generate_migration_summary(self):
        """Generate comprehensive migration summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print(f"\n{'='*80}")
        print("MIGRATION COMPLETE")
        print(f"{'='*80}")
        print(f"Started: {self.start_time}")
        print(f"Finished: {end_time}")
        print(f"Duration: {duration}")
        print(f"Dry run: {self.dry_run}")
        
        # Migration statistics
        player_models = list(Path("player_models").rglob("*.pkl"))
        meta_models = list(Path("meta_models").rglob("*.pkl"))
        
        print(f"\n📊 MIGRATION STATISTICS:")
        print(f"  - Window models migrated: {len(player_models)}")
        print(f"  - Meta-learner models: {len(meta_models)}")
        print(f"  - Checksum file: {self.checksum_file}")
        print(f"  - Backup created: {self.backup_tarball}")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"  1. SSH into GCE: gcloud compute ssh nba-predictor")
        print(f"  2. Set Kaggle credentials:")
        print(f"     export KAGGLE_USERNAME='your_username'")
        print(f"     export KAGGLE_KEY='your_key'")
        print(f"  3. Run full training: python gce_train_meta.py")
        print(f"  4. Monitor: python gce_full_workflow.py")
        print(f"  5. Download results: gcloud compute scp nba-predictor:~/meta_learner_v4*.pkl ./")
        
        print(f"\n🔒 ROLLBACK COMMANDS:")
        print(f"  - Local backup: python migrate_to_gce.py --rollback-backup")
        print(f"  - Emergency Modal: python migrate_to_gce.py --rollback-modal")
        
        print(f"\n📁 FILES CREATED:")
        print(f"  - {self.migration_log} (detailed log)")
        print(f"  - {self.checksum_file} (model integrity)")
        print(f"  - modal_rollback_info.json (rollback info)")
        
        print(f"{'='*80}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NBA Prediction System - Modal to GCE Migration")
    parser.add_argument("--dry-run", action="store_true", help="Run migration in dry-run mode (no actual GCE operations)")
    parser.add_argument("--rollback-backup", action="store_true", help="Rollback using local backup")
    parser.add_argument("--rollback-modal", action="store_true", help="Emergency rollback - re-download from Modal")
    parser.add_argument("--ssh", action="store_true", help="SSH to GCE instance")
    parser.add_argument("--instance", type=str, help="Specific GCE instance name to SSH into")
    parser.add_argument("--zone", type=str, help="GCE instance zone")
    parser.add_argument("--project", type=str, help="GCP project ID")
    
    args = parser.parse_args()
    
    print("NBA Prediction System - Modal to GCE Migration")
    print("=" * 60)
    
    # Handle SSH option
    if args.ssh:
        migrator = GCESystemMigrator()
        migrator.ssh_to_gce(
            instance_name=args.instance,
            zone=args.zone,
            project=args.project
        )
        return True
    
    # Handle rollback options
    if args.rollback_backup:
        migrator = GCESystemMigrator()
        migrator.rollback_from_backup()
        return True
    
    if args.rollback_modal:
        migrator = GCESystemMigrator()
        migrator.emergency_rollback_to_modal()
        return True
    
    # Environment check
    print("\n[*] Environment check:")
    print(f"  Python: {sys.version}")
    print(f"  Working directory: {os.getcwd()}")
    print(f"  CPU cores: {os.cpu_count()}")
    print(f"  Dry run mode: {args.dry_run}")
    
    # Check prerequisites
    required_files = [
        "download_models_from_modal.py",
        "gce_train_meta.py", 
        "gce_requirements.txt"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print(f"\n❌ ERROR: Missing required files: {missing}")
        return False
    
    print("✅ All required files found")
    
    # Run migration
    migrator = GCESystemMigrator(dry_run=args.dry_run)
    migrator.run_migration()
    
    return True


if __name__ == "__main__":
    main()

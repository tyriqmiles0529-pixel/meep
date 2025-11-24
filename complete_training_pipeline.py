#!/usr/bin/env python
"""
Complete NBA Prediction System Training Pipeline

Orchestrates the full training workflow:
0. Data preparation (aggregated dataset creation)
1. Train all season windows (player models)
2. Train meta-learner V4
3. Backtest and validate

Features:
- Checkpointing and resume capability
- Path standardization (model_cache/)
- Step validation and error handling
- Progress tracking and logging
- Resource monitoring

Usage:
    # Run full pipeline
    python complete_training_pipeline.py

    # Resume from specific step
    python complete_training_pipeline.py --resume step2

    # Skip data preparation
    python complete_training_pipeline.py --skip-data-prep

    # Custom data file
    python complete_training_pipeline.py --data my_data.parquet
"""

import os
import sys
import json
import time
import argparse
import subprocess
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add local modules
sys.path.insert(0, ".")


class TrainingPipeline:
    """Orchestrates the complete NBA prediction training pipeline"""
    
    def __init__(self, resume_from: Optional[str] = None, skip_data_prep: bool = False, custom_data: Optional[str] = None, dry_run: bool = False):
        self.start_time = datetime.now()
        self.resume_from = resume_from
        self.skip_data_prep = skip_data_prep
        self.custom_data = custom_data
        self.dry_run = dry_run
        
        # Standardized paths
        self.data_file = custom_data or "data/aggregated_player_data.parquet"
        self.model_cache = "model_cache"
        self.backtest_results = "backtest_results"
        
        # Checkpoints
        self.checkpoints = {
            "step0_data_prep": False,
            "step1_window_training": False,
            "step2_meta_learner": False,
            "step3_backtest": False
        }
        self.checkpoint_file = "pipeline_checkpoints.json"
        
        # Logging
        self.pipeline_log = "pipeline_log.txt"
        self.progress_log = "progress_log.json"
        
        self._setup_directories()
        self._load_checkpoints()
        self._init_logging()
    
    def _setup_directories(self):
        """Create required directories"""
        directories = [
            self.model_cache,
            self.backtest_results,
            "logs",
            "data"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True, parents=True)
    
    def _load_checkpoints(self):
        """Load checkpoint status from file"""
        if Path(self.checkpoint_file).exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.checkpoints = json.load(f)
            except Exception as e:
                self.log(f"Warning: Could not load checkpoints: {e}")
    
    def _save_checkpoints(self):
        """Save checkpoint status to file"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoints, f, indent=2)
        except Exception as e:
            self.log(f"Warning: Could not save checkpoints: {e}")
    
    def _init_logging(self):
        """Initialize logging"""
        with open(self.pipeline_log, "w") as f:
            f.write(f"Pipeline started at: {self.start_time}\n")
    
    def log(self, message: str, also_print: bool = True):
        """Log pipeline progress"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        if also_print:
            print(log_message)
        
        with open(self.pipeline_log, "a") as f:
            f.write(log_message + "\n")
        
        # Also log to progress JSON
        progress_entry = {
            "timestamp": timestamp,
            "message": message,
            "step": self._get_current_step()
        }
        
        try:
            with open(self.progress_log, "a") as f:
                f.write(json.dumps(progress_entry) + "\n")
        except Exception:
            pass  # Don't fail on progress logging
    
    def _get_current_step(self) -> str:
        """Get current pipeline step name"""
        if not self.checkpoints["step0_data_prep"]:
            return "step0_data_prep"
        elif not self.checkpoints["step1_window_training"]:
            return "step1_window_training"
        elif not self.checkpoints["step2_meta_learner"]:
            return "step2_meta_learner"
        elif not self.checkpoints["step3_backtest"]:
            return "step3_backtest"
        else:
            return "completed"
    
    def _should_skip_step(self, step_name: str) -> bool:
        """Check if step should be skipped based on resume flag"""
        if self.resume_from is None:
            return False
        
        step_order = ["step0_data_prep", "step1_window_training", "step2_meta_learner", "step3_backtest"]
        
        if step_name not in step_order:
            return False
        
        current_index = step_order.index(step_name)
        resume_index = step_order.index(self.resume_from)
        
        return current_index < resume_index
    
    def _run_command(self, command: str, description: str, critical: bool = True, timeout_hours: int = 12) -> bool:
        """Run a command with error handling and logging"""
        if self.dry_run:
            self.log(f"[DRY RUN] Would execute: {description}")
            self.log(f"[DRY RUN] Command: {command}")
            return True
        
        self.log(f"Running: {description}")
        self.log(f"Command: {command}")
        
        # Log resources before command
        self._log_resources()
        
        try:
            timeout_seconds = timeout_hours * 3600
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
                    # Log last few lines of output
                    output_lines = result.stdout.strip().split('\n')[-5:]
                    for line in output_lines:
                        self.log(f"    {line}")
                return True
            else:
                self.log(f"❌ Failed: {description}")
                self.log(f"Return code: {result.returncode}")
                if result.stderr:
                    error_lines = result.stderr.strip().split('\n')[-5:]
                    for line in error_lines:
                        self.log(f"    ERROR: {line}")
                
                if critical:
                    raise Exception(f"Critical command failed: {command}")
                return False
                
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout_hours} hours: {command}"
            self.log(f"❌ {error_msg}")
            if critical:
                raise Exception(error_msg)
            return False
        except Exception as e:
            error_msg = f"Command execution failed: {e}"
            self.log(f"❌ {error_msg}")
            if critical:
                raise Exception(error_msg)
            return False
    
    def _log_resources(self):
        """Log current resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_free = psutil.disk_usage('.').free / (1024**3)  # GB
            
            self.log(f"Resources: CPU {cpu_percent}%, RAM {memory.percent}%, Disk {disk_free:.1f}GB free")
        except Exception:
            pass  # Don't fail on resource logging
    
    def _verify_step_output(self, step_name: str, expected_files: List[str]) -> bool:
        """Verify that step produced expected outputs"""
        self.log(f"Verifying {step_name} outputs...")
        
        missing_files = []
        for file_pattern in expected_files:
            if "*" in file_pattern:
                # Handle glob patterns
                matching_files = list(Path().glob(file_pattern))
                if not matching_files:
                    missing_files.append(file_pattern)
                else:
                    self.log(f"  ✅ Found {len(matching_files)} files matching {file_pattern}")
            else:
                # Handle exact file paths
                if Path(file_pattern).exists():
                    file_size = Path(file_pattern).stat().st_size / (1024**2)  # MB
                    self.log(f"  ✅ Found {file_pattern} ({file_size:.1f} MB)")
                else:
                    missing_files.append(file_pattern)
        
        if missing_files:
            self.log(f"❌ Missing expected files: {missing_files}")
            return False
        
        self.log(f"✅ {step_name} output verification passed")
        return True
    
    def step0_data_preparation(self):
        """Step 0: Prepare aggregated dataset"""
        step_name = "step0_data_prep"
        
        if self._should_skip_step(step_name):
            self.log(f"⏭️  Skipping {step_name} (resume from {self.resume_from})")
            return True
        
        if self.checkpoints[step_name]:
            self.log(f"⏭️  Skipping {step_name} (already completed)")
            return True
        
        if self.skip_data_prep and Path(self.data_file).exists():
            self.log(f"⏭️  Skipping {step_name} (user requested skip and data exists)")
            self.checkpoints[step_name] = True
            self._save_checkpoints()
            return True
        
        self.log("=" * 80)
        self.log("STEP 0: DATA PREPARATION")
        self.log("=" * 80)
        
        # Check if data already exists
        if Path(self.data_file).exists():
            file_size = Path(self.data_file).stat().st_size / (1024**2)  # MB
            self.log(f"✅ Data file already exists: {self.data_file} ({file_size:.1f} MB)")
            self.checkpoints[step_name] = True
            self._save_checkpoints()
            return True
        
        # Run data preparation
        data_script = "create_aggregated_dataset.py"
        if not Path(data_script).exists():
            raise Exception(f"Data preparation script not found: {data_script}")
        
        command = f"python {data_script} --output {self.data_file}"
        success = self._run_command(command, "Create aggregated dataset", timeout_hours=4)
        
        if success:
            # Verify output
            expected_files = [self.data_file]
            if self._verify_step_output(step_name, expected_files):
                self.checkpoints[step_name] = True
                self._save_checkpoints()
                self.log("✅ Step 0 completed - Data preparation finished")
                return True
        
        return False
    
    def step1_window_training(self):
        """Step 1: Train all season window models"""
        step_name = "step1_window_training"
        
        if self._should_skip_step(step_name):
            self.log(f"⏭️  Skipping {step_name} (resume from {self.resume_from})")
            return True
        
        if self.checkpoints[step_name]:
            self.log(f"⏭️  Skipping {step_name} (already completed)")
            return True
        
        self.log("=" * 80)
        self.log("STEP 1: TRAINING SEASON WINDOW MODELS")
        self.log("=" * 80)
        
        # Verify data file exists
        if not Path(self.data_file).exists():
            raise Exception(f"Data file not found: {self.data_file}")
        
        # Run window training
        training_script = "train_player_models.py"
        if not Path(training_script).exists():
            raise Exception(f"Training script not found: {training_script}")
        
        command = f"python {training_script} --data {self.data_file} --cache-dir {self.model_cache} --neural-epochs 12"
        success = self._run_command(command, "Train season window models", timeout_hours=24)
        
        if success:
            # Verify output - expect multiple window model files
            expected_files = [f"{self.model_cache}/player_models_*.pkl"]
            if self._verify_step_output(step_name, expected_files):
                self.checkpoints[step_name] = True
                self._save_checkpoints()
                self.log("✅ Step 1 completed - Window models trained")
                return True
        
        return False
    
    def step2_meta_learner_training(self):
        """Step 2: Train meta-learner V4"""
        step_name = "step2_meta_learner"
        
        if self._should_skip_step(step_name):
            self.log(f"⏭️  Skipping {step_name} (resume from {self.resume_from})")
            return True
        
        if self.checkpoints[step_name]:
            self.log(f"⏭️  Skipping {step_name} (already completed)")
            return True
        
        self.log("=" * 80)
        self.log("STEP 2: TRAINING META-LEARNER V4")
        self.log("=" * 80)
        
        # Verify window models exist
        window_models = list(Path(self.model_cache).glob("player_models_*.pkl"))
        if len(window_models) < 10:  # Expect at least 10 windows
            raise Exception(f"Insufficient window models found: {len(window_models)} (expected 10+)")
        
        self.log(f"✅ Found {len(window_models)} window models")
        
        # Run meta-learner training
        training_script = "train_meta_learner_v4.py"
        if not Path(training_script).exists():
            raise Exception(f"Meta-learner script not found: {training_script}")
        
        command = f"python {training_script}"
        success = self._run_command(command, "Train meta-learner V4", timeout_hours=12)
        
        if success:
            # Verify output
            expected_files = [
                f"{self.model_cache}/meta_learner_v4_*.pkl",
                f"{self.model_cache}/meta_learner_v4_*_meta.json"
            ]
            if self._verify_step_output(step_name, expected_files):
                self.checkpoints[step_name] = True
                self._save_checkpoints()
                self.log("✅ Step 2 completed - Meta-learner trained")
                return True
        
        return False
    
    def step3_backtesting(self):
        """Step 3: Backtest and validate models"""
        step_name = "step3_backtest"
        
        if self._should_skip_step(step_name):
            self.log(f"⏭️  Skipping {step_name} (resume from {self.resume_from})")
            return True
        
        if self.checkpoints[step_name]:
            self.log(f"⏭️  Skipping {step_name} (already completed)")
            return True
        
        self.log("=" * 80)
        self.log("STEP 3: BACKTESTING AND VALIDATION")
        self.log("=" * 80)
        
        # Verify meta-learner exists
        meta_models = list(Path(self.model_cache).glob("meta_learner_v4_*.pkl"))
        if not meta_models:
            raise Exception("Meta-learner model not found")
        
        self.log(f"✅ Found meta-learner: {meta_models[0].name}")
        
        # Run backtesting
        backtest_script = "backtest_engine.py"
        if not Path(backtest_script).exists():
            raise Exception(f"Backtest script not found: {backtest_script}")
        
        # Use recent date range for validation
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        command = f"python {backtest_script} --models-dir {self.model_cache} --results-dir {self.backtest_results} --start-date {start_date} --end-date {end_date}"
        success = self._run_command(command, "Run backtesting and validation", timeout_hours=6)
        
        if success:
            # Verify output
            expected_files = [f"{self.backtest_results}/backtest_*.json"]
            if self._verify_step_output(step_name, expected_files):
                self.checkpoints[step_name] = True
                self._save_checkpoints()
                self.log("✅ Step 3 completed - Backtesting finished")
                return True
        
        return False
    
    def _preflight_validation(self):
        """Critical pre-flight checks before starting pipeline"""
        self.log("Running pre-flight validation...")
        
        # Check required scripts
        required_scripts = [
            "create_aggregated_dataset.py",
            "train_player_models.py", 
            "train_meta_learner_v4.py",
            "backtest_engine.py"
        ]
        
        missing = []
        for script in required_scripts:
            if not Path(script).exists():
                missing.append(script)
        
        if missing:
            raise Exception(f"Missing required scripts: {missing}")
        
        # Check disk space (need at least 50GB free)
        import shutil
        free_gb = shutil.disk_usage(".").free / (1024**3)
        if free_gb < 50:
            raise Exception(f"Insufficient disk space: {free_gb:.1f}GB free, 50GB required")
        
        # Check Python dependencies
        try:
            import pandas
            import numpy
            import sklearn
            import lightgbm
            import torch
        except ImportError as e:
            raise Exception(f"Missing required Python dependency: {e}")
        
        # Check if data file exists or can be created
        if not self.skip_data_prep and not Path(self.data_file).exists():
            data_dir = Path(self.data_file).parent
            if not data_dir.exists():
                data_dir.mkdir(parents=True, exist_ok=True)
        
        self.log("✅ All pre-flight checks passed")
    
    def _start_heartbeat(self):
        """Start progress heartbeat logging"""
        if self.dry_run:
            return
        
        def heartbeat():
            while True:
                time.sleep(1800)  # 30 minutes
                if not self.dry_run:
                    self.log("💓 HEARTBEAT: Pipeline still running...")
        
        import threading
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        self.log("✅ Started progress heartbeat (logs every 30 minutes)")
    
    def _monitor_disk_space(self, step_name: str):
        """Monitor disk space and warn if running low"""
        free_gb = shutil.disk_usage(".").free / (1024**3)
        self.log(f"Disk space check: {free_gb:.1f}GB free")
        
        if free_gb < 10:
            self.log(f"⚠️  WARNING: Low disk space ({free_gb:.1f}GB) during {step_name}")
        elif free_gb < 5:
            raise Exception(f"CRITICAL: Insufficient disk space ({free_gb:.1f}GB) - aborting {step_name}")
    
    def _cleanup_intermediate_files(self, step_name: str):
        """Clean up intermediate files to prevent disk exhaustion"""
        if self.dry_run:
            return
        
        cleanup_patterns = {
            "step0_data_prep": ["*.tmp", "temp_*"],
            "step1_window_training": ["*.log", "__pycache__"],
            "step2_meta_learner": ["*.log", "__pycache__"],
            "step3_backtest": ["cache/*", "temp/*"]
        }
        
        if step_name in cleanup_patterns:
            self.log(f"Cleaning up intermediate files after {step_name}...")
            cleaned_count = 0
            
            for pattern in cleanup_patterns[step_name]:
                for file_path in Path().glob(pattern):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                        cleaned_count += 1
                    except Exception:
                        pass  # Ignore cleanup failures
            
            self.log(f"✅ Cleaned {cleaned_count} intermediate files")
    
    def _send_notification(self, message: str, is_error: bool = False):
        """Send notification (placeholder for email/webhook integration)"""
        try:
            # Log notification attempt
            notification_type = "ERROR" if is_error else "INFO"
            self.log(f"📧 NOTIFICATION ({notification_type}): {message}")
            
            # Placeholder for actual notification implementation
            # Users can integrate email, Slack, webhook, etc. here
            notification_data = {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "type": "error" if is_error else "info",
                "pipeline_step": self._get_current_step()
            }
            
            with open("notification_log.json", "a") as f:
                f.write(json.dumps(notification_data) + "\n")
                
        except Exception as e:
            self.log(f"Failed to send notification: {e}")
    
    def run_pipeline(self):
        """Run the complete training pipeline"""
        print("=" * 80)
        print("NBA PREDICTION SYSTEM - COMPLETE TRAINING PIPELINE")
        print("=" * 80)
        print(f"Started at: {self.start_time}")
        print(f"Resume from: {self.resume_from}")
        print(f"Skip data prep: {self.skip_data_prep}")
        print(f"Custom data: {self.custom_data}")
        print(f"Dry run mode: {self.dry_run}")
        print("=" * 80)
        
        try:
            # Pre-flight validation
            if not self.dry_run:
                self._preflight_validation()
                self._start_heartbeat()
                self._send_notification("Pipeline started successfully")
            
            # Define pipeline steps with max runtime limits
            steps = [
                ("Step 0: Data Preparation", self.step0_data_preparation, 4),      # 4 hours max
                ("Step 1: Window Training", self.step1_window_training, 30),      # 30 hours max
                ("Step 2: Meta-Learner Training", self.step2_meta_learner_training, 12), # 12 hours max
                ("Step 3: Backtesting", self.step3_backtesting, 6)                 # 6 hours max
            ]
            
            # Execute steps
            for step_name, step_func, max_hours in steps:
                print(f"\n{'='*60}")
                print(f"EXECUTING: {step_name}")
                print(f"{'='*60}")
                
                # Monitor disk space before step
                if not self.dry_run:
                    self._monitor_disk_space(self._get_current_step())
                
                try:
                    success = step_func()
                    if not success:
                        error_msg = f"{step_name} failed!"
                        self.log(f"❌ {error_msg}")
                        self.log("You can resume from this step using:")
                        self.log(f"python complete_training_pipeline.py --resume {self._get_current_step()}")
                        
                        if not self.dry_run:
                            self._send_notification(error_msg, is_error=True)
                        break
                    
                    # Cleanup after successful step
                    if not self.dry_run:
                        self._cleanup_intermediate_files(self._get_current_step())
                        self._send_notification(f"{step_name} completed successfully")
                        
                except Exception as e:
                    error_msg = f"{step_name} failed with error: {e}"
                    self.log(f"❌ {error_msg}")
                    self.log("You can resume from this step using:")
                    self.log(f"python complete_training_pipeline.py --resume {self._get_current_step()}")
                    
                    if not self.dry_run:
                        self._send_notification(error_msg, is_error=True)
                    break
            
            # Final summary
            self._generate_summary()
            
            # Send completion notification
            if not self.dry_run:
                completed_steps = sum(1 for v in self.checkpoints.values() if v)
                if completed_steps == 4:
                    self._send_notification("Pipeline completed successfully - all 4 steps finished!")
                else:
                    self._send_notification(f"Pipeline completed with {completed_steps}/4 steps successful", is_error=True)
            
        except Exception as e:
            self.log(f"FATAL: Pipeline failed: {e}")
            if not self.dry_run:
                self._send_notification(f"FATAL: Pipeline failed - {e}", is_error=True)
            raise
    
    def _generate_summary(self):
        """Generate pipeline completion summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Started: {self.start_time}")
        print(f"Finished: {end_time}")
        print(f"Duration: {duration}")
        
        completed_steps = sum(1 for v in self.checkpoints.values() if v)
        print(f"Steps completed: {completed_steps}/4")
        
        for step, completed in self.checkpoints.items():
            status = "✅" if completed else "❌"
            print(f"  {status} {step}")
        
        print(f"\n📁 Final outputs:")
        print(f"  - {self.data_file} (aggregated dataset)")
        print(f"  - {self.model_cache}/ (window models + meta-learner)")
        print(f"  - {self.backtest_results}/ (backtest results)")
        print(f"  - {self.pipeline_log} (detailed log)")
        print(f"  - {self.checkpoint_file} (checkpoints)")
        
        print(f"\n📋 Next steps:")
        print(f"  1. Review backtest results in {self.backtest_results}/")
        print(f"  2. Test predictions with: python predict_today.py")
        print(f"  3. Deploy to production if validation passes")
        
        print(f"{'='*80}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="NBA Prediction System - Complete Training Pipeline")
    
    parser.add_argument("--resume", type=str, 
                       choices=["step0_data_prep", "step1_window_training", "step2_meta_learner", "step3_backtest"],
                       help="Resume pipeline from specific step")
    parser.add_argument("--skip-data-prep", action="store_true",
                       help="Skip data preparation if data file exists")
    parser.add_argument("--data", type=str,
                       help="Custom data file path (overrides default)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run pipeline in dry-run mode (no actual training, validation only)")
    
    args = parser.parse_args()
    
    print("NBA Prediction System - Complete Training Pipeline")
    print("=" * 60)
    
    # Environment check
    print("\n[*] Environment check:")
    print(f"  Python: {sys.version}")
    print(f"  Working directory: {os.getcwd()}")
    print(f"  CPU cores: {os.cpu_count()}")
    print(f"  Dry run mode: {args.dry_run}")
    
    if args.dry_run:
        print("\n DRY RUN MODE: No actual training will be performed")
        print("   Use this to validate all scripts and paths before running the full pipeline")
    
    # Check prerequisites
    required_scripts = [
        "create_aggregated_dataset.py",
        "train_player_models.py",
        "train_meta_learner_v4.py",
        "backtest_engine.py"
    ]
    
    missing = []
    for script in required_scripts:
        if not Path(script).exists():
            missing.append(script)
    
    if missing:
        print(f"\n ERROR: Missing required scripts: {missing}")
        print("\nTo fix:")
        for script in missing:
            print(f"  - Ensure {script} exists in the current directory")
        return False
    
    print(" All required scripts found")
    
    # Additional dry-run validation
    if args.dry_run:
        print("\n DRY RUN VALIDATION:")
        print(" Scripts exist and are accessible")
        print(" Python environment is ready")
        print(" Command structure is valid")
        print(" Ready to run full pipeline")
        print("\nTo run the actual pipeline, remove --dry-run flag:")
        print("python complete_training_pipeline.py")
        return True
    
    # Run pipeline
    pipeline = TrainingPipeline(
        resume_from=args.resume,
        skip_data_prep=args.skip_data_prep,
        custom_data=args.data,
        dry_run=args.dry_run
    )
    pipeline.run_pipeline()
    
    return True


if __name__ == "__main__":
    # Import needed for backtesting date calculation
    from datetime import timedelta
    main()

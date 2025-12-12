#!/usr/bin/env python3
"""
Check MLflow experiments and runs to verify database state.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from mlflow.tracking import MlflowClient

def check_mlflow_state():
    """Check current MLflow state."""
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    print("=" * 70)
    print("MLflow Database State Check")
    print("=" * 70)
    
    # Get all experiments
    try:
        experiments = client.search_experiments(view_type=1)  # 1 = ALL (including deleted)
        print(f"\nFound {len(experiments)} experiment(s):")
        
        for exp in experiments:
            lifecycle = getattr(exp, "lifecycle_stage", "active")
            print(f"\n  Experiment: {exp.name}")
            print(f"    ID: {exp.experiment_id}")
            print(f"    Lifecycle: {lifecycle}")
            
            # Count runs
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=1000
            )
            print(f"    Runs: {len(runs)}")
            
            if len(runs) > 0:
                print(f"    First run: {runs[0].info.run_name}")
                print(f"    Last run: {runs[-1].info.run_name}")
                print(f"    Run IDs: {[r.info.run_id[:8] for r in runs[:5]]}...")
        
        # Check for deleted experiments
        deleted = [e for e in experiments if getattr(e, "lifecycle_stage", "active") == "deleted"]
        if deleted:
            print(f"\n⚠️  Found {len(deleted)} deleted experiment(s) (soft deleted):")
            for exp in deleted:
                print(f"    - {exp.name} (ID: {exp.experiment_id})")
                print(f"      These can be restored by MLflow!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total experiments: {len(experiments)}")
    print(f"Active experiments: {len([e for e in experiments if getattr(e, 'lifecycle_stage', 'active') == 'active'])}")
    print(f"Deleted experiments: {len(deleted)}")
    
    total_runs = sum(len(client.search_runs(experiment_ids=[e.experiment_id], max_results=1000)) 
                     for e in experiments)
    print(f"Total runs: {total_runs}")
    
    return True

if __name__ == "__main__":
    check_mlflow_state()


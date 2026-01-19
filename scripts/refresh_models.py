#!/usr/bin/env python3
"""
Refresh/Regenerate All Models and Datasets
Trains all models from scratch and generates fresh datasets
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bondtrader.config import get_config


def refresh_training_data():
    """Generate fresh training dataset"""
    print("=" * 70)
    print("GENERATING TRAINING DATASET")
    print("=" * 70)

    from bondtrader.data.training_data_generator import (
        TrainingDataGenerator,
        save_training_dataset,
    )

    config = get_config()

    print(f"\n📊 Configuration:")
    print(f"   Number of bonds: {config.training_num_bonds}")
    print(f"   Time periods: {config.training_time_periods}")
    print(f"   Batch size: {config.training_batch_size}")
    print(f"   Random seed: {config.ml_random_state}")

    print("\n🔄 Generating dataset...")
    generator = TrainingDataGenerator(seed=config.ml_random_state)
    dataset = generator.generate_comprehensive_dataset(
        total_bonds=config.training_num_bonds,
        time_periods=config.training_time_periods,
        bonds_per_period=config.training_batch_size,
    )

    # Save dataset
    os.makedirs(config.data_dir, exist_ok=True)
    dataset_path = os.path.join(config.data_dir, "training_dataset.joblib")
    save_training_dataset(dataset, dataset_path)

    print(f"✅ Training dataset saved to: {dataset_path}")
    return dataset_path


def refresh_evaluation_data():
    """Generate fresh evaluation dataset"""
    print("\n" + "=" * 70)
    print("GENERATING EVALUATION DATASET")
    print("=" * 70)

    from bondtrader.data.evaluation_dataset_generator import (
        EvaluationDatasetGenerator,
        save_evaluation_dataset,
    )

    config = get_config()

    print("\n🔄 Generating evaluation dataset...")
    generator = EvaluationDatasetGenerator(seed=42)
    evaluation_dataset = generator.generate_evaluation_dataset(
        num_bonds=2000,
        scenarios=None,  # All scenarios
        include_benchmarks=True,
        point_in_time=True,
    )

    # Save dataset
    os.makedirs(config.evaluation_data_dir, exist_ok=True)
    eval_path = os.path.join(config.evaluation_data_dir, "evaluation_dataset.joblib")
    save_evaluation_dataset(evaluation_dataset, eval_path)

    print(f"✅ Evaluation dataset saved to: {eval_path}")
    return eval_path


def refresh_models(dataset_path: str = None, skip_training_data: bool = False):
    """Train all models from scratch"""
    print("\n" + "=" * 70)
    print("TRAINING ALL MODELS")
    print("=" * 70)

    from scripts.train_all_models import ModelTrainer

    config = get_config()

    # Use provided dataset or default
    if dataset_path is None:
        dataset_path = os.path.join(config.data_dir, "training_dataset.joblib")

    print(f"\n📊 Configuration:")
    print(f"   Dataset: {dataset_path}")
    print(f"   Model directory: {config.model_dir}")
    print(f"   Checkpoint directory: {config.checkpoint_dir}")
    print(f"   Max workers: {config.max_workers or 'auto'}")

    # Initialize trainer
    print("\n🔄 Initializing trainer...")
    trainer = ModelTrainer(
        dataset_path=dataset_path if not skip_training_data else None,
        generate_new=skip_training_data,
        use_parallel=True,
        max_workers=config.max_workers,
    )

    # Train all models
    print("\n🚀 Training all models...")
    print("   This may take several minutes...\n")

    results = trainer.train_all_models()

    # Save models
    print("\n💾 Saving models...")
    trainer.save_models(results)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results.values() if r.get("status") == "success")
    failed = sum(1 for r in results.values() if r.get("status") == "failed")

    print(f"\n✅ Successfully trained: {successful} models")
    if failed > 0:
        print(f"❌ Failed: {failed} models")

    # Show test performance if available
    if "test_evaluations" in results:
        print("\n📊 Test Set Performance:")
        for model_name, eval_data in results["test_evaluations"].items():
            if "r2" in eval_data:
                print(
                    f"   {model_name}: R² = {eval_data['r2']:.4f}, RMSE = {eval_data['rmse']:.2f}"
                )

    return results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Refresh/regenerate all models and datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full refresh (datasets + models)
  python scripts/refresh_models.py
  
  # Only refresh models (use existing datasets)
  python scripts/refresh_models.py --skip-training-data
  
  # Only generate datasets (don't train models)
  python scripts/refresh_models.py --datasets-only
  
  # Use specific dataset path
  python scripts/refresh_models.py --dataset training_data/my_dataset.joblib
        """,
    )

    parser.add_argument(
        "--skip-training-data",
        action="store_true",
        help="Skip training data generation, use existing dataset",
    )

    parser.add_argument(
        "--datasets-only", action="store_true", help="Only generate datasets, do not train models"
    )

    parser.add_argument(
        "--dataset", type=str, default=None, help="Path to training dataset (if not using default)"
    )

    parser.add_argument(
        "--skip-evaluation-data", action="store_true", help="Skip evaluation dataset generation"
    )

    args = parser.parse_args()

    try:
        # Generate training data
        dataset_path = None
        if not args.skip_training_data and not args.datasets_only:
            dataset_path = refresh_training_data()
        elif args.dataset:
            dataset_path = args.dataset

        # Generate evaluation data
        if not args.skip_evaluation_data:
            refresh_evaluation_data()

        # Train models
        if not args.datasets_only:
            refresh_models(
                dataset_path=dataset_path,
                skip_training_data=args.skip_training_data and not args.dataset,
            )

        print("\n" + "=" * 70)
        print("✅ REFRESH COMPLETE")
        print("=" * 70)
        print("\n💡 All models and datasets have been regenerated!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

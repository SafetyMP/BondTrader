#!/usr/bin/env python3
"""
Regenerate Evaluation Dataset Script

This script regenerates the evaluation dataset to fix the missing 'bonds' issue.
The new dataset will include all bonds in each scenario, which is required for evaluation.

Usage:
    python regenerate_evaluation_dataset.py [--num-bonds N] [--scenarios scenario1,scenario2]
    
Options:
    --num-bonds N      Number of bonds to generate (default: 1000)
    --scenarios LIST   Comma-separated list of scenarios (default: all scenarios)
    --help             Show this help message
"""

import os
import sys
import argparse
from datetime import datetime

from model_scoring_evaluator import ModelEvaluator
from bondtrader.data.evaluation_dataset_generator import EvaluationDatasetGenerator, save_evaluation_dataset


def main():
    """Main function to regenerate evaluation dataset"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Regenerate evaluation dataset with bonds included",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate with default settings (1000 bonds, all scenarios)
  python regenerate_evaluation_dataset.py
  
  # Regenerate with 2000 bonds
  python regenerate_evaluation_dataset.py --num-bonds 2000
  
  # Regenerate specific scenarios only
  python regenerate_evaluation_dataset.py --scenarios normal_market,rate_shock_up_200bps
        """
    )
    
    parser.add_argument(
        '--num-bonds',
        type=int,
        default=1000,
        help='Number of bonds to generate (default: 1000)'
    )
    
    parser.add_argument(
        '--scenarios',
        type=str,
        default=None,
        help='Comma-separated list of scenario names (default: all scenarios). '
             'Available: normal_market, rate_shock_up_200bps, rate_shock_down_200bps, '
             'credit_spread_widening, liquidity_crisis, market_crash, '
             'low_volatility, high_volatility, recovery'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Backup existing dataset before regenerating'
    )
    
    args = parser.parse_args()
    
    # Parse scenarios if provided
    scenarios = None
    if args.scenarios:
        scenarios = [s.strip() for s in args.scenarios.split(',')]
        print(f"  Scenarios specified: {', '.join(scenarios)}")
    
    print("=" * 80)
    print("EVALUATION DATASET REGENERATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Number of bonds: {args.num_bonds}")
    print(f"  Scenarios: {len(scenarios) if scenarios else 'All scenarios'}")
    print(f"  Backup existing: {args.backup}")
    
    # Backup existing dataset if requested
    eval_path = 'evaluation_data/evaluation_dataset.joblib'
    if args.backup and os.path.exists(eval_path):
        backup_path = f'evaluation_data/evaluation_dataset_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
        import shutil
        shutil.copy2(eval_path, backup_path)
        print(f"\n  ✓ Existing dataset backed up to: {backup_path}")
    
    # Check existing dataset
    if os.path.exists(eval_path):
        file_size = os.path.getsize(eval_path) / (1024 * 1024)  # MB
        mtime = datetime.fromtimestamp(os.path.getmtime(eval_path))
        print(f"\n  Existing dataset found:")
        print(f"    Size: {file_size:.2f} MB")
        print(f"    Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        response = input(f"\n  Overwrite existing dataset? (yes/no) [yes]: ").strip().lower()
        if response and response not in ['yes', 'y']:
            print("  Cancelled. Exiting.")
            return
    else:
        print(f"\n  No existing dataset found. Creating new one.")
    
    print(f"\n" + "=" * 80)
    print("GENERATING EVALUATION DATASET")
    print("=" * 80)
    print("\n  This may take several minutes depending on num_bonds...")
    print("  Please be patient...\n")
    
    start_time = datetime.now()
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_dir='trained_models',
            evaluation_data_dir='evaluation_data'
        )
        
        # Generate new dataset
        evaluation_dataset = evaluator.generate_or_load_evaluation_dataset(
            generate_new=True,
            num_bonds=args.num_bonds
        )
        
        # If scenarios were specified, we need to regenerate with those specific scenarios
        if scenarios:
            print(f"\n  Regenerating with specified scenarios...")
            generator = EvaluationDatasetGenerator(seed=42)
            evaluation_dataset = generator.generate_evaluation_dataset(
                num_bonds=args.num_bonds,
                scenarios=scenarios,
                include_benchmarks=True,
                point_in_time=True
            )
            
            # Save the regenerated dataset
            save_evaluation_dataset(evaluation_dataset, eval_path)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Verify the dataset
        print(f"\n" + "=" * 80)
        print("VERIFICATION")
        print("=" * 80)
        
        from bondtrader.data.evaluation_dataset_generator import load_evaluation_dataset
        loaded_dataset = load_evaluation_dataset(eval_path)
        
        scenarios_dict = loaded_dataset.get('scenarios', {})
        valid_scenarios = 0
        total_bonds = 0
        
        print(f"\n  Checking dataset structure...")
        for sc_name, sc_data in scenarios_dict.items():
            if sc_name == 'benchmarks':
                continue
            
            if isinstance(sc_data, dict):
                bonds_count = len(sc_data.get('bonds', []))
                has_bonds = 'bonds' in sc_data and bonds_count > 0
                
                if has_bonds:
                    valid_scenarios += 1
                    total_bonds += bonds_count
                    print(f"    ✓ {sc_name}: {bonds_count} bonds")
                else:
                    print(f"    ✗ {sc_name}: Missing bonds!")
        
        print(f"\n  Summary:")
        print(f"    Valid scenarios: {valid_scenarios}/{len([s for s in scenarios_dict.keys() if s != 'benchmarks'])}")
        print(f"    Total bonds across scenarios: {total_bonds}")
        print(f"    Evaluation bonds available: {len(loaded_dataset.get('evaluation_bonds', []))}")
        
        print(f"\n" + "=" * 80)
        print("REGENERATION COMPLETE")
        print("=" * 80)
        print(f"\n  ✓ Dataset generated successfully!")
        print(f"  ✓ Saved to: {eval_path}")
        print(f"  ✓ Generation time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"  ✓ All scenarios now include bonds")
        
        if valid_scenarios == len([s for s in scenarios_dict.keys() if s != 'benchmarks']):
            print(f"\n  ✓ Dataset is ready for evaluation!")
        else:
            print(f"\n  ⚠️  Warning: Some scenarios may be missing bonds.")
            print(f"     Please check the output above.")
        
    except KeyboardInterrupt:
        print(f"\n\n  ⚠️  Regeneration cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n  ✗ Error during regeneration: {e}")
        import traceback
        print(f"\n  Full traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Grid search with visible logs."""
import subprocess
import sys
import json
import gc
import torch
import pandas as pd
from code.config import Config


def run_grid_search():
    """Run hyperparameter grid search for all models"""

    config = Config()
    hyperparam_configs = Config.get_reduced_grid()

    print(f"Running grid search: {len(hyperparam_configs)} configurations per model")

    models = ['resnet50', 'densenet']

    print(f"\n{'=' * 70}")
    print(f"Models: {', '.join(models)}")
    print(f"Total experiments: {len(models) * len(hyperparam_configs)}")
    print(f"{'=' * 70}\n")

    all_results = []
    total_experiments = len(models) * len(hyperparam_configs)
    experiment_count = 0

    for model in models:
        for config_idx, hyperparam_config in enumerate(hyperparam_configs):
            experiment_count += 1
            config_name = hyperparam_config['config_name']

            print(f"\n{'=' * 70}")
            print(f"Experiment {experiment_count}/{total_experiments}")
            print(f"Model: {model} | Config: {config_name}")
            print(f"{'=' * 70}\n")

            # Train with visible output
            train_result = subprocess.run(
                [sys.executable, 'train.py',
                 '--model', model,
                 '--config-index', str(config_idx)],
                capture_output=False  # Show all output
            )

            if train_result.returncode != 0:
                print(f"\n✗ Training failed for {model} with {config_name}")
            else:
                print(f"\n✓ Training completed for {model} | {config_name}")

            # Clean up GPU memory between experiments
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Evaluate with visible output
            eval_result = subprocess.run(
                [sys.executable, 'evaluate_grid.py',
                 '--model', model,
                 '--config-name', config_name],
                capture_output=False  # Show all output
            )

            if eval_result.returncode != 0:
                print(f"\n✗ Evaluation failed for {model} with {config_name}")
            else:
                print(f"\n✓ Evaluation completed for {model} | {config_name}")

            # Load evaluation results
            report_path = config.GRID_SEARCH_DIR / f'{model}_{config_name}_report.json'
            if report_path.exists():
                try:
                    with open(report_path, 'r') as f:
                        results = json.load(f)
                    results['model'] = model
                    results['config_name'] = config_name
                    results.update(hyperparam_config)
                    all_results.append(results)
                except Exception as e:
                    print(f"Warning: Could not load results for {model}_{config_name}: {e}")

            # Clean up again after evaluation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save all results
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        results_csv = config.GRID_SEARCH_DIR / 'all_results.csv'
        results_df.to_csv(results_csv, index=False)

        print(f"\n{'=' * 70}")
        print("GRID SEARCH COMPLETE!")
        print(f"{'=' * 70}")
        print(f"Total experiments: {len(all_results)}")
        print(f"Results saved to: {results_csv}")

        # Generate comparison plots
        print("\nGenerating comparison plots...")
        subprocess.run([sys.executable, 'analyze_grid_search.py'], capture_output=False)

        print("\n✓ All done! Check results/grid_search/ for detailed analysis.")
    else:
        print(f"\n{'=' * 70}")
        print("GRID SEARCH FAILED!")
        print(f"{'=' * 70}")
        print("No successful experiments completed.")


if __name__ == '__main__':
    run_grid_search()
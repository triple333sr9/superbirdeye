import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from config import Config
import json
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from dataset import CUBDataset
from models import PartLocalizer


def analyze_grid_search_results():
    """Analyze and visualize grid search results"""

    config = Config()

    print(f"\n{'=' * 70}")
    print("COMPREHENSIVE GRID SEARCH ANALYSIS")
    print(f"{'=' * 70}\n")

    # STEP 1: Scan for all trained models
    print("Step 1: Scanning for trained models...")
    trained_models = scan_trained_models(config)
    print(f"  Found {len(trained_models)} trained model files\n")

    if len(trained_models) == 0:
        print("No trained models found!")
        return

    # STEP 2: Check which models have evaluation reports
    print("Step 2: Checking evaluation status...")
    models_needing_eval = check_evaluation_status(config, trained_models)
    print(f"  {len(models_needing_eval)} models need evaluation\n")

    # STEP 3: Run missing evaluations
    if len(models_needing_eval) > 0:
        print("Step 3: Running missing evaluations...")
        run_missing_evaluations(config, models_needing_eval)
        print()
    else:
        print("Step 3: All models already evaluated ✓\n")

    # STEP 4: Collect all results and regenerate CSV
    print("Step 4: Collecting all evaluation results...")
    all_results = collect_all_results(config, trained_models)

    if len(all_results) == 0:
        print("No successful evaluations found!")
        return

    # Save regenerated CSV
    results_df = pd.DataFrame(all_results)
    results_csv = config.GRID_SEARCH_DIR / 'all_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"  Regenerated all_results.csv with {len(all_results)} models\n")

    # STEP 5: Add YOLOv8 if available
    yolo_report = config.REPORTS_DIR / 'yolov8_report.json'
    if yolo_report.exists():
        print("Step 5: Adding YOLOv8 results...")
        with open(yolo_report, 'r') as f:
            yolo_results = json.load(f)

        yolo_row = {
            'model': 'yolov8',
            'config_name': 'default',
            'loss_function': 'yolo_default',
            'learning_rate': 0.01,
            'weight_decay': 0.0005,
            'hidden_dim': 'N/A',
            'augmentation': 'yolo_default',
            'dropout': 'N/A',
            'avg_mae': yolo_results.get('avg_mae', 0),
            'avg_mae_x': yolo_results.get('avg_mae_x', 0),
            'avg_mae_y': yolo_results.get('avg_mae_y', 0),
            'avg_pck_01': yolo_results.get('avg_pck_01', 0),
            'beak_mae_x': yolo_results.get('beak_mae_x', 0),
            'beak_mae_y': yolo_results.get('beak_mae_y', 0),
            'training_epochs': 'N/A',
            'vis_accuracy': yolo_results.get('vis_accuracy', 0),
        }
        results_df = pd.concat([results_df, pd.DataFrame([yolo_row])], ignore_index=True)
        results_df.to_csv(results_csv, index=False)
        print(f"  Added YOLOv8 to results\n")

    print(f"{'=' * 70}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total models analyzed: {len(results_df)}")
    print(f"Results saved to: {results_csv}\n")

    # STEP 6: Create comprehensive visualizations
    print("Step 6: Creating visualizations...")
    create_overview_plots(results_df, config)
    create_hyperparameter_analysis(results_df, config)
    create_beak_focus_analysis(results_df, config)
    create_model_comparison(results_df, config)
    create_best_config_report(results_df, config)
    create_best_models_detailed_analysis(results_df, config)
    create_all_models_metric_graphs(results_df, config)
    visualize_best_5_models(results_df, config)

    print(f"\n{'=' * 70}")
    print(f"ALL VISUALIZATIONS COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Results location: {config.GRID_SEARCH_DIR}")
    print(f"{'=' * 70}\n")


def scan_trained_models(config):
    """Scan saved_models directory for all trained model files"""
    import re

    trained_models = []
    model_pattern = re.compile(r'(.+?)_(.+?)_best\.pth')

    for model_file in config.SAVE_DIR.glob('*_best.pth'):
        match = model_pattern.match(model_file.name)
        if match:
            backbone = match.group(1)
            config_name = match.group(2)

            # Skip YOLOv8 (handled separately)
            if backbone == 'yolov8':
                continue

            trained_models.append({
                'backbone': backbone,
                'config_name': config_name,
                'model_path': model_file
            })
            print(f"  Found: {backbone} | {config_name}")

    return trained_models


def check_evaluation_status(config, trained_models):
    """Check which trained models are missing evaluation reports"""
    models_needing_eval = []

    for model_info in trained_models:
        backbone = model_info['backbone']
        config_name = model_info['config_name']
        report_path = config.GRID_SEARCH_DIR / f'{backbone}_{config_name}_report.json'

        if not report_path.exists():
            models_needing_eval.append(model_info)
            print(f"  Missing eval: {backbone} | {config_name}")
        else:
            print(f"  ✓ Has eval: {backbone} | {config_name}")

    return models_needing_eval


def run_missing_evaluations(config, models_needing_eval):
    """Run evaluation for models missing reports"""
    import subprocess
    import sys

    total = len(models_needing_eval)
    for idx, model_info in enumerate(models_needing_eval, 1):
        backbone = model_info['backbone']
        config_name = model_info['config_name']

        print(f"\n  [{idx}/{total}] Evaluating {backbone} | {config_name}...")

        try:
            result = subprocess.run(
                [sys.executable, 'evaluate_grid.py',
                 '--model', backbone,
                 '--config-name', config_name],
                capture_output=True,
                encoding='utf-8',
                errors='replace',
                timeout=600  # 10 minute timeout per evaluation
            )

            if result.returncode == 0:
                print(f"      ✓ Evaluation successful")
            else:
                print(f"      ✗ Evaluation failed")
                if result.stderr:
                    print(f"      Error: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            print(f"      ✗ Evaluation timed out (>10 minutes)")
        except Exception as e:
            print(f"      ✗ Evaluation error: {str(e)[:200]}")

        # Clean up GPU memory
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass


def collect_all_results(config, trained_models):
    """Collect all evaluation results from JSON reports"""
    all_results = []

    for model_info in trained_models:
        backbone = model_info['backbone']
        config_name = model_info['config_name']
        report_path = config.GRID_SEARCH_DIR / f'{backbone}_{config_name}_report.json'

        if report_path.exists():
            try:
                with open(report_path, 'r') as f:
                    results = json.load(f)

                # Ensure it has the required fields
                if 'avg_mae' in results and 'avg_pck_01' in results:
                    all_results.append(results)
                    print(f"  ✓ Loaded: {backbone} | {config_name}")
                else:
                    print(f"  ✗ Incomplete report: {backbone} | {config_name}")

            except Exception as e:
                print(f"  ✗ Error loading {backbone} | {config_name}: {str(e)[:100]}")
        else:
            print(f"  ✗ No report: {backbone} | {config_name}")

    return all_results

    # Create comprehensive visualizations
    create_overview_plots(df, config)
    create_hyperparameter_analysis(df, config)
    create_beak_focus_analysis(df, config)
    create_model_comparison(df, config)
    create_best_config_report(df, config)

    # NEW: Create detailed visualizations for best models
    create_best_models_detailed_analysis(df, config)
    create_all_models_metric_graphs(df, config)
    visualize_best_5_models(df, config)

    print(f"\nAll visualizations saved to: {config.GRID_SEARCH_DIR}")


def create_best_models_detailed_analysis(df, config):
    """Create detailed analysis for the top 5 best models overall"""

    print("\nGenerating detailed analysis for best 5 models...")

    # Get top 5 models overall by avg_mae
    top_5 = df.nsmallest(5, 'avg_mae')

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 5, hspace=0.4, wspace=0.3)

    fig.suptitle('TOP 5 BEST MODELS - DETAILED ANALYSIS',
                 fontsize=18, fontweight='bold', y=0.995)

    # For each of the top 5 models
    for idx, (_, row) in enumerate(top_5.iterrows()):
        col = idx

        # Model title
        model_name = f"{row['model']}\n{row['config_name'][:25]}"

        # 1. Performance metrics bar chart
        ax1 = fig.add_subplot(gs[0, col])
        metrics = ['avg_mae', 'avg_mae_x', 'avg_mae_y', 'beak_mae_x']
        values = [row['avg_mae'], row['avg_mae_x'], row['avg_mae_y'], row['beak_mae_x']]
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

        bars = ax1.bar(range(len(metrics)), values, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels(['MAE', 'MAE X', 'MAE Y', 'Beak X'], rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Error', fontsize=9)
        ax1.set_title(f'#{idx + 1}: {model_name}', fontsize=10, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=7)

        # 2. PCK score
        ax2 = fig.add_subplot(gs[1, col])
        pck_score = row['avg_pck_01']
        ax2.barh([0], [pck_score], color='#27ae60', alpha=0.7)
        ax2.barh([0], [100 - pck_score], left=[pck_score], color='#e0e0e0', alpha=0.3)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlabel('PCK@0.1 (%)', fontsize=9)
        ax2.set_yticks([])
        ax2.text(pck_score / 2, 0, f'{pck_score:.1f}%',
                 ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax2.set_title('PCK@0.1 Score', fontsize=9)

        # 3. Hyperparameters table
        ax3 = fig.add_subplot(gs[2, col])
        ax3.axis('tight')
        ax3.axis('off')

        hyperparams = [
            ['Loss', str(row['loss_function'])[:12]],
            ['LR', f"{row['learning_rate']:.0e}"],
            ['WD', f"{row['weight_decay']:.0e}"],
            ['Hidden', str(row['hidden_dim'])],
            ['Dropout', str(row['dropout'])],
            ['Aug', str(row['augmentation'])]
        ]

        table = ax3.table(cellText=hyperparams,
                          cellLoc='left',
                          loc='center',
                          bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.5)

        for i in range(len(hyperparams)):
            table[(i, 0)].set_facecolor('#ecf0f1')
            table[(i, 0)].set_text_props(weight='bold')

        ax3.set_title('Hyperparameters', fontsize=9, pad=10)

        # 4. Rank comparison
        ax4 = fig.add_subplot(gs[3, col])

        # Calculate ranks for different metrics
        rank_mae = (df['avg_mae'] <= row['avg_mae']).sum()
        rank_pck = (df['avg_pck_01'] >= row['avg_pck_01']).sum()
        rank_beak = (df['beak_mae_x'] <= row['beak_mae_x']).sum()

        ranks = [rank_mae, rank_pck, rank_beak]
        rank_labels = ['MAE\nRank', 'PCK\nRank', 'Beak\nRank']
        colors_rank = ['#3498db', '#2ecc71', '#e74c3c']

        bars = ax4.bar(range(3), ranks, color=colors_rank, alpha=0.7)
        ax4.set_xticks(range(3))
        ax4.set_xticklabels(rank_labels, fontsize=8)
        ax4.set_ylabel('Rank', fontsize=9)
        ax4.set_title('Rankings', fontsize=9)
        ax4.invert_yaxis()  # Lower rank = better
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, ranks):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'#{val}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.savefig(config.GRID_SEARCH_DIR / 'top_5_models_detailed.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created top_5_models_detailed.png")


def create_all_models_metric_graphs(df, config):
    """Create comprehensive metric graphs for ALL grid search models"""

    print("\nGenerating metric graphs for all models...")

    # Filter out YOLO for fair comparison
    df_grid = df[df['model'] != 'yolov8'].copy()

    if len(df_grid) == 0:
        print("No grid search models found")
        return

    # Add model labels - handle missing values
    df_grid['model_label'] = df_grid.apply(
        lambda row: f"{str(row['model'])[:3]}-{str(row['config_name'])[:15]}"
        if pd.notna(row['model']) and pd.notna(row['config_name'])
        else "unknown",
        axis=1
    )

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    fig.suptitle('ALL GRID SEARCH MODELS - COMPREHENSIVE METRICS',
                 fontsize=18, fontweight='bold')

    # 1. All models MAE distribution
    ax1 = fig.add_subplot(gs[0, :])
    models_sorted = df_grid.sort_values('avg_mae').reset_index(drop=True)
    x = np.arange(len(models_sorted))
    colors = ['#2ecc71' if i < 5 else '#3498db' if i < 10 else '#95a5a6'
              for i in range(len(models_sorted))]

    bars = ax1.bar(x, models_sorted['avg_mae'], color=colors, alpha=0.8)
    ax1.axhline(y=models_sorted['avg_mae'].median(), color='red',
                linestyle='--', linewidth=2, label=f'Median: {models_sorted["avg_mae"].median():.4f}')
    ax1.axhline(y=models_sorted['avg_mae'].mean(), color='orange',
                linestyle='--', linewidth=2, label=f'Mean: {models_sorted["avg_mae"].mean():.4f}')

    # Highlight top 5
    for i in range(min(5, len(models_sorted))):
        bars[i].set_edgecolor('gold')
        bars[i].set_linewidth(3)

    # Add labels for every model
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_sorted['model_label'], rotation=90, ha='right', fontsize=7)
    ax1.set_xlabel('Model Configuration (sorted by MAE)', fontsize=11)
    ax1.set_ylabel('Average MAE', fontsize=11)
    ax1.set_title('All Models Ranked by Average MAE (Gold border = Top 5)',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # 2. MAE X vs MAE Y scatter with labels
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = ax2.scatter(df_grid['avg_mae_x'], df_grid['avg_mae_y'],
                          c=df_grid['avg_mae'], cmap='RdYlGn_r',
                          s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Highlight and label best model
    best = df_grid.loc[df_grid['avg_mae'].idxmin()]
    ax2.scatter(best['avg_mae_x'], best['avg_mae_y'],
                color='gold', s=400, marker='*',
                edgecolors='black', linewidth=2, label='Best Model', zorder=5)
    ax2.annotate(f"Best: {best['model_label']}",
                 xy=(best['avg_mae_x'], best['avg_mae_y']),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=8, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Label top 5 models
    top_5_grid = df_grid.nsmallest(5, 'avg_mae')
    for idx, (_, row) in enumerate(top_5_grid.iterrows()):
        if idx > 0:  # Skip best (already labeled)
            ax2.annotate(f"#{idx + 1}",
                         xy=(row['avg_mae_x'], row['avg_mae_y']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=7, fontweight='bold',
                         bbox=dict(boxstyle='circle,pad=0.3', facecolor='lightgreen', alpha=0.7))

    ax2.set_xlabel('Average MAE X', fontsize=10)
    ax2.set_ylabel('Average MAE Y', fontsize=10)
    ax2.set_title('X vs Y Error Trade-off (Top 5 labeled)', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Overall MAE')

    # 3. PCK Distribution with labels
    ax3 = fig.add_subplot(gs[1, 1])
    pck_sorted = df_grid.sort_values('avg_pck_01', ascending=False).reset_index(drop=True)
    colors_pck = ['#2ecc71' if i < 5 else '#3498db' if i < 10 else '#95a5a6'
                  for i in range(len(pck_sorted))]

    bars = ax3.barh(range(len(pck_sorted)), pck_sorted['avg_pck_01'],
                    color=colors_pck, alpha=0.8)

    # Add labels for top 5
    for i in range(min(5, len(pck_sorted))):
        value = pck_sorted.iloc[i]['avg_pck_01']
        label = pck_sorted.iloc[i]['model_label']
        ax3.text(value + 0.5, i, f"#{i + 1}: {label[:20]}",
                 va='center', fontsize=7, fontweight='bold')

    ax3.set_xlabel('PCK@0.1 (%)', fontsize=10)
    ax3.set_ylabel('Model Configuration', fontsize=10)
    ax3.set_title('All Models by PCK@0.1 (Top 5 labeled)',
                  fontsize=11, fontweight='bold')
    ax3.set_yticks([])
    ax3.grid(axis='x', alpha=0.3)

    # 4. Beak X Error Focus with labels
    ax4 = fig.add_subplot(gs[1, 2])
    beak_sorted = df_grid.sort_values('beak_mae_x').reset_index(drop=True)
    colors_beak = ['#e74c3c' if i < 5 else '#e67e22' if i < 10 else '#95a5a6'
                   for i in range(len(beak_sorted))]

    bars = ax4.barh(range(len(beak_sorted)), beak_sorted['beak_mae_x'],
                    color=colors_beak, alpha=0.8)

    # Add labels for top 5
    for i in range(min(5, len(beak_sorted))):
        value = beak_sorted.iloc[i]['beak_mae_x']
        label = beak_sorted.iloc[i]['model_label']
        ax4.text(value + 0.001, i, f"#{i + 1}: {label[:20]}",
                 va='center', fontsize=7, fontweight='bold')

    ax4.set_xlabel('Beak MAE X', fontsize=10)
    ax4.set_title('Beak X-Axis Error (Top 5 labeled)',
                  fontsize=11, fontweight='bold')
    ax4.set_yticks([])
    ax4.grid(axis='x', alpha=0.3)

    # 5. Hyperparameter impact on MAE
    ax5 = fig.add_subplot(gs[2, 0])
    loss_impact = df_grid.groupby('loss_function')['avg_mae'].agg(['mean', 'std', 'min'])

    if len(loss_impact) > 0:
        loss_impact.plot(kind='bar', ax=ax5, alpha=0.8)
        ax5.set_title('Loss Function Impact', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Loss Function', fontsize=10)
        ax5.set_ylabel('Average MAE', fontsize=10)
        ax5.legend(['Mean', 'Std', 'Min'], fontsize=8)
        ax5.grid(axis='y', alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax5.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax5.transAxes)

    # 6. Learning rate impact
    ax6 = fig.add_subplot(gs[2, 1])
    lr_impact = df_grid.groupby('learning_rate')['avg_mae'].agg(['mean', 'std', 'min'])

    if len(lr_impact) > 0:
        x_lr = np.arange(len(lr_impact))
        width = 0.25
        ax6.bar(x_lr - width, lr_impact['mean'], width, label='Mean', alpha=0.8)
        ax6.bar(x_lr, lr_impact['std'], width, label='Std', alpha=0.8)
        ax6.bar(x_lr + width, lr_impact['min'], width, label='Min', alpha=0.8)
        ax6.set_xticks(x_lr)
        ax6.set_xticklabels([f'{x:.0e}' for x in lr_impact.index], rotation=45, ha='right')
        ax6.set_title('Learning Rate Impact', fontsize=11, fontweight='bold')
        ax6.set_xlabel('Learning Rate', fontsize=10)
        ax6.set_ylabel('Average MAE', fontsize=10)
        ax6.legend(fontsize=8)
        ax6.grid(axis='y', alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax6.transAxes)

    # 7. Model architecture comparison
    ax7 = fig.add_subplot(gs[2, 2])
    model_stats = df_grid.groupby('model')['avg_mae'].agg(['mean', 'min', 'max', 'std'])

    if len(model_stats) > 0:
        x_model = np.arange(len(model_stats))

        ax7.bar(x_model, model_stats['mean'], color='#3498db', alpha=0.7, label='Mean')
        ax7.errorbar(x_model, model_stats['mean'], yerr=model_stats['std'],
                     fmt='none', color='black', capsize=5)
        ax7.scatter(x_model, model_stats['min'], color='#2ecc71',
                    s=100, marker='v', label='Best', zorder=5)
        ax7.scatter(x_model, model_stats['max'], color='#e74c3c',
                    s=100, marker='^', label='Worst', zorder=5)

        ax7.set_xticks(x_model)
        ax7.set_xticklabels(model_stats.index, rotation=45, ha='right')
        ax7.set_title('Model Architecture Comparison', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Model', fontsize=10)
        ax7.set_ylabel('Average MAE', fontsize=10)
        ax7.legend(fontsize=8)
        ax7.grid(axis='y', alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax7.transAxes)

    plt.savefig(config.GRID_SEARCH_DIR / 'all_models_metrics.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created all_models_metrics.png")


def visualize_best_5_models(df, config):
    """Generate prediction visualizations for the best 5 models"""

    print("\nGenerating prediction visualizations for best 5 models...")

    # Get top models, excluding YOLO
    df_no_yolo = df[df['model'] != 'yolov8'].copy()
    top_models = df_no_yolo.nsmallest(6, 'avg_mae')  # Get 6 in case we need extras

    # Take first 5 non-YOLO models
    top_5_grid = top_models.head(5)

    if len(top_5_grid) == 0:
        print("No grid search models found")
        return

    print(f"Selected top 5 models for visualization:")
    for idx, (_, row) in enumerate(top_5_grid.iterrows()):
        print(f"  #{idx + 1}: {row['model']} - {row['config_name'][:30]} (MAE: {row['avg_mae']:.4f})")

    # Load validation dataset
    dataset = CUBDataset(config, mode='val')

    # Pick 5 random samples
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(len(dataset), min(5, len(dataset)), replace=False)

    # Create visualization directory
    vis_dir = config.GRID_SEARCH_DIR / 'best_5_predictions'
    vis_dir.mkdir(exist_ok=True)

    for sample_idx in sample_indices:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Sample {sample_idx} - Top 5 Models Comparison',
                     fontsize=16, fontweight='bold')

        # Load ground truth - handle new 4-tuple format
        batch_data = dataset[sample_idx]
        if len(batch_data) == 4:
            image_tensor, targets, visibility_gt, (orig_w, orig_h, img_id) = batch_data
        else:
            image_tensor, targets, (orig_w, orig_h, img_id) = batch_data
            visibility_gt = torch.ones(config.NUM_PARTS)

        row = dataset.image_data.iloc[sample_idx]
        img_path = config.IMAGES_DIR / row['path']
        original_img = Image.open(img_path).convert('RGB')

        # First subplot: ground truth
        ax = axes[0, 0]
        gt_img = original_img.copy()
        gt_draw = ImageDraw.Draw(gt_img)

        colors = ['red', 'blue', 'green']
        targets_reshaped = targets.view(-1, 2).numpy()

        # Draw ground truth keypoints
        for j, (color, part_name) in enumerate(zip(colors, config.PART_NAMES)):
            gt = targets_reshaped[j]
            if gt[0] >= 0 and gt[1] >= 0:
                gt_x = gt[0] * orig_w
                gt_y = gt[1] * orig_h
                radius = max(8, int(min(orig_w, orig_h) * 0.02))
                gt_draw.ellipse(
                    [gt_x - radius, gt_y - radius, gt_x + radius, gt_y + radius],
                    fill=color, outline='white', width=3
                )

        # Add legend to ground truth
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        legend_y = 10
        for j, (color, part_name) in enumerate(zip(colors, config.PART_NAMES)):
            gt_draw.text((10, legend_y + j * 20), f"● {part_name}",
                         fill=color, font=font, stroke_width=2, stroke_fill='black')

        ax.imshow(gt_img)
        ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Next 5 subplots: predictions from top 5 models
        for idx, (_, model_row) in enumerate(top_5_grid.iterrows()):
            if idx >= 5:
                break

            row_idx = (idx + 1) // 3
            col_idx = (idx + 1) % 3
            ax = axes[row_idx, col_idx]

            backbone = model_row['model']
            config_name = model_row['config_name']

            # Load model
            model_path = config.SAVE_DIR / f'{backbone}_{config_name}_best.pth'

            if not model_path.exists():
                ax.text(0.5, 0.5, f'Model not found\n{backbone}\n{config_name}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            try:
                # Load model
                checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
                hyperparam_config = checkpoint.get('hyperparameters', {})

                # Create temporary config
                temp_config = Config()
                if hyperparam_config:
                    temp_config.apply_hyperparameters(hyperparam_config)
                else:
                    temp_config.HIDDEN_DIM = 512
                    temp_config.DROPOUT = 0.4

                actual_backbone = 'resnet50' if backbone == 'superanimal_bird' else backbone
                model = PartLocalizer(actual_backbone, config.NUM_PARTS, temp_config).to(config.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                # Get prediction
                with torch.no_grad():
                    image_input = image_tensor.unsqueeze(0).to(config.DEVICE)
                    coords_pred, vis_pred = model(image_input)
                    coords_pred = coords_pred.cpu().squeeze()
                    vis_pred = vis_pred.cpu().squeeze()

                # Visualize
                pred_img = original_img.copy()
                pred_draw = ImageDraw.Draw(pred_img)

                # Calculate distances for this prediction
                coords_pred_np = coords_pred.numpy()
                vis_pred_binary = (vis_pred.numpy() > 0.5)
                distances = []

                for j, (color, part_name) in enumerate(zip(colors, config.PART_NAMES)):
                    gt = targets_reshaped[j]

                    # Only draw prediction if model predicts it's visible
                    if vis_pred_binary[j]:
                        # Calculate distance only if GT is also visible
                        if gt[0] >= 0 and gt[1] >= 0:
                            pred_point = coords_pred_np[j]
                            dist = np.linalg.norm(pred_point - gt)
                            distances.append(dist)

                        pred_x = np.clip(coords_pred_np[j, 0], 0, 1) * orig_w
                        pred_y = np.clip(coords_pred_np[j, 1], 0, 1) * orig_h

                        cross_size = max(12, int(min(orig_w, orig_h) * 0.025))
                        pred_draw.line(
                            [(pred_x - cross_size, pred_y - cross_size),
                             (pred_x + cross_size, pred_y + cross_size)],
                            fill=color, width=4
                        )
                        pred_draw.line(
                            [(pred_x - cross_size, pred_y + cross_size),
                             (pred_x + cross_size, pred_y - cross_size)],
                            fill=color, width=4
                        )

                # Add legend
                legend_y = 10
                for j, (color, part_name) in enumerate(zip(colors, config.PART_NAMES)):
                    pred_draw.text((10, legend_y + j * 20), f"✕ {part_name}",
                                   fill=color, font=font, stroke_width=2, stroke_fill='black')

                # Add distance info
                if distances:
                    avg_dist = np.mean(distances)
                    pred_draw.text((10, orig_h - 25), f"Avg Dist: {avg_dist:.4f}",
                                   fill='white', font=font, stroke_width=2, stroke_fill='black')

                # Add visibility count info
                vis_count_text = f"Predicted: {int(vis_pred_binary.sum())}/3 visible"
                pred_draw.text((10, orig_h - 50), vis_count_text,
                               fill='white', font=font, stroke_width=2, stroke_fill='black')

                ax.imshow(pred_img)
                title = f"Rank #{idx + 1}: {backbone}\n{config_name[:30]}\nMAE: {model_row['avg_mae']:.4f}"
                ax.set_title(title, fontsize=9, fontweight='bold')
                ax.axis('off')

            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading model\n{str(e)[:50]}',
                        ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.axis('off')

        # Hide unused subplots
        for idx in range(len(top_5_grid), 5):
            row_idx = (idx + 1) // 3
            col_idx = (idx + 1) % 3
            if row_idx < 2 and col_idx < 3:
                axes[row_idx, col_idx].axis('off')

        plt.tight_layout()
        plt.savefig(vis_dir / f'sample_{sample_idx}_comparison.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✓ Created {len(sample_indices)} comparison visualizations in {vis_dir}/")

def create_overview_plots(df, config):
    """Create overview comparison plots"""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Best configs per model
    ax1 = fig.add_subplot(gs[0, :])

    # Get top 5 configs per model
    top_configs = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model].nsmallest(5, 'avg_mae')
        top_configs.append(model_df)

    top_df = pd.concat(top_configs)

    x = np.arange(len(top_df))
    colors = ['#1f77b4' if m == 'resnet50' else '#ff7f0e' for m in top_df['model']]

    bars = ax1.bar(x, top_df['avg_mae'], color=colors, alpha=0.8)
    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_ylabel('Average MAE', fontsize=12)
    ax1.set_title('Top 5 Configurations per Model (Lower is Better)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{row['model'][:3]}\n{row['config_name'][:20]}"
                         for _, row in top_df.iterrows()], rotation=45, ha='right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=df['avg_mae'].median(), color='red', linestyle='--', label='Median', alpha=0.5)
    ax1.legend()

    # 2. MAE X vs Y scatter
    ax2 = fig.add_subplot(gs[1, 0])
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        ax2.scatter(model_df['avg_mae_x'], model_df['avg_mae_y'],
                    label=model, alpha=0.6, s=80)

    ax2.set_xlabel('Average MAE X', fontsize=11)
    ax2.set_ylabel('Average MAE Y', fontsize=11)
    ax2.set_title('X vs Y Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. PCK@0.1 distribution
    ax3 = fig.add_subplot(gs[1, 1])
    df.boxplot(column='avg_pck_01', by='model', ax=ax3)
    ax3.set_xlabel('Model', fontsize=11)
    ax3.set_ylabel('PCK@0.1 (%)', fontsize=11)
    ax3.set_title('PCK Distribution by Model', fontsize=12, fontweight='bold')
    plt.sca(ax3)
    plt.xticks(rotation=0)

    # 4. Beak X error distribution
    ax4 = fig.add_subplot(gs[1, 2])
    df.boxplot(column='beak_mae_x', by='model', ax=ax4)
    ax4.set_xlabel('Model', fontsize=11)
    ax4.set_ylabel('Beak MAE X', fontsize=11)
    ax4.set_title('Beak X-Axis Error (Focus Area)', fontsize=12, fontweight='bold')
    plt.sca(ax4)
    plt.xticks(rotation=0)

    # 5. Training epochs vs performance
    ax5 = fig.add_subplot(gs[2, 0])
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        ax5.scatter(model_df['training_epochs'], model_df['avg_mae'],
                    label=model, alpha=0.6, s=80)
    ax5.set_xlabel('Training Epochs', fontsize=11)
    ax5.set_ylabel('Average MAE', fontsize=11)
    ax5.set_title('Training Duration vs Performance', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 6. Best vs worst comparison
    ax6 = fig.add_subplot(gs[2, 1])
    best_row = df.loc[df['avg_mae'].idxmin()]
    worst_row = df.loc[df['avg_mae'].idxmax()]

    metrics = ['avg_mae', 'avg_mae_x', 'avg_mae_y', 'beak_mae_x']
    x_pos = np.arange(len(metrics))
    width = 0.35

    ax6.bar(x_pos - width / 2, [best_row[m] for m in metrics], width,
            label='Best Config', color='green', alpha=0.7)
    ax6.bar(x_pos + width / 2, [worst_row[m] for m in metrics], width,
            label='Worst Config', color='red', alpha=0.7)

    ax6.set_ylabel('MAE', fontsize=11)
    ax6.set_title('Best vs Worst Configuration', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(['Avg MAE', 'MAE X', 'MAE Y', 'Beak X'], rotation=45, ha='right')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)

    # 7. Summary table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('tight')
    ax7.axis('off')

    summary_data = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        summary_data.append([
            model,
            f"{model_df['avg_mae'].min():.4f}",
            f"{model_df['avg_mae'].mean():.4f}",
            f"{model_df['beak_mae_x'].min():.4f}"
        ])

    table = ax7.table(cellText=summary_data,
                      colLabels=['Model', 'Best MAE', 'Avg MAE', 'Best Beak X'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax7.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.savefig(config.GRID_SEARCH_DIR / 'overview_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created overview_comparison.png")


def create_hyperparameter_analysis(df, config):
    """Analyze impact of each hyperparameter"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hyperparameter Impact Analysis', fontsize=16, fontweight='bold')

    # 1. Loss function impact
    ax = axes[0, 0]
    loss_data = df.groupby('loss_function')['avg_mae'].agg(['mean', 'std', 'min'])
    loss_data.plot(kind='bar', ax=ax, alpha=0.8)
    ax.set_title('Loss Function Impact', fontweight='bold')
    ax.set_ylabel('Average MAE')
    ax.set_xlabel('Loss Function')
    ax.legend(['Mean', 'Std', 'Min'])
    ax.grid(axis='y', alpha=0.3)
    plt.sca(ax)
    plt.xticks(rotation=45)

    # 2. Learning rate impact
    ax = axes[0, 1]
    lr_data = df.groupby('learning_rate')['avg_mae'].agg(['mean', 'std', 'min'])
    lr_data.plot(kind='bar', ax=ax, alpha=0.8, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_title('Learning Rate Impact', fontweight='bold')
    ax.set_ylabel('Average MAE')
    ax.set_xlabel('Learning Rate')
    ax.set_xticklabels([f'{x:.0e}' for x in lr_data.index], rotation=45)
    ax.legend(['Mean', 'Std', 'Min'])
    ax.grid(axis='y', alpha=0.3)

    # 3. Weight decay impact
    ax = axes[0, 2]
    wd_data = df.groupby('weight_decay')['avg_mae'].agg(['mean', 'std', 'min'])
    wd_data.plot(kind='bar', ax=ax, alpha=0.8, color=['coral', 'gold', 'lightseagreen'])
    ax.set_title('Weight Decay Impact', fontweight='bold')
    ax.set_ylabel('Average MAE')
    ax.set_xlabel('Weight Decay')
    ax.set_xticklabels([f'{x:.0e}' for x in wd_data.index], rotation=45)
    ax.legend(['Mean', 'Std', 'Min'])
    ax.grid(axis='y', alpha=0.3)

    # 4. Hidden dimension impact
    ax = axes[1, 0]
    hidden_data = df.groupby('hidden_dim')['avg_mae'].agg(['mean', 'std', 'min'])
    hidden_data.plot(kind='bar', ax=ax, alpha=0.8, color=['mediumpurple', 'palegreen'])
    ax.set_title('Hidden Dimension Impact', fontweight='bold')
    ax.set_ylabel('Average MAE')
    ax.set_xlabel('Hidden Dimension')
    ax.legend(['Mean', 'Std', 'Min'])
    ax.grid(axis='y', alpha=0.3)
    plt.sca(ax)
    plt.xticks(rotation=0)

    # 5. Augmentation impact
    ax = axes[1, 1]
    aug_data = df.groupby('augmentation')['avg_mae'].agg(['mean', 'std', 'min'])
    aug_data.plot(kind='bar', ax=ax, alpha=0.8, color=['salmon', 'khaki', 'plum'])
    ax.set_title('Data Augmentation Impact', fontweight='bold')
    ax.set_ylabel('Average MAE')
    ax.set_xlabel('Augmentation Level')
    ax.legend(['Mean', 'Std', 'Min'])
    ax.grid(axis='y', alpha=0.3)
    plt.sca(ax)
    plt.xticks(rotation=0)

    # 6. Combined heatmap
    ax = axes[1, 2]
    pivot = df.pivot_table(values='avg_mae',
                           index='loss_function',
                           columns='learning_rate',
                           aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Avg MAE'})
    ax.set_title('Loss Function × Learning Rate', fontweight='bold')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss Function')

    plt.tight_layout()
    plt.savefig(config.GRID_SEARCH_DIR / 'hyperparameter_analysis.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created hyperparameter_analysis.png")


def create_beak_focus_analysis(df, config):
    """Special analysis focusing on beak X-axis issue"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Beak X-Axis Performance Analysis', fontsize=16, fontweight='bold')

    # 1. Beak X by loss function
    ax = axes[0, 0]
    df.boxplot(column='beak_mae_x', by='loss_function', ax=ax)
    ax.set_title('Beak X Error by Loss Function', fontweight='bold')
    ax.set_ylabel('Beak MAE X')
    ax.set_xlabel('Loss Function')
    plt.sca(ax)
    plt.xticks(rotation=45)

    # 2. Beak X by learning rate
    ax = axes[0, 1]
    lr_beak = df.groupby('learning_rate')['beak_mae_x'].agg(['mean', 'min', 'max'])
    x = np.arange(len(lr_beak))
    ax.bar(x, lr_beak['mean'], alpha=0.7, label='Mean')
    ax.errorbar(x, lr_beak['mean'],
                yerr=[lr_beak['mean'] - lr_beak['min'], lr_beak['max'] - lr_beak['mean']],
                fmt='none', color='black', capsize=5, label='Min-Max Range')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lr:.0e}' for lr in lr_beak.index], rotation=45)
    ax.set_title('Beak X Error by Learning Rate', fontweight='bold')
    ax.set_ylabel('Beak MAE X')
    ax.set_xlabel('Learning Rate')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 3. Beak X vs Y scatter
    ax = axes[1, 0]
    scatter = ax.scatter(df['beak_mae_x'], df['beak_mae_y'],
                         c=df['avg_pck_01'], cmap='RdYlGn', s=100, alpha=0.6)
    ax.set_xlabel('Beak MAE X', fontsize=12)
    ax.set_ylabel('Beak MAE Y', fontsize=12)
    ax.set_title('Beak X vs Y Error (colored by PCK@0.1)', fontweight='bold')
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'k--', alpha=0.3, label='X=Y line')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='PCK@0.1 (%)')

    # 4. Best configs for beak X
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    best_beak_x = df.nsmallest(5, 'beak_mae_x')[
        ['model', 'config_name', 'beak_mae_x', 'beak_mae_y', 'avg_pck_01']
    ]

    table_data = []
    for _, row in best_beak_x.iterrows():
        table_data.append([
            row['model'][:8],
            row['config_name'][:15],
            f"{row['beak_mae_x']:.4f}",
            f"{row['beak_mae_y']:.4f}",
            f"{row['avg_pck_01']:.1f}%"
        ])

    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'Config', 'Beak X', 'Beak Y', 'PCK'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    ax.set_title('Top 5 Configs for Beak X-Axis', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(config.GRID_SEARCH_DIR / 'beak_x_focus_analysis.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created beak_x_focus_analysis.png")


def create_model_comparison(df, config):
    """Compare models across all metrics"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Architecture Comparison', fontsize=16, fontweight='bold')

    metrics = ['avg_mae', 'avg_mae_x', 'avg_mae_y', 'avg_pck_01']
    titles = ['Average MAE', 'Average MAE X', 'Average MAE Y', 'Average PCK@0.1']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        model_stats = df.groupby('model')[metric].agg(['mean', 'std', 'min', 'max'])

        x = np.arange(len(model_stats))
        width = 0.2

        ax.bar(x - width, model_stats['min'], width, label='Best', alpha=0.8, color='green')
        ax.bar(x, model_stats['mean'], width, label='Mean', alpha=0.8, color='blue')
        ax.bar(x + width, model_stats['max'], width, label='Worst', alpha=0.8, color='red')

        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_stats.index)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.GRID_SEARCH_DIR / 'model_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created model_comparison.png")


def create_best_config_report(df, config):
    """Create detailed report of best configurations"""

    report_path = config.GRID_SEARCH_DIR / 'best_configurations_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER GRID SEARCH - BEST CONFIGURATIONS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Overall best
        best_overall = df.loc[df['avg_mae'].idxmin()]
        f.write("OVERALL BEST CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model: {best_overall['model']}\n")
        f.write(f"Config: {best_overall['config_name']}\n")
        f.write(f"Loss Function: {best_overall['loss_function']}\n")
        f.write(f"Learning Rate: {best_overall['learning_rate']:.0e}\n")
        f.write(f"Weight Decay: {best_overall['weight_decay']:.0e}\n")
        f.write(f"Hidden Dim: {int(best_overall['hidden_dim']) if best_overall['hidden_dim'] != 'N/A' else 'N/A'}\n")
        f.write(f"Augmentation: {best_overall['augmentation']}\n")
        f.write(f"Dropout: {best_overall['dropout'] if best_overall['dropout'] != 'N/A' else 'N/A'}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Average MAE: {best_overall['avg_mae']:.4f}\n")
        f.write(f"  Average MAE X: {best_overall['avg_mae_x']:.4f}\n")
        f.write(f"  Average MAE Y: {best_overall['avg_mae_y']:.4f}\n")
        f.write(f"  Beak MAE X: {best_overall['beak_mae_x']:.4f}\n")
        f.write(f"  PCK@0.1: {best_overall['avg_pck_01']:.2f}%\n\n")

        # Best for beak X
        best_beak_x = df.loc[df['beak_mae_x'].idxmin()]
        f.write("BEST FOR BEAK X-AXIS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model: {best_beak_x['model']}\n")
        f.write(f"Config: {best_beak_x['config_name']}\n")
        f.write(f"Loss Function: {best_beak_x['loss_function']}\n")
        f.write(f"Learning Rate: {best_beak_x['learning_rate']:.0e}\n")
        f.write(f"Weight Decay: {best_beak_x['weight_decay']:.0e}\n")
        f.write(f"Beak MAE X: {best_beak_x['beak_mae_x']:.4f}\n\n")

        # Best per model
        f.write("BEST CONFIGURATION PER MODEL:\n")
        f.write("-" * 80 + "\n")
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            best_model = model_df.loc[model_df['avg_mae'].idxmin()]
            f.write(f"\n{model.upper()}:\n")
            f.write(f"  Config: {best_model['config_name']}\n")
            f.write(f"  Avg MAE: {best_model['avg_mae']:.4f}\n")
            f.write(f"  Beak MAE X: {best_model['beak_mae_x']:.4f}\n")
            f.write(f"  PCK@0.1: {best_model['avg_pck_01']:.2f}%\n")

        # Hyperparameter recommendations
        f.write("\n" + "=" * 80 + "\n")
        f.write("HYPERPARAMETER RECOMMENDATIONS:\n")
        f.write("=" * 80 + "\n\n")

        f.write("Based on the grid search results:\n\n")

        best_loss = df.groupby('loss_function')['avg_mae'].mean().idxmin()
        f.write(f"- Best Loss Function: {best_loss}\n")

        best_lr = df.groupby('learning_rate')['avg_mae'].mean().idxmin()
        f.write(f"- Best Learning Rate: {best_lr:.0e}\n")

        best_wd = df.groupby('weight_decay')['avg_mae'].mean().idxmin()
        f.write(f"- Best Weight Decay: {best_wd:.0e}\n")

        best_hidden = df[df['hidden_dim'] != 'N/A'].groupby('hidden_dim')['avg_mae'].mean().idxmin()
        f.write(f"- Best Hidden Dimension: {int(best_hidden)}\n")

        best_aug = df.groupby('augmentation')['avg_mae'].mean().idxmin()
        f.write(f"- Best Augmentation: {best_aug}\n")

        best_dropout = df[df['dropout'] != 'N/A'].groupby('dropout')['avg_mae'].mean().idxmin()
        f.write(f"- Best Dropout: {best_dropout}\n")

    print(f"✓ Created best_configurations_report.txt")

    # Print to console
    with open(report_path, 'r') as f:
        print("\n" + f.read())


if __name__ == '__main__':
    analyze_grid_search_results()
#!/usr/bin/env python3
"""
Complete Model Optimization Pipeline
Runs training, testing, and creates comprehensive reports
"""

import os
import sys
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional for styling

# Import our modules
from optimized_training_system import run_comprehensive_training
from model_testing_suite import ModelTester

def create_final_report(training_results, test_results, best_training_config, best_test_model):
    """Create final comprehensive report"""
    print("📋 Creating final comprehensive report...")

    # Create report dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Training strategies comparison
    ax = axes[0, 0]
    strategies = []
    scores = []

    for strategy, results in training_results.items():
        if 'mlp_standard_logger' in results:
            logger = results['mlp_standard_logger']
            strategies.append(strategy.replace('_', '\n'))
            scores.append(max(logger.metrics['val_r2']))

    bars = ax.bar(range(len(strategies)), scores, alpha=0.7, color='skyblue')
    ax.set_title('Training Strategies Performance', fontweight='bold', fontsize=14)
    ax.set_ylabel('Best Validation R² Score')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, value in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Test results comparison
    ax = axes[0, 1]
    test_models = []
    test_scores = []

    for model_name, results in test_results.items():
        test_models.append(model_name.replace('_', '\n'))
        test_scores.append(results['overall_metrics']['r2'])

    # Show top 10 models
    if len(test_models) > 10:
        sorted_indices = sorted(range(len(test_scores)), key=lambda i: test_scores[i], reverse=True)[:10]
        test_models = [test_models[i] for i in sorted_indices]
        test_scores = [test_scores[i] for i in sorted_indices]

    bars = ax.bar(range(len(test_models)), test_scores, alpha=0.7, color='lightgreen')
    ax.set_title('Top Model Test Performance', fontweight='bold', fontsize=14)
    ax.set_ylabel('Test R² Score')
    ax.set_xticks(range(len(test_models)))
    ax.set_xticklabels(test_models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Functional goals achievement
    ax = axes[1, 0]

    # Count functional goals achievement across all models
    goal_counts = {}
    total_models = len(test_results)

    for model_name, results in test_results.items():
        for goal, passed in results['functional_goals'].items():
            if goal not in goal_counts:
                goal_counts[goal] = 0
            if passed:
                goal_counts[goal] += 1

    goals = list(goal_counts.keys())
    percentages = [goal_counts[goal] / total_models * 100 for goal in goals]

    bars = ax.barh(range(len(goals)), percentages, alpha=0.7, color='orange')
    ax.set_title('Functional Goals Achievement Rate', fontweight='bold', fontsize=14)
    ax.set_xlabel('Achievement Rate (%)')
    ax.set_yticks(range(len(goals)))
    ax.set_yticklabels([goal.replace('_', ' ').title() for goal in goals])
    ax.grid(True, alpha=0.3, axis='x')

    # Add percentage labels
    for bar, value in zip(bars, percentages):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}%', ha='left', va='center', fontweight='bold')

    # Plot 4: Executive summary
    ax = axes[1, 1]
    ax.axis('off')

    # Calculate summary statistics
    total_strategies = len(training_results)
    total_models_tested = len(test_results)

    best_training_score = 0
    best_test_score = 0

    for strategy, results in training_results.items():
        if 'mlp_standard_logger' in results:
            score = max(results['mlp_standard_logger'].metrics['val_r2'])
            best_training_score = max(best_training_score, score)

    for model_name, results in test_results.items():
        score = results['overall_metrics']['r2']
        best_test_score = max(best_test_score, score)

    # Count models meeting functional goals
    models_meeting_goals = 0
    for model_name, results in test_results.items():
        if sum(results['functional_goals'].values()) >= len(results['functional_goals']) * 0.8:
            models_meeting_goals += 1

    summary_text = "🏆 OPTIMIZATION RESULTS SUMMARY\n\n"
    summary_text += f"📊 Training Overview:\n"
    summary_text += f"  • Strategies Tested: {total_strategies}\n"
    summary_text += f"  • Best Training R²: {best_training_score:.4f}\n"
    summary_text += f"  • Best Configuration: {best_training_config}\n\n"

    summary_text += f"🧪 Testing Overview:\n"
    summary_text += f"  • Models Tested: {total_models_tested}\n"
    summary_text += f"  • Best Test R²: {best_test_score:.4f}\n"
    summary_text += f"  • Best Model: {best_test_model}\n\n"

    summary_text += f"🎯 Success Metrics:\n"
    summary_text += f"  • Models Meeting Goals: {models_meeting_goals}/{total_models_tested}\n"
    summary_text += f"  • Success Rate: {models_meeting_goals/total_models_tested*100:.1f}%\n\n"

    summary_text += f"📈 Recommendations:\n"
    if best_test_score > 0.8:
        summary_text += f"  ✅ Excellent performance achieved\n"
    elif best_test_score > 0.7:
        summary_text += f"  ✅ Good performance achieved\n"
    else:
        summary_text += f"  ⚠️ Consider additional optimization\n"

    summary_text += f"  • Use {best_test_model} for production\n"
    summary_text += f"  • Monitor performance regularly\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.savefig('output/training/plots/final_optimization_report.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create detailed text report
    report_content = f"""
# Model Optimization Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Total Training Strategies**: {total_strategies}
- **Total Models Tested**: {total_models_tested}
- **Best Training R²**: {best_training_score:.4f}
- **Best Test R²**: {best_test_score:.4f}
- **Success Rate**: {models_meeting_goals/total_models_tested*100:.1f}%

## Best Configuration
- **Training Strategy**: {best_training_config}
- **Best Test Model**: {best_test_model}

## Detailed Results
### Training Results:
"""

    for strategy, results in training_results.items():
        if 'mlp_standard_logger' in results:
            logger = results['mlp_standard_logger']
            best_score = max(logger.metrics['val_r2'])
            report_content += f"- {strategy}: R² = {best_score:.4f}\n"

    report_content += "\n### Test Results:\n"
    for model_name, results in test_results.items():
        score = results['overall_metrics']['r2']
        goals_met = sum(results['functional_goals'].values())
        total_goals = len(results['functional_goals'])
        report_content += f"- {model_name}: R² = {score:.4f}, Goals = {goals_met}/{total_goals}\n"

    with open('output/training/final_optimization_report.md', 'w') as f:
        f.write(report_content)

    print("✅ Final report created: final_optimization_report.png and .md")

def main():
    """Main execution function"""
    print("🚀 STARTING COMPLETE MODEL OPTIMIZATION PIPELINE")
    print("=" * 100)

    start_time = time.time()

    try:
        # Step 1: Run comprehensive training
        print("\n📚 PHASE 1: COMPREHENSIVE TRAINING")
        print("-" * 50)

        training_results, best_strategy, best_model, best_score = run_comprehensive_training()

        print(f"\n✅ Training completed in {time.time() - start_time:.1f} seconds")

        # Step 2: Run comprehensive testing
        print("\n🧪 PHASE 2: COMPREHENSIVE TESTING")
        print("-" * 50)

        tester = ModelTester()
        test_results, best_test_model, best_test_score = tester.run_comprehensive_tests()

        print(f"\n✅ Testing completed in {time.time() - start_time:.1f} seconds")

        # Step 3: Create final report
        print("\n📋 PHASE 3: FINAL REPORTING")
        print("-" * 50)

        best_training_config = f"{best_strategy}_{best_model}"
        create_final_report(training_results, test_results, best_training_config, best_test_model)

        # Final summary
        total_time = time.time() - start_time

        print("\n🎉 OPTIMIZATION PIPELINE COMPLETED!")
        print("=" * 100)
        print(f"⏱️  Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"🏆 Best Training: {best_training_config} (R² = {best_score:.4f})")
        print(f"🥇 Best Test: {best_test_model} (R² = {best_test_score:.4f})")
        print(f"📁 Results Location: output/training/")
        print(f"📊 Visualizations: output/training/plots/")
        print(f"📋 Logs: output/training/logs/")
        print(f"📄 Final Report: output/training/final_optimization_report.md")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)

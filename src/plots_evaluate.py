import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import probplot

def create_evaluation_plots(all_labels, all_preds):
    # Plotting
    plt.figure(figsize=(20, 20))

    # 1. Predicted vs Actual Scatter Plot
    plt.subplot(2, 2, 1)
    plt.scatter(all_labels, all_preds, alpha=0.5)
    plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], 'r--', lw=2)
    plt.xlabel('Actual Interaction Score')
    plt.ylabel('Predicted Interaction Score')
    plt.title('Predicted vs Actual Interaction Scores')

    # 2. Residuals Distribution
    plt.subplot(2, 2, 2)
    residuals = all_preds - all_labels
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Count')
    plt.title('Distribution of Residuals')

    # 3. Q-Q Plot
    plt.subplot(2, 2, 3)
    probplot(residuals, plot=plt)
    plt.title('Q-Q Plot of Residuals')

    # 4. Residuals vs Predicted Values
    plt.subplot(2, 2, 4)
    plt.scatter(all_preds, residuals, alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.axhline(y=0, color='r', linestyle='--')

    plt.tight_layout()
    plt.savefig('evaluation_plots.png')
    plt.close()

    # 5. Actual vs Predicted Line Plot
    plt.figure(figsize=(15, 5))
    sorted_indices = np.argsort(all_labels)
    plt.plot(all_labels[sorted_indices], label='Actual', alpha=0.7)
    plt.plot(all_preds[sorted_indices], label='Predicted', alpha=0.7)
    plt.xlabel('Sorted Sample Index')
    plt.ylabel('Interaction Score')
    plt.title('Actual vs Predicted Interaction Scores (Sorted)')
    plt.legend()
    plt.savefig('actual_vs_predicted_line.png')
    plt.close()

    # 6. Error Distribution Plot
    plt.figure(figsize=(10, 5))
    sns.kdeplot(all_labels, shade=True, label='Actual')
    sns.kdeplot(all_preds, shade=True, label='Predicted')
    plt.xlabel('Interaction Score')
    plt.ylabel('Density')
    plt.title('Distribution of Actual vs Predicted Scores')
    plt.legend()
    plt.savefig('score_distribution.png')
    plt.close()

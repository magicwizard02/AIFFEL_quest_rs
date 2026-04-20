import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

class ExperimentVisualizer:
    def __init__(self, lr, batch, train_samples=116945):
        """
        Initialize the visualizer with training hyperparameters.
        :param lr: Learning rate used in the filename (e.g., 0.001)
        :param batch: Batch size used in the filename (e.g., 256)
        """
        self.lr = lr
        self.batch = batch
        self.train_samples = train_samples
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def plot_summary(self, model_names):
        """
        Generates a 3-panel plot for Mean Loss, Train Accuracy, and Validation Accuracy.
        """
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        data_found = False

        for i, name in enumerate(model_names):
            acc_path = f"results/{name}/metrics/accuracy_LR{self.lr}_B{self.batch}.csv"
            loss_path = f"results/{name}/metrics/loss_LR{self.lr}_B{self.batch}.csv"
            
            if os.path.exists(acc_path) and os.path.exists(loss_path):
                df_acc = pd.read_csv(acc_path)
                df_loss = pd.read_csv(loss_path)
                
                num_epochs = len(df_acc)
                epochs = range(1, num_epochs + 1)
                data_found = True

                epoch_losses = [x.mean() for x in np.array_split(df_loss['batch_loss'].values, num_epochs)]

                axes[0].plot(epochs, epoch_losses, label=name, marker='o', color=self.colors[i % len(self.colors)], markersize=4)
                axes[1].plot(epochs, df_acc['train_acc'], label=name, marker='o', color=self.colors[i % len(self.colors)], markersize=4)
                axes[2].plot(epochs, df_acc['val_acc'], label=name, marker='o', color=self.colors[i % len(self.colors)], markersize=4)

        if data_found:
            titles = ['Mean Training Loss', 'Training Accuracy (%)', 'Validation Accuracy (%)']
            for ax, title in zip(axes, titles):
                ax.set_title(title, fontweight='bold', fontsize=14)
                ax.set_xlabel('Training Epochs')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend()
            plt.tight_layout()
            plt.show()

    def plot_overfitting(self, model_names):
        """
        Visualizes the generalization gap by shading the area between Train and Validation accuracy.
        This is critical for identifying the point where the model begins to memorize training data.
        """
        num_models = len(model_names)
        fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 6))
        
        # Ensure axes is always iterable even for a single model
        if num_models == 1:
            axes = [axes]

        for i, name in enumerate(model_names):
            acc_path = f"results/{name}/metrics/accuracy_LR{self.lr}_B{self.batch}.csv"
            
            if os.path.exists(acc_path):
                df = pd.read_csv(acc_path)
                epochs = range(1, len(df) + 1)
                
                # Plotting both curves for comparison
                axes[i].plot(epochs, df['train_acc'], 'o-', label='Train Accuracy', color='#1f77b4', linewidth=2)
                axes[i].plot(epochs, df['val_acc'], 's--', label='Validation Accuracy', color='#d62728', linewidth=2)
                
                # Shading the gap to highlight the Generalization Error
                axes[i].fill_between(epochs, df['train_acc'], df['val_acc'], color='gray', alpha=0.15, label='Generalization Gap')
                
                axes[i].set_title(f'Overfitting Analysis: {name}', fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Training Epochs', fontsize=12)
                axes[i].set_ylabel('Accuracy (%)', fontsize=12)
                axes[i].legend(loc='best')
                axes[i].grid(True, alpha=0.2, linestyle='--')
            else:
                print(f"File Error: Could not find accuracy data for overfitting plot at {acc_path}")
        
        plt.tight_layout()
        plt.show()

# Execution 
# Ensure that the model_names list contains the exact folder names found within your 'results' directory.
# viz = ExperimentVisualizer(lr=0.001, batch=256)
# viz.plot_summary(['Seq2Seq_Attention_Baseline'])
#viz.plot_overfitting(['Seq2Seq_Attention_Baseline'])
import os.path

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from src.models.transformer_artifical_data import artificial_transformer
from src.models.transformer_artifical_data import artificial_dataset
from src import utils
import csv
import time
#
# def initialize_csv(file_path, description):
#     """
#     Creates a new CSV file with a free-text description at the top.
#
#     Args:
#         file_path (str): The path to the CSV file.
#         description (str): A description of the experiment.
#     """
#     with open(file_path, 'w') as f:
#         # Write the description as a comment at the top
#         f.write(f"# Experiment Description: {description}\n")
#         f.write("# Tracking losses and metrics for each epoch\n")
#         # Write the headers for the epoch data
#         f.write("epoch,train_loss,val_loss\n")


def evaluate_model(model, test_dataset, test_df, batch_size=1, masked_model=True, save=False, file_name="", desc=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    result_folder = os.path.join(utils.get_project_root(), 'src/analysis/results')
    if save:
        if not (file_name and desc):
            raise ValueError("Please provide a file name and a description to save the results.")
        # if file_name does not end with .csv, add it
        if not file_name.endswith('.csv'):
                file_name += '.csv'
        # check if there is a file_name.csv in the result_folder
        if file_name in os.listdir(result_folder):
            raise ValueError(f"File name {file_name} already exists in {result_folder}. Please choose a different name.")

    test_df_w_analysis = test_df.copy()

    # Prepare DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Ensures alignment with test_df
        collate_fn=artificial_dataset.collate_fn
    )

    # Retrieve scaling factors from test_dataset
    targets_mean = test_dataset.targets_mean.to(device)
    targets_std = test_dataset.targets_std.to(device)

    # Lists to store results
    predicted_frequencies = []
    true_frequencies = []
    widths = []
    mses = []
    maes = []
    mes = []


    with torch.no_grad():
        for idx, (padded_sequences, attention_mask, targets) in enumerate(test_loader):
            padded_sequences = padded_sequences.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            # Model prediction
            if masked_model:
                outputs = model(padded_sequences, attention_mask).squeeze(-1)
            else:
                outputs = model(padded_sequences).squeeze(-1)

            # Rescale outputs and targets
            outputs_rescaled = outputs * targets_std + targets_mean
            targets_rescaled = targets * targets_std + targets_mean

            # Undo the logarithm transformation
            # outputs_original = torch.exp(outputs_rescaled) - 1e-6
            # targets_original = torch.exp(targets_rescaled) - 1e-6
            outputs_original = outputs_rescaled
            targets_original = targets_rescaled

            # Convert to CPU and numpy for compatibility
            predicted_frequency = outputs_original.cpu().numpy()
            true_frequency = targets_original.cpu().numpy()

            # Compute MSE for each sample in the batch
            mse = (predicted_frequency - true_frequency) ** 2
            # Compute MAE for each sample in the batch
            mae = np.abs(predicted_frequency - true_frequency)
            # Compute ME for each sample in the batch
            me = predicted_frequency - true_frequency

            # Store results
            predicted_frequencies.extend(predicted_frequency)
            true_frequencies.extend(true_frequency)
            mses.extend(mse)
            maes.extend(mae)
            mes.extend(me)

    test_df_w_analysis['predicted_frequency'] = predicted_frequencies
    test_df_w_analysis['mse'] = mses
    test_df_w_analysis['mae'] = maes
    test_df_w_analysis['me'] = mes

    # Convert lists to numpy arrays
    predicted_frequencies = np.array(predicted_frequencies)
    true_frequencies = np.array(true_frequencies)
    widths = np.array(widths)
    mses = np.array(mses)

    if save:

        save_loc = os.path.join(utils.get_project_root(), 'src/analysis/results', file_name)
        test_df_w_analysis.to_csv(save_loc)
        print(f"Results saved to {save_loc}")
        with open(os.path.join(utils.get_project_root(), 'src/analysis/results/result_desc.csv'), 'a') as file:
            writer = csv.writer(file)
            # write filename, timestamp format yyyy-mm-dd hh:mm:ss, description
            writer.writerow([file_name, time.strftime("%Y-%m-%d %H:%M:%S"), desc])


    return test_df_w_analysis

def analyze_results(df_w_analysis):

    predicted_frequencies = df_w_analysis['predicted_frequency'].values
    true_frequencies = df_w_analysis['frequency'].values
    widths = df_w_analysis['width'].values
    mses = df_w_analysis['mse'].values
    maes = df_w_analysis['mae'].values
    mes = df_w_analysis['me'].values

    # Compute Pearson correlation between frequency and MSE
    freq_mse_corr, freq_mse_pval = pearsonr(true_frequencies, mses)
    print(f"Correlation between True Frequency and MSE: {freq_mse_corr:.4f}, p-value: {freq_mse_pval:.4e}")

    # Compute Pearson correlation betwee<n width and MSE
    width_mse_corr, width_mse_pval = pearsonr(widths, mses)
    print(f"Correlation between Width and MSE: {width_mse_corr:.4f}, p-value: {width_mse_pval:.4e}")
    print(f"Mean squared error: {np.mean(mses):.4f}, Mean error: {np.mean(mes):.4f}, Mean absolute error: {np.mean(maes):.4f}")

    # # Plot True Frequency vs. MSE
    # plt.figure(figsize=(8, 6))
    # plt.scatter(true_frequencies, mses)
    # plt.xlabel('True Frequency')
    # plt.ylabel('Mean Squared Error (MSE)')
    # plt.title('True Frequency vs. MSE')
    # plt.show()

    # Plot Width vs. MSE
    plt.figure(figsize=(8, 6))
    plt.scatter(widths, mses)
    plt.xlabel('Width')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Width vs. MSE')
    plt.show()
    #
    # # Additional Plot: True vs. Predicted Frequencies
    # plt.figure(figsize=(8, 6))
    # plt.scatter(true_frequencies, predicted_frequencies)
    # plt.plot([true_frequencies.min(), true_frequencies.max()],
    #          [true_frequencies.min(), true_frequencies.max()], 'k--', lw=2)
    # plt.xlabel('True Frequency')
    # plt.ylabel('Predicted Frequency')
    # plt.title('Predicted vs. True Frequency')
    # plt.show()

def width_against_mse(df_w_analysis, ax=None):
    if ax is None:
        ax = plt.gca()  # Use current axis if none provided
    widths = df_w_analysis['width'].values
    mses = df_w_analysis['mse'].values
    ax.scatter(widths, mses, s=10)
    ax.set_xlabel('Width')
    ax.set_ylabel('Squared Error')
    ax.set_title('Width vs. MSE')
    return ax

def true_freq_against_mse(df_w_analysis, ax=None):
    if ax is None:
        ax = plt.gca()  # Use current axis if none provided
    true_frequencies = df_w_analysis['frequency'].values
    mses = df_w_analysis['mse'].values
    widths = df_w_analysis['width'].values
    min_width = 400
    max_width = 600
    widths_normalized = (widths - min_width) / (max_width - min_width)
    ax.scatter(true_frequencies, mses, s=10, c=widths_normalized, cmap='viridis')
    ax.set_xlabel('True Frequency')
    ax.set_ylabel('Squared Error')
    return ax

def true_freq_against_pred_freq(df_w_analysis, ax=None):
    if ax is None:
        ax = plt.gca()  # Use current axis if none provided
    true_frequencies = df_w_analysis['frequency'].values
    predicted_frequencies = df_w_analysis['predicted_frequency'].values
    widths = df_w_analysis['width'].values
    min_width = 400
    max_width = 600
    widths_normalized = (widths - min_width) / (max_width - min_width)
    ax.scatter(true_frequencies, predicted_frequencies, s=10, c=widths_normalized, cmap='viridis')
    ax.plot([true_frequencies.min(), true_frequencies.max()],
             [true_frequencies.min(), true_frequencies.max()], 'k--', lw=2)
    ax.set_xlabel('True Frequency')
    ax.set_ylabel('Predicted Frequency')
    return ax



if __name__ == '__main__':
    pass
    # # Assuming you have your trained model and test_dataset, test_df
    # # Load datasets and dataframes
    # datasets, dataframes = artificial_dataset.split_data_into_datasets()
    # test_dataset = datasets['test']
    # test_df = dataframes['test']
    #
    # # Load your trained model
    # # If you haven't trained the model yet, make sure to do so before this step
    # # model = ...  # Load or define your trained model
    #
    # # Evaluate the model on the test dataset
    # predicted_freqs, true_freqs, widths, mses = evaluate_model(model, test_dataset, test_df)
    #
    # # Analyze results
    # analyze_results(predicted_freqs, true_freqs, widths, mses)
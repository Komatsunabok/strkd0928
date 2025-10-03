import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import glob
from matplotlib import gridspec

def visualize_cka_matrix(csv_path=None, cka_matrix=None, 
                         model1_name='model1', model2_name='model2',
                         show_img=False, save_img=False):
    """
    入力：
        csv_path: ckaマトリクスをほぞんするcsvファイルのパス
        cka_matrix: 
        csv_pathかcka_matrix
    """
    if cka_matrix==None:
        cka_matrix = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip the header row
            for row in reader:
                cka_matrix.append([float(value) for value in row])

    cka_array = np.array(cka_matrix)

    # Plot the CKA matrix
    plt.figure(figsize=(7, 3))
    plt.imshow(cka_array, cmap='inferno')
    plt.colorbar()
    plt.xlabel(f'{model1_name} Layers')
    plt.ylabel(f'{model2_name} Layers')
    plt.title(f'CKA Matrix - {os.path.basename(csv_path)}')
    plt.show()


def visualize_all_cka_matrices(folder_path, model1_name='vgg13', model2_name='vgg13'):
    """
    Visualizes all CKA matrices from CSV files in the specified folder.

    Parameters:
    - folder_path (str): Path to the folder containing CKA CSV files.
    - model_name_s (str): Name of the student model (for Y-axis label).
    - model_name_t (str): Name of the teacher model (for X-axis label).
    """
    # Find all CSV files matching the pattern
    csv_files = sorted(glob.glob(os.path.join(folder_path, 'cka_epoch_*.csv')))

    if not csv_files:
        print("No CSV files found in the specified folder.")
        return
    count=0
    for csv_path in csv_files:
        count += 1
        if count >5: break
        # Read the CSV and parse the matrix
        visualize_cka_matrix(csv_path, model1_name, model2_name, show_img=True)

def visualize_and_save_cka_grid(folder_path, model_name_s='vgg13', model_name_t='vgg13'):
    """
    Reads all CKA matrix CSV files from the specified folder, visualizes each matrix,
    and combines them into a single image grid. Saves the combined image in an 'image'
    directory inside a subfolder named after the input folder's basename.

    Parameters:
    - folder_path (str): Path to the folder containing CKA CSV files.
    - model_name_s (str): Name of the student model (for Y-axis label).
    - model_name_t (str): Name of the teacher model (for X-axis label).
    """
    # Find all CSV files matching the pattern
    csv_files = sorted(glob.glob(os.path.join(folder_path, 'cka_epoch_*.csv')))
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    # Create output directory
    base_folder_name = os.path.basename(folder_path.rstrip('/'))
    output_dir = os.path.join('image', base_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare figure grid
    num_files = len(csv_files)
    cols = min(4, num_files)
    rows = (num_files + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 4, rows * 3))
    spec = gridspec.GridSpec(rows, cols, figure=fig)

    for idx, csv_path in enumerate(csv_files):
        # Read the CSV and parse the matrix
        cka_matrix = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip the header row
            for row in reader:
                cka_matrix.append([float(value) for value in row])
        cka_array = np.array(cka_matrix)

        # Plot each matrix in a subplot
        ax = fig.add_subplot(spec[idx])
        im = ax.imshow(cka_array, cmap='inferno')
        ax.set_title(os.path.basename(csv_path), fontsize=8)
        ax.set_xlabel(f'{model_name_t} Layers')
        ax.set_ylabel(f'{model_name_s} Layers')

    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # Save the combined image
    output_path = os.path.join(output_dir, 'cka_grid.png')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"CKA grid image saved to {output_path}")


if __name__ == '__main__':
    # visualize_all_cka_matrices('../save/cka_logs/cka_log_S-vgg13_T-vgg13_cifar10_kd_r-1.0_a-1.0_b-400.0_0')
    visualize_and_save_cka_grid('../save/cka_logs/cka_log_S-vgg13_T-vgg13_cifar10_ckad_r-1.0_a-1.0_b-400.0_0_Distill_gn-4_me-mean_red-mean_sgrp-uniform')

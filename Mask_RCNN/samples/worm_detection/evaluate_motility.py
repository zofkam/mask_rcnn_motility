import pandas as pd
import os
import numpy as np

# threshold for motility above value dead below motile
THRESHOLD = 0.8
MIN_FRAMES = 10
WORM_MULTIPLIER = 1.1


def calculate_motility(input_folder, output_file):
    """
    Calculate the motility for multiple videos from the links csv files and store the results as a csv

    :param input_folder:
    :param output_file:
    :return:
    """

    all_files = sorted(os.scandir(input_folder), key=lambda e: e.name)

    output = []

    # process all files within the input folder
    for entry in all_files:
        # process only files ending with "links.csv"
        if entry.name.endswith("links.csv") and entry.is_file():
            # print('Processing file: {0}'.format(entry.name))
            out = {}
            df = pd.read_csv(os.path.join(input_folder, entry.name))
            # get the min/max number for each frame
            particle_max = df.groupby('frame').size().max()
            particle_min = df.groupby('frame').size().min()

            # remove ones where the particle didn't exist for the comparison
            df = df.loc[df['IoU'] != -1,]
            # count the number of observations
            df['size'] = df.groupby('particle')['frame'].transform('size')
            # group per particle and calculate mean
            df_grouped = df.groupby('particle')[['IoU', 'size']].mean().reset_index()
            # get the particles with the highest number of frames up to the max detected
            df_grouped = df_grouped.nlargest(int(particle_max * WORM_MULTIPLIER), 'size')
            # remove records with less than min_frames
            # df_grouped = df_grouped.loc[df_grouped['size'] >= MIN_FRAMES,]
            # mark as motile or non-motile based on threshold value
            df_grouped['motile'] = np.where(df_grouped['IoU'] >= THRESHOLD, 0, 1)
            # extract required data
            out['group'] = entry.name.split('z')[0]
            out['file_name'] = entry.name
            # calculate motility percentage
            out['motility'] = df_grouped['motile'].sum() / df_grouped.shape[0]
            out['total_motile'] = df_grouped['motile'].sum()
            out['total_worms'] = df_grouped.shape[0]
            output.append(out)

    df_mask = pd.DataFrame(output)

    df_mask.to_csv(output_file,index=False)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate motility output')
    parser.add_argument('--in_folder', required=True,
                        metavar="/path/to/files/",
                        help="Path to folder containing motility output")
    parser.add_argument('--out_path', required=True,
                        metavar="/path/to/files/my_output.csv",
                        help="Path to save outputs from motility evaluation")

    args = parser.parse_args()

    calculate_motility(args.in_folder, args.out_path)


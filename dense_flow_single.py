import dense_flow_cv as df
from pathlib import Path
import argparse

# run on specific files instead of an entire directory


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process videos and analyze motility.")
    parser.add_argument('vidlist', nargs='+', type=str,
                        help='video(s) to be analyzed')
    args = parser.parse_args()
    vidlist = args.vidlist

    fix_list = []
    wd = Path.cwd()
    for file in vidlist:
        path = wd.joinpath(file)
        fix_list.append(path)

    for file in fix_list:
        vid = file
        motility = df.optical_flow(wd, vid)
        normalization_factor = df.segment_worms(wd, vid)
        outfile = str(wd) + '.csv'
        df.wrap_up(vid, outfile, motility, normalization_factor)
        print("Video:", file, "Total motility:", motility,
              "Normalization factor:", normalization_factor,
              "Normalized motility:", motility / normalization_factor)

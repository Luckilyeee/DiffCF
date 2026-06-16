

import argparse

from grid_search import plot_only_main


def main():
    parser = argparse.ArgumentParser(description="Plot Pareto frontier from an existing grid_search_results.csv")
    parser.add_argument("--config", required=True, help="Path to a single-dataset config yaml")
    args = parser.parse_args()

    plot_only_main(args.config)


if __name__ == "__main__":
    main()


import argparse
from appliance_energy.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run appliance-energy forecasting pipeline")
    parser.add_argument("--skip-foundation", action="store_true", help="Skip Chronos-Bolt")
    parser.add_argument("--full-sarimax-grid", action="store_true", help="Repeat the full AIC search before fitting the selected model")
    args = parser.parse_args()
    _, scores = run_pipeline(skip_foundation=args.skip_foundation, full_sarimax_grid=args.full_sarimax_grid)
    print(scores.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

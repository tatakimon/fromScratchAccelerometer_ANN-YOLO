"""Quick environment check for the learning workspace."""

import importlib.util


PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "sklearn",
    "torch",
    "ultralytics",
    "cv2",
]


def main() -> None:
    missing = []

    for package in PACKAGES:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
            print(f"missing: {package}")
        else:
            print(f"ok:      {package}")

    if missing:
        print("\nInstall dependencies with:")
        print("pip install -r requirements.txt")
    else:
        print("\nEnvironment looks ready.")


if __name__ == "__main__":
    main()


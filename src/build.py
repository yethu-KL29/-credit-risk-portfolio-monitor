"""One command to build data + model + anomaly flags: python -m src.build"""
from . import anomaly, ingest, model


def main():
    ingest.main()
    model.main()
    anomaly.main()


if __name__ == "__main__":
    main()

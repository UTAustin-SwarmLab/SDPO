import argparse
import json
import pathlib

import datasets


DEFAULT_PARTITIONS = ["biology", "chemistry", "material", "physics"]


def merge_partitions(
    input_dir: str,
    output_dir: str,
    partitions: list[str],
) -> None:
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        records = []
        for partition in partitions:
            json_file = input_path / partition / f"{split}.json"
            if not json_file.exists():
                raise FileNotFoundError(f"Missing {json_file}")

            with json_file.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    record["dataset"] = f"sciknoweval_{partition}"
                    records.append(record)

        for idx, record in enumerate(records):
            record["idx"] = idx

        ds = datasets.Dataset.from_list(records)
        out_file = output_path / f"{split}.json"
        ds.to_json(out_file)
        print(f"Saved {len(records)} {split} examples to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge SciKnowEval partition train/test splits into combined files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/sciknoweval",
        help="Directory containing per-partition subfolders (biology, chemistry, ...)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/sciknoweval/all",
        help="Directory where merged train.json and test.json will be written",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        nargs="+",
        default=DEFAULT_PARTITIONS,
        help="Partition subfolders to merge",
    )
    args = parser.parse_args()
    merge_partitions(args.input_dir, args.output_dir, args.partitions)

import json
import argparse
import os

def main(x_shot):
    # Path to prediction results
    input_path = f"output/vitl/dataset1_{x_shot}shot/inference/coco_instances_results.json"
    output_path = f"./dataset1_{x_shot}shot.json"

    # Load predictions
    with open(input_path, "r") as f:
        predictions = json.load(f)

    # Adjust category_id: from 1-based → 0-based
    for pred in predictions:
        pred["category_id"] -= 1  # Shift down by 1

    # Save new prediction file
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Modified predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix category_id for dataset1 x-shot predictions")
    parser.add_argument("-x", "--xshot", required=True, type=int, help="Number of shots (e.g. 1, 5, 10)")
    args = parser.parse_args()
    main(args.xshot)
import datasets
import argparse



# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()
    
    print(f"Downloading... dataset: {args.name} | split: {args.split} | config name: {args.config_name} | save path: {args.save_path}")
    
    load_kwargs = {}
    if args.config_name is not None:
        load_kwargs["name"] = args.config_name
    if args.split is not None:
        load_kwargs["split"] = args.split

    dataset = datasets.load_dataset(args.name, **load_kwargs)
    print(f"Dataset downloaded successfully: {dataset}")

    dataset.save_to_disk(args.save_path)
    print(f"Dataset saved successfully to {args.save_path}")
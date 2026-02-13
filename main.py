import yaml
import pandas as pd
from classifier import FakeNewsClassifier # Assuming the class name in your file

def load_config(path="config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_pipeline():
    # 1. Load Configurations
    cfg = load_config()
    print(f"--- Starting {cfg['project_name']} v{cfg['version']} ---")

    # 2. Load Dataset
    try:
        df = pd.read_csv(cfg['data']['raw_path'])
        print(f"Successfully loaded {len(df)} articles.")
    except FileNotFoundError:
        print("Error: Dataset not found. Please run the generation script first.")
        return

    # 3. Initialize Classifier (Integrated with config.yml)
    # We pass the hyperparams from your config to your uploaded classifier logic
    model = FakeNewsClassifier(
        nb_alpha=cfg['models']['naive_bayes']['alpha'],
        svm_kernel=cfg['models']['svm']['kernel'],
        test_size=cfg['data']['test_size']
    )

    # 4. Execute Workflow
    print("Preprocessing text and training models...")
    model.train(df)
    
    # 5. Evaluate
    metrics = model.evaluate()
    print("\nModel Performance:")
    for metric in cfg['output']['metrics']:
        print(f"{metric.capitalize()}: {metrics.get(metric, 'N/A')}")

    # 6. Save Model
    model.save(cfg['output']['model_save_dir'])
    print(f"\nModel weights saved to {cfg['output']['model_save_dir']}")

if __name__ == "__main__":
    run_pipeline()

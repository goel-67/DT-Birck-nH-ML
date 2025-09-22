import sys
import json


def main():
    # Ensure project root is on path
    if '.' not in sys.path:
        sys.path.append('.')

    from src.data.data_manager import DataManager

    dm = DataManager()

    print("[check] Loading dataset...")
    df = dm.load_dataset()
    print(f"[check] Dataset rows: {len(df)}")

    print("[check] Reading recipes Excel...")
    recipes = dm.read_recipes_excel()
    if recipes is None:
        print("[check] ERROR: Could not read Recipes Excel (SharePoint). Aborting.")
        sys.exit(1)

    # Determine maximum iteration from Excel
    max_iter = 0
    if 'Iteration_num' in recipes.columns:
        try:
            max_iter = int(recipes['Iteration_num'].max())
        except Exception:
            max_iter = 6
    else:
        max_iter = 6

    results = {}
    for iter_num in range(1, max_iter + 1):
        try:
            print(f"[check] Computing training data for iteration {iter_num}...")
            train_df = dm.get_training_data_for_iteration(df, recipes, iter_num)
            results[iter_num] = len(train_df)
            print(f"[check] Iteration {iter_num}: training points = {len(train_df)}")
        except Exception as e:
            results[iter_num] = f"ERROR: {e}"
            print(f"[check] Iteration {iter_num}: ERROR {e}")

    print("\n[check] Summary (iteration -> training points):")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()



import torch

path = "models/best_model (1).pt"

obj = torch.load(path, map_location="cpu", weights_only=False)

print("Type of loaded object:", type(obj))
print(f"\nTotal keys: {len(obj)}")

print("\n--- FIRST 10 keys ---")
for k in list(obj.keys())[:10]:
    print(" -", k, "shape:", obj[k].shape if hasattr(obj[k], "shape") else "")

print("\n--- LAST 15 keys ---")
for k in list(obj.keys())[-15:]:
    print(" -", k, "shape:", obj[k].shape if hasattr(obj[k], "shape") else "")


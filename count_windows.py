import modal

app = modal.App("count-windows")

# Volume
model_volume = modal.Volume.from_name("nba-models")

# Simple image
image = modal.Image.debian_slim()

@app.function(
    image=image,
    timeout=300,
    volumes={"/models": model_volume}
)
def count_window_files():
    """Just count the window model files"""
    import os
    from pathlib import Path
    
    models_dir = Path("/models")
    window_files = list(models_dir.glob("player_models_*.pkl"))
    
    return len(window_files)

@app.local_entrypoint()
def main():
    count = count_window_files.remote()
    print(f"Window count: {count}")
    return count

if __name__ == "__main__":
    main()

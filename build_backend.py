import os
import subprocess
import shutil
import platform

# Telepítsük a szükséges függőségeket
print("Installing dependencies...")
subprocess.check_call(["pip", "install", "pyinstaller", "flask", "flask-cors"])

# Hozzuk létre a PyInstaller számára a futtatható fájlt
print("Building with PyInstaller...")
if os.path.exists("dist"):
    shutil.rmtree("dist")

subprocess.check_call(["pyinstaller", "football_predictor.spec"])

# A kész futtatható fájl elérési útja
dist_path = os.path.join("dist", "football_predictor_backend")
if platform.system() == "Windows":
    backend_exe = os.path.join(dist_path, "football_predictor_backend.exe")
else:
    backend_exe = os.path.join(dist_path, "football_predictor_backend")

# Ellenőrizzük, hogy létezik-e
if os.path.exists(backend_exe):
    print(f"Successfully built backend: {backend_exe}")
    
    # Másoljuk át a Tauri projektbe
    tauri_backend_dir = os.path.join("football-predictor-ui", "src-tauri", "backend")
    os.makedirs(tauri_backend_dir, exist_ok=True)
    
    if platform.system() == "Windows":
        target_path = os.path.join(tauri_backend_dir, "football_predictor_backend.exe")
    else:
        target_path = os.path.join(tauri_backend_dir, "football_predictor_backend")
    
    # Töröljük, ha már létezik
    if os.path.exists(target_path):
        os.remove(target_path)
    
    # Másoljuk a teljes dist mappát
    shutil.copytree(dist_path, tauri_backend_dir, dirs_exist_ok=True)
    print(f"Copied backend to Tauri project: {tauri_backend_dir}")
else:
    print(f"Error: Backend executable not found at {backend_exe}")

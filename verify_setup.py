import subprocess
import sys

def check_and_install_dependencies():
    required_packages = [
        'streamlit',
        'pandas',
        'sentence-transformers',
        'scikit-learn',
        'python-dotenv',
        'numpy',
        'google-generativeai',
        'google-cloud-aiplatform',
        'requests',
        'protobuf'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} has been installed")

if __name__ == "__main__":
    print("Verifying and installing dependencies...")
    check_and_install_dependencies()
    print("\nSetup complete! You can now run the main application.")

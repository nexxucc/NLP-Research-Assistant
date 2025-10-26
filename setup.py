"""
Setup Script for NLP Research Assistant

Automates initial project setup.
"""

import os
from pathlib import Path


def create_directories():
    """Create necessary project directories."""
    directories = [
        "data/papers",
        "data/faiss_index",
        "data/processed",
        "data/training",
        "models/custom_embeddings",
        "models/finetuned_llm",
        "outputs/reports",
        "outputs/evaluations",
    ]

    print("Creating project directories...")

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   [OK] Created {dir_path}")

    print("\nDirectories created successfully!")


def check_env_file():
    """Check if .env file exists."""
    env_path = Path(".env")

    if not env_path.exists():
        print("\nWARNING: .env file not found!")
        print("   Creating from .env.example...")

        example_path = Path(".env.example")
        if example_path.exists():
            import shutil
            shutil.copy(example_path, env_path)
            print("   [OK] .env file created")
            print("\nIMPORTANT: Edit .env and add your GOOGLE_API_KEY")
            print("   Get key from: https://makersuite.google.com/app/apikey")
        else:
            print("   ERROR: .env.example not found")
    else:
        print("\n[OK] .env file exists")


def main():
    """Run setup."""
    print("=" * 60)
    print("  NLP Research Assistant - Project Setup")
    print("=" * 60)
    print()

    # Create directories
    create_directories()

    # Check .env
    check_env_file()

    print("\n" + "=" * 60)
    print("  Setup Complete!")
    print("=" * 60)

    print("\nNext Steps:")
    print("   1. Edit .env and add your Google Gemini API key")
    print("   2. Add PDF papers to data/papers/")
    print("   3. Run: python scripts/build_index.py")
    print("   4. Run: streamlit run src/ui/streamlit_app.py")

    print("\nHappy researching!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cleanup script to remove vectorstore and clear cached data.
This should be run before deploying to Streamlit Cloud.
"""

import os
import shutil
from pathlib import Path

def cleanup_vectorstore():
    """Remove vectorstore directory and clear cached data"""
    print("🧹 Starting cleanup process...")
    
    # Remove vectorstore directory
    vectorstore_path = Path("vectorstore")
    if vectorstore_path.exists():
        try:
            shutil.rmtree(vectorstore_path)
            print("✅ Removed vectorstore directory")
        except Exception as e:
            print(f"❌ Error removing vectorstore: {e}")
    else:
        print("ℹ️ Vectorstore directory not found")
    
    # Remove any __pycache__ directories
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                cache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_path)
                    print(f"✅ Removed {cache_path}")
                except Exception as e:
                    print(f"❌ Error removing {cache_path}: {e}")
    
    print("🎉 Cleanup complete!")
    print("\n📝 Next steps:")
    print("1. Commit these changes to git")
    print("2. Push to GitHub")
    print("3. Deploy to Streamlit Cloud")
    print("\n💡 The vectorstore will be automatically recreated when needed")

if __name__ == "__main__":
    cleanup_vectorstore() 
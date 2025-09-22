#!/usr/bin/env python3
"""
Simple server to run the Multimodal RAG Backend.
Fixed all inconsistencies and ready to run.
"""

import sys
import os
from pathlib import Path

def main():
    """Main function to start the server."""
    print("🚀 Multimodal RAG Backend Server")
    print("=" * 40)
    
    # Add project to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Create necessary directories
    dirs = ['data', 'models', 'cache', 'storage', 'logs', 'temp']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"📁 Directory ready: {d}")
    
    try:
        # Check essential imports
        print("\n🔍 Checking dependencies...")
        
        try:
            import fastapi
            import uvicorn
            import pydantic
            import numpy
            print("✅ Core dependencies available")
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("📦 Install with: pip install fastapi uvicorn pydantic python-multipart numpy")
            return
        
        # Import and start
        print("\n🏗️ Starting server...")
        from src.api.main import app
        
        print("✅ Server ready!")
        print("🌐 Server URL: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
        print("\n⚠️  Note: Using mock implementations for development/testing")
        print("📝 All endpoints are functional with sample data")
        print("\nPress Ctrl+C to stop the server")
        
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level="info")
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
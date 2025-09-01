#!/usr/bin/env python3
"""
Simple test to verify RAG system works
"""

from rag_system import get_rag_system

def test_simple():
    print("🧪 Simple RAG System Test")
    print("=" * 30)
    
    try:
        # Initialize RAG system
        print("🔄 Initializing RAG system...")
        rag = get_rag_system()
        
        # Check status
        print("\n📊 Checking processing status...")
        status = rag.get_processing_status()
        
        print(f"Total documents in folder: {status.get('total_count', 0)}")
        print(f"Processed documents: {status.get('processed_count', 0)}")
        print(f"New documents: {status.get('new_count', 0)}")
        print(f"Up to date: {status.get('is_up_to_date', False)}")
        
        if status.get('processed_count', 0) > 0:
            print("\n✅ System is working! Documents are processed.")
            
            # Test a simple query
            print("\n🔍 Testing a simple query...")
            try:
                response = rag.query("What documents are available?")
                print(f"Response: {response[:200]}...")
            except Exception as e:
                print(f"Query failed: {e}")
        else:
            print("\n⚠️ No documents processed yet. Use the Force Process button in the frontend.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_simple()

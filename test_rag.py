#!/usr/bin/env python3
"""
Test script to verify RAG system is working with new PDFs
"""

from rag_system import get_rag_system
import os

def test_rag_system():
    print("🧪 Testing RAG System with New PDFs")
    print("=" * 50)
    
    # Initialize RAG system
    rag = get_rag_system()
    
    # Check documents in folder
    print("\n📁 Documents in folder:")
    documents = rag.list_documents()
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    
    # Check vector store status
    print(f"\n📊 Vector store status:")
    doc_count = rag.get_document_count()
    print(f"  Documents in vector store: {doc_count}")
    
    # Force process all documents
    print("\n⚡ Force processing all documents...")
    success = rag.force_process_all_documents()
    
    if success:
        print("✅ Documents processed successfully!")
        
        # Test queries
        print("\n🔍 Testing queries...")
        
        # Test agriculture query
        print("\n🌾 Testing Agriculture Query:")
        agri_question = "What are the key agricultural statistics for 2023?"
        agri_answer = rag.query(agri_question)
        print(f"Q: {agri_question}")
        print(f"A: {agri_answer[:200]}...")
        
        # Test population query
        print("\n👥 Testing Population Query:")
        pop_question = "What are the population statistics for 2016?"
        pop_answer = rag.query(pop_question)
        print(f"Q: {pop_question}")
        print(f"A: {pop_answer[:200]}...")
        
        # Test combined query
        print("\n🔗 Testing Combined Query:")
        combined_question = "How do agricultural statistics relate to population data?"
        combined_answer = rag.query(combined_question)
        print(f"Q: {combined_question}")
        print(f"A: {combined_answer[:200]}...")
        
    else:
        print("❌ Failed to process documents")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    test_rag_system()

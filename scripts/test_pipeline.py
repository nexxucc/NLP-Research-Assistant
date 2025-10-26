"""
Test Script for RAG Pipeline

Tests the complete end-to-end pipeline with sample data.

Usage:
    python scripts/test_pipeline.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.pipeline import RAGPipeline


def test_pipeline():
    """Test complete RAG pipeline."""
    print("=" * 80)
    print("TESTING RAG PIPELINE")
    print("=" * 80)

    # Check API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == 'your_api_key_here':
        print("\n❌ ERROR: GOOGLE_API_KEY not set in .env file")
        print("Please add your API key to .env and try again")
        return False

    print("\n✅ API key found")

    # Check for papers
    papers_dir = "data/papers"
    papers_path = Path(papers_dir)

    if not papers_path.exists() or not list(papers_path.glob("*.pdf")):
        print(f"\n❌ ERROR: No PDF files found in {papers_dir}")
        print(f"Please add PDF files to {papers_dir} and try again")
        return False

    pdf_files = list(papers_path.glob("*.pdf"))
    print(f"\n✅ Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")

    # Initialize pipeline
    print("\n" + "=" * 80)
    print("INITIALIZING PIPELINE")
    print("=" * 80)

    pipeline = RAGPipeline(
        chunk_size=512,
        chunk_overlap=50,
        top_k=5
    )

    # Load papers
    print("\n📄 Loading papers...")
    num_papers = pipeline.load_papers(papers_dir)
    print(f"✅ Loaded {num_papers} papers")

    # Build index
    print("\n🔨 Building FAISS index...")
    num_chunks = pipeline.build_index()
    print(f"✅ Created {num_chunks} chunks")

    # Save index
    print("\n💾 Saving index...")
    pipeline.save_index("data/faiss_index")
    print("✅ Index saved")

    # Test queries
    print("\n" + "=" * 80)
    print("TESTING QUERIES")
    print("=" * 80)

    test_questions = [
        "What is the main contribution of this research?",
        "What methodology was used?",
        "What were the key findings?"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}: {question}")
        print(f"{'='*80}")

        try:
            result = pipeline.query(question, top_k=3)

            print(f"\n📝 ANSWER:")
            print(result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'])

            print(f"\n📚 SOURCES:")
            for source in result['sources']:
                print(f"   - {source['filename']}, Page {source.get('page', 'N/A')}")

            print("\n✅ Query successful")

        except Exception as e:
            print(f"\n❌ Query failed: {e}")
            return False

    # Show stats
    print("\n" + "=" * 80)
    print("PIPELINE STATISTICS")
    print("=" * 80)

    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"{key:.<40} {value}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)

    print("\n🎉 Your RAG pipeline is working correctly!")
    print("\nNext steps:")
    print("   • Run the Streamlit app: streamlit run src/ui/streamlit_app.py")
    print("   • Try different questions")
    print("   • Experiment with hybrid retrieval")

    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)

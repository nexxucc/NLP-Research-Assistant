"""
PDF Loader Module

Handles loading and extracting text from research papers (PDFs).
Includes metadata extraction and text cleaning.
"""

from typing import List, Dict, Optional
from pathlib import Path
import logging
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Loads and extracts text from PDF research papers.

    This class handles PDF reading, text extraction, and basic cleaning.
    Preserves structure by tracking page numbers for citations.
    """

    def __init__(self):
        """Initialize PDF loader."""
        self.supported_extensions = ['.pdf']

    def load_single_pdf(self, file_path: str) -> Dict[str, any]:
        """
        Load a single PDF file and extract text with metadata.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing:
                - text: Full extracted text
                - metadata: Dict with filename, page_count, etc.
                - pages: List of dicts with page-level text and page numbers

        Example:
            >>> loader = PDFLoader()
            >>> doc = loader.load_single_pdf("paper.pdf")
            >>> print(doc['metadata']['page_count'])
            >>> print(doc['pages'][0]['text'][:100])
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        try:
            reader = PdfReader(str(file_path))

            # Extract metadata
            metadata = {
                'filename': file_path.name,
                'filepath': str(file_path.absolute()),
                'page_count': len(reader.pages),
            }

            # Try to extract PDF metadata if available
            if reader.metadata:
                metadata['title'] = reader.metadata.get('/Title', '')
                metadata['author'] = reader.metadata.get('/Author', '')
                metadata['subject'] = reader.metadata.get('/Subject', '')

            # Extract text page by page
            pages = []
            full_text = []

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()

                if page_text.strip():  # Only add non-empty pages
                    pages.append({
                        'page_num': page_num,
                        'text': self._clean_text(page_text)
                    })
                    full_text.append(page_text)

            result = {
                'text': '\n\n'.join(full_text),
                'metadata': metadata,
                'pages': pages
            }

            logger.info(f"Loaded PDF: {file_path.name} ({metadata['page_count']} pages)")
            return result

        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise

    def load_directory(self, directory_path: str) -> List[Dict[str, any]]:
        """
        Load all PDF files from a directory.

        Args:
            directory_path: Path to directory containing PDFs

        Returns:
            List of document dictionaries (same format as load_single_pdf)

        Example:
            >>> loader = PDFLoader()
            >>> docs = loader.load_directory("data/papers")
            >>> print(f"Loaded {len(docs)} papers")
        """
        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        pdf_files = list(directory.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []

        documents = []
        successful = 0
        failed = 0

        for pdf_file in pdf_files:
            try:
                doc = self.load_single_pdf(str(pdf_file))
                documents.append(doc)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to load {pdf_file.name}: {str(e)}")
                failed += 1

        logger.info(f"Loaded {successful} PDFs successfully, {failed} failed")
        return documents

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and formatting artifacts.

        Args:
            text: Raw text from PDF

        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = ' '.join(text.split())

        # Remove common PDF artifacts
        text = text.replace('\x00', '')
        text = text.replace('\uf0b7', '•')  # Replace bullet point encoding

        return text.strip()


if __name__ == "__main__":
    # Example usage
    loader = PDFLoader()

    # Test loading a single file
    # doc = loader.load_single_pdf("data/papers/example.pdf")
    # print(f"Loaded: {doc['metadata']['filename']}")
    # print(f"Pages: {doc['metadata']['page_count']}")
    # print(f"First 200 chars: {doc['text'][:200]}")

    # Test loading a directory
    # docs = loader.load_directory("data/papers")
    # print(f"\nLoaded {len(docs)} documents total")

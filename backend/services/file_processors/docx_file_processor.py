from docx import Document

from backend.services.file_processors.file_processor import FileProcessor


class DocxFileProcessor(FileProcessor):
    def process_file(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            return '\n'.join(p.text for p in doc.paragraphs)
        except Exception as e:
            return f'Error reading DOCX: {e}'

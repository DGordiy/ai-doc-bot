from pdfminer.high_level import extract_text

from backend.services.file_processors.file_processor import FileProcessor


class PdfFileProcessor(FileProcessor):
    def process_file(self, file_path: str) -> str:
        try:
            text = extract_text(file_path)
            return text
        except Exception as e:
            return f'Error reading PDF: {e}'

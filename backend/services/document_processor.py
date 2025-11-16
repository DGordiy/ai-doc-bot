from pathlib import Path

from backend.services.file_processors.docx_file_processor import DocxFileProcessor
from backend.services.file_processors.pdf_file_processor import PdfFileProcessor
from backend.services.file_processors.txt_file_processor import TxtFileProcessor


class DocumentProcessor:
    def __init__(self):
        self.file_handlers = {
            "pdf": PdfFileProcessor(),
            "docx": DocxFileProcessor(),
            "txt": TxtFileProcessor()
        }
        pass

    # Master function
    def process_file(self, file_path: str) -> str:
        ext = get_file_extension(file_path)
        handler = self.file_handlers.get(ext)

        if handler:
            return handler.process_file(file_path)
        else:
            raise ValueError(f'Unsupported file extension: {ext}')


def get_file_extension(file_path: str) -> str:
    return Path(file_path).suffix[1:].lower()

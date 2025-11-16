from backend.services.file_processors.file_processor import FileProcessor


class TxtFileProcessor(FileProcessor):
    def process_file(self, file_path: str) -> str:
        with open(file_path, 'r') as f:
            return f.read()
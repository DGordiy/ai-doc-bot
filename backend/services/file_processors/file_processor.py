from abc import abstractmethod


class FileProcessor:
    @abstractmethod
    def process_file(self, file_path: str) -> str:
        pass

from abc import abstractmethod, ABC


class FileProcessor(ABC):
    @abstractmethod
    def process_file(self, file_path: str) -> str:
        pass

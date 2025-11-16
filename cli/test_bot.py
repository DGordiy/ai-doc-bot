# CLI tests will go here
from backend.services.document_processor import DocumentProcessor

dp = DocumentProcessor()

# for file_name in ["example.pdf", "example.docx", "example.txt", "example.xyz"]:
#     try:
#         print(dp.process_file(file_name))
#     except ValueError as e:
#         print(e)

MAX_LEN = 1000

print("Testing PDF\n")
pdf_text = dp.process_file("../tests/example.pdf")
print(pdf_text[:MAX_LEN])

print('-' * 50)
print("Testing DOCX\n")
docx_text = dp.process_file("../tests/example.docx")
print(docx_text[:MAX_LEN])

print('-' * 50)
print("Testing TXT\n")
txt_text = dp.process_file("../tests/example.txt")
print(txt_text[:MAX_LEN])
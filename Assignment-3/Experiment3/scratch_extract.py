import PyPDF2
import sys

pdf_path = r"c:\Drive F\HPC labs\Assignment-3\Experiment no 1.pdf"

with open(pdf_path, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    print(f"Total pages: {len(reader.pages)}")
    print("=" * 80)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        print(f"\n{'='*30} PAGE {i+1} {'='*30}")
        print(text)
        print()

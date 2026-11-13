"""PDF raster QA for an explicitly synthetic report, never a real evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert any('synthetic' in part.lower() for part in args.synthetic_output.parts)
    receipt = json.loads((args.synthetic_output / 'private' / 'REPORT_RECEIPT.json').read_bytes())
    assert receipt['status'] == 'SYNTHETIC_REPORT_ONLY'
    args.output.mkdir(parents=True, exist_ok=False)
    import fitz
    from PIL import Image, ImageOps
    metadata = []; thumbnails = []
    for name, pin in receipt['public_whitelist_sha256'].items():
        if not name.endswith('.pdf'): continue
        raw = (args.synthetic_output / 'public' / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pin
        with fitz.open(stream=raw, filetype='pdf') as doc:
            assert len(doc) == 1
            page = doc[0]; text = page.get_text()
            assert 'SYNTHETIC DEMONSTRATION' in text
            assert re.search(r'\b[0-9a-fA-F]{64}\b', text) is None
            assert 'hotpotqa_' not in text
            preview = page.get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False)
            destination = args.output / (Path(name).stem + '_pdf_preview.png')
            preview.save(destination)
            metadata.append({'file': name, 'pages': len(doc), 'text_chars': len(text),
                'font_types': sorted({font[2] for font in page.get_fonts()}),
                'source_pdf_sha256': pin, 'preview_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
            with Image.open(destination) as image:
                thumbnail = ImageOps.contain(image.convert('RGB'), (1050, 1000))
                cell = Image.new('RGB', (1090, 1040), 'white')
                cell.paste(thumbnail, ((1090 - thumbnail.width) // 2, (1040 - thumbnail.height) // 2))
                thumbnails.append(cell)
    assert len(thumbnails) == 4
    sheet = Image.new('RGB', (2180, 2080), 'white')
    for index, thumbnail in enumerate(thumbnails): sheet.paste(thumbnail, ((index % 2) * 1090, (index // 2) * 1040))
    sheet.save(args.output / 'PDF_CONTACT_SHEET.png')
    (args.output / 'RECEIPT.json').write_text(json.dumps({'status': 'PASS_SYNTHETIC_PDF_STRUCTURE_REQUIRES_VISUAL_REVIEW',
        'files': metadata, 'real_data_read': False, 'new_model_calls': 0, 'new_statistics': False}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'SYNTHETIC_PDF_RASTERS_READY', 'pages': len(metadata)}))


if __name__ == '__main__': main()

import re

def clean_bib_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove month lines (both braced/quoted AND unbraced macros like 'month = aug,')
    content = re.sub(r'^\s*month\s*=\s*[^,\n]+,?\s*$\n?', '', content, flags=re.MULTILINE | re.IGNORECASE)

    # 2. Remove urldate, url, doi, isbn, issn fields
    content = re.sub(r'^\s*(urldate|url|doi|isbn|issn)\s*=\s*[^,\n]+,?\s*$\n?', '', content, flags=re.MULTILINE | re.IGNORECASE)

    # 3. Fix unclosed year fields like 'year = {1993,' -> 'year = {1993},'
    content = re.sub(r'(\byear\s*=\s*\{)(\d{4})\s*,?\s*$', r'\1\2},', content, flags=re.MULTILINE | re.IGNORECASE)

    # Clean up blank lines
    content = re.sub(r'\n\s*\n', '\n', content)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("Successfully cleaned .bib file!")

clean_bib_file('serotonin_bib.bib', 'serotonin_bib_clean.bib')
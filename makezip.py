import os
import zipfile

with zipfile.ZipFile("files.zip", "w") as zf:
    for dirname, subdirs, files in os.walk("./"):
        if dirname == './':
            for filename in files:
                if filename in ['files.zip', '.gitignore', 'README.md', 'LICENSE']:
                    continue
                zf.write(filename)
            continue
        continuehere = False
        for notallowed in ['./checkpoints', './.git', './.idea', './report', './paper']:
            if dirname.startswith(notallowed):
                continuehere = True
                break
        if continuehere:
            continue
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
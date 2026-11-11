"""List a zip archive's layout without extracting it, and flag unsafe paths."""

import collections
import sys
import zipfile

z = zipfile.ZipFile(sys.argv[1])
infos = z.infolist()
unsafe = [i.filename for i in infos if i.filename.startswith(("/", "\\")) or ".." in i.filename]
print("entries", len(infos), "| uncompressed GB", round(sum(i.file_size for i in infos) / 1e9, 2))
print("unsafe paths", unsafe[:5])
ext = collections.Counter(
    i.filename.rsplit(".", 1)[-1] if "." in i.filename.split("/")[-1] else "<dir>" for i in infos
)
print(ext.most_common())
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
dirs = collections.Counter("/".join(i.filename.split("/")[:depth]) for i in infos)
for d, n in sorted(dirs.items())[:60]:
    print(f"{n:6d}  {d}")

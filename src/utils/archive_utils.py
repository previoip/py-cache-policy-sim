import zipfile
from pathlib import Path
from io import IOBase

def unload_zip_from_io(buf: IOBase, export_folder):
  export_folder = Path(export_folder)

  with zipfile.ZipFile(buf) as zp:
    # print('extracting zip file content:')
    for member in zp.infolist():
      extract_tgt = export_folder / member.filename
      if extract_tgt.exists():
        # print(f'file already exists: {extract_tgt}')
        pass
      else:
        print('extracting',
          f'size: {member.file_size}',
          f'filename: {member.filename}',
          sep=' - '
        )      
        zp.extract(member, path=export_folder)
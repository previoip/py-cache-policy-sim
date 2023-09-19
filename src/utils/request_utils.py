import requests
from io import BytesIO
from pathlib import Path
from urllib.request import url2pathname
from urllib.parse import urlparse
from re import compile as re_compile
from os.path import join as path_join
from os.path import pathsep, relpath
import tempfile

__re_illegal_chars = re_compile(r'[^\w_. -]')

def sanitize_illegal_chars(string, substring):
  return __re_illegal_chars.sub(substring, string)

def url_to_fp(url):
  url_parsed = urlparse(url)
  netloc_path, file_path = url2pathname(url_parsed.netloc), url2pathname(url_parsed.path)
  netloc_path = sanitize_illegal_chars('_', netloc_path)
  if url_parsed.path == '/':
    return netloc_path.strip('/').strip('\\')
  else:
    return path_join(netloc_path, file_path).strip('/').strip('\\')

def req_header(url, header=None, default_fallback_value=None):
  resp = requests.head(url)
  resp.raise_for_status()
  if header is None:
    return resp.headers
  return resp.headers.get(header, default_fallback_value)

def get_request(url, caching_enable=True, cache_folder='cache', params=None, chunk_size=1024*1024) -> BytesIO:

  cache_folder = Path(cache_folder).resolve()
  cache_file = cache_folder / url_to_fp(url) 

  if caching_enable and cache_file.exists() and cache_file.is_file():
    print(f'using cached file {relpath(cache_file)}')
    buf = BytesIO(cache_file.read_bytes())
  else:
    print(f'downloading file to {relpath(cache_file)}')

    content_length = int(req_header(url, 'Content-Length', 0))
    print('file size:', content_length)

    resp = requests.get(url, params=params, stream=True)
    resp.raise_for_status()

    buf = BytesIO()

    for chunk in resp.iter_content(chunk_size=chunk_size):
      buf.write(chunk)
      progress_frac = buf.getbuffer().nbytes / content_length
      progress_frac = progress_frac if progress_frac < 1 else 1
      
      progress = int(progress_frac * 20)
      print('progress: [',  ('='*progress).ljust(20), f'] {progress_frac*100:.2f}%', sep='')

    resp.close()
  
  buf.seek(0)
  if caching_enable:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(buf.read())
    buf.seek(0)

  return buf


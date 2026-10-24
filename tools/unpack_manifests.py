from pathlib import Path
import argparse,gzip,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def digest(b):return hashlib.sha256(b).hexdigest()
def inside(rel):
 p=(ROOT/rel).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('Path escapes release')
 return p
def main():
 ap=argparse.ArgumentParser(description='Verify and unpack original research manifests')
 ap.add_argument('--check-only',action='store_true')
 args=ap.parse_args()
 entries=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))['files']
 count=0
 for e in entries:
  if 'uncompressed_path' not in e:continue
  payload=inside(e['path']).read_bytes()
  if digest(payload)!=e['sha256']:raise ValueError('Compressed checksum mismatch')
  raw=gzip.decompress(payload)
  if digest(raw)!=e['uncompressed_sha256']:raise ValueError('Manifest checksum mismatch')
  target=inside(e['uncompressed_path'])
  if not args.check_only:
   if target.exists() and digest(target.read_bytes())!=e['uncompressed_sha256']:
    raise ValueError('Existing file differs: '+e['uncompressed_path'])
   target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():target.write_bytes(raw)
  count+=1
 print(str(count)+' original manifests verified'+(' (check only)' if args.check_only else ' and available'))
if __name__=='__main__':main()

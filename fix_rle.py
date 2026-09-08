import pathlib

p = pathlib.Path('src/train.py')
t = p.read_text()

# 1. Add an RLE decode function right after the imports
old_marker = 'logger = logging.getLogger(__name__)'
rle_func = '''logger = logging.getLogger(__name__)


def rle_decode(rle_string: str, shape: tuple, class_id: int) -> np.ndarray:
    \"\"\"Decode a Severstal-style RLE string into a 2D mask.

    Severstal encodes pixels column-major (top-to-bottom, then left-to-right),
    1-indexed start positions. Decode into (H, W) with class_id where set.
    \"\"\"
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if not isinstance(rle_string, str) or not rle_string.strip():
        return mask.reshape(h, w)
    parts = rle_string.split()
    starts = np.asarray(parts[0::2], dtype=int) - 1
    lengths = np.asarray(parts[1::2], dtype=int)
    for start, length in zip(starts, lengths):
        mask[start:start + length] = class_id
    return mask.reshape(w, h).T
'''

assert old_marker in t, 'logger init line not found - inspect manually'
t = t.replace(old_marker, rle_func, 1)

# 2. Replace the sample-building loop to store the RLE string instead of a fake mask path
old_loop = '''        if os.path.exists(csv_path):
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row[\"ImageId\"]
                    cid = row[\"ClassId\"]
                    cname = class_map.get(cid)
                    if cname is None:
                        continue
                    samples.append(
                        {
                            \"source\": \"severstal\",
                            \"class_name\": cname,
                            \"group_id\": f\"severstal/{fname}\",
                            \"image_rel\": f\"train_images/{fname}\",
                            \"mask_rel\": f\"train_images/{fname}\",  # placeholder
                        }
                    )'''

new_loop = '''        if os.path.exists(csv_path):
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row[\"ImageId\"]
                    cid = row[\"ClassId\"]
                    cname = class_map.get(cid)
                    if cname is None:
                        continue
                    samples.append(
                        {
                            \"source\": \"severstal\",
                            \"class_name\": cname,
                            \"group_id\": f\"severstal/{fname}\",
                            \"image_rel\": f\"train_images/{fname}\",
                            \"class_id\": int(cid),
                            \"rle\": row[\"EncodedPixels\"],
                        }
                    )'''

assert old_loop in t, 'sample-building loop not found - inspect manually'
t = t.replace(old_loop, new_loop, 1)

# 3. Replace TrainDataset.__getitem__ mask loading to decode RLE instead of imread
old_getitem = '''        mask_path = os.path.join(self.data_root, s[\"mask_rel\"])
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)'''

new_getitem = '''        native_h, native_w = image.shape[0], image.shape[1]
        mask = rle_decode(s.get(\"rle\", \"\"), (native_h, native_w), s.get(\"class_id\", 1))'''

assert old_getitem in t, 'mask loading block not found - inspect manually'
t = t.replace(old_getitem, new_getitem, 1)

p.write_text(t)
print('train.py fixed: real RLE mask decoding added')

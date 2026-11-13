from pathlib import Path
import shutil

import numpy as np
from google.colab import files
from PIL import Image
from skimage import data

out = Path("/content/lab3_demo")
out.mkdir(exist_ok=True)

hr = Image.fromarray(data.astronaut()).crop((70, 12, 326, 268))
lr = hr.resize((64, 64), Image.Resampling.BICUBIC)

hr.save(out / "hr.png")
lr.save(out / "lr.png")
lr.resize(hr.size, Image.Resampling.BICUBIC).save(out / "bicubic.png")

zero = np.zeros((64, 64), dtype=np.uint8)
mixed = zero.copy()
mixed[8:44, 18:50] = 1

np.save(out / "mask_zero.npy", zero)
np.save(out / "mask_one.npy", np.ones_like(zero))
np.save(out / "mask_mixed.npy", mixed)

archive = shutil.make_archive(str(out), "zip", root_dir=out)
files.download(archive)

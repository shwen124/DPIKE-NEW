import json
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset


class PrivacyPrefixDataset(Dataset):
  """PMET rewrite records built by `build_privacy_requests.py`."""

  def __init__(self, data_dir: str, json_name: str = "privacy_prefix_requests.json", size: Optional[int] = None):
    path = Path(data_dir) / json_name
    if not path.is_file():
      raise FileNotFoundError(
        f"{path} not found. Run dsets/build_privacy_requests.py first."
      )
    with path.open("r", encoding="utf-8") as handle:
      self.data = json.load(handle)
    if size is not None:
      self.data = self.data[:size]
    print(f"Loaded privacy prefix dataset with {len(self)} elements from {path}")

  def __len__(self):
    return len(self.data)

  def __getitem__(self, item):
    return self.data[item]

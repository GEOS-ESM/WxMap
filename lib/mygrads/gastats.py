from dataclasses import dataclass, field
from typing import Optional, Dict, List
import re


# --- Sub-structures ---

@dataclass
class DimensionInfo:
    start: int
    end: int
    type: str
    origin: float
    delta: float


@dataclass
class ContourInfo:
    min: float
    max: float
    interval: float


@dataclass
class StatsBlock:
    values: List[float]


# --- Main structure ---

@dataclass
class GradsStat:
    data_type: Optional[str] = None
    dimensions: Optional[List[int]] = None

    i_dimension: Optional[DimensionInfo] = None
    j_dimension: Optional[DimensionInfo] = None

    sizes: Optional[List[int]] = None

    undef_value: Optional[float] = None
    undef_count: Optional[int] = None
    valid_count: Optional[int] = None

    min: Optional[float] = None
    max: Optional[float] = None

    cmin: Optional[float] = None
    cmax: Optional[float] = None
    cint: Optional[float] = None

    stats: Dict[str, StatsBlock] = field(default_factory=dict)

    contouring: Optional[ContourInfo] = None

    raw_lines: List[str] = field(default_factory=list)

    # --- convenience ---
    @property
    def all_missing(self) -> Optional[bool]:
        if self.valid_count is None:
            return None
        return self.valid_count == 0

    @property
    def total_count(self) -> Optional[bool]:
        if self.valid_count is None:
            if self.undef_count is None:
                return 0
            else:
                return self.undef_count
        elif self.undef_count is None:
            return self.valid_count
        return self.valid_count + self.undef_count

    @property
    def valid_ratio(self) -> str:
        return f'{self.valid_count or 0} / {self.total_count}'

# --- Parser ---

def parse_grads_stat(Lines: List[str]) -> GradsStat:
    result = GradsStat()

    for line in Lines:
        line = line.strip()
        if not line:
            continue

        # --- Undef / Valid counts ---
        if line.startswith("Undef count"):
            m = re.search(r"Undef count = (\d+)\s+Valid count = (\d+)", line)
            if m:
                result.undef_count = int(m.group(1))
                result.valid_count = int(m.group(2))
            continue

        # --- Dimensions ---
        if line.startswith("I Dimension") or line.startswith("J Dimension"):
            key, rest = line.split("=", 1)
            parts = rest.strip().split()
            if len(parts) >=5:
                dim = DimensionInfo(
                    start=int(parts[0]),
                    end=int(parts[2]),
                    type=parts[3],
                    origin=float(parts[4]),
                    delta=float(parts[5]),
                )

                if key.startswith("I"):
                    result.i_dimension = dim
                else:
                    result.j_dimension = dim

            continue

        # --- Stats blocks ---
        if line.startswith("Stats["):
            label, values = line.split("]:", 1)
            label = label.replace("Stats[", "").strip()

            nums = [float(x) for x in values.split()]
            result.stats[label] = StatsBlock(nums)
            continue

        # --- Contouring ---
        if line.startswith("Contouring"):
            m = re.search(r"Contouring:\s+(\S+)\s+to\s+(\S+)\s+interval\s+(\S+)", line)
            if m:
                result.contouring = ContourInfo(
                    min=float(m.group(1)),
                    max=float(m.group(2)),
                    interval=float(m.group(3)),
                )
            continue
        # --- Simple key = value ---
        if "=" in line and not line.startswith("Stats["):
            key, val = [x.strip() for x in line.split("=", 1)]

            if key == "Data Type":
                result.data_type = val

            elif key == "Dimensions":
                result.dimensions = [int(x) for x in val.split()]

            elif key == "Sizes":
                result.sizes = [int(x) for x in val.split()]

            elif key == "Undef value":
                result.undef_value = float(val)

            elif key == "Min, Max":
                nums = val.split()
                result.min = float(nums[0])
                result.max = float(nums[1])

            elif key == "Cmin, cmax, cint":
                nums = val.split()
                result.cmin = float(nums[0])
                result.cmax = float(nums[1])
                result.cint = float(nums[2])

            continue
        # fallback
        result.raw_lines.append(line)

    return result
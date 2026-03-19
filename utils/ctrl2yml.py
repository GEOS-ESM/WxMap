#!/usr/bin/env python3
"""
Convert a GrADS ctl/latest file VARS block into YAML.

Example:
    python ctl_to_yaml.py xgc_tavg_1hr_glo_L1440x721_slv.latest
    python ctl_to_yaml.py htf_inst_15mn_glo_L1440x721_slv.latest -o vars.yml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import yaml


VAR_LINE_RE = re.compile(
    r"""
    ^(?P<grads>\S+)
    =>
    (?P<nc>\S+)
    \s+
    (?P<levels>\d+)
    \s+
    (?P<dims>\S+)
    \s+
    (?P<desc>.+)
    $
    """,
    re.VERBOSE,
)


def parse_ctl(path: str | Path) -> tuple[str, list[dict]]:
    path = Path(path)
    lines = path.read_text().splitlines()

    dset = None
    in_vars = False
    variables: list[dict] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("DSET "):
            dset = line.split(None, 1)[1].strip()
            continue

        if line.startswith("VARS "):
            in_vars = True
            continue

        if line == "ENDVARS":
            in_vars = False
            continue

        if in_vars:
            match = VAR_LINE_RE.match(line)
            if not match:
                raise ValueError(f"Could not parse VARS line:\n{raw_line}")

            variables.append(
                {
                    "grads_name": match.group("grads"),
                    "nc_name": match.group("nc"),
                    "levels": int(match.group("levels")),
                    "dims": match.group("dims"),
                    "description": match.group("desc").strip(),
                }
            )

    if dset is None:
        raise ValueError("No DSET line found.")
    if not variables:
        raise ValueError("No VARS block found.")

    return dset, variables


def dataset_stem_from_dset(dset: str) -> str:
    """
    Example:
    /path/GEOS.cf.fcst.xgc_tavg_1hr_glo_L1440x721_slv.20260318_09z+%y4%m2%d2_%h2%n2z.R0.nc4
    ->
    xgc_tavg_1hr_glo_L1440x721_slv
    """
    name = Path(dset).name
    name = re.sub(r"\.nc4?$", "", name)

    match = re.match(
        r"^(?:GEOS\.[^.]+\.[^.]+\.)?(?P<stem>.+?)\.\d{8}_\d{2}z\+.*$",
        name,
    )
    if match:
        return match.group("stem")

    return name


def extract_units(description: str) -> str:
    """
    Use the last parenthetical group as units.
    """
    matches = re.findall(r"\(([^()]*)\)", description)
    return matches[-1].strip() if matches else ""


def clean_description(description: str) -> str:
    """
    Remove all parenthetical text and collapse whitespace.
    """
    text = re.sub(r"\([^()]*\)", "", description)
    text = text.replace('_',' ')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def move_of_phrase(text: str) -> str:
    """
    Convert:
      "Tropospheric column density of ozone"
    to:
      "ozone Tropospheric column density"
    """
    match = re.match(r"^(.*)\s+of\s+(.+)$", text, flags=re.IGNORECASE)
    if match:
        left = match.group(1).strip()
        right = match.group(2).strip()
        return f"{right} {left}"
    return text


def make_long_name(var: dict) -> str:
    """
    Simpler rules:
    - if , use grads variable name, e.g. SO2
    - otherwise use cleaned description with "X of Y" -> "Y X"
    """
    # if var["levels"] > 0:
    #     return var["grads_name"]
    text = clean_description(var["description"])
    text = move_of_phrase(text)
    return text.title()


def make_expression(nc_name: str, dataset_stem: str, levels: int) -> str:
    expr = f"{nc_name}.{dataset_stem}"
    if levels > 0 and any([s in dataset_stem for s in ['_slv','_x1','_Nx']]):
        expr += "(z=1)"
    return expr


def build_yaml_dict(dset: str, variables: list[dict], suffix: Optional[str]) -> dict:
    suffix = suffix or ''
    dataset_stem = dataset_stem_from_dset(dset)
    output = {}
    for var in variables:
        key = f"_{var['nc_name']}{suffix}"
        output[key] = {
            "long_name": make_long_name(var),
            "units": extract_units(var["description"]),
            "expression": make_expression(
                var["nc_name"],
                dataset_stem,
                var["levels"],
            ),
        }

    return output


def get_yml_text(ctl_file, suffix: Optional[str]='') -> None:

    dset, variables = parse_ctl(ctl_file)
    result = build_yaml_dict(dset, variables,suffix)

    yaml_text = yaml.safe_dump(
        result,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    return yaml_text

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ctl_file", help="Path to ctl/latest file")
    parser.add_argument("-o", "--output", help="Output YAML file")
    parser.add_argument("-s", "--suffix", help="Field Key Suffix")
    args = parser.parse_args()

    yaml_text = get_yml_text(args.ctl_file, args.suffix)
    if args.output:
        Path(args.output).write_text(yaml_text)
    else:
        print(yaml_text)


if __name__ == "__main__":
    main()
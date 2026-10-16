import numpy as np, pandas as pd
import astropy.coordinates as coord, astropy.table as atpy, astropy.units as u
from astroquery.vizier import Vizier
from astropy.io.votable import parse
from scipy import stats
import os
from pathlib import Path
from urllib.request import urlretrieve
import requests
import inspect


def to_hms_format(s):
    parts = s.split()
    return f"{parts[0]}h{parts[1]}m{parts[2]}s"


def to_dms_format(s):
    parts = s.split()
    return f"{parts[0]}d{parts[1]}m{parts[2]}s"


import re


def transform_name(name):
    # Use regex to split the numeric part and check for missing signs
    match = re.match(r"(ComPACT_G\d+\.\d{3})([-+]?\d+\.\d+)", name)
    if match:
        prefix, second_part = match.groups()
        # Check if the second part starts with a sign; add `+` if it's missing
        if not second_part.startswith(("+", "-")):
            second_part = f"+{second_part}"
        return f"{prefix}{second_part}"
    return name


"""
zazn = pd.read_csv('compact_zphot.dat', sep=',')

grouped = zazn.groupby("name").agg(list)

max_columns = grouped["zphot"].map(len).max()  # Максимальное число дубликатов
for i in range(max_columns):
    df[f"zphot_{i+1}"] = df["Name"].map(lambda x: grouped["zphot"][x][i] if x in grouped.index and i < len(grouped["zphot"][x]) else None)
    df[f"IRamp_{i+1}"] = df["Name"].map(lambda x: grouped["IRamp"][x][i] if x in grouped.index and i < len(grouped["IRamp"][x]) else None)
    df[f"signi_{i+1}"] = df["Name"].map(lambda x: grouped["signi"][x][i] if x in grouped.index and i < len(grouped["signi"][x]) else None)

df.to_csv('zazn.csv', index=False)
"""


def download_file_if_missing(path, url):
    """
    Скачивает файл по ссылке, если он отсутствует локально.

    path: путь к локальному файлу
    url:  ссылка на скачивание
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {os.path.basename(path)} from {url}...")
        urlretrieve(url, path)
        print("Download complete.")
    return path


import zipfile


def download_and_extract_if_missing(path, url, extract_filename=None):
    """
    Проверяет, есть ли файл по path. Если нет — скачивает zip по url,
    распаковывает нужный файл (extract_filename) или весь архив,
    удаляет zip и сохраняет файл(ы) по path.

    Parameters:
    -----------
    path : str
        Путь к файлу, который должен быть после распаковки.
    url : str
        Ссылка на zip-архив для скачивания.
    extract_filename : str or None
        Имя файла внутри архива для распаковки.
        Если None — распаковывает все файлы архива.
    """
    if os.path.exists(path):
        # Файл уже есть — не скачиваем
        return

    filename = url.split("?")[0].split("/")[-1]
    if not filename.endswith(".zip"):
        filename += ".zip"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    zip_path = os.path.join(os.path.dirname(path), filename)

    # Скачиваем zip архив
    print(f"Downloading {url} ...")
    r = requests.get(url)
    r.raise_for_status()

    with open(zip_path, "wb") as f:
        f.write(r.content)

    # Распаковываем
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        if extract_filename is None:
            zip_ref.extractall(os.path.dirname(path))
        else:
            zip_ref.extract(extract_filename, os.path.dirname(path))

    os.remove(zip_path)
    print(f"File extracted to {path}")


def download_catalogs(catalog_names, base_path=None):
    """
    Download and process astronomical catalogs by their names.

    Parameters:
    -----------
    catalog_names : str or list
        Catalog name or list of names. Available options:
        'PSZ2', 'ACT_DR5', 'PSZSPT', 'ComPRASS', 'ACT_MCMF',
        'SPT_SZ', 'SPT2500D', 'SPTECS', 'SPT_DEEP', 'SPT_FAR',
        'MCXC2', 'MaDCoWS', 'MaDCoWS_II', 'MaDCoWS_AGN', 'eRASS'

    Returns:
    --------
    dict or pd.DataFrame
        If single catalog requested - returns DataFrame.
        If multiple catalogs - returns dictionary of DataFrames.

    Examples:
    ---------
    >>> # Download single catalog
    >>> psz2 = download_catalogs('PSZ2')
    >>>
    >>> # Download multiple catalogs
    >>> catalogs = download_catalogs(['ACT_DR5', 'SPT_SZ'])
    >>> act_data = catalogs['ACT_DR5']
    """
    project_root = Path(__file__).resolve().parents[2]
    path = (
        Path(base_path).expanduser().resolve()
        if base_path is not None
        else project_root / "data/raw"
    )
    current_path = path
    parent_path = path

    # Catalog configuration dictionary
    CATALOG_CONFIG = {
        "PSZ2": {
            "source": "vizier",
            "code": "J/ApJS/272/7/table2",
            "columns": None,
            "postprocess": lambda df: df.assign(
                M=df["MSZ"],
                Mtype="SZ",
                M_Err=df.E_MSZ,
                M_err=df.e_MSZ,
                zType=np.where(df.f_z == "spec", "spec", "phot"),
            ),
        },
        "PSZ1": {
            "source": "vizier",
            "code": "J/A+A/581/A14/psz1v2_1",
            "columns": [
                "PSZ1",
                "RAJ2000",
                "DEJ2000",
                "z",
                "q_z",
                "MYZ500",
                "E_MYZ500",
                "e_MYZ500",
            ],
            "postprocess": lambda df: df.assign(
                M=df["MYZ500"],
                Mtype="SZ",
                M_Err=df.E_MYZ500,
                M_err=df.e_MYZ500,
                zType=np.where(df.q_z == "spec", "spec", "phot"),
            ),
        },
        "PSZ1_Barenna": {
            "source": "vizier",
            "code": "J/A+A/616/A42/tablea1",
            "columns": [
                "ID",
                "RAJ2000",
                "DEJ2000",
                "<zsp>",
                "zphot",
                "e_zphot",
                "_RA",
                "_DE",
            ],
            "postprocess": lambda df: df.assign(
                RAJ2000=df["_RA"],
                DEJ2000=df["_DE"],
                M=np.nan,
                Mtype="SZ",
                M_Err=np.nan,
                M_err=np.nan,
                z=np.where(df["<zsp>"] > 0, df["<zsp>"], df.zphot),
                zType=np.where(df["<zsp>"] > 0, "spec", "phot"),
            ),
        },
        "PSZ-MCMF": {
            "source": "local_vot",
            "path": os.path.join(
                parent_path,
                "stad2319_supplemental_file",
                "PSZ-SN3_cluster_catalogue.vot",
            ),
            "url": "https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/mnras/525/1/10.1093_mnras_stad2319/2/stad2319_supplemental_file.zip?Expires=1756825473&Signature=yYybm9B~zUzm8a0CP7fif17SWai9BszSr0gX6-MmaMzZk8Aa690b081Shodbsn126Mf1ZlIo4VdfpqYNdOjDR5k2hl25ru5TGlJoUL4dyQYiv-wnSmf3oC~YCCP1IGtVZQIKMD~o~cYTz-Ah86QIiRlq~dT3LZ4Yl0ExAruVE8~WXmB6E2JJfPPadZt-UU61KPSd2ygd0iUzYdJ1br9Ki5vI4k1Zq9AHyEt-EteAFgoTKOYO4jRVhNg7qVF7dchaG89Q3eVYFikkOur1kf1YKquinIXTI9SCcAe16Z1v~xHODSg7YuTwwN8KJD6qA5m5nApE062E0QEoIgQd37TAtQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA",
            "postprocess": lambda df: df.assign(
                RAJ2000=df["RA"],
                DEJ2000=df["DEC"],
                M=df["M500"],
                Mtype="SZ",
                M_Err=df["M500__inf_"],
                M_err=df["M500__sup_"],
                zType="phot",
            ),
        },
        "ACTDR6": {
            "source": "local",
            "path": os.path.join(path, "DR6_cluster-catalog_v0.10.fits"),
            "url": "https://extragalactic.phys.wits.ac.za/act-dr6-clusters/v0.10/products/DR6_cluster-catalog_v0.10.fits",
            "postprocess": lambda df: df.assign(
                name=[str(x, encoding="utf-8") for x in df["name"]],
                zType=[str(x, encoding="utf-8") for x in df["redshiftType"]],
                RAJ2000=df["RADeg"],
                DEJ2000=df["decDeg"],
                M=df["M500cUncorr"],
                M_Err=df["M500cUncorr_errPlus"],
                M_err=df["M500cUncorr_errMinus"],
                Mtype="SZ",
                z=df["redshift"],
            ),
        },
        "ACTDR5": {
            "source": "local",
            "path": os.path.join(path, "DR5_cluster-catalog_v1.1.fits"),
            "url": "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/DR5_cluster-catalog_v1.1.fits",
            "postprocess": lambda df: df.assign(
                name=[str(x, encoding="utf-8") for x in df["name"]],
                zType=[str(x, encoding="utf-8") for x in df["redshiftType"]],
                RAJ2000=df["RADeg"],
                DEJ2000=df["decDeg"],
                M=df["M500cUncorr"],
                M_Err=df["M500cUncorr_errPlus"],
                M_err=df["M500cUncorr_errMinus"],
                Mtype="SZ",
                z=df["redshift"],
            ),
        },
        "PSZSPT": {
            "source": "vizier",
            "code": "J/A+A/647/A106",
            "columns": [
                "r_z",
                "Name",
                "RAJ2000",
                "DEJ2000",
                "M500",
                "b_M500",
                "B_M500",
                "z",
            ],
            "postprocess": lambda df: df.assign(
                M=df["M500"] * 1e-14,
                Mtype="SZ",
                M_Err=df["B_M500"],
                M_err=df["b_M500"],
                zType="phot",
            ),
        },
        # https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/934/129
        "PSZSPT_Salvati": {
            "source": "vizier",
            "code": "J/ApJ/934/129/table5",
            "columns": [
                "PSZ2",
                "RAJ2000",
                "DEJ2000",
                "M500free",
                "E_M500free",
                "e_M500free",
                "z",
            ],
            "postprocess": lambda df: df.assign(
                M=df["M500free"] * 1e-14,
                Mtype="SZ",
                M_Err=df["E_M500free"],
                M_err=df["e_M500free"],
                zType="phot",
            ),
        },
        "ComPRASS": {
            "source": "vizier",
            "code": "J/A+A/626/A7/comprass",
            "columns": [
                "Name",
                "RAJ2000",
                "DEJ2000",
                "M500",
                "z",
                "e_M500",
                "E_M500",
                "r_z",
            ],
            "postprocess": lambda df: df.assign(
                M=df["M500"],
                Mtype="X/SZ",
                M_Err=df.E_M500,
                M_err=df.e_M500,
                zType="phot",
            ),
        },
        "ACT-MCMF": {
            "source": "vizier",
            "code": "J/A+A/690/A322/catalog",
            "columns": [
                "Name",
                "RAJ2000",
                "DEJ2000",
                "zsp1",
                "z1C",
                "z2C",
                "M500-1C",
                "E_M500-1C",
                "e_M500-1C",
                "zspType",
            ],
            "postprocess": lambda df: df.assign(
                z=pd.Series(np.where(df["zsp1"] > 0, df.zsp1, np.nan))
                .combine_first(df["z1C"])
                .combine_first(df["z2C"]),
                M=df["M500-1C"] * 1e-14,
                Mtype="SZ",
                M_Err=df["E_M500-1C"],
                M_err=df["e_M500-1C"],
                zType=np.where(df["zspType"] > 0, "spec", "phot"),
            ),
        },
        "SPT-SZ": {
            "source": "vizier",
            "code": "J/ApJS/216/27/table4",
            "columns": [
                "SPT-CL",
                "RAJ2000",
                "DEJ2000",
                "M500c",
                "e_M500c",
                "z",
                "n_z",
                "f_z",
            ],
            "postprocess": lambda df: df.assign(
                M=df["M500c"],
                Mtype="SZ",
                M_Err=df.e_M500c,
                M_err=df.e_M500c,
                zType=np.where(df["n_z"] == "+", "spec", "phot"),
            ),
        },
        "SPT2500D": {
            "source": "vizier",
            "code": "J/ApJ/878/55/table5",
            "columns": [
                "SPT-CL",
                "RAJ2000",
                "DEJ2000",
                "M500",
                "e_M500",
                "E_M500",
                "z",
                "n_z",
            ],
            "postprocess": lambda df: df.assign(
                M=df["M500"],
                Mtype="SZ",
                M_Err=df.E_M500,
                M_err=df.e_M500,
                zType=np.where(df["n_z"] == "+", "spec", "phot"),
            ),
        },
        "SPTECS": {
            "source": "vizier",
            "code": ["J/ApJS/247/25/table10", "J/ApJS/247/25/cand"],
            "columns": None,
            "postprocess": lambda dfs: pd.concat(dfs)
            .assign(
                M=lambda x: x.M500c,
                Mtype="SZ",
                M_Err=lambda x: x.E_M500c,
                M_err=lambda x: x.e_M500c,
                zType=lambda x: np.where(x.e_z > 0, "phot", "spec"),
            )
            .reset_index(drop=True),
        },
        "SPTDEEP": {
            "source": "local",
            "path": os.path.join(path, "SPT3G_pol_Deep_joint_catalog.fits"),
            "url": "https://pole.uchicago.edu/public/data/spt3g_deep_cluster_sample/SPT3G_pol_Deep_joint_catalog.fits",
            "postprocess": lambda df: df.assign(
                RAJ2000=df["RA"],
                DEJ2000=df["DEC"],
                z=df["REDSHIFT"],
                M=df["M500"],
                Mtype="SZ",
                M_Err=df["M500_UERR"],
                M_err=df["M500_LERR"],
                zType=np.where(df["SPECZ"] == 1, "spec", "phot"),
            ),
        },
        "SPTFAR": {
            "source": "manual",
            "postprocess": lambda: pd.DataFrame(
                {
                    "SPT-CL": [
                        "J0421-4845",
                        "J0607-4448",
                        "J2040-4451",
                        "J0446-4606",
                        "J0459-4947",
                    ],
                    "z": [1.38, 1.401, 1.478, 1.52, 1.72],
                    "M": [2.90, 3.28, 1.47, 1.52, 1.80],
                    "M_Err": [np.nan] * 5,
                    "M_err": [np.nan] * 5,
                    "Mtype": ["SZ"] * 5,
                    "zType": ["phot", "spec", "spec", "phot", "spec"],
                    "RAJ2000": [
                        65.3206,
                        91.8984,
                        310.2483,
                        71.7325,
                        74.9269,
                    ],  # Placeholder - needs actual coordinates
                    "DEJ2000": [
                        -48.7612,
                        -44.8033,
                        -44.8602,
                        -46.1012,
                        -49.7872,
                    ],  # Placeholder - needs actual coordinates
                }
            ),
        },
        "SPT100deg": {
            "source": "local",
            "path": os.path.join(path, "sptpol100d_catalog_huang19.fits"),
            "url": "https://lambda.gsfc.nasa.gov/data/suborbital/SPT/sptpol100/sptpol100d_catalog_huang19.fits",
            "postprocess": lambda df: df.assign(
                RAJ2000=df["RA"],
                DEJ2000=df["Dec"],
                M=df["M500"],
                Mtype="SZ",
                M_Err=df["M500_uerr"],
                M_err=df["M500_lerr"],
                z=df["redshift"],
                zType=np.where(df["redshift_unc"].to_numpy() == 0, "spec", "phot"),
            ),
        },
        "SPT-MCMF": {
            "source": "vizier",
            "code": "J/MNRAS/531/3973/catalog",
            "columns": [
                "SPT",
                "RAJ2000",
                "DEJ2000",
                "zBest",
                "zspec",
                "M500Best",
                "E_M500Best",
                "e_M500Best",
            ],
            "postprocess": lambda df: df.assign(
                M=df["M500Best"],
                Mtype="SZ",
                M_Err=df["E_M500Best"],
                M_err=df["e_M500Best"],
                z=pd.Series(np.where(df["zspec"] > 0, df.zspec, np.nan)).combine_first(
                    df["zBest"]
                ),
                zType=np.where(df["zspec"].to_numpy() > 0, "spec", "phot"),
            ).reset_index(drop=True),
        },
        "MCXC2": {
            "source": "vizier",
            "code": "J/A+A/688/A187/mcxcii",
            "columns": [
                "MCXC",
                "RAJ2000",
                "DEJ2000",
                "z",
                "zType",
                "M500",
                "e_M500",
                "E_M500",
            ],
            "postprocess": lambda df: df.assign(
                M=df["M500"],
                Mtype="X",
                M_Err=df["E_M500"],
                M_err=df["e_M500"],
                zType=np.where(df.zType == "S", "spec", "phot"),
            ),
        },
        "RASS-MCMF": {
            "source": "vizier",
            "code": "J/MNRAS/526/3757/catalog",
            "columns": [
                "Name",
                "RAJ2000",
                "DEJ2000",
                "zsp1",
                "zph1",
                "zph2",
                "zph3",
                "M500-1",
            ],
            "postprocess": lambda df: df.assign(
                M=df["M500-1"] * 1e-14,
                Mtype="X",
                M_Err=np.nan,
                M_err=np.nan,
                z=np.where(
                    df["zsp1"] > 0,
                    df.zsp1,
                    np.where(
                        df.zph1 > 0,
                        df.zph1,
                        np.where(
                            df.zph2 > 0, df.zph2, np.where(df.zph3 > 0, df.zph3, np.nan)
                        ),
                    ),
                ),
                zType=np.where(df["zsp1"] > 0, "spec", "phot"),
            ),
        },
        "MaDCoWS": {
            "source": "vizier",
            "code": "J/ApJS/240/33/table3",
            "columns": None,
            "postprocess": lambda df: df.assign(
                RAJ2000=df["RAJ2000"].apply(
                    lambda x: coord.Angle(to_hms_format(x)).degree
                ),
                DEJ2000=df["DEJ2000"].apply(
                    lambda x: coord.Angle(to_dms_format(x)).degree
                ),
                M=df["M500"],
                Mtype="IK",
                M_Err=df["E_M500"],
                M_err=df["e_M500"],
                z=df["Specz"].combine_first(df["Photz"]).combine_first(df["Photz2"]),
                zType=["spec" if not np.isnan(a) else "phot" for a in df["Specz"]],
            ),
        },
        "MaDCoWS_II": {
            "source": "vizier",
            "code": "J/ApJ/967/123/table6",
            "columns": None,
            "postprocess": lambda df: df.assign(
                M=np.nan,
                Mtype="IK",
                M_Err=np.nan,
                M_err=np.nan,
                z=df["zspec"].combine_first(df["zphot"]),
                zType=[
                    "spec" if not np.isnan(a) else "phot"
                    for a, b in zip(df["zspec"].to_numpy(), df["zphot"].to_numpy())
                ],
            ),
        },
        "MaDCoWS_AGN": {
            "source": "manual",
            "postprocess": lambda: pd.DataFrame(
                {
                    "name": [
                        "MOO J0105+1323",
                        "MOO J0319−0025",
                        "MOO J1014+0038",
                        "MOO J1046+2758",
                        "MOO J1059+5454",
                        "MOO J1142+1527",
                        "MOO J1229+6521",
                        "MOO J1514+1346",
                        "MOO J2206+0906",
                    ],
                    "M": [3.87, 4.81, 5.08, 4.27, 3.42, 6.02, 5.70, 1.84, 5.35],
                    "M_Err": [0.53, 1.22, 1.72, 0.23, 0.66, 0.72, 3.83, 0.93, 0.86],
                    "M_err": [0.91, 1.1, 1, 0.46, 0.59, 0.7, 1.28, 0.32, 0.49],
                    "Mtype": ["IK"] * 9,
                    "z": [1.143, 1.194, 1.230, 1.16, 1.14, 1.189, 0.819, 1.059, 0.926],
                    "zType": [
                        "spec",
                        "spec",
                        "spec",
                        "phot",
                        "phot",
                        "spec",
                        "spec",
                        "spec",
                        "spec",
                    ],
                    "RA": [
                        "01 05 31.5",
                        "03 19 24.4",
                        "10 14 08.4",
                        "10 46 51.6",
                        "10 59 49.4",
                        "11 42 45.1",
                        "12 29 44.6",
                        "15 14 42.7",
                        "22 06 28.6",
                    ],
                    "DEC": [
                        "+13 23 55",
                        "-00 25 21",
                        "+00 38 26",
                        "+27 58 05",
                        "+54 54 37",
                        "+15 27 05",
                        "+65 21 22",
                        "+13 46 31",
                        "+09 06 32",
                    ],
                }
            ).assign(
                RAJ2000=lambda x: x["RA"].apply(
                    lambda r: coord.Angle(to_hms_format(r)).degree
                ),
                DEJ2000=lambda x: x["DEC"].apply(
                    lambda d: coord.Angle(to_dms_format(d)).degree
                ),
            ),
        },
        "eRASS": {
            "source": "vizier",
            "code": "J/A+A/685/A106/emain",
            "columns": [
                "Name",
                "RAJ2000",
                "DEJ2000",
                "RAXdeg",
                "DEXdeg",
                "ExtLike",
                "DetLike0",
                "Exp",
                "zBest",
                "zType",
                "CR500",
                "F500",
                "L500",
                "Mgas500",
                "M500",
                "b_M500",
                "B_M500",
                "R500",
                "F500T",
            ],
            "postprocess": lambda df: df.assign(
                RAJ2000=df["RAXdeg"],
                DEJ2000=df["DEXdeg"],
                M=df["M500"].to_numpy() / 10,
                Mtype="X",
                M_Err=df["B_M500"],
                M_err=df["b_M500"],
                z=df["zBest"],
                zType=np.where(df["zType"] == "Spectroscopic", "spec", "phot"),
            ),
        },
        "XXL-365": {
            # FIND FILE on a cite with proper mass!
            "source": "vizier",
            "code": "J/IX/52/xxl365gc",
            "columns": None,
            "postprocess": lambda df: df.assign(
                RAJ2000=df["RAJ2000"],
                DEJ2000=df["DEJ2000"],
                M=np.nan,  # df['M500scal']/10,
                Mtype="X",
                M_Err=np.nan,  # df['M500scal_err']/10,
                M_err=np.nan,  # df['M500scal_err']/10,
                z=df["z"],
                zType=np.where(df["Ngal"].to_numpy() > 0, "spec", "phot"),
            ),
        },
        "XXL365": {
            "source": "local",
            "path": os.path.join(path, "XXL_365_GC.fits"),
            "postprocess": lambda df: df.assign(
                RAJ2000=df["ra"],
                DEJ2000=df["decl"],
                M=df["M500scal"] / 10,
                Mtype="X",
                M_Err=df["M500scal_err"] / 10,
                M_err=df["M500scal_err"] / 10,
                z=df["z"],
                zType=np.where(df["ngal"].to_numpy() > 0, "spec", "phot"),
            ),
        },
        "XCLASS": {
            "source": "vizier",
            "code": "J/MNRAS/423/3561/xclass",
            "columns": None,
            "postprocess": lambda df: df.assign(
                M=np.nan,
                z=df["z"],
                Mtype="X",
                M_Err=np.nan,
                M_err=np.nan,
                zType=np.where(df["q_z"].to_numpy() == "confirmed", "spec", "phot"),
            ),
        },
        "XCSDR1": {
            "source": "vizier",
            "code": "J/MNRAS/423/1024/xcsdr1",
            "columns": None,
            "postprocess": lambda df: df.assign(
                M=np.nan,
                z=df["z"],
                Mtype="X",
                M_Err=np.nan,
                M_err=np.nan,
                zType=np.where(df["n_z"].to_numpy() == "spec", "spec", "phot"),
            ),
        },
        "RXGCC": {
            "source": "vizier",
            "code": "J/A+A/658/A59/table3",
            "columns": ["RXGCC", "RAJ2000", "DEJ2000", "z", "M500", "e_M500"],
            "postprocess": lambda df: df.assign(
                M=df["M500"],
                z=df["z"],
                Mtype="X",
                M_Err=df["e_M500"],
                M_err=df["e_M500"],
                zType="phot",
            ),
        },
        "REDMAPPER-DES-SVA": {
            "source": "vizier",
            "code": "J/ApJS/224/1/sva1exp",
            "columns": None,
            "postprocess": lambda df: df.assign(
                M=np.nan,
                Mtype="Opt",
                M_Err=np.nan,
                M_err=np.nan,
                z=np.where(df["zspec"] > 0, df["zspec"], df["zlambda"]),
                zType=np.where(df["zspec"].to_numpy() > 0, "spec", "phot"),
            ),
        },
        "REDMAPPER-DES-SDSS": {
            "source": "vizier",
            "code": "J/ApJS/224/1/cat_dr8",
            "columns": None,
            "postprocess": lambda df: df.assign(
                M=np.nan,
                Mtype="Opt",
                M_err=np.nan,
                M_Err=np.nan,
                z=np.where(df["zspec"] > 0, df["zspec"], df["zlambda"]),
                zType=np.where(df["zspec"].to_numpy() > 0, "spec", "phot"),
            ),
        },
        "NEURALENS": {
            "source": "vizier",
            "code": "J/ApJ/909/27/cand",
            "columns": None,
            "postprocess": lambda df: df.assign(
                RAJ2000=df["_RA"],
                DEJ2000=df["_DE"],
                M=np.nan,
                Mtype="Opt",
                M_err=np.nan,
                M_Err=np.nan,
                z=np.where(df["zsp"] > 0, df["zsp"], df["zph"]),
                zType=np.where(df["zsp"].to_numpy() > 0, "spec", "phot"),
            ),
        },
        "MARD-Y3": {
            "source": "vizier",
            "code": "J/MNRAS/488/739/mard-y3",
            "columns": None,
            "postprocess": lambda df: df.assign(
                M=np.nan,
                Mtype="X",
                M_err=np.nan,
                M_Err=np.nan,
                z=df["zLFCR"],
                zType="phot",
            ),
        },
        "LP15": {
            "source": "vizier",
            "code": ["J/A+A/631/A148/tablea1", "J/A+A/631/A148/tablea2"],
            "columns": None,
            "postprocess": lambda dfs: pd.concat(dfs).assign(
                RAJ2000=lambda x: x["_RA"],
                DEJ2000=lambda x: x["_DE"],
                z=lambda x: [
                    a if not np.isnan(a) else (b if not np.isnan(b) else c)
                    for a, b, c in zip(x["zsp"], x["zsp2"], x["zph"])
                ],
                zType=lambda x: [
                    (
                        "spec"
                        if not np.isnan(a)
                        else ("spec" if not np.isnan(b) else "phot")
                    )
                    for a, b, c in zip(x["zsp"], x["zsp2"], x["zph"])
                ],
                M=np.nan,
                M_Err=np.nan,
                M_err=np.nan,
                Mtype="Opt",
            ),
        },
        "GALWEIGHT": {
            "source": "vizier",
            "code": "J/ApJS/246/2/galwcls",
            "columns": None,
            "postprocess": lambda df: df.assign(
                M=df["M500"] * 1e-14,
                Mtype="Opt",
                M_err=np.nan,
                M_Err=np.nan,
                z=df["z"],
                zType="phot",
            ),
        },
        "CODEX": {
            "source": "vizier",
            "code": "J/A+A/638/A114/catalog",
            "columns": None,
            "postprocess": lambda df: df.assign(
                M=np.nan, Mtype="X", M_Err=np.nan, M_err=np.nan, zType="phot"
            ),
        },
        "Wen&Han": {
            "source": "vizier",
            "code": "J/MNRAS/481/4158/table1",
            "columns": None,
            "postprocess": lambda df: df.assign(
                M=df["M500"], Mtype="Opt", M_err=np.nan, M_Err=np.nan, zType="spec"
            ),
        },
        "SDSSDR6": {
            "source": "vizier",
            "code": "J/ApJ/736/21/gc_cat",
            "columns": None,
            "postprocess": lambda df: df.assign(
                M=np.nan,
                Mtype="Opt",
                M_err=np.nan,
                M_Err=np.nan,
                z=df["zc"],
                zType=np.where(df["c"] == 1, "spec", "phot"),
            ),
        },
        "zCluster": {
            "source": "local_csv",
            "path": os.path.join(current_path, "ComPACT_withzCluster.csv"),
            "postprocess": lambda df: df.assign(
                RAJ2000=df["RA"],
                DEJ2000=df["DEC"],
                zType="phot",
                Mtype=np.nan,
                M_err=np.nan,
                M_Err=np.nan,
                M=np.nan,
            ),
        },
        "zaznobin": {
            "source": "local_csv",
            "path": os.path.join(current_path, "zazn.csv"),
            "postprocess": lambda df: df.assign(
                z=df["zphot_1"],
                z_signi=df["signi_1"],
                zType="phot",
                Mtype=np.nan,
                M_err=np.nan,
                M_Err=np.nan,
                M=np.nan,
            ),
        },
        "ComPACT": {
            "source": "vizier",
            "code": "J/MNRAS/531/1998/compact",
            "columns": None,
            "postprocess": lambda df: df.assign(
                Priority=pd.to_numeric(df["Priority"], errors="coerce").astype("Int64")
            ),
        },
        "ComPACT_update": {
            "source": "local_csv",
            "path": os.path.join(parent_path, "ComPACT.csv"),
            "columns": None,
            "postprocess": lambda df: df.assign(
                Priority=pd.to_numeric(df["Priority"], errors="coerce").astype("Int64")
            ),
        },
        "ComPACT_one": {
            "source": "manual",
            "postprocess": lambda: pd.DataFrame(
                {
                    "name": ["ComPACT_GG16.016+74.451"],
                    "M": [7.2],
                    "M_Err": [1.7],
                    "M_err": [1.7],
                    "Mtype": ["SZ"],
                    "z": [1.072],
                    "zType": ["spec"],
                    "RAJ2000": [208.868219],
                    "DEJ2000": [21.714945],
                }
            ),
        },
    }

    # Handle input (convert to list if single catalog requested)
    if isinstance(catalog_names, str):
        catalog_names = [catalog_names]
        return_single = True
    else:
        return_single = False

    results = {}

    for name in catalog_names:
        if name not in CATALOG_CONFIG:
            raise ValueError(
                f"Unknown catalog: {name}. Available options: {list(CATALOG_CONFIG.keys())}"
            )

        config = CATALOG_CONFIG[name]

        try:
            # Data loading
            if config["source"] == "vizier":
                vizier_client = Vizier(row_limit=-1)
                if config["columns"] is not None:  # Explicit None check
                    vizier_client.columns = config["columns"]

                if isinstance(config["code"], list):
                    # Multiple tables case (e.g., SPTECS)
                    tables = [
                        vizier_client.get_catalogs(code)[0] for code in config["code"]
                    ]
                    df = config["postprocess"]([t.to_pandas() for t in tables])
                else:
                    df = vizier_client.get_catalogs(config["code"])
                    df = df[df.keys()[0]]
                    df = df.to_pandas().reset_index(drop=True)
                    df = config["postprocess"](df)
            elif config["source"] == "local":
                if "url" in config:
                    download_file_if_missing(config["path"], config["url"])
                df = atpy.Table().read(config["path"], format="fits").to_pandas()
                df = config["postprocess"](df)
            elif config["source"] == "local_csv":
                if "url" in config:
                    download_file_if_missing(config["path"], config["url"])
                df = pd.read_csv(config["path"])
                df = config["postprocess"](df)
            elif config["source"] == "local_vot":
                if "url" in config:
                    download_and_extract_if_missing(
                        config["path"],
                        config["url"],
                        extract_filename=os.path.basename(config["path"]),
                    )
                votable = parse(config["path"])
                table = votable.get_first_table()
                data = table.array.data
                columns = table.array.dtype.names
                df = pd.DataFrame(data, columns=columns)
                df = config["postprocess"](df)
            elif config["source"] == "manual":
                df = config["postprocess"]()
            else:
                raise ValueError(f"Unknown source type for catalog {name}")

            # Standardize columns
            # required_cols = ['RAJ2000', 'DEJ2000', 'z', 'M', 'zType']
            # for col in required_cols:
            #    if col not in df.columns:
            #        raise ValueError(f"Catalog {name} is missing required column: {col}")

            results[name] = df.reset_index(drop=True)

        except Exception as e:
            raise RuntimeError(f"Error processing catalog {name}: {str(e)}")

    return results[list(results.keys())[0]] if return_single else results


def cross_match_catalogs(catalogs, priority_order, radius=5):
    # Итоговый DataFrame для сохранения результата
    result = pd.DataFrame()

    # Преобразовать радиус из минут в градусы
    radius_deg = radius / 60

    # Обработка каталогов в порядке приоритета
    for cat_name in priority_order:
        current_cat = catalogs[cat_name].copy()  # Копия, чтобы не изменять оригинал
        current_cat["Source"] = cat_name  # Источник координат
        if "zSource" not in current_cat.columns:
            current_cat["zSource"] = cat_name
        if "mSource" not in current_cat.columns:
            current_cat["mSource"] = cat_name

        # Преобразуем координаты текущего каталога в SkyCoord
        current_coords = coord.SkyCoord(
            ra=current_cat["RAJ2000"].values * u.degree,
            dec=current_cat["DEJ2000"].values * u.degree,
        )

        # Если result не пустой, находим пересечения
        if not result.empty:
            result_coords = coord.SkyCoord(
                ra=result["RAJ2000"].values * u.degree,
                dec=result["DEJ2000"].values * u.degree,
            )

            # Найти ближайшие совпадения
            idx, d2d, _ = current_coords.match_to_catalog_sky(result_coords)

            # Найти объекты, которые находятся вне радиуса (те новые, не встречаемые до этого)
            mask = d2d > radius_deg * u.deg
            new_entries = current_cat[mask]

            # Обработка совпадающих объектов
            matched = current_cat[~mask].reset_index(drop=True)

            for i, row in matched.iterrows():
                match_idx = idx[~mask][i]  # Индекс совпадающего объекта в result

                # Если в result уже есть spec, не обновляем
                if result.at[match_idx, "zType"] == "spec":
                    continue

                # Если новый объект имеет zType == 'spec', обновляем z, Mass, zType, Source и Msource
                if row["zType"] == "spec":
                    result.at[match_idx, "z"] = row["z"]
                    result.at[match_idx, "M"] = row["M"]
                    result.at[match_idx, "mSource"] = row["mSource"]
                    result.at[match_idx, "zType"] = "spec"
                    result.at[match_idx, "zSource"] = row["zSource"]
                    result.at[match_idx, "Source"] = cat_name

                # Если z или Mass отрицательные, заменяем на положительные при наличии
                if (
                    np.isnan(result.at[match_idx, "z"]) or result.at[match_idx, "z"] < 0
                ) and row["z"] > 0:
                    result.at[match_idx, "z"] = row["z"]
                    result.at[match_idx, "zSource"] = row["zSource"]

                if (
                    np.isnan(result.at[match_idx, "M"]) or result.at[match_idx, "M"] < 0
                ) and row["M"] > 0:
                    result.at[match_idx, "M"] = row["M"]
                    result.at[match_idx, "mSource"] = row["mSource"]

            # Добавляем новые объекты, не найденные в result
            result = pd.concat([result, new_entries], ignore_index=True)
        else:
            # Если результат пуст, добавляем первый каталог целиком
            result = current_cat.copy()

    # Удаляем строки, где z < 0 или Mass < 0, если они не были исправлены
    result = result[(result["z"] >= 0) & (result["M"] >= 0)].reset_index(drop=True)

    return result


def plot_contours(z, M):
    df = np.stack((z, np.log10(M * 1e14)), axis=1)

    xmin = df.min()
    xmax = df.max()
    ymin = df.min()
    ymax = df.max()

    kernel = stats.gaussian_kde(df.T)

    X, Y = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]

    positions = np.c_[X.ravel(), Y.ravel()]

    Z = np.reshape(kernel(positions.T).T, X.T.shape)
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    return X, Y, Z

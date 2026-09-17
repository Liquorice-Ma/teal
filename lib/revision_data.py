"""九条意见补实验的数据基础；与历史 STK 数据及结果隔离。"""
from dataclasses import asdict, dataclass
import csv
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
from scipy import sparse

EARTH_KM = 6378.135
SNAPSHOTS = 432
STEP_SECONDS = 300
SPLITS = {"train": [0, 240], "val": [240, 288], "test": [288, 432]}


@dataclass(frozen=True)
class Shell:
    name: str
    planes: int
    satellites_per_plane: int
    altitude_km: float
    inclination_deg: float
    mean_motion: float
    min_elevation_deg: float

    @property
    def nodes(self):
        return self.planes * self.satellites_per_plane


SHELLS = (
    Shell("HypatiaStarlink550", 72, 22, 550, 53, 15.19, 25),
    Shell("HypatiaKuiper630", 34, 34, 630, 51.9, 14.80, 30),
    Shell("HypatiaTelesat1015", 27, 13, 1015, 98.98, 13.66, 10),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cities(archive, limit=100):
    """GeoNames 第15列才是人口；同人口时按 geonameid 固定排序。"""
    with zipfile.ZipFile(archive) as zipped:
        with zipped.open("cities15000.txt") as source:
            rows = csv.reader(io.TextIOWrapper(source, encoding="utf-8"), delimiter="\t")
            cities = []
            for row in rows:
                if len(row) != 19:
                    raise ValueError("GeoNames 记录必须有19列")
                population = int(row[14])
                if population <= 0 or row[6] != "P":
                    continue
                cities.append(dict(id=int(row[0]), name=row[1], latitude=float(row[4]),
                                   longitude=float(row[5]), population=population,
                                   modified=row[18], feature_code=row[7]))
    cities.sort(key=lambda item: (-item["population"], item["id"]))
    if len(cities) < limit:
        raise ValueError("城市数据不足，禁止静默缩小城市集")
    return cities[:limit]


def city_positions(cities):
    """采用与轨道模型一致的 WGS72 半径、零海拔球面地面站。"""
    lat = np.deg2rad([c["latitude"] for c in cities])
    lon = np.deg2rad([c["longitude"] for c in cities])
    return EARTH_KM * np.column_stack((np.cos(lat) * np.cos(lon),
                                      np.cos(lat) * np.sin(lon), np.sin(lat)))


def segment_clearance(a, b):
    """线段到地心的最短距离减去地球半径，单位 km。"""
    direction = b - a
    norm2 = np.sum(direction * direction, axis=-1)
    fraction = np.clip(-np.sum(a * direction, axis=-1) / norm2, 0, 1)
    return np.linalg.norm(a + fraction[..., None] * direction, axis=-1) - EARTH_KM


def access_mapping(satellites, ground, min_elevation_deg):
    delta = satellites[None, :, :] - ground[:, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    up = ground / np.linalg.norm(ground, axis=-1, keepdims=True)
    sine = np.sum(delta * up[:, None, :], axis=-1) / distance
    visible = sine >= np.sin(np.deg2rad(min_elevation_deg))
    feasible_distance = np.where(visible, distance, np.inf)
    nearest = feasible_distance.argmin(axis=1)
    nearest[~visible.any(axis=1)] = -1
    return nearest


def propagate_tles(tle_path, snapshots=SNAPSHOTS):
    """satgenpy 生成 TLE；SGP4 传播，GMST 转换 TEME 到球面 ECEF。"""
    from sgp4.api import Satrec, SatrecArray, jday
    lines = Path(tle_path).read_text().splitlines()
    planes, per_plane = map(int, lines[0].split())
    if len(lines) != 1 + 3 * planes * per_plane:
        raise ValueError("TLE 数量不匹配")
    records = [Satrec.twoline2rv(lines[i + 1], lines[i + 2])
               for i in range(1, len(lines), 3)]
    epoch, fraction = jday(2000, 1, 1, 0, 0, 0)
    jd = np.full(snapshots, epoch)
    fractions = fraction + np.arange(snapshots) * STEP_SECONDS / 86400
    error, positions, _ = SatrecArray(records).sgp4(jd, fractions)
    if np.any(error):
        raise RuntimeError(f"SGP4 错误码: {np.unique(error).tolist()}")
    days = jd + fractions - 2451545.0
    centuries = days / 36525
    angle = np.deg2rad((280.46061837 + 360.98564736629 * days
                       + 0.000387933 * centuries**2 - centuries**3 / 38710000) % 360)
    x, y, z = positions.transpose(2, 0, 1)
    ecef = np.stack((x * np.cos(angle) + y * np.sin(angle),
                     -x * np.sin(angle) + y * np.cos(angle), z), axis=-1)
    return ecef.transpose(1, 0, 2)


def city_traffic(cities, seed=0, snapshots=SNAPSHOTS):
    """两个模型共享总业务包络；返回 city OD，尚未按训练 LP 定标。"""
    population = np.asarray([c["population"] for c in cities], dtype=float)
    count = len(cities)
    if count < 2 or np.any(population <= 0):
        raise ValueError("至少需要两个正人口城市")
    probability = population / population.sum()
    pair_probability = probability[:, None] * probability[None, :]
    np.fill_diagonal(pair_probability, 0)
    pair_probability /= pair_probability.sum()
    # 球面大圆距离，避免直接相乘原始人口导致溢出。
    xyz = city_positions(cities) / EARTH_KM
    distance = EARTH_KM * np.arccos(np.clip(xyz @ xyz.T, -1, 1))
    gravity = pair_probability / np.maximum(distance, 100)
    gravity /= gravity.sum()
    rng = np.random.default_rng(seed)
    envelope = 1 + 0.3 * np.sin(2 * np.pi * np.arange(snapshots) / 288)
    sampled = np.stack([rng.multinomial(10000, pair_probability.ravel()).reshape(count, count)
                        for _ in range(snapshots)]) / 10000
    return {"population": sampled * envelope[:, None, None],
            "gravity": gravity[None, :, :] * envelope[:, None, None]}


def satellite_traffic(city_tm, attachment, nodes):
    """保存为 time×flattened OD 的 CSR；本地卫星内流量单独返回。"""
    rows, cols, values, local = [], [], [], []
    for t, (demand, mapping) in enumerate(zip(city_tm, attachment)):
        if np.any(mapping < 0):
            raise ValueError("公共城市中仍有不可接入城市")
        source, target = np.nonzero(demand)
        s, d = mapping[source], mapping[target]
        remote = s != d
        rows.extend([t] * int(remote.sum()))
        cols.extend((s[remote] * nodes + d[remote]).tolist())
        values.extend(demand[source[remote], target[remote]].tolist())
        local.append(float(demand[source[~remote], target[~remote]].sum()))
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(city_tm), nodes * nodes))
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, np.asarray(local)


def traffic_inventory(matrix, nodes, paths=4, width=128, history=3):
    """全序列并集仅供离线资源统计，不作为在线需求身份。"""
    train_union = np.unique(matrix[:SPLITS["train"][1]].indices)
    union = np.unique(matrix.indices)
    all_pairs = nodes * (nodes - 1)
    # DOTE 全 OD 固定输入/输出，单精度参数、梯度和 Adam 两个状态的下界。
    parameters = history * all_pairs * width + width
    parameters += 3 * (width * width + width)
    parameters += width * paths * all_pairs + paths * all_pairs
    return dict(nodes=nodes, snapshots=matrix.shape[0], nonzero_per_snapshot=matrix.getnnz(axis=1).tolist(),
                training_pair_union=len(train_union), full_pair_union=len(union),
                full_union_path_nodes=len(union) * paths,
                fixed_all_pairs=all_pairs, dote_parameters=parameters,
                dote_fp32_adam_lower_bound_gib=parameters * 16 / 2**30,
                note="显存下界不含激活、临时张量及路径图；不是实测峰值。并集不用于在线输入。")

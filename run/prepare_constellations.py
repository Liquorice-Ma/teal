#!/usr/bin/env python3
"""生成独立的 Hypatia 三星座/双流量预检数据，不写入历史实验目录。

用法：python run/prepare_constellations.py --hypatia .cache/revision/hypatia
输出为未定标 CSR；manifest 未通过完整预检时，不允许进入正式训练。
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

import networkx as nx
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from revision_data import (SHELLS, SNAPSHOTS, STEP_SECONDS, SPLITS, access_mapping,
                           city_positions, city_traffic, propagate_tles, read_cities,
                           satellite_traffic, segment_clearance, sha256, traffic_inventory)

GEONAMES_URL = "https://download.geonames.org/export/dump/cities15000.zip"
HYPATIA_COMMIT = "0ac531c313eba2335f6344b46347140c3a0d4230"


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_generator(hypatia, relative_path):
    """仅导入所需的 MIT 生成模块，不加载 ns-3 或整套分析依赖。"""
    path = hypatia / "satgenpy" / "satgen" / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_cities(cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "cities15000.zip"
    if not path.exists():
        request = urllib.request.Request(GEONAMES_URL, headers={"User-Agent": "SpaTE-research-data/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read(32 * 1024 * 1024 + 1)
            if len(content) > 32 * 1024 * 1024:
                raise ValueError("城市数据超过预期大小")
            # 先验证 ZIP 和字段，失败下载不能变成可复用缓存。
            import io
            import zipfile
            with zipfile.ZipFile(io.BytesIO(content)) as zipped:
                if "cities15000.txt" not in zipped.namelist():
                    raise ValueError("下载文件不包含城市数据")
            with path.open("xb") as stream:
                stream.write(content)
            write_json(cache / "source.json", {
                "url": GEONAMES_URL, "license": "CC BY 4.0",
                "license_url": "https://www.geonames.org/export/",
                "attribution": "GeoNames", "sha256": sha256(path),
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "last_modified": response.headers.get("Last-Modified"),
                "population_note": "滚动人口代理；不是统一统计年份或 Hypatia 2025 预测值",
            })
    source = json.loads((cache / "source.json").read_text())
    if sha256(path) != source["sha256"]:
        raise ValueError("城市缓存校验值不匹配")
    return path, source


def generate(args):
    hypatia = Path(args.hypatia).resolve()
    commit = subprocess.check_output(["git", "-C", str(hypatia), "rev-parse", "HEAD"], text=True).strip()
    if commit != HYPATIA_COMMIT:
        raise ValueError(f"Hypatia 版本不匹配：{commit}，要求 {HYPATIA_COMMIT}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive, source = fetch_cities(Path(args.cache).resolve())
    cities = read_cities(archive)
    ground = city_positions(cities)
    tle_generator = load_generator(hypatia, "tles/generate_tles_from_scratch.py")
    isl_generator = load_generator(hypatia, "isls/generate_plus_grid_isls.py")
    mappings, summaries = {}, {}
    common = np.ones(len(cities), dtype=bool)
    blockers = []
    for shell in SHELLS:
        target = output / shell.name
        target.mkdir()
        tle = target / "tles.txt"
        tle_generator.generate_tles_from_scratch_manual(
            str(tle), shell.name, shell.planes, shell.satellites_per_plane,
            True, shell.inclination_deg, 0.0000001, 0.0, shell.mean_motion)
        edges = np.asarray(isl_generator.generate_plus_grid_isls(
            str(target / "isls.txt"), shell.planes, shell.satellites_per_plane, 0), dtype=int)
        graph = nx.Graph()
        graph.add_nodes_from(range(shell.nodes))
        graph.add_edges_from(edges.tolist())
        positions = propagate_tles(tle)
        clearance = segment_clearance(positions[:, edges[:, 0]], positions[:, edges[:, 1]])
        mapping = np.stack([access_mapping(p, ground, shell.min_elevation_deg) for p in positions])
        visible_all = (mapping >= 0).all(axis=0)
        common &= visible_all
        mappings[shell.name] = mapping
        summaries[shell.name] = dict(
            config=asdict(shell), nodes=shell.nodes, directed_edges=2 * len(edges),
            min_isl_clearance_km=float(clearance.min()),
            invalid_isl_time_pairs=int((clearance < 80).sum()),
            connected=nx.is_connected(graph), continuously_covered_cities=int(visible_all.sum()),
            city_uncovered_snapshots=(mapping < 0).sum(axis=0).tolist(),
            tle_sha256=sha256(tle), isl_sha256=sha256(target / "isls.txt"))
        if not nx.is_connected(graph) or np.any(clearance < 80):
            blockers.append(f"{shell.name}: 固定 +Grid 不满足连通性或80km遮挡边界")
        directed = graph.to_directed()
        for edge_id, (u, v) in enumerate(sorted(directed.edges())):
            directed[u][v].update(capacity=1000.0, edge_id=edge_id)
        # 稳定节点/边标识；不同 NetworkX 版本均能明确读取 links 字段。
        write_json(target / "topology.json", nx.node_link_data(directed, edges="links"))
        np.savez_compressed(target / "geometry.npz", positions_ecef_km=positions,
                            attachment_all_cities=mapping)
        print(json.dumps({shell.name: summaries[shell.name]}, ensure_ascii=False), flush=True)
    selected = [city for city, keep in zip(cities, common) if keep]
    population_total = sum(c["population"] for c in cities)
    manifest = dict(
        schema_version=1, stage="geometry_checked", status="blocked" if blockers else "unscaled",
        generator_sha256=sha256(Path(__file__)),
        data_module_sha256=sha256(ROOT / "lib" / "revision_data.py"),
        hypatia_commit=commit, population_source=source, shells=summaries,
        snapshots=SNAPSHOTS, interval_seconds=STEP_SECONDS, epoch_utc="2000-01-01T00:00:00Z",
        splits=SPLITS, traffic_seed=args.seed, capacity_mbps=1000,
        calibration=None, candidates=cities, retained_cities=selected,
        excluded_city_ids=[c["id"] for c, keep in zip(cities, common) if not keep],
        retained_population_fraction=sum(c["population"] for c in selected) / population_total,
        access_rule="最近可见卫星，球面 WGS72、真实几何仰角阈值25/30/10度、零海拔城市",
        geometry_note="5分钟采样时刻检查；固定逻辑ISL，不建模终端转向与GSL容量",
        identity_policy="全有序卫星对身份来自拓扑；全序列非零并集仅作离线预检",
        blockers=blockers, inventory={})
    if len(selected) < 2:
        blockers.append("全时段跨三星座公共城市少于2个；不自动放宽接入规则")
    if blockers:
        manifest["status"] = "blocked"
        write_json(output / "manifest.json", manifest)
        return 2
    traces = city_traffic(selected, args.seed)
    np.savez_compressed(output / "city_traffic.npz", **traces)
    for shell in SHELLS:
        attachment = mappings[shell.name][:, common]
        np.save(output / shell.name / "attachment.npy", attachment)
        for model, city_tm in traces.items():
            matrix, local = satellite_traffic(city_tm, attachment, shell.nodes)
            path = output / shell.name / f"{model}_unscaled.npz"
            sparse.save_npz(path, matrix)
            np.save(output / shell.name / f"{model}_local.npy", local)
            inventory = traffic_inventory(matrix, shell.nodes)
            inventory["sha256"] = sha256(path)
            inventory["local_traffic_fraction_mean"] = float(np.mean(local / city_tm.sum(axis=(1, 2))))
            manifest["inventory"][f"{shell.name}/{model}"] = inventory
    manifest["stage"] = "unscaled_full_sequence_precheck"
    manifest["pending_gates"] = ["训练段边流LP定标", "全部方法单任务峰值与耗时", "基线原文/实现对应核验"]
    # 先用解析下界拒绝必然超预算的固定全OD网络；不尝试分配以制造 OOM。
    for name, inventory in manifest["inventory"].items():
        if inventory["dote_fp32_adam_lower_bound_gib"] > args.gpu_gib * 0.75:
            blockers.append(f"{name}: DOTE 参数/梯度/Adam下界超过保留25%余量后的显存预算")
    manifest["status"] = "blocked" if blockers else "pending_calibration_and_runtime"
    write_json(output / "manifest.json", manifest)
    print(json.dumps(dict(output=str(output), status=manifest["status"],
                          retained_cities=len(selected), blockers=blockers), ensure_ascii=False), flush=True)
    return 2 if blockers else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypatia", default=str(ROOT / ".cache/revision/hypatia"))
    parser.add_argument("--cache", default=str(ROOT / ".cache/revision/geonames"))
    parser.add_argument("--output", required=True, help="必须为不存在的独立目录")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-gib", type=float, default=24.0)
    sys.exit(generate(parser.parse_args()))

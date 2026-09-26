"""Inspect packet paths and TCP limits with retained evidence and bounded probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import ipaddress
import json
import platform
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.diagnostics.evidence import capture, finite_number, write_bundle

SCHEMA = "aisp.network-diagnosis.v1"
PACKET_FIELDS = ("frame.time_epoch", "ip.src", "ipv6.src", "ip.dst", "ipv6.dst",
                 "tcp.stream", "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.reset",
                 "tcp.analysis.retransmission", "tcp.analysis.zero_window",
                 "icmp.type", "icmp.code", "icmpv6.type", "icmp.mtu", "icmpv6.mtu",
                 "arp.opcode", "arp.src.proto_ipv4", "arp.dst.proto_ipv4", "arp.src.hw_mac",
                 "eth.src", "eth.dst", "ip.ttl", "ipv6.hlim")


def bdp_analysis(bandwidth_gbps, rtt_ms, window_bytes=None):
    bandwidth = finite_number(bandwidth_gbps, "bandwidth_gbps", positive=True)
    rtt = finite_number(rtt_ms, "rtt_ms", positive=True)
    result = {"bandwidth_gbps": bandwidth, "rtt_ms": rtt,
              "bdp_bytes": bandwidth * 1e9 / 8 * rtt / 1000,
              "basis": "window/RTT is an upper bound, not measured throughput"}
    if window_bytes is not None:
        window = finite_number(window_bytes, "window_bytes", positive=True)
        result.update(window_bytes=window, window_limit_gbps=window * 8 / (rtt / 1000) / 1e9,
                      window_fraction_of_bdp=window / result["bdp_bytes"])
    return result


def parse_ss(text):
    sockets = []
    current = None
    for line in text.splitlines():
        if re.match(r"^\S+\s+\d+\s+\d+\s+\S+\s+\S+", line):
            fields = line.split()
            current = {"state": fields[0], "recv_queue_bytes": int(fields[1]),
                       "send_queue_bytes": int(fields[2]), "local": fields[3], "peer": fields[4]}
            sockets.append(current)
        if current is None:
            continue
        for key, value in re.findall(r"\b(rtt|mss|cwnd|snd_wnd|bytes_retrans|bytes_sent|retrans):([\d./]+)", line):
            if key == "rtt":
                current["rtt_ms"] = float(value.split("/")[0])
            elif key == "retrans":
                current["retransmissions_total"] = int(value.split("/")[-1])
            elif value.isdigit():
                current[key] = int(value)
        if "mss" in current and "cwnd" in current:
            current["congestion_window_bytes"] = current["mss"] * current["cwnd"]
    return sockets


def parse_packets(text):
    """Analyze observed events, without assigning a reset to an unseen endpoint."""
    if not isinstance(text, str):
        raise ValueError("Packet fields must be text")
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    required = {"frame.time_epoch", "tcp.flags.reset", "tcp.flags.syn"}
    if not required.issubset(rows[0] if rows else text.splitlines()[0].split("\t") if text else []):
        raise ValueError("Packet input must be a tshark field export with the documented header")
    resets, pmtu, neighbors = [], [], []
    counts = {"packets": len(rows), "syn": 0, "syn_ack": 0, "retransmissions": 0, "zero_windows": 0}
    counts.update(arp_requests=0, arp_replies=0, echo_requests=0, echo_replies=0,
                  neighbor_solicitations=0, neighbor_advertisements=0)
    for row in rows:
        if not row.get("frame.time_epoch"):
            raise ValueError("Each packet must have a timestamp")
        timestamp = finite_number(float(row["frame.time_epoch"]), "packet timestamp")
        syn, ack = row.get("tcp.flags.syn") == "1", row.get("tcp.flags.ack") == "1"
        counts["syn_ack" if ack else "syn"] += int(syn)
        counts["retransmissions"] += int(row.get("tcp.analysis.retransmission", "") not in ("", "0"))
        counts["zero_windows"] += int(row.get("tcp.analysis.zero_window", "") not in ("", "0"))
        counts["echo_requests"] += int(row.get("icmp.type") == "8" or row.get("icmpv6.type") == "128")
        counts["echo_replies"] += int(row.get("icmp.type") == "0" or row.get("icmpv6.type") == "129")
        counts["neighbor_solicitations"] += int(row.get("icmpv6.type") == "135")
        counts["neighbor_advertisements"] += int(row.get("icmpv6.type") == "136")
        opcode = row.get("arp.opcode")
        if opcode in {"1", "2"}:
            counts["arp_requests" if opcode == "1" else "arp_replies"] += 1
            neighbors.append({"timestamp_unix_s": timestamp, "operation": "request" if opcode == "1" else "reply",
                              "sender_ip": row.get("arp.src.proto_ipv4"), "target_ip": row.get("arp.dst.proto_ipv4"),
                              "sender_mac": row.get("arp.src.hw_mac"), "ethernet_destination": row.get("eth.dst")})
        if row.get("tcp.flags.reset") == "1":
            resets.append({"timestamp_unix_s": timestamp, "source": row.get("ip.src") or row.get("ipv6.src"),
                           "destination": row.get("ip.dst") or row.get("ipv6.dst"), "stream": row.get("tcp.stream")})
        if (row.get("icmp.type") == "3" and row.get("icmp.code") == "4") or row.get("icmpv6.type") == "2":
            pmtu.append({"timestamp_unix_s": timestamp, "advertised_mtu": row.get("icmp.mtu") or row.get("icmpv6.mtu")})
    return {**counts, "resets": resets, "packet_too_big": pmtu, "arp_events": neighbors,
            "limits": ["Capture vantage point and offloads affect observed segmentation and checksums.",
                       "A reset source address does not prove the remote application generated it.",
                       "Missing ICMP does not prove path MTU is healthy."]}


def _json_command(commands, name):
    command = commands.get(name, {})
    if command.get("status") != "ok":
        return None
    try:
        return json.loads(command["stdout"])
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"Invalid JSON in successful {name} capture") from exc


def _measurement_signal(snapshot, command, metric, value, unit):
    """Export only captures that retain their original clock and host identity."""
    identity = snapshot.get("collector", {})
    if not identity.get("host") or not identity.get("clock_domain"):
        return []
    if "start_unix_s" not in command or not command.get("duration_s"):
        return []
    start = finite_number(command["start_unix_s"], "capture start")
    duration = finite_number(command["duration_s"], "capture duration", positive=True)
    scope = {"host": identity["host"]}
    if snapshot.get("interface"):
        scope["interface"] = snapshot["interface"]
    if snapshot.get("target"):
        scope["peer"] = snapshot["target"]
    return [{"metric": metric, "role": "measurement", "value": finite_number(value, metric),
             "unit": unit, "start_unix_s": start, "end_unix_s": start + duration,
             "clock_domain": identity["clock_domain"], "scope": scope}]


def analyze(snapshot):
    if not isinstance(snapshot, dict) or snapshot.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA}")
    commands = snapshot.get("commands", {})
    if not isinstance(commands, dict) or any(not isinstance(row, dict) for row in commands.values()):
        raise ValueError("Capture commands must be an object of command results")
    if not isinstance(snapshot.get("collector", {}), dict):
        raise ValueError("Collector identity must be an object")
    findings, paths = [], []
    routes = _json_command(commands, "route")
    neighbors = _json_command(commands, "neighbors") or []
    for name, rows in (("route", routes), ("neighbors", neighbors)):
        if rows is not None and (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)):
            raise ValueError(f"{name} capture must contain a JSON list of objects")
    if routes:
        for route in routes:
            next_hop = route.get("gateway") or snapshot["target"]
            neighbor = next((n for n in neighbors if n.get("dst") == next_hop and n.get("dev") == route.get("dev")), None)
            paths.append({"destination": snapshot["target"], "source": route.get("prefsrc", route.get("src")),
                          "interface": route.get("dev"), "next_hop": next_hop,
                          "forwarding": "gateway" if route.get("gateway") else "on_link",
                          "neighbor": neighbor,
                          "explanation": "The IP destination stays the target. Link-layer delivery uses the next-hop neighbor."})
            if neighbor is None or set(neighbor.get("state", [])) & {"FAILED", "INCOMPLETE"}:
                findings.append({"summary": "Next-hop neighbor resolution is absent or incomplete in this snapshot.",
                                 "next_measurement": "Run an explicit reachability probe, then capture the neighbor table again."})
    ss_command = commands.get("sockets", {})
    sockets = parse_ss(ss_command.get("stdout", "")) if ss_command.get("status") == "ok" else []
    for name, command in commands.items():
        if name.startswith("socket_sample_") and command.get("status") == "ok":
            for socket in parse_ss(command.get("stdout", "")):
                sockets.append({**socket, "sample": name, "sample_unix_s": command["start_unix_s"]})
    ping = commands.get("ping", {})
    match = re.search(r"(?:rtt|round-trip).*?=\s*[\d.]+/([\d.]+)/", ping.get("stdout", "")) if ping.get("status") == "ok" else None
    rtt = float(match.group(1)) if match else None
    bdp = None
    if snapshot.get("bandwidth_gbps") is not None and rtt is not None and rtt > 0:
        bdp = bdp_analysis(snapshot["bandwidth_gbps"], rtt, snapshot.get("window_bytes"))
    for sock in sockets:
        if sock.get("rtt_ms", 0) > 0 and snapshot.get("bandwidth_gbps"):
            windows = [sock[k] for k in ("congestion_window_bytes", "snd_wnd") if sock.get(k, 0) > 0]
            if windows:
                sock["bdp"] = bdp_analysis(snapshot["bandwidth_gbps"], sock["rtt_ms"], min(windows))
        if sock.get("bytes_retrans", 0) or sock.get("retransmissions_total", 0):
            findings.append({"summary": f"Socket to {sock['peer']} has cumulative retransmission evidence.",
                             "next_measurement": "Sample this socket during the workload to measure the change and correlate both endpoints."})
    signals = _measurement_signal(snapshot, ping, "ping_rtt_mean", rtt, "milliseconds") if rtt is not None else []
    throughput = []
    for name, command in commands.items():
        if not name.startswith("iperf_") or command.get("status") != "ok":
            continue
        data = _json_command(commands, name)
        if not isinstance(data, dict):
            raise ValueError(f"{name} capture must contain a JSON object")
        if data.get("error"):
            findings.append({"summary": f"{name} reported {data['error']}", "next_measurement": "Check the selected server and interface."})
            continue
        received = data.get("end", {}).get("sum_received", {})
        rate = received.get("bits_per_second")
        if rate is None:
            raise ValueError(f"Missing receiver throughput in {name}")
        throughput.append({"probe": name, "receiver_gbps": finite_number(rate, name) / 1e9,
                           "retransmits": data.get("end", {}).get("sum_sent", {}).get("retransmits"),
                           "source": name, "network_plane": snapshot.get("network_plane", "unspecified")})
        signals.extend(_measurement_signal(snapshot, command, name + "_receiver_rate", rate / 1e9, "Gbps"))
    mtu_probes = []
    for name, command in commands.items():
        if name.startswith("mtu_"):
            output = command.get("stdout", "") + command.get("stderr", "")
            state = "size_passed" if command.get("status") == "ok" else "too_big" if re.search(r"too long|mtu[ =]|Frag needed|Packet too big", output, re.I) else "inconclusive"
            mtu_probes.append({"ip_packet_bytes": int(name[4:]), "status": state, "source": name})
    packets = parse_packets(snapshot["packet_fields"]) if "packet_fields" in snapshot else None
    if packets and packets["resets"]:
        findings.append({"summary": f"Observed {len(packets['resets'])} TCP resets.",
                         "next_measurement": "Correlate capture location, socket logs and the peer capture before attributing the reset."})
    if packets and packets["packet_too_big"]:
        findings.append({"summary": "The capture contains explicit path-MTU feedback.",
                         "next_measurement": "Compare the advertised MTU with the selected route and repeat a bounded payload probe."})
    statuses = {k: v.get("status", "invalid") for k, v in commands.items()}
    missing = [name for name, status in statuses.items() if status != "ok"]
    if missing:
        findings.append({"summary": f"{len(missing)} requested command captures did not complete successfully.",
                         "next_measurement": "Inspect command_status and retained stderr, resolve the missing tool or endpoint, then rerun that probe."})
    constrained = [s for s in sockets if s.get("bdp", {}).get("window_fraction_of_bdp", 1) < 1]
    if constrained:
        findings.append({"summary": "At least one sampled TCP window is below the configured bandwidth-delay product.",
                         "next_measurement": "Compare window growth, receive-window limits and retransmission deltas during steady traffic before changing buffers."})
    observed = bool(paths or sockets or throughput or (packets and packets["packets"] > 0)
                    or rtt is not None or any(p["status"] != "inconclusive" for p in mtu_probes))
    status = "ok" if observed and all(v == "ok" for v in statuses.values()) else "partial" if observed else "unavailable"
    return {"schema": SCHEMA, "status": status, "target": snapshot.get("target"),
            "packet_paths": paths, "sockets": sockets, "rtt_ms": rtt, "bdp": bdp,
            "throughput": throughput, "mtu_probes": mtu_probes, "packet_analysis": packets,
            "command_status": statuses, "findings": findings, "reason": snapshot.get("reason"), "signals": signals,
            "valid_for_performance_claim": False}


def collect(args):
    target = str(ipaddress.ip_address(args.target))
    host = hashlib.sha256(platform.node().encode()).hexdigest()[:12]
    result = {"schema": SCHEMA, "target": target, "collected_unix_s": time.time(),
              "collector": {"host": host, "clock_domain": f"collector_wall_clock:{host}"},
              "interface": args.interface,
              "bandwidth_gbps": args.bandwidth_gbps, "window_bytes": args.window_bytes,
              "network_plane": args.network_plane, "commands": {}}
    if platform.system() != "Linux":
        result["reason"] = "SKIPPED: live network collection requires Linux and iproute2"
        return result
    family = "-6" if ":" in target else "-4"
    commands = result["commands"]
    specs = {"route": ["ip", "-j", family, "route", "get", target],
             "neighbors": ["ip", "-j", family, "neigh", "show"],
             "links": ["ip", "-j", "-s", "link", "show"],
             "sockets": ["ss", "-H", "-t", "-i", "-n", "-m", "dst", target],
             "tcp_settings": ["sysctl", "net.ipv4.tcp_rmem", "net.ipv4.tcp_wmem", "net.ipv4.tcp_window_scaling", "net.ipv4.tcp_mtu_probing"],
             "ip_version": ["ip", "-Version"]}
    if args.interface:
        specs["route"] += ["oif", args.interface]
        specs["ethtool"] = ["ethtool", args.interface]
    for name, argv in specs.items():
        commands[name] = capture(argv, args.timeout)
    if args.probe:
        ping = ["ping", family, "-n", "-c", "3", "-W", "2"]
        if args.interface:
            ping += ["-I", args.interface]
        commands["ping"] = capture([*ping, target], args.timeout)
        header = 48 if family == "-6" else 28
        for size in args.mtu_bytes:
            if size <= header or size > 65535:
                raise ValueError("MTU probes must fit an IP packet and include its headers")
            commands[f"mtu_{size}"] = capture([*ping, "-M", "do", "-s", str(size - header), target], args.timeout)
    if args.iperf:
        for flows in args.parallel:
            for reverse in (False, True):
                name = f"iperf_{flows}_{'reverse' if reverse else 'forward'}"
                argv = ["iperf3", family, "-c", target, "-p", str(args.port), "-J", "-t", str(args.duration), "-P", str(flows)]
                if args.bind_address:
                    argv += ["-B", str(ipaddress.ip_address(args.bind_address))]
                if args.interface:
                    argv += ["--bind-dev", args.interface]
                if reverse:
                    argv += ["-R"]
                with ThreadPoolExecutor(max_workers=1) as pool:
                    probe = pool.submit(capture, argv, args.duration + args.timeout)
                    index = 0
                    while not probe.done():
                        commands[f"socket_sample_{name}_{index}"] = capture(specs["sockets"], min(args.timeout, 2))
                        index += 1
                        time.sleep(0.5)
                    commands[name] = probe.result()
    if args.pcap:
        argv = ["tshark", "-n", "-r", str(args.pcap), "-T", "fields", "-E", "header=y", "-E", "separator=/t", "-E", "occurrence=f"]
        for field in PACKET_FIELDS:
            argv += ["-e", field]
        commands["pcap"] = capture(argv, args.timeout)
        if commands["pcap"]["status"] == "ok":
            result["packet_fields"] = commands["pcap"]["stdout"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    live = sub.add_parser("collect", help="Capture Linux state and explicitly selected traffic probes")
    live.add_argument("--target", required=True, help="Destination IP address")
    live.add_argument("--interface")
    live.add_argument("--bind-address")
    live.add_argument("--probe", action="store_true", help="Run ping and DF payload probes")
    live.add_argument("--iperf", action="store_true", help="Run bounded tests against an existing iperf3 server")
    live.add_argument("--port", type=int, default=5201)
    live.add_argument("--parallel", type=int, nargs="+", default=[1, 4])
    live.add_argument("--duration", type=int, default=5)
    live.add_argument("--timeout", type=float, default=15)
    live.add_argument("--mtu-bytes", type=int, nargs="+", default=[1500, 9000])
    live.add_argument("--bandwidth-gbps", type=float)
    live.add_argument("--window-bytes", type=int)
    live.add_argument("--network-plane", choices=["data", "management", "unspecified"], default="unspecified")
    live.add_argument("--pcap", type=Path)
    offline = sub.add_parser("analyze", help="Analyze retained raw capture JSON")
    offline.add_argument("--input", type=Path, required=True)
    packet = sub.add_parser("packets", help="Analyze a retained tshark field export")
    packet.add_argument("--input", type=Path, required=True)
    calc = sub.add_parser("bdp", help="Calculate bandwidth-delay product and window limit")
    calc.add_argument("--bandwidth-gbps", type=float, required=True)
    calc.add_argument("--rtt-ms", type=float, required=True)
    calc.add_argument("--window-bytes", type=int)
    for command in (live, offline, packet, calc):
        command.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "collect":
            if not 1 <= args.duration <= 300 or not 1 <= args.port <= 65535 or any(p < 1 or p > 128 for p in args.parallel):
                raise ValueError("Invalid duration, port or parallel flow count")
            finite_number(args.timeout, "timeout", positive=True)
            if args.bandwidth_gbps is not None:
                finite_number(args.bandwidth_gbps, "bandwidth_gbps", positive=True)
            if args.window_bytes is not None:
                finite_number(args.window_bytes, "window_bytes", positive=True)
            raw = collect(args)
            report = analyze(raw)
        elif args.operation == "analyze":
            raw = json.loads(args.input.read_text())
            report = analyze(raw)
        elif args.operation == "packets":
            raw = {"schema": SCHEMA, "packet_fields": args.input.read_text()}
            report = analyze(raw)
        else:
            raw = {"bandwidth_gbps": args.bandwidth_gbps, "rtt_ms": args.rtt_ms, "window_bytes": args.window_bytes}
            report = {"schema": SCHEMA, "status": "calculated", "bdp": bdp_analysis(**raw), "findings": []}
        output = write_bundle(args.run_dir, "network-diagnose", raw, report)
        print(json.dumps({"status": report["status"], "report": str(output)}))
        return 2 if report["status"] == "unavailable" else 0
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

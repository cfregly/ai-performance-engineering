# Network diagnosis

Use `aisp tools network-diagnose` to explain the route a connection uses, measure
TCP limits, and analyze retained captures. This tool does not change interfaces,
routes, TCP settings or switch configuration. It writes raw evidence, JSON results
and a Markdown report under the selected run directory.

Run from `code/`. Start with state collection on Linux:

```bash
python -m cli.aisp tools network-diagnose -- collect \
  --target 192.0.2.2 --interface eth0 --run-dir /tmp/network-state
```

The report distinguishes direct neighbor delivery from delivery through a gateway.
For IPv4, ARP resolves that next hop to a link-layer address. For IPv6, Neighbor
Discovery serves that purpose. The destination IP remains the target across a
routed hop while link-layer addresses change. A host snapshot cannot establish a
switch's forwarding-table contents or prove the entire path is healthy.

Add explicit reachability and DF payload probes:

```bash
python -m cli.aisp tools network-diagnose -- collect \
  --target 192.0.2.2 --probe --mtu-bytes 1500 9000 \
  --bandwidth-gbps 10 --run-dir /tmp/network-mtu
```

Sizes include IP and ICMP headers. A successful small ping does not establish that
a large packet can pass. Explicit packet-too-big feedback supports an MTU diagnosis.
A timeout is inconclusive because filtering and loss can produce it too.

To measure throughput, first run `iperf3 -s` on the intended peer. Then select the
interface and network plane explicitly:

```bash
python -m cli.aisp tools network-diagnose -- collect \
  --target 192.0.2.2 --interface eth0 --bind-address 192.0.2.1 \
  --iperf --duration 5 --parallel 1 4 --network-plane data \
  --probe --bandwidth-gbps 10 --run-dir /tmp/network-throughput
```

The tool runs forward and reverse tests and samples `ss` during each test. Some
hosts require privileges for interface binding. Failed commands remain in the
report. Management-network throughput must not be presented as RDMA-fabric
throughput. One flow improving with parallel flows suggests an avenue to test,
but does not identify whether the constraint is a window, a path or an endpoint.

The bandwidth-delay calculator also works without Linux:

```bash
python -m cli.aisp tools network-diagnose -- bdp \
  --bandwidth-gbps 10 --rtt-ms 200 --window-bytes 16777216 \
  --run-dir /tmp/network-bdp
```

This calculates a 250 MB bandwidth-delay product and a roughly 0.671 Gbit/s window
ceiling. These are calculations, not measured network performance. Socket
congestion windows use `cwnd * mss`. When available, the peer's advertised receive
window also constrains the sender. Socket buffer settings alone do not establish
the effective window.

For packet analysis, `collect --pcap capture.pcapng` invokes `tshark` and retains
its field export. `packets --input packets.tsv --run-dir /tmp/packet-analysis`
analyzes an existing export. The exact field list is `PACKET_FIELDS` in
[network_diagnosis_tool.py](network_diagnosis_tool.py). Keep the tab-separated
header and export the first occurrence of each field.

The analyzer reports ARP requests and replies, ICMP echo, IPv6 neighbor discovery,
SYN, SYN/ACK, retransmissions, zero windows, reset addresses and path-MTU feedback.
ARP events preserve sender addresses and the observed Ethernet destination, so
readers can inspect neighbor resolution before the first ping. A reset shows
where that packet appears to originate at the capture point. It does not establish
which process or middlebox generated it.
Offloads can change apparent packet sizes and checksums. Correlate both endpoint
captures and socket/application logs before assigning a cause.

Replay an existing raw result without generating traffic:

```bash
python -m cli.aisp tools network-diagnose -- analyze \
  --input /tmp/network-throughput/raw/network-diagnose.json \
  --run-dir /tmp/network-reanalysis
```

Use a new output directory for each analysis. Exit code 2 means no usable
measurement was available, including unsupported live collection on non-Linux
hosts. A partial report identifies failed or unavailable commands explicitly.

Live ping and iperf measurements also export timed signals for
[cross-layer analysis](../core/analysis/cross_layer_diagnosis.md). Each signal
retains its collector clock and hashed host identity. A retained capture without
that identity is still analyzable, but does not gain an invented clock identity.

Command references: [ss](https://man7.org/linux/man-pages/man8/ss.8.html),
[iperf3](https://software.es.net/iperf/invoking.html), and
[tshark](https://www.wireshark.org/docs/man-pages/tshark.html).

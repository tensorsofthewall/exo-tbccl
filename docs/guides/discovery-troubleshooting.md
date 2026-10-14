# Discovery troubleshooting

Use this when the two exo nodes do not form one cluster, or form one and then lose a peer. Work through the questions in order; each one has a command. You do not need to know exo's internals.

exo discovers nodes with IPv6 multicast on UDP port 52413 (group `ff12::e0a1:de89`), then connects them with TCP on port 52414. A node that fails to announce, or a peer that is announced but cannot be reached, shows up as "never discovered" or "discovered, then expired about ten seconds later".

## 1. Are both nodes visible?

```sh
curl -s localhost:52415/state | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['topology']['nodes']), {k[:6]: v['friendlyName'] for k, v in d['nodeIdentities'].items()})"
```

If you see `1` on both hosts, the nodes have not discovered each other. If you see `2`, discovery works; continue with question 3.

## 2. Same namespace? Same version?

Both nodes must run with the same `EXO_ZENOH_NAMESPACE` (the exo log prints `EXO_ZENOH_NAMESPACE: <name>` at start) and the same exo version. Two nodes in different namespaces never see each other, and that is by design.

## 3. Does one node become master, and does the peer expire?

The logs should show each node `discovered` the other's id, and exactly one master after a short election (the other node logs `elected master - demoting self` once). If a peer is `discovered` and then `expired` about ten seconds later, the nodes see each other's announcements but cannot keep a connection: check question 6 (TCP reachability).

## 4. Is UDP port 52413 already in use on the host?

```sh
ss -lun | grep 52413            # Linux
netstat -an | grep 52413        # macOS
lsof -nP -iUDP:52413            # who listens
```

Only exo should listen there. On macOS another listener on that port (a packet-capture or debugging script, a second exo) can take the unicast discovery replies and stop the nodes from discovering each other. Stop the other program.

## 5. Was the Mac node started from a live session?

On macOS a node started from a **detached tmux session or a nohup job** was observed to fail every discovery send with `No route to host`, after which exo stops announcing on every interface for the rest of the process's life: the Mac is then silent on the network and the Linux node never sees it. Restart the Mac node from an interactive Terminal or a live `ssh -tt` session ([running across two hosts](two-host.md)). Also make sure the macOS firewall / Local Network prompt for exo's Python was answered once, from a local session.

## 6. Does scoped IPv6 connectivity work?

Discovery uses IPv6 link-local addresses, which need the interface name as a scope:

```sh
ping -6 -c3 'fe80::PEER-ADDRESS%thunderbolt0'      # Linux; PEER-ADDRESS = the Mac's bridge0 link-local address
ping6 -c3 'fe80::PEER-ADDRESS%bridge0'              # macOS; PEER-ADDRESS = the Linux thunderbolt0 link-local address
```

Both directions must work. Find the addresses with `ip -6 addr show thunderbolt0` (Linux) and `ifconfig bridge0` (macOS). Also check IPv4 (`ping 192.168.3.x`) and that nothing blocks TCP port 52414 between the nodes. The validated link had IPv6 link-local addresses on both ends; whether other macOS Thunderbolt Bridge IPv6 settings matter was not isolated, so do not assume a particular setting is required.

## 7. Is `MlxTbccl` actually selected?

If the cluster forms but placement fails or the instance is not an `MlxTbcclInstance`:

```sh
.venv/bin/python -c "import exo_tbccl; print(exo_tbccl.is_available())"       # on every node
curl -s localhost:52415/state | python3 -c "import json,sys; d=json.load(sys.stdin); print([next(iter(i)) for i in d['instances'].values()])"
```

`(False, reason)` names what is missing. See [verify](verify-active.md).

## Still stuck?

Collect: the exo log of each node from start-up (with `-vv` for verbose logging), the output of questions 1 to 6 on both hosts, and `ping`/`ping6` results. Do not paste credentials or private network details.

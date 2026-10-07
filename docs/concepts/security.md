# Security model

exo-tbccl is an adapter over TBCCL and has the same trust model: **trusted peers on a trusted network**. There is no peer authentication, no transport encryption, no message authentication and no authorization. Read the TBCCL security model first; this page lists what exo-tbccl adds.

exo-tbccl moves activations between pipeline stages in clear text over TBCCL. The endpoint blobs are exchanged through exo's own control plane (a runner byte exchange between runners and workers), so who may join an exo cluster and read those blobs is decided by exo, not by this package. Anyone who can reach a rank's listener during bootstrap can attempt a connection; TBCCL's handshake rejects everything that does not match the communicator id, the rank and the world size, but those values are identifiers, not credentials.

The package does not unpickle or execute anything received from a peer; it passes payload bytes to MLX arrays of a size the receiver fixed beforehand.

## Deployment assumptions

- All ranks are run by the same trusted party, on hosts and links that no untrusted party can reach: loopback, a private network, or a direct link such as Thunderbolt.
- If traffic must cross a network that is not trusted, put it inside a tunnel that provides authentication and encryption (a VPN or an encrypted overlay). exo-tbccl does not provide one.
- Do not expose rank listeners to the public internet. Bind to the specific address of the trusted link rather than a wildcard where the interface allows it.

## Reporting a vulnerability

See the security policy in the repository (`SECURITY.md`). A clean error on malformed input is a robustness property and does not mean the component is safe against a hostile peer.

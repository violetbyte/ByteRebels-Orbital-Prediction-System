#!/usr/bin/env python3
"""
privacy_agg_final.py

Team-4: Privacy + Secure Aggregation (final, hardened)

What this file provides:
- Local differential privacy (Laplace mechanism) per node (configurable epsilon & clip)
- Fixed-point integer encoding + additive secret-sharing (2-server model)
- Per-server packets (each server receives only its share)
- HMAC integrity on per-server packets
- Server-side optional HMAC verification
- Robust error handling, input validation, and clear exceptions/logging
- Demo simulation that keeps cryptographic secrets separate from RNG used for Laplace sampling

Important notes:
- This is an MVP/prototype. For production: use a tested secure-aggregation/MPC library,
  enforce TLS, use hardware-backed key storage, and use vetted DP libraries for noise generation.
- The demo seeds the demo RNG for reproducibility only; secrets use `secrets` (crypto-grade).
"""

from __future__ import annotations

import random
import math
import json
import hmac
import hashlib
import secrets
import logging
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("privacy_agg")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SCALE = 10**6  # fixed-point scale factor (1 -> 1_000_000)
MODULUS = 2**61 - 1  # modulus for share arithmetic (very large)
DEFAULT_EPSILON = 0.5  # default privacy budget (tunable)
DEFAULT_CLIP_MAX = 0.2  # default clip bound for node values
# RNG used for Laplace sampling (not crypto-grade). Seeded only in demo for reproducibility.
RNG = random.Random()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class PrivacyAggError(Exception):
    pass


class PacketVerificationError(PrivacyAggError):
    pass


class InvalidPacketError(PrivacyAggError):
    pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _validate_probability(v: float) -> None:
    if not isinstance(v, (float, int)):
        raise ValueError("probability must be numeric")
    if math.isnan(v) or math.isinf(v):
        raise ValueError("probability must be finite")
    # values can be outside [0,1] before clipping; caller should clip


def laplace_sample(scale: float) -> float:
    """Laplace(0, scale) sample using RNG (not crypto-grade).
    For production use a vetted cryptographic mechanism or library.
    """
    if scale <= 0:
        return 0.0
    u = RNG.random() - 0.5  # in (-0.5, 0.5)
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


def add_laplace_noise(value: float, epsilon: float, sensitivity: float) -> float:
    """Add Laplace noise with given epsilon and sensitivity.
    Caller must ensure `value` was clipped to match `sensitivity`.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be > 0")
    scale = sensitivity / epsilon
    noisy = value + laplace_sample(scale)
    # enforce numerical bounds
    return max(0.0, min(1.0, noisy))


def float_to_int(v: float) -> int:
    """Convert float in [0,1] to fixed-point int modulo MODULUS."""
    if not isinstance(v, (float, int)):
        raise ValueError("float_to_int expects numeric input")
    # clip for safety
    v_clipped = max(0.0, min(1.0, float(v)))
    return int(round(v_clipped * SCALE)) % MODULUS


def int_to_float(i: int) -> float:
    """Convert fixed-point int (mod MODULUS) back to float."""
    if not isinstance(i, int):
        raise ValueError("int_to_float expects int input")
    if i > MODULUS // 2:
        i = i - MODULUS
    return float(i) / SCALE


def make_shares(value_int: int, n_shares: int = 2) -> List[int]:
    """Create n_shares integers whose sum modulo MODULUS equals value_int."""
    if not isinstance(value_int, int):
        raise ValueError("make_shares expects integer value_int")
    if n_shares < 2:
        raise ValueError("n_shares must be >= 2")
    shares = [secrets.randbelow(MODULUS) for _ in range(n_shares - 1)]
    ssum = sum(shares) % MODULUS
    last = (value_int - ssum) % MODULUS
    shares.append(last)
    # shuffle securely
    for idx in range(len(shares) - 1, 0, -1):
        j = secrets.randbelow(idx + 1)
        shares[idx], shares[j] = shares[j], shares[idx]
    return shares


def aggregate_shares(shares: List[int]) -> int:
    """Aggregate a list of shares (mod MODULUS)."""
    if not all(isinstance(s, int) for s in shares):
        raise ValueError("aggregate_shares expects list of ints")
    return sum(shares) % MODULUS


# ---------------------------------------------------------------------------
# Node: creates per-server packets containing only that server's share
# ---------------------------------------------------------------------------
class Node:
    def __init__(
        self,
        node_id: str,
        secret_key: Optional[bytes] = None,
        clip_max: float = DEFAULT_CLIP_MAX,
    ):
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be non-empty string")
        if not (0.0 < clip_max <= 1.0):
            raise ValueError("clip_max must be in (0, 1]")
        self.node_id = node_id
        # crypto-grade key for HMAC; secrets.token_bytes(32) recommended for real deployments
        self.secret_key = secret_key or secrets.token_bytes(32)
        self.clip_max = float(clip_max)

    def _clip(self, v: float) -> float:
        return max(0.0, min(self.clip_max, v))

    def prepare_share_packets(
        self,
        cell_id: str,
        true_prob: float,
        epsilon: float = DEFAULT_EPSILON,
        mode: str = "local_dp",
        n_servers: int = 2,
    ) -> Dict[str, Any]:
        """
        Prepare per-server packets. Returns dict:
          {
            "packets": [ {server_index, node_id, cell_id, share, share_index, hmac, mode}, ... ],
            "audit": { "noisy_value": float, "v_int": int }  # only for demo/audit; do not transmit publicly
          }

        mode: 'local_dp' | 'central_dp' | 'no_dp'
        """
        if n_servers < 2:
            raise ValueError("n_servers must be >= 2")
        _validate_probability(true_prob)
        if mode == "local_dp" and epsilon <= 0:
            raise ValueError("epsilon must be > 0 for local_dp mode")

        # 1) clip
        clipped = self._clip(float(true_prob))

        # 2) add local DP noise if requested
        if mode == "local_dp":
            noisy = add_laplace_noise(clipped, epsilon, sensitivity=self.clip_max)
        elif mode in ("central_dp", "no_dp"):
            noisy = clipped
        else:
            raise ValueError("Unknown mode")

        # 3) fixed-point
        v_int = float_to_int(noisy)

        # 4) make shares
        shares = make_shares(v_int, n_shares=n_servers)

        # 5) create per-server packets with HMAC over node_id:cell_id:share_index:share
        packets = []
        for idx, share in enumerate(shares):
            payload = f"{self.node_id}:{cell_id}:{idx}:{share}".encode()
            hm = hmac.new(self.secret_key, payload, hashlib.sha256).hexdigest()
            pkt = {
                "node_id": self.node_id,
                "cell_id": cell_id,
                "share_index": idx,
                "share": share,
                "hmac": hm,
                "mode": mode,
            }
            packets.append(pkt)

        # audit info useful for demo/testing only (do not expose in real deployment)
        audit = {"noisy_value": noisy, "v_int": v_int}
        return {"packets": packets, "audit": audit}


# ---------------------------------------------------------------------------
# Server: receives exactly one share per node per cell
# ---------------------------------------------------------------------------
class Server:
    def __init__(self, name: str, known_node_keys: Optional[Dict[str, bytes]] = None):
        if not isinstance(name, str) or not name:
            raise ValueError("Server name must be non-empty string")
        self.name = name
        # known_node_keys maps node_id -> secret_key for HMAC verification
        self.known_node_keys = known_node_keys or {}
        self.cell_shares: Dict[str, List[int]] = defaultdict(list)
        self.hmacs: Dict[str, List[str]] = defaultdict(list)

    def _validate_packet_shape(self, packet: Dict[str, Any]) -> None:
        # minimal validation of required fields
        if not isinstance(packet, dict):
            raise InvalidPacketError("packet must be a dict")
        required = {"node_id", "cell_id", "share_index", "share", "hmac", "mode"}
        if not required.issubset(set(packet.keys())):
            raise InvalidPacketError(
                f"packet missing required fields: {required - set(packet.keys())}"
            )
        if not isinstance(packet["share_index"], int) or packet["share_index"] < 0:
            raise InvalidPacketError("share_index must be non-negative int")
        if not isinstance(packet["share"], int):
            raise InvalidPacketError("share must be int")

    def verify_hmac(self, packet: Dict[str, Any]) -> bool:
        """Verify HMAC for a per-server packet using known node key.
        Returns True if valid, False otherwise.
        """
        node_id = packet.get("node_id")
        if node_id not in self.known_node_keys:
            logger.debug(
                "Server %s: unknown node_id %s for HMAC verification",
                self.name,
                node_id,
            )
            return False
        key = self.known_node_keys[node_id]
        payload = f"{node_id}:{packet['cell_id']}:{packet['share_index']}:{packet['share']}".encode()
        expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, packet.get("hmac", ""))

    def receive_packet(self, packet: Dict[str, Any], verify: bool = True) -> None:
        """Store the single share included in packet. If verify=True, attempt HMAC verification and raise on failure."""
        try:
            self._validate_packet_shape(packet)
        except InvalidPacketError as e:
            logger.error("Server %s received invalid packet: %s", self.name, e)
            raise

        if verify:
            if not self.verify_hmac(packet):
                logger.error(
                    "Server %s: HMAC verification failed for node %s cell %s",
                    self.name,
                    packet.get("node_id"),
                    packet.get("cell_id"),
                )
                raise PacketVerificationError(
                    f"HMAC verification failed at server {self.name} for node {packet.get('node_id')}"
                )
        # store share
        cid = packet["cell_id"]
        share = packet["share"]
        self.cell_shares[cid].append(int(share))
        self.hmacs[cid].append(packet["hmac"])
        logger.debug(
            "Server %s stored share for cell %s (current count %d)",
            self.name,
            cid,
            len(self.cell_shares[cid]),
        )

    def aggregate(self) -> Dict[str, int]:
        """Return per-cell aggregated int sum (mod MODULUS) of the shares this server holds."""
        out: Dict[str, int] = {}
        for cid, shares in self.cell_shares.items():
            out[cid] = aggregate_shares(shares)
        return out


# ---------------------------------------------------------------------------
# Aggregator: combines multiple servers (assumes non-colluding) to reconstruct totals
# ---------------------------------------------------------------------------
class Aggregator:
    def __init__(self, servers: List[Server]):
        if not servers or len(servers) < 2:
            raise ValueError("Aggregator requires at least 2 servers")
        self.servers = servers

    def reconstruct_totals(self) -> Dict[str, float]:
        """Combine per-cell aggregates from all servers and return totals as floats."""
        aggregated_per_server = [s.aggregate() for s in self.servers]
        all_cell_ids = set()
        for d in aggregated_per_server:
            all_cell_ids.update(d.keys())
        totals: Dict[str, float] = {}
        for cid in all_cell_ids:
            total_int = 0
            for d in aggregated_per_server:
                total_int = (total_int + d.get(cid, 0)) % MODULUS
            totals[cid] = int_to_float(total_int)
        return totals

    def fused_average(
        self,
        totals: Dict[str, float],
        counts: Dict[str, int],
        central_dp: bool = False,
        epsilon: float = DEFAULT_EPSILON,
        clip_max: float = DEFAULT_CLIP_MAX,
    ) -> Dict[str, float]:
        """
        Compute per-cell average = totals / counts.
        If central_dp is True, add Laplace noise with sensitivity = clip_max / n and epsilon.
        Returns fused map with values in [0,1].
        """
        if central_dp and (epsilon <= 0 or clip_max <= 0):
            raise ValueError("central_dp requires epsilon>0 and clip_max>0")

        fused: Dict[str, float] = {}
        for cid, total in totals.items():
            n = counts.get(cid, 0)
            if n <= 0:
                logger.warning("No contributions for cell %s; skipping", cid)
                continue
            avg = total / float(n)
            if central_dp:
                sensitivity = clip_max / float(n)
                # scale = sensitivity / epsilon
                scale = sensitivity / epsilon
                avg = avg + laplace_sample(scale)
            fused[cid] = max(0.0, min(1.0, avg))
        return fused


# ---------------------------------------------------------------------------
# Optional: robust statistics if nodes consent to share per-node noisy values (debug/trusted)
# ---------------------------------------------------------------------------
def robust_stats_from_noisy_per_node(
    per_node_values: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """
    Given per-node noisy floats for each cell, compute median and trimmed mean.
    Only for debugging/trusted mode where per-node noisy values are available.
    """
    out: Dict[str, Dict[str, float]] = {}
    for cid, vals in per_node_values.items():
        if not vals:
            continue
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        median = (
            vals_sorted[n // 2]
            if n % 2 == 1
            else 0.5 * (vals_sorted[n // 2 - 1] + vals_sorted[n // 2])
        )
        trim = max(1, n // 10)
        trimmed = vals_sorted[trim : n - trim] if n > 2 * trim else vals_sorted
        trimmed_mean = sum(trimmed) / len(trimmed)
        out[cid] = {"median": median, "trimmed_mean": trimmed_mean}
    return out


# ---------------------------------------------------------------------------
# Demo simulation (self-contained). Safe to run locally for testing.
# ---------------------------------------------------------------------------
def demo_simulation(repro_seed: Optional[int] = 42) -> None:
    """
    Demonstrates end-to-end flow:
    - Create nodes with per-node secrets
    - Nodes prepare per-server packets (local DP)
    - Servers receive only their share packet and verify HMAC
    - Aggregator reconstructs totals and computes fused averages
    - Debug optionally reconstructs per-node noisy values (only for demo)
    """
    # Setup demo RNG for Laplace reproducibility only. Do NOT seed secrets.
    if repro_seed is not None:
        RNG.seed(repro_seed)

    # Create node keys (crypto-grade)
    node_keys: Dict[str, bytes] = {
        f"node{i}": secrets.token_bytes(32) for i in range(5)
    }
    nodes = [Node(f"node{i}", secret_key=node_keys[f"node{i}"]) for i in range(5)]

    # Simulated ground-truth probabilities per node per cell (node3 includes an outlier)
    true_values = {
        "A-12": [0.02, 0.03, 0.025, 0.5, 0.03],
        "B-05": [0.15, 0.12, 0.14, 0.13, 0.11],
    }

    epsilon = 0.5
    mode = "local_dp"
    n_servers = 2

    # Servers know node keys to verify HMACs
    serverA = Server("serverA", known_node_keys=node_keys)
    serverB = Server("serverB", known_node_keys=node_keys)
    servers = [serverA, serverB]

    counts: Dict[str, int] = defaultdict(int)
    # For demo only: we collect per-node noisy values (reconstructed from node a priori)
    debug_per_node: Dict[str, List[float]] = defaultdict(list)

    # Simulate nodes preparing packets and sending them to each server (each server only receives its own packet)
    for node_idx, node in enumerate(nodes):
        for cid, arr in true_values.items():
            true_p = arr[node_idx]
            result = node.prepare_share_packets(
                cell_id=cid,
                true_prob=true_p,
                epsilon=epsilon,
                mode=mode,
                n_servers=n_servers,
            )
            packets: List[Dict[str, Any]] = result["packets"]
            audit = result["audit"]  # contains noisy_value and v_int for demo

            # send exactly one packet to each server (server i gets packets[i])
            for i, server in enumerate(servers):
                pkt = packets[i]
                # servers verify HMAC and store share
                try:
                    server.receive_packet(pkt, verify=True)
                except PacketVerificationError as e:
                    logger.error("Packet verification failed during demo: %s", e)
                    raise

            counts[cid] += 1
            # debug reconstructed noisy value (from node audit) - not available to servers in real flow
            debug_per_node[cid].append(audit["noisy_value"])

    # Aggregation
    aggregator = Aggregator(servers=servers)
    totals = aggregator.reconstruct_totals()
    fused = aggregator.fused_average(
        totals, counts, central_dp=False, epsilon=epsilon, clip_max=DEFAULT_CLIP_MAX
    )
    robust = robust_stats_from_noisy_per_node(debug_per_node)

    # Print results
    print("\n--- RESULTS (secure aggregation, local DP) ---")
    print("Per-cell totals (sum of noisy fixed-point values):")
    print(json.dumps(totals, indent=2))
    print("\nFused average (after secure aggregation):")
    print(json.dumps(fused, indent=2))
    print("\n--- DEBUG (per-node noisy values, demo only) ---")
    print(json.dumps(debug_per_node, indent=2))
    print("\nRobust stats (median, trimmed_mean) from per-node noisy values:")
    print(json.dumps(robust, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        demo_simulation()
    except Exception as exc:
        logger.exception("Demo failed: %s", exc)
        raise

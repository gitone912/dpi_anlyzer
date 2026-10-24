#!/usr/bin/env python3
"""Small dependency-free PCAP analyzer with hostname application prediction."""
from __future__ import annotations

import argparse
import ipaddress
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


# Curated labeled hostname examples are the training set for the built-in model.
# Users can add examples here; labels are application names.
TRAINING_DOMAINS = {
    "Google": ["google.com", "googlevideo.com", "gstatic.com", "googleapis.com"],
    "YouTube": ["youtube.com", "googlevideo.com", "ytimg.com", "youtube-nocookie.com"],
    "Facebook": ["facebook.com", "fbcdn.net", "messenger.com"],
    "Instagram": ["instagram.com", "cdninstagram.com"],
    "WhatsApp": ["whatsapp.com", "whatsapp.net"],
    "Twitter/X": ["twitter.com", "x.com", "twimg.com"],
    "Netflix": ["netflix.com", "nflxvideo.net", "nflximg.net"],
    "Amazon": ["amazon.com", "amazonaws.com", "primevideo.com"],
    "Microsoft": ["microsoft.com", "live.com", "office.com", "windows.com", "msn.com"],
    "Apple": ["apple.com", "icloud.com", "mzstatic.com"],
    "Telegram": ["telegram.org", "t.me", "telegra.ph"],
    "TikTok": ["tiktok.com", "tiktokcdn.com", "byteoversea.com"],
    "Spotify": ["spotify.com", "scdn.co"],
    "Zoom": ["zoom.us", "zoom.com"],
    "Discord": ["discord.com", "discord.gg", "discordapp.net"],
    "GitHub": ["github.com", "githubusercontent.com", "github.io"],
    "Cloudflare": ["cloudflare.com", "cloudflare-dns.com"],
}


def features(host: str) -> set[str]:
    host = host.lower().strip(".")
    tokens = set(re.findall(r"[a-z0-9]+", host))
    padded = f"^{host}$"
    tokens.update(padded[i:i + 3] for i in range(max(0, len(padded) - 2)))
    return tokens


class HostnameModel:
    """Multinomial Naive Bayes trained on labeled hostname examples."""
    def __init__(self, training_domains=None):
        training_domains = training_domains or TRAINING_DOMAINS
        self.class_docs = Counter()
        self.feature_counts = defaultdict(Counter)
        self.vocabulary = set()
        for label, domains in training_domains.items():
            for domain in domains:
                fs = features(domain.strip())
                self.class_docs[label] += 1
                self.feature_counts[label].update(fs)
                self.vocabulary.update(fs)

    def predict(self, host: str) -> tuple[str | None, float]:
        fs = features(host)
        if not host or not fs:
            return None, 0.0
        scores = {}
        total_docs = sum(self.class_docs.values())
        vocab_size = max(1, len(self.vocabulary | fs))
        for label, docs in self.class_docs.items():
            total = sum(self.feature_counts[label].values())
            score = math.log(docs / total_docs)
            for feat in fs:
                score += math.log((self.feature_counts[label][feat] + 1) / (total + vocab_size))
            scores[label] = score
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Normalize scores to provide a rough model confidence estimate.
        peak = top[0][1]
        exps = [math.exp(max(-700, value - peak)) for _, value in top]
        confidence = exps[0] / sum(exps)
        # Unknown hosts should remain unknown instead of forcing a poor guess.
        if confidence < 0.70:
            return None, confidence
        return top[0][0], confidence


@dataclass
class Packet:
    timestamp: tuple[int, int]
    data: bytes
    header: bytes


class Pcap:
    MAGIC = {
        b"\xd4\xc3\xb2\xa1": ("<", 1_000_000),
        b"\xa1\xb2\xc3\xd4": (">", 1_000_000),
        b"\x4d\x3c\xb2\xa1": ("<", 1_000),
        b"\xa1\xb2\x3c\x4d": (">", 1_000),
    }

    @classmethod
    def read(cls, path: Path):
        with path.open("rb") as f:
            gh = f.read(24)
            if len(gh) != 24 or gh[:4] not in cls.MAGIC:
                raise ValueError("Unsupported or invalid PCAP file (classic PCAP required).")
            endian, resolution = cls.MAGIC[gh[:4]]
            _, _, _, _, _, _, linktype = struct.unpack(endian + "IHHIIII", gh)
            packets = []
            while True:
                h = f.read(16)
                if not h:
                    break
                if len(h) != 16:
                    raise ValueError("Truncated PCAP packet header.")
                sec, frac, included, _ = struct.unpack(endian + "IIII", h)
                data = f.read(included)
                if len(data) != included:
                    raise ValueError("Truncated PCAP packet data.")
                packets.append(Packet((sec, frac * (1_000_000 // resolution)), data, h))
            return gh, endian, resolution, linktype, packets

    @staticmethod
    def write(path: Path, gh: bytes, endian: str, resolution: int, packets: list[Packet]):
        with path.open("wb") as f:
            f.write(gh)
            for p in packets:
                sec, usec = p.timestamp
                frac = usec if resolution == 1_000_000 else usec // 1000
                f.write(struct.pack(endian + "IIII", sec, frac, len(p.data), len(p.data)))
                f.write(p.data)


def inspect_packet(data: bytes, linktype: int):
    """Return (src IP, dst IP, protocol, src port, dst port, host, app hint)."""
    if linktype != 1 or len(data) < 14:  # Ethernet only
        return None
    etype = struct.unpack_from("!H", data, 12)[0]
    off = 14
    while etype in (0x8100, 0x88A8, 0x9100):
        if len(data) < off + 4:
            return None
        etype = struct.unpack_from("!H", data, off + 2)[0]
        off += 4
    if etype == 0x0800:
        if len(data) < off + 20 or data[off] >> 4 != 4:
            return None
        ihl = (data[off] & 15) * 4
        if ihl < 20 or len(data) < off + ihl:
            return None
        proto = data[off + 9]
        src = str(ipaddress.IPv4Address(data[off + 12:off + 16]))
        dst = str(ipaddress.IPv4Address(data[off + 16:off + 20]))
        frag = struct.unpack_from("!H", data, off + 6)[0] & 0x1FFF
        trans = off + ihl
        if frag:
            return src, dst, proto, 0, 0, "", ""
    elif etype == 0x86DD:
        if len(data) < off + 40 or data[off] >> 4 != 6:
            return None
        proto = data[off + 6]
        src = str(ipaddress.IPv6Address(data[off + 8:off + 24]))
        dst = str(ipaddress.IPv6Address(data[off + 24:off + 40]))
        trans = off + 40
    else:
        return None
    sport = dport = 0
    payload = b""
    if proto == 6 and len(data) >= trans + 20:
        sport, dport = struct.unpack_from("!HH", data, trans)
        hlen = (data[trans + 12] >> 4) * 4
        if hlen >= 20:
            payload = data[min(len(data), trans + hlen):]
    elif proto == 17 and len(data) >= trans + 8:
        sport, dport = struct.unpack_from("!HH", data, trans)
        payload = data[trans + 8:]
    host = extract_host(payload, proto, sport, dport)
    hint = "DNS" if 53 in (sport, dport) else "HTTP" if 80 in (sport, dport) else "HTTPS" if 443 in (sport, dport) and not host else ""
    return src, dst, proto, sport, dport, host, hint


def extract_host(payload: bytes, proto: int, sport: int, dport: int) -> str:
    if proto == 6 and payload:
        # HTTP Host header
        m = re.search(rb"(?im)^host:\s*([^\r\n:]+)", payload[:8192])
        if m:
            return m.group(1).decode("ascii", "ignore").strip().lower()
        # TLS ClientHello SNI extension
        if len(payload) >= 9 and payload[0] == 22 and payload[5] == 1:
            body = payload[9: min(len(payload), 65536)]
            try:
                p = 2 + 32
                sid = body[p]; p += 1 + sid
                cs = struct.unpack_from("!H", body, p)[0]; p += 2 + cs
                comp = body[p]; p += 1 + comp
                extlen = struct.unpack_from("!H", body, p)[0]; p += 2
                end = min(len(body), p + extlen)
                while p + 4 <= end:
                    typ, size = struct.unpack_from("!HH", body, p); p += 4
                    ext_end = min(end, p + size)
                    if typ == 0 and p + 5 <= ext_end:
                        name_len = struct.unpack_from("!H", body, p + 3)[0]
                        return body[p + 5:p + 5 + name_len].decode("ascii", "ignore").lower()
                    p = ext_end
            except (IndexError, struct.error):
                pass
    if proto == 17 and (sport == 53 or dport == 53) and len(payload) >= 13:
        try:
            p = 12
            labels = []
            while p < len(payload) and payload[p]:
                n = payload[p]; p += 1
                if n & 0xC0 or p + n > len(payload):
                    break
                labels.append(payload[p:p + n].decode("ascii", "ignore")); p += n
            return ".".join(labels).lower()
        except IndexError:
            pass
    return ""


def flow_key(parsed):
    src, dst, proto, sport, dport, *_ = parsed
    endpoints = sorted(((src, sport), (dst, dport)))
    return proto, endpoints[0], endpoints[1]


def analyze_packets(packets, linktype, blocked_ips=(), blocked_apps=(), blocked_domains=()):
    """Classify and filter packets, carrying hostname labels across each flow."""
    model = HostnameModel()
    output, app_counts, domain_apps = [], Counter(), {}
    flow_labels = {}
    blocked_ips = {str(ipaddress.ip_address(x)) for x in blocked_ips}
    blocked_apps = {x.lower() for x in blocked_apps}
    blocked_domains = [x.lower() for x in blocked_domains]
    dropped = 0
    for packet in packets:
        parsed = inspect_packet(packet.data, linktype)
        src = host = app = ""
        if parsed:
            src, _, _, _, _, host, hint = parsed
            key = flow_key(parsed)
            if host:
                predicted_app, confidence = model.predict(host)
                if predicted_app is None:
                    app, confidence = hint, 0.0
                else:
                    app = predicted_app
                flow_labels[key] = (app, host, confidence)
            elif key in flow_labels:
                app, host, confidence = flow_labels[key]
            else:
                app, confidence = hint, 0.0
            if app:
                app_counts[app] += 1
            if host and app:
                domain_apps[host] = (app, confidence)
        should_drop = src in blocked_ips or app.lower() in blocked_apps or any(d in host for d in blocked_domains)
        if should_drop:
            dropped += 1
        else:
            output.append(packet)
    return output, app_counts, domain_apps, dropped


def main():
    ap = argparse.ArgumentParser(description="Inspect a classic PCAP and predict applications from hostnames.")
    ap.add_argument("input", type=Path, help="Input classic PCAP (Ethernet link type)")
    ap.add_argument("output", type=Path, help="Output PCAP containing forwarded packets")
    ap.add_argument("--block-app", action="append", default=[], help="Drop packets predicted as this app (repeatable)")
    ap.add_argument("--block-ip", action="append", default=[], help="Drop packets with this source IP (repeatable)")
    ap.add_argument("--block-domain", action="append", default=[], help="Drop packets whose hostname contains this text")
    args = ap.parse_args()
    try:
        gh, endian, resolution, linktype, packets = Pcap.read(args.input)
        if linktype != 1:
            raise ValueError(f"Only Ethernet PCAPs are supported (link type {linktype}).")
        output, app_counts, domain_apps, dropped = analyze_packets(
            packets, linktype, args.block_ip, args.block_app, args.block_domain
        )
        Pcap.write(args.output, gh, endian, resolution, output)
        print(f"Packets read: {len(packets)}")
        print(f"Forwarded: {len(output)}")
        print(f"Dropped: {dropped}")
        print("Application predictions:")
        for app, count in app_counts.most_common():
            print(f"  {app}: {count}")
        if domain_apps:
            print("Detected hostnames:")
            for host, (app, confidence) in sorted(domain_apps.items()):
                print(f"  {host} -> {app} (model confidence {confidence:.1%})")
        print(f"Output written to: {args.output}")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import struct
import tempfile
import unittest
from pathlib import Path

from dpi_engine import HostnameModel, Packet, Pcap, analyze_packets, extract_host, inspect_packet


def tls_client_hello(host):
    name = host.encode("ascii")
    sni_entry = b"\x00" + struct.pack("!H", len(name)) + name
    sni_list = struct.pack("!H", len(sni_entry)) + sni_entry
    extension = struct.pack("!HH", 0, len(sni_list)) + sni_list
    body = b"\x03\x03" + bytes(32) + b"\x00" + b"\x00\x02\x13\x01" + b"\x01\x00"
    body += struct.pack("!H", len(extension)) + extension
    handshake = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + struct.pack("!H", len(handshake)) + handshake


def ethernet_tcp(payload=b"", src_port=50000, dst_port=443):
    tcp = struct.pack("!HHII", src_port, dst_port, 1, 0) + b"\x50\x18" + struct.pack("!HHH", 65535, 0, 0) + payload
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(tcp), 1, 0x4000, 64, 6, 0,
                     bytes((192, 0, 2, 1)), bytes((198, 51, 100, 2)))
    return bytes(12) + b"\x08\x00" + ip + tcp


class HostnameModelTests(unittest.TestCase):
    def test_known_domain_is_classified(self):
        label, confidence = HostnameModel().predict("www.youtube.com")
        self.assertEqual(label, "YouTube")
        self.assertGreaterEqual(confidence, 0.70)

    def test_unknown_domain_abstains(self):
        label, _ = HostnameModel().predict("unrelated-random-host.invalid")
        self.assertIsNone(label)


class PacketTests(unittest.TestCase):
    def test_extracts_tls_sni_from_ethernet_ipv4_tcp(self):
        parsed = inspect_packet(ethernet_tcp(tls_client_hello("www.youtube.com")), 1)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed[5], "www.youtube.com")

    def test_rejects_truncated_and_non_ethernet_packets(self):
        self.assertIsNone(inspect_packet(b"\x00" * 8, 1))
        self.assertIsNone(inspect_packet(b"\x00" * 64, 101))

    def test_blocked_app_applies_to_the_rest_of_its_flow(self):
        frames = [ethernet_tcp(tls_client_hello("www.youtube.com")), ethernet_tcp()]
        packets = [Packet((10, i), frame, b"") for i, frame in enumerate(frames)]
        forwarded, _, domains, dropped = analyze_packets(packets, 1, blocked_apps=["YouTube"])
        self.assertEqual(len(forwarded), 0)
        self.assertEqual(dropped, 2)
        self.assertEqual(domains["www.youtube.com"][0], "YouTube")

    def test_unknown_hostname_falls_back_without_claiming_model_confidence(self):
        request = b"GET / HTTP/1.1\r\nHost: example.invalid\r\n\r\n"
        frame = ethernet_tcp(request, src_port=50000, dst_port=80)
        packets = [Packet((10, 0), frame, b"")]
        _, _, domains, _ = analyze_packets(packets, 1)
        self.assertEqual(domains["example.invalid"], ("HTTP", 0.0))

    def test_classic_pcap_round_trip(self):
        header = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
        expected = [Packet((1700000000, 123456), ethernet_tcp(), b"")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.pcap"
            Pcap.write(path, header, "<", 1_000_000, expected)
            read_header, endian, resolution, linktype, actual = Pcap.read(path)
        self.assertEqual(read_header, header)
        self.assertEqual((endian, resolution, linktype), ("<", 1_000_000, 1))
        self.assertEqual(actual[0].timestamp, expected[0].timestamp)
        self.assertEqual(actual[0].data, expected[0].data)


if __name__ == "__main__":
    unittest.main()

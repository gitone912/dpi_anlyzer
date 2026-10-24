#!/usr/bin/env python3
"""Evaluate hostname predictions on examples excluded from model training."""
from collections import Counter

from dpi_engine import HostnameModel


# Keep this set separate from TRAINING_DOMAINS. Update only with labeled,
# independently chosen hostnames; do not copy these into the training set.
HELD_OUT_DOMAINS = {
    "Google": ["maps.google.com", "google.co.uk"],
    "YouTube": ["music.youtube.com", "youtu.be"],
    "Facebook": ["m.facebook.com", "facebook.net"],
    "Instagram": ["l.instagram.com", "instagram.net"],
    "WhatsApp": ["web.whatsapp.com", "whatsapp.net"],
    "Twitter/X": ["mobile.twitter.com", "twitter.net"],
    "Netflix": ["netflix.net", "netflix.co.uk"],
    "Amazon": ["amazon.co.uk", "aws.amazon.com"],
    "Microsoft": ["azure.microsoft.com", "microsoftonline.com"],
    "Apple": ["appleid.apple.com", "apple.co.uk"],
    "Telegram": ["telegram.me", "telegram.dog"],
    "TikTok": ["vm.tiktok.com", "tiktokv.com"],
    "Spotify": ["open.spotify.com", "spotifycdn.com"],
    "Zoom": ["us02web.zoom.us", "zoomgov.com"],
    "Discord": ["canary.discord.com", "discordapp.com"],
    "GitHub": ["gist.github.com", "github.dev"],
    "Cloudflare": ["cdn.cloudflare.net", "pages.dev"],
}


def main():
    model = HostnameModel()
    total = correct = covered = accepted_correct = 0
    per_class = Counter()
    mistakes = []
    for expected, domains in HELD_OUT_DOMAINS.items():
        for domain in domains:
            predicted, confidence = model.predict(domain)
            total += 1
            if predicted:
                covered += 1
                accepted_correct += predicted == expected
            correct += predicted == expected
            per_class[expected] += predicted == expected
            if predicted != expected:
                mistakes.append((domain, expected, predicted or "abstained", confidence))
    print(f"Held-out examples: {total}")
    print(f"Correct predictions: {correct}/{total} ({correct / total:.1%})")
    print(f"Coverage: {covered}/{total} ({covered / total:.1%})")
    print(f"Accuracy when model predicts: {accepted_correct}/{covered} ({accepted_correct / covered:.1%})" if covered else "Accuracy when model predicts: n/a")
    print("Per-class recall:")
    for label in HELD_OUT_DOMAINS:
        print(f"  {label}: {per_class[label]}/{len(HELD_OUT_DOMAINS[label])}")
    if mistakes:
        print("Errors (hostname: expected -> predicted; confidence):")
        for domain, expected, predicted, confidence in mistakes:
            print(f"  {domain}: {expected} -> {predicted} ({confidence:.1%})")


if __name__ == "__main__":
    main()

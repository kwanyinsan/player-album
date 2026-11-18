from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from player_tracking import LocalTrack
from utils import ensure_dir


def export_gallery_html(
    output_root: Path,
    player_groups: list[dict[str, Any]],
    tracks_by_id: dict[str, LocalTrack],
    include_review_rejected: bool = False,
) -> Path:
    global_dir = ensure_dir(output_root / "global")
    gallery_path = global_dir / "gallery.html"
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Player Album Gallery</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:0;background:#f5f7fa;color:#1d2430}",
        "header{padding:20px 28px;background:#243447;color:#fff}",
        "main{padding:24px;display:grid;gap:20px}",
        ".player{background:#fff;border:1px solid #d9e0e8;border-radius:8px;padding:16px}",
        ".player.review{border-color:#d9a441;background:#fffaf0}",
        ".player.rejected{border-color:#d2d6dc;background:#f6f7f9;opacity:.78}",
        ".top{display:flex;gap:16px;align-items:center;margin-bottom:12px}",
        ".rep{width:132px;height:180px;object-fit:contain;border-radius:6px;border:1px solid #ccd5df;background:#fff}",
        ".review{color:#a33;font-weight:700}",
        ".status{font-weight:700;text-transform:uppercase}",
        ".tracks{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}",
        ".track{border-top:1px solid #e3e8ef;padding-top:10px}",
        ".thumbs{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}",
        ".thumbs img{width:104px;height:140px;object-fit:contain;border-radius:4px;border:1px solid #d8dee7;background:#fff}",
        ".meta{font-size:13px;color:#5e6a78;line-height:1.4}",
        ".top>div{min-width:0;overflow-wrap:anywhere}",
        ".rep{flex-shrink:0}",
        ".video{border-top:1px solid #d9e0e8;padding-top:12px;margin-top:14px;min-width:0}",
        ".video-heading{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}",
        ".video-heading h3{font-size:16px;margin:0;overflow-wrap:anywhere;min-width:0}",
        ".coverage{font-size:14px;font-variant-numeric:tabular-nums}",
        ".track{min-width:0;overflow-wrap:anywhere}",
        "@media(max-width:400px){main{padding:12px}.top{align-items:flex-start;flex-wrap:wrap}.tracks{grid-template-columns:minmax(0,1fr)}}",
        "</style>",
        "</head>",
        "<body>",
        "<header><h1>Accepted Player Albums</h1></header>",
        "<main>",
    ]

    visible_groups = [
        group for group in player_groups
        if include_review_rejected or group.get("export_status") == "accepted"
    ]

    if not visible_groups:
        parts.append('<section class="player"><p>No player groups were created.</p></section>')

    for group in visible_groups:
        player_id = html.escape(str(group["player_id"]))
        representative = _gallery_src(group.get("representative_crop"))
        review = bool(group.get("needs_review"))
        export_status = html.escape(str(group.get("export_status", "accepted")))
        quality_score = group.get("album_quality_score")
        quality_text = f"{float(quality_score):.3f}" if isinstance(quality_score, (int, float)) else "n/a"
        reasons = ", ".join(html.escape(str(reason)) for reason in group.get("quality_reasons", [])) or "none"
        identity = group.get("identity_validation") or {}
        identity_score = identity.get("identity_consistency_score")
        identity_text = f"{float(identity_score):.3f}" if isinstance(identity_score, (int, float)) else "n/a"
        color_score = identity.get("color_mean")
        color_text = f"{float(color_score):.3f}" if isinstance(color_score, (int, float)) else "n/a"
        review_text = '<span class="review">Conflict review</span>' if review else "No conflict"
        videos = len(group.get("video_ids", group.get("videos", [])))
        parts.append(f'<section class="player {export_status}" data-player-id="{player_id}">')
        parts.append('<div class="top">')
        if representative:
            parts.append(f'<img class="rep" src="{representative}" alt="{player_id} representative">')
        else:
            parts.append('<div class="rep"></div>')
        parts.append(
            f'<div><h2>{player_id}</h2><div class="meta"><span class="status">{export_status}</span> '
            f'quality: {quality_text}<br>{review_text}<br>'
            f'Reasons: {reasons}<br>Identity: {identity_text} / color: {color_text}<br>'
            f'Cluster size: {len(group.get("local_tracks", []))}<br>Videos: {videos}</div></div>'
        )
        parts.append("</div>")
        rankings = group.get("video_rankings")
        if rankings is None:
            rankings = [{"local_tracks": group.get("local_tracks", [])}]
        for entry in rankings:
            video_id = html.escape(str(entry.get("video_id", "")), quote=True)
            parts.append(f'<section class="video" data-video-id="{video_id}">')
            if "rank" in entry:
                filename = html.escape(str(entry["original_filename"]))
                percentage = entry.get("appearance_percentage")
                if percentage is None:
                    reason = html.escape(str(entry["coverage_status"]).replace("_", " "))
                    coverage = f"Coverage unavailable: {reason}"
                else:
                    coverage = f"{percentage:.1f}% ({entry['visible_frames']} of {entry['total_frames']} frames)"
                parts.append(f'<div class="video-heading"><h3>{entry["rank"]}. {filename}</h3>'
                             f'<span class="coverage">{coverage}</span></div>')
            parts.append('<div class="tracks">')
            for local_track_id in entry["local_tracks"]:
                track = tracks_by_id.get(local_track_id)
                if track is None:
                    continue
                parts.append('<div class="track">')
                parts.append(f'<div class="meta"><strong>{html.escape(local_track_id)}</strong><br>{html.escape(track.source_video_name)}</div>')
                parts.append('<div class="thumbs">')
                for crop_path in track.crop_paths[:8]:
                    parts.append(f'<img src="{_gallery_src(crop_path)}" alt="{html.escape(local_track_id)} crop">')
                parts.append("</div></div>")
            parts.append("</div></section>")
        parts.append("</section>")

    parts.extend(["</main>", "</body>", "</html>"])
    gallery_path.write_text("\n".join(parts), encoding="utf-8")
    return gallery_path


def _gallery_src(root_relative_path: object) -> str:
    if not root_relative_path:
        return ""
    return "../" + html.escape(str(root_relative_path).replace("\\", "/"), quote=True)


def export_standalone_gallery(output_root: Path, source_root: Path | None = None) -> Path:
    """Validate gallery references and embed local images into a shareable HTML file."""
    import base64
    import mimetypes
    from html.parser import HTMLParser
    from artifact_paths import artifact_path

    gallery = output_root / "global/gallery.html"
    sources = set()

    class Images(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "img":
                sources.add(dict(attrs).get("src", ""))

    content = gallery.read_text(encoding="utf-8")
    Images().feed(content)
    for source in sources:
        path = (gallery.parent / source).resolve()
        if not path.is_relative_to(output_root.resolve()):
            raise ValueError(f"Invalid gallery image: {source}")
        path = artifact_path(output_root, path.relative_to(output_root.resolve()).as_posix(), source_root)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content = content.replace(f'src="{html.escape(source, quote=True)}"', f'src="data:{mime};base64,{encoded}"')
    destination = gallery.with_name("gallery_standalone.html")
    destination.write_text(content, encoding="utf-8")
    return destination

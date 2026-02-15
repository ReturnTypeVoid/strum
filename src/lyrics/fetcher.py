"""
Lyrics fetching from free web sources.

Priority:
1. LRCLIB - Free API with synced/timed lyrics (LRC format)
2. Lyrics.ovh - Free API with plain lyrics
3. Fallback to Whisper transcription
"""

import logging
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class SyncedLyric:
    """A single line of synced lyrics with timestamp."""
    time: float  # seconds
    text: str


@dataclass
class LyricsResult:
    """Result from lyrics fetching."""
    text: str  # Plain text lyrics
    synced: Optional[list[SyncedLyric]] = None  # Timed lyrics if available
    source: str = "unknown"  # Where lyrics came from


def parse_lrc_time(time_str: str) -> float:
    """Parse LRC timestamp [mm:ss.xx] to seconds."""
    # Format: [mm:ss.xx] or [mm:ss:xx]
    match = re.match(r'\[(\d+):(\d+)[.:](\d+)\]', time_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        centiseconds = int(match.group(3))
        return minutes * 60 + seconds + centiseconds / 100
    return 0.0


def parse_lrc(lrc_content: str) -> list[SyncedLyric]:
    """Parse LRC format lyrics into synced lyrics list."""
    synced = []
    
    for line in lrc_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Find all timestamps at start of line (can be multiple)
        timestamps = re.findall(r'\[(\d+:\d+[.:]\d+)\]', line)
        
        # Get text after timestamps
        text = re.sub(r'\[\d+:\d+[.:]\d+\]', '', line).strip()
        
        if timestamps and text:
            for ts in timestamps:
                time_sec = parse_lrc_time(f'[{ts}]')
                synced.append(SyncedLyric(time=time_sec, text=text))
    
    # Sort by time
    synced.sort(key=lambda x: x.time)
    return synced


def fetch_from_lrclib(artist: str, title: str, duration_sec: Optional[float] = None) -> Optional[LyricsResult]:
    """
    Fetch lyrics from LRCLIB (free, no API key).
    
    Returns synced lyrics when available (best for charting).
    
    API: https://lrclib.net/api/get?artist_name={artist}&track_name={title}
    """
    try:
        params = {
            'artist_name': artist,
            'track_name': title,
        }
        if duration_sec:
            params['duration'] = int(duration_sec)
        
        url = 'https://lrclib.net/api/get?' + urllib.parse.urlencode(params)
        logger.info(f"Fetching lyrics from LRCLIB: {artist} - {title}")
        
        response = requests.get(url, timeout=(5, 10), headers={
            'User-Agent': 'STRUM/1.0 (https://github.com/strum)'
        })
        
        if response.status_code == 404:
            logger.debug("LRCLIB: Not found")
            return None
        
        response.raise_for_status()
        data = response.json()
        
        # LRCLIB returns both plain and synced lyrics
        plain_lyrics = data.get('plainLyrics', '')
        synced_lyrics = data.get('syncedLyrics', '')
        
        if not plain_lyrics and not synced_lyrics:
            return None
        
        synced = None
        if synced_lyrics:
            synced = parse_lrc(synced_lyrics)
            logger.info(f"LRCLIB: Got {len(synced)} synced lyric lines")
        
        return LyricsResult(
            text=plain_lyrics or '\n'.join(s.text for s in synced),
            synced=synced,
            source='lrclib'
        )
        
    except requests.RequestException as e:
        logger.warning(f"LRCLIB request failed: {e}")
        return None
    except Exception as e:
        logger.warning(f"LRCLIB parsing failed: {e}")
        return None


def fetch_from_lyrics_ovh(artist: str, title: str) -> Optional[LyricsResult]:
    """
    Fetch lyrics from Lyrics.ovh (free, no API key).
    
    Returns plain text lyrics only (no timing).
    
    API: https://api.lyrics.ovh/v1/{artist}/{title}
    """
    try:
        # URL encode artist and title
        artist_enc = urllib.parse.quote(artist)
        title_enc = urllib.parse.quote(title)
        
        url = f'https://api.lyrics.ovh/v1/{artist_enc}/{title_enc}'
        logger.info(f"Fetching lyrics from Lyrics.ovh: {artist} - {title}")
        
        response = requests.get(url, timeout=(5, 10), headers={
            'User-Agent': 'STRUM/1.0'
        })
        
        if response.status_code == 404:
            logger.debug("Lyrics.ovh: Not found")
            return None
        
        response.raise_for_status()
        data = response.json()
        
        lyrics = data.get('lyrics', '').strip()
        if not lyrics:
            return None
        
        # Clean up common issues
        lyrics = re.sub(r'\r\n', '\n', lyrics)
        lyrics = re.sub(r'\n{3,}', '\n\n', lyrics)
        
        logger.info(f"Lyrics.ovh: Got {len(lyrics.split())} words")
        
        return LyricsResult(
            text=lyrics,
            synced=None,
            source='lyrics.ovh'
        )
        
    except (requests.RequestException, ConnectionError, OSError) as e:
        logger.warning(f"Lyrics.ovh request failed: {e}")
        return None
    except Exception as e:
        logger.warning(f"Lyrics.ovh error: {e}")
        return None


def fetch_lyrics(
    artist: str,
    title: str,
    duration_sec: Optional[float] = None
) -> Optional[LyricsResult]:
    """
    Fetch lyrics from available free sources.
    
    Priority:
    1. LRCLIB (has synced/timed lyrics)
    2. Lyrics.ovh (plain text)
    
    Args:
        artist: Artist name
        title: Song title
        duration_sec: Optional song duration for better matching
        
    Returns:
        LyricsResult if found, None if no lyrics available
    """
    # Clean up artist/title
    artist = artist.strip()
    title = title.strip()
    
    # Remove common suffixes that interfere with search
    title_clean = re.sub(r'\s*\(.*?\)\s*$', '', title)  # Remove (feat. X), (Remix), etc.
    title_clean = re.sub(r'\s*\[.*?\]\s*$', '', title_clean)  # Remove [Explicit], etc.
    
    def _do_fetch():
        """Inner function run in thread with hard timeout."""
        # Try LRCLIB first (best - has synced lyrics)
        result = fetch_from_lrclib(artist, title_clean, duration_sec)
        if result:
            return result
        
        # Try with original title if cleaned version failed
        if title_clean != title:
            result = fetch_from_lrclib(artist, title, duration_sec)
            if result:
                return result
        
        # Try Lyrics.ovh
        result = fetch_from_lyrics_ovh(artist, title_clean)
        if result:
            return result
        
        # Try with original title
        if title_clean != title:
            result = fetch_from_lyrics_ovh(artist, title)
            if result:
                return result
        
        return None
    
    # Use thread-based timeout (15s hard limit) because socket/SSL
    # timeouts can hang indefinitely on Windows
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_fetch)
            result = future.result(timeout=15)
            if result:
                return result
    except FuturesTimeoutError:
        logger.warning(f"Lyrics fetch timed out (15s) for: {artist} - {title}")
    except Exception as e:
        logger.warning(f"Lyrics fetch failed: {e}")
    
    logger.info(f"No lyrics found online for: {artist} - {title}")
    return None


def extract_artist_title_from_path(file_path: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract artist and title from file path.
    
    Common formats:
    - Artist - Title/song.ogg
    - Artist - Title.mp3
    - /path/to/Artist - Title/vocals.wav
    """
    import os
    from pathlib import Path
    
    path = Path(file_path)
    
    # Try parent directory name first (common for game charts)
    parent_name = path.parent.name
    
    # Pattern: "Artist - Title"
    match = re.match(r'^(.+?)\s*-\s*(.+)$', parent_name)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    
    # Try filename
    stem = path.stem
    match = re.match(r'^(.+?)\s*-\s*(.+)$', stem)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    
    return None, None


if __name__ == "__main__":
    # Test
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) >= 3:
        artist = sys.argv[1]
        title = sys.argv[2]
    else:
        artist = "Bring Me The Horizon"
        title = "DiE4u"
    
    print(f"\nSearching for: {artist} - {title}")
    result = fetch_lyrics(artist, title)
    
    if result:
        print(f"\nSource: {result.source}")
        if result.synced:
            print(f"Synced lyrics: {len(result.synced)} lines")
            for lyric in result.synced[:10]:
                print(f"  [{lyric.time:.2f}] {lyric.text}")
            if len(result.synced) > 10:
                print(f"  ... and {len(result.synced) - 10} more lines")
        else:
            print(f"Plain lyrics: {len(result.text)} chars")
            print(result.text[:500])
    else:
        print("No lyrics found")

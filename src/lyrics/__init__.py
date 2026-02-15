"""Lyrics fetching and processing."""

from .fetcher import (
    fetch_lyrics,
    extract_artist_title_from_path,
    LyricsResult,
    SyncedLyric,
)

__all__ = [
    'fetch_lyrics',
    'extract_artist_title_from_path', 
    'LyricsResult',
    'SyncedLyric',
]

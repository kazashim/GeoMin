#!/usr/bin/env python
"""
Command-line interface for GeoMin library.
"""

import argparse
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='GeoMin - Geophysics Library for Satellite-Based Mining Detection'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    subparsers = parser.add_subparsers(
        title='commands',
        dest='command',
        help='Available commands'
    )
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search satellite imagery')
    search_parser.add_argument(
        '--provider', '-p',
        choices=['sentinel', 'landsat', 'planet', 'maxar'],
        default='sentinel',
        help='Satellite data provider'
    )
    search_parser.add_argument(
        '--bbox', '-b',
        required=True,
        help='Bounding box (minx,miny,maxx,maxy)'
    )
    search_parser.add_argument(
        '--start-date', '-s',
        help='Start date (YYYY-MM-DD)'
    )
    search_parser.add_argument(
        '--end-date', '-e',
        help='End date (YYYY-MM-DD)'
    )
    search_parser.add_argument(
        '--cloud-cover', '-c',
        type=float,
        default=20,
        help='Maximum cloud cover percentage'
    )
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download satellite imagery')
    download_parser.add_argument(
        '--scene-id', '-i',
        required=True,
        help='Scene ID to download'
    )
    download_parser.add_argument(
        '--provider', '-p',
        choices=['sentinel', 'landsat'],
        default='sentinel',
        help='Satellite data provider'
    )
    download_parser.add_argument(
        '--output', '-o',
        help='Output directory'
    )
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process satellite data')
    process_parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input file or directory'
    )
    process_parser.add_argument(
        '--output', '-o',
        help='Output directory'
    )
    process_parser.add_argument(
        '--indices', '-x',
        nargs='+',
        default=['all'],
        help='Spectral indices to calculate'
    )
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect changes/minerals')
    detect_parser.add_argument(
        '--before', '-b',
        required=True,
        help='Before image path'
    )
    detect_parser.add_argument(
        '--after', '-a',
        required=True,
        help='After image path'
    )
    detect_parser.add_argument(
        '--output', '-o',
        help='Output directory'
    )
    detect_parser.add_argument(
        '--method', '-m',
        choices=['vegetation', 'pca', 'kmeans'],
        default='vegetation',
        help='Change detection method'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'search':
        print(f"Searching {args.provider} imagery...")
        print(f"Bbox: {args.bbox}")
        print(f"Date range: {args.start_date} to {args.end_date}")
        print(f"Cloud cover max: {args.cloud-cover}%")
    
    elif args.command == 'download':
        print(f"Downloading scene {args.scene_id} from {args.provider}")
    
    elif args.command == 'process':
        print(f"Processing {args.input}")
        print(f"Indices: {args.indices}")
    
    elif args.command == 'detect':
        print(f"Detecting changes between {args.before} and {args.after}")
        print(f"Method: {args.method}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export OpenAPI specification from FastAPI application
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import api_server
sys.path.insert(0, str(Path(__file__).parent))

from api_server import app


def export_openapi(output_format: str = "json", output_file: str = None):
    """
    Export OpenAPI specification

    Args:
        output_format: Output format ('json' or 'yaml')
        output_file: Output file path (defaults to openapi.json or openapi.yaml)
    """
    openapi_schema = app.openapi()

    if output_format == "yaml":
        try:
            import yaml
        except ImportError:
            print("Error: PyYAML required for YAML export. Install with: pip install pyyaml")
            sys.exit(1)

        output_file = output_file or "openapi.yaml"
        with open(output_file, "w") as f:
            yaml.dump(openapi_schema, f, default_flow_style=False, sort_keys=False)
        print(f"✅ OpenAPI specification exported to {output_file}")

    else:
        output_file = output_file or "openapi.json"
        with open(output_file, "w") as f:
            json.dump(openapi_schema, f, indent=2)
        print(f"✅ OpenAPI specification exported to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export OpenAPI specification from BondTrader API")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format (default: json)")
    parser.add_argument("--output", help="Output file path (default: openapi.json or openapi.yaml)")

    args = parser.parse_args()
    export_openapi(args.format, args.output)

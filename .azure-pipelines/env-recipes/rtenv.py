"""
Configures a conda recipe for runtime dependencies on a platform with a python version.

The generated script is deposited next to this script, and is named 'rtenv.yml'

command line arguments are the python version and platform
    - valid python versions -v/--version: 3.8, 3.9, 3.10, 3.11
    - valid platforms -p/--platform: win64, linux64, osx64, osxarm64
"""

import argparse
import os
import sys
from typing import List

supported_py_version = ['3.9', '3.10', '3.11', '3.12']
supported_platform = ['win64', 'linux64', 'osx64', 'osxarm64']
this_dir = os.path.dirname(os.path.abspath(__file__))
py_version_token = '@PYTHON_VERSION@'
env_output_script = 'rtenv.yml'
env_output_script_path = os.path.join(this_dir, env_output_script)


class RTEnvParser:
    """Command line argument parser"""

    def __init__(self, argv: List[str]):

        parser = argparse.ArgumentParser(description='Tissue Forge runtime environment configuration')

        parser.add_argument('-v', '--version', required=True, type=str, dest='py_version',
                            help='Python version (3.9, 3.10, 3.11, 3.12)',
                            choices=supported_py_version)
        parser.add_argument('-p', '--platform', required=True, type=str, dest='platform',
                            help='Target platform',
                            choices=supported_platform)

        self.parsed_args = parser.parse_args(argv)

    @property
    def py_version(self) -> str:
        """Python version"""

        return self.parsed_args.py_version

    @property
    def platform(self) -> str:
        """Target platform"""

        return self.parsed_args.platform

    @staticmethod
    def help_str() -> str:
        """Simple help string when incorrectly called"""

        return 'Command line arguments are the python version (-v) and platform (-p)'


def main(py_version: str, platform: str):
    if py_version not in supported_py_version:
        raise ValueError('Unsupported python version. Supported versions are ' + ','.join(supported_py_version))
    elif platform not in supported_platform:
        raise ValueError('Unsupported platform. Supported versions are ' + ','.join(supported_platform))

    env_script = f'rtenv_{platform}.yml.in'
    env_script_path = os.path.join(this_dir, env_script)
    with open(env_script_path, 'r') as f:
        fstr = f.read()
    with open(env_output_script_path, 'w') as f:
        f.write(fstr.replace(py_version_token, py_version))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError(RTEnvParser.help_str())
    argvp = RTEnvParser(sys.argv[1:])
    main(py_version=argvp.py_version, platform=argvp.platform)

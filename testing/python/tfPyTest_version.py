# ******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022-2024 T.J. Sego
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# ******************************************************************************

import tissue_forge as tf

print("Tissue Forge Version:")
print("version: ", tf.version.version)
print("build date: ", tf.version.build_date)
print("compiler: ", tf.version.compiler)
print("compiler_version: ", tf.version.compiler_version)
print("system_version: ", tf.version.system_version)

for k, v in tf.system.cpu_info().items():
    print("cpuinfo[", k, "]: ",  v)

for k, v in tf.system.compile_flags().items():
    print("compile_flags[", k, "]: ",  v)


def test_pass():
    pass

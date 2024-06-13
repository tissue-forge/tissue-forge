/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

%{

#include <rendering/tfColorMapper.h>
#include <rendering/tfStyle.h>

%}


%import <tf_style.h>
%import <rendering/tfColorMaps.h>

%ignore TissueForge::rendering::ColorMapperFunc;
%ignore TissueForge::rendering::Style::flags;
%ignore TissueForge::rendering::Style::setFlag;
%ignore TissueForge::rendering::Style::map_color;
%ignore TissueForge::rendering::Style::Style(const std::string &, const bool &, uint32_t, TissueForge::rendering::ColorMapper*);
%ignore TissueForge::rendering::Style::init;
%ignore TissueForge::rendering::Style::setColorMapper;

%rename(_hasMapParticle) TissueForge::rendering::ColorMapper::hasMapParticle;
%rename(_hasMapAngle) TissueForge::rendering::ColorMapper::hasMapAngle;
%rename(_hasMapBond) TissueForge::rendering::ColorMapper::hasMapBond;
%rename(_hasMapDihedral) TissueForge::rendering::ColorMapper::hasMapDihedral;
%rename(clear_map_particle) TissueForge::rendering::ColorMapper::clearMapParticle;
%rename(clear_map_angle) TissueForge::rendering::ColorMapper::clearMapAngle;
%rename(clear_map_bond) TissueForge::rendering::ColorMapper::clearMapBond;
%rename(clear_map_dihedral) TissueForge::rendering::ColorMapper::clearMapDihedral;
%rename(set_map_particle_position_x) TissueForge::rendering::ColorMapper::setMapParticlePositionX;
%rename(set_map_particle_position_y) TissueForge::rendering::ColorMapper::setMapParticlePositionY;
%rename(set_map_particle_position_z) TissueForge::rendering::ColorMapper::setMapParticlePositionZ;
%rename(set_map_particle_velocity_x) TissueForge::rendering::ColorMapper::setMapParticleVelocityX;
%rename(set_map_particle_velocity_y) TissueForge::rendering::ColorMapper::setMapParticleVelocityY;
%rename(set_map_particle_velocity_z) TissueForge::rendering::ColorMapper::setMapParticleVelocityZ;
%rename(set_map_particle_speed) TissueForge::rendering::ColorMapper::setMapParticleSpeed;
%rename(set_map_particle_force_x) TissueForge::rendering::ColorMapper::setMapParticleForceX;
%rename(set_map_particle_force_y) TissueForge::rendering::ColorMapper::setMapParticleForceY;
%rename(set_map_particle_force_z) TissueForge::rendering::ColorMapper::setMapParticleForceZ;
%rename(set_map_particle_species) TissueForge::rendering::ColorMapper::setMapParticleSpecies;
%rename(set_map_angle_angle) TissueForge::rendering::ColorMapper::setMapAngleAngle;
%rename(set_map_angle_angle_eq) TissueForge::rendering::ColorMapper::setMapAngleAngleEq;
%rename(set_map_bond_length) TissueForge::rendering::ColorMapper::setMapBondLength;
%rename(set_map_bond_length_eq) TissueForge::rendering::ColorMapper::setMapBondLengthEq;
%rename(set_map_dihedral_angle) TissueForge::rendering::ColorMapper::setMapDihedralAngle;
%rename(set_map_dihedral_angle_eq) TissueForge::rendering::ColorMapper::setMapDihedralAngleEq;

%rename(_rendering_ColorMapper) ColorMapper;
%rename(_rendering_Style) Style;

%include <rendering/tfColorMapper.h>
%include <rendering/tfStyle.h>

%extend TissueForge::rendering::ColorMapper {
    %pythoncode %{
        @property
        def has_map_particle(self) -> bool:
            return self._hasMapParticle()

        @property
        def has_map_angle(self) -> bool:
            return self._hasMapAngle()

        @property
        def has_map_bond(self) -> bool:
            return self._hasMapBond()

        @property
        def has_map_dihedral(self) -> bool:
            return self._hasMapDihedral()
        
    %}
}

%extend TissueForge::rendering::Style {
    %pythoncode %{
        @property
        def visible(self) -> bool:
            """Visibility flag"""
            return self.getVisible()

        @visible.setter
        def visible(self, visible: bool):
            self.setVisible(visible)

        @property
        def colormap(self) -> str:
            """
            Name of color map
            """
            return self.getColorMap()

        @colormap.setter
        def colormap(self, _name: str):
            self.setColorMap(_name)

        def __reduce__(self):
            return self.fromString, (self.toString(),)
    %}
}

/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego
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

#include <rendering/tfStyle.h>

%}


%import <tf_style.h>
%import <rendering/tfColorMapper.h>

%ignore TissueForge::rendering::ColorMapperFunc;
%ignore TissueForge::rendering::Style::flags;
%ignore TissueForge::rendering::Style::mapper;
%ignore TissueForge::rendering::Style::mapper_func;
%ignore TissueForge::rendering::Style::setFlag;
%ignore TissueForge::rendering::Style::map_color;
%ignore TissueForge::rendering::Style::getColorMap;
%ignore TissueForge::rendering::Style::Style(const std::string &, const bool &, uint32_t, TissueForge::rendering::ColorMapper*);
%ignore TissueForge::rendering::Style::init;
%ignore TissueForge::rendering::Style::setColorMapper;

%rename(_rendering_Style) Style;

%include <rendering/tfStyle.h>

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

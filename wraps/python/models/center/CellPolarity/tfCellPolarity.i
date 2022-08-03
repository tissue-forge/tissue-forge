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
    #include <models/center/CellPolarity/tfCellPolarity.h>
%}

%rename(_models_center_CellPolarity_getVectorAB) TissueForge::models::center::CellPolarity::getVectorAB;
%rename(_models_center_CellPolarity_getVectorPCP) TissueForge::models::center::CellPolarity::getVectorPCP;
%rename(_models_center_CellPolarity_setVectorAB) TissueForge::models::center::CellPolarity::setVectorAB;
%rename(_models_center_CellPolarity_setVectorPCP) TissueForge::models::center::CellPolarity::setVectorPCP;
%rename(_models_center_CellPolarity_update) TissueForge::models::center::CellPolarity::update;
%rename(_models_center_CellPolarity_registerParticle) TissueForge::models::center::CellPolarity::registerParticle;
%rename(_models_center_CellPolarity_unregister) TissueForge::models::center::CellPolarity::unregister;
%rename(_models_center_CellPolarity_registerType) TissueForge::models::center::CellPolarity::registerType;
%rename(_models_center_CellPolarity_getInitMode) TissueForge::models::center::CellPolarity::getInitMode;
%rename(_models_center_CellPolarity_setInitMode) TissueForge::models::center::CellPolarity::setInitMode;
%rename(_models_center_CellPolarity_getInitPolarAB) TissueForge::models::center::CellPolarity::getInitPolarAB;
%rename(_models_center_CellPolarity_setInitPolarAB) TissueForge::models::center::CellPolarity::setInitPolarAB;
%rename(_models_center_CellPolarity_getInitPolarPCP) TissueForge::models::center::CellPolarity::getInitPolarPCP;
%rename(_models_center_CellPolarity_setInitPolarPCP) TissueForge::models::center::CellPolarity::setInitPolarPCP;
%rename(_models_center_CellPolarity_PersistentForce) TissueForge::models::center::CellPolarity::PersistentForce;
%rename(_models_center_CellPolarity_createPersistentForce) TissueForge::models::center::CellPolarity::createPersistentForce;
%rename(_models_center_CellPolarity_PolarityArrowData) TissueForge::models::center::CellPolarity::PolarityArrowData;
%rename(_models_center_CellPolarity_setDrawVectors) TissueForge::models::center::CellPolarity::setDrawVectors;
%rename(_models_center_CellPolarity_setArrowColors) TissueForge::models::center::CellPolarity::setArrowColors;
%rename(_models_center_CellPolarity_setArrowScale) TissueForge::models::center::CellPolarity::setArrowScale;
%rename(_models_center_CellPolarity_setArrowLength) TissueForge::models::center::CellPolarity::setArrowLength;
%rename(_models_center_CellPolarity_getVectorArrowAB) TissueForge::models::center::CellPolarity::getVectorArrowAB;
%rename(_models_center_CellPolarity_getVectorArrowPCP) TissueForge::models::center::CellPolarity::getVectorArrowPCP;
%rename(_models_center_CellPolarity_load) TissueForge::models::center::CellPolarity::load;
%rename(_models_center_CellPolarity_ContactPotential) TissueForge::models::center::CellPolarity::ContactPotential;
%rename(_models_center_CellPolarity_createContactPotential) TissueForge::models::center::CellPolarity::createContactPotential;

%include <models/center/CellPolarity/tfCellPolarity.h>

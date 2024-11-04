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

#ifndef _SOURCE_RENDERING_TFSUBRENDERER_H_
#define _SOURCE_RENDERING_TFSUBRENDERER_H_

#include <TissueForge_private.h>

#include "tfArcBallCamera.h"
#include <Magnum/Platform/GlfwApplication.h>


namespace TissueForge {


    namespace rendering {


        struct SubRenderer {

            virtual ~SubRenderer() {}

            /**
             * @brief Starts the renderer. 
             * 
             * Called by parent renderer once backend is initialized. 
             * 
             * @param clipPlanes clip plane specification
             * @return HRESULT 
             */
            virtual HRESULT start(const std::vector<fVector4> &clipPlanes) = 0;

            /**
             * @brief Updates visualization. 
             * 
             * @param camera scene camera
             * @param viewportSize scene viewport size
             * @param modelViewMat scene model view matrix
             * @return HRESULT 
             */
            virtual HRESULT draw(ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) = 0;

            virtual void keyPressEvent(Magnum::Platform::GlfwApplication::KeyEvent& event) {}
            virtual void keyReleaseEvent(Platform::GlfwApplication::KeyEvent& event) {}
            virtual void mousePressEvent(Magnum::Platform::GlfwApplication::MouseEvent& event) {}
            virtual void mouseReleaseEvent(Magnum::Platform::GlfwApplication::MouseEvent& event) {}
            virtual void mouseMoveEvent(Magnum::Platform::GlfwApplication::MouseMoveEvent& event) {}
            virtual void mouseScrollEvent(Magnum::Platform::GlfwApplication::MouseScrollEvent& event) {}
            virtual void textInputEvent(Platform::GlfwApplication::TextInputEvent& event) {}

            /**
             * @brief Adds a clip plane equation
             * 
             * @param pe clip plane equation
             * @return const unsigned 
             */
            virtual const unsigned addClipPlaneEquation(const Magnum::Vector4& pe) { return E_NOTIMPL; }
            
            /**
             * @brief Removes a clip plane equation
             * 
             * @param id id of clip plane equation
             * @return const unsigned 
             */
            virtual const unsigned removeClipPlaneEquation(const unsigned int &id) { return E_NOTIMPL; }

            /**
             * @brief Sets a clip plane equation
             * 
             * @param id id of clip plane equation
             * @param pe clip plane equation
             */
            virtual void setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {}

            /**
             * @brief Sets the ambient color
             * 
             * @param color 
             */
            virtual void setAmbientColor(const Magnum::Color3& color) {}

            /**
             * @brief Set the diffuse color
             * 
             * @param color 
             */
            virtual void setDiffuseColor(const Magnum::Color3& color) {}

            /**
             * @brief Set the specular color
             * 
             * @param color 
             */
            virtual void setSpecularColor(const Magnum::Color3& color) {}

            /**
             * @brief Sets the shininess
             * 
             * @param shininess 
             */
            virtual void setShininess(float shininess) {}

            /**
             * @brief Sets the light direction
             * 
             * @param lightDir 
             */
            virtual void setLightDirection(const fVector3& lightDir) {}

            /**
             * @brief Sets the light color
             * 
             * @param color 
             */
            virtual void setLightColor(const Magnum::Color3 &color) {}

        };

}};

#endif // _SOURCE_RENDERING_TFSUBRENDERER_H_
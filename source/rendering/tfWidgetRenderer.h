/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
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

#ifndef _SOURCE_RENDERING_TFWIDGETRENDERER_H_
#define _SOURCE_RENDERING_TFWIDGETRENDERER_H_

#include "tfSubRenderer.h"


namespace TissueForge::rendering {

    struct WidgetRenderer : SubRenderer {

        template <typename T> using CallbackInput = void(*)(const T&);
        template <typename T> using FunctionInput = std::function<void(const T&)>;
        using CallbackOutput = void(*)();
        using FunctionOutput = std::function<void()>;

        ~WidgetRenderer();

        HRESULT start(const std::vector<fVector4> &clipPlanes) override;
        HRESULT draw(ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) override;

        /**
         * @brief Gets the global instance of the renderer. 
         * 
         * Cannot be used until the universe renderer has been initialized. 
         * 
         * @return WidgetRenderer* 
         */
        static WidgetRenderer *get();

        void keyPressEvent(Magnum::Platform::GlfwApplication::KeyEvent& event) override;
        void mousePressEvent(Magnum::Platform::GlfwApplication::MouseEvent& event) override;
        void mouseReleaseEvent(Magnum::Platform::GlfwApplication::MouseEvent& event) override;
        void textInputEvent(Magnum::Platform::GlfwApplication::TextInputEvent& event) override;

        int addButton(rendering::WidgetRenderer::CallbackOutput cb, const std::string& label);
        int addButton(rendering::WidgetRenderer::FunctionOutput cb, const std::string& label);

        int addOutputField(const int& val, const std::string& label);
        int addOutputField(const int& val);
        int addOutputField(const float& val, const std::string& label);
        int addOutputField(const float& val);
        int addOutputField(const double& val, const std::string& label);
        int addOutputField(const double& val);
        int addOutputField(const std::string& val, const std::string& label);
        int addOutputField(const std::string& val);

        int addInputField(CallbackInput<int> cb, const int& val, const std::string& label);
        int addInputField(CallbackInput<int> cb, const int& val);
        int addInputField(CallbackInput<float> cb, const float& val, const std::string& label);
        int addInputField(CallbackInput<float> cb, const float& val);
        int addInputField(CallbackInput<double> cb, const double& val, const std::string& label);
        int addInputField(CallbackInput<double> cb, const double& val);
        int addInputField(CallbackInput<std::string> cb, const std::string& val, const std::string& label);
        int addInputField(CallbackInput<std::string> cb, const std::string& val);
        int addInputField(FunctionInput<int> cb, const int& val, const std::string& label);
        int addInputField(FunctionInput<int> cb, const int& val);
        int addInputField(FunctionInput<float> cb, const float& val, const std::string& label);
        int addInputField(FunctionInput<float> cb, const float& val);
        int addInputField(FunctionInput<double> cb, const double& val, const std::string& label);
        int addInputField(FunctionInput<double> cb, const double& val);
        int addInputField(FunctionInput<std::string> cb, const std::string& val, const std::string& label);
        int addInputField(FunctionInput<std::string> cb, const std::string& val);

        HRESULT pressButton(const unsigned int& idx);

        HRESULT getOutputInt(const unsigned int& idx, int& val);
        HRESULT getOutputFloat(const unsigned int& idx, float& val);
        HRESULT getOutputDouble(const unsigned int& idx, double& val);
        HRESULT getOutputString(const unsigned int& idx, std::string& val);

        HRESULT getInputInt(const unsigned int& idx, int& val);
        HRESULT getInputFloat(const unsigned int& idx, float& val);
        HRESULT getInputDouble(const unsigned int& idx, double& val);
        HRESULT getInputString(const unsigned int& idx, std::string& val);

        HRESULT setOutputInt(const unsigned int& idx, const int& val);
        HRESULT setOutputFloat(const unsigned int& idx, const float& val);
        HRESULT setOutputDouble(const unsigned int& idx, const double& val);
        HRESULT setOutputString(const unsigned int& idx, const std::string& val);

    };
}


#endif // _SOURCE_RENDERING_TFWIDGETRENDERER_H_
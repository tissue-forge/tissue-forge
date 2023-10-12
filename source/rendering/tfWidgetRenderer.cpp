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

#include "tfWidgetRenderer.h"
#include "tfApplication.h"
#include "tfGlfwApplication.h"
#include "tfGlfwWindow.h"

#include <Corrade/Containers/Optional.h>
#include <Corrade/Interconnect/Receiver.h>
#include <Corrade/Utility/FormatStl.h>
#include <Magnum/Magnum.h>
#include <Magnum/GL/AbstractFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Text/Alignment.h>
#include <Magnum/Ui/Anchor.h>
#include <Magnum/Ui/Button.h>
#include <Magnum/Ui/Label.h>
#include <Magnum/Ui/Plane.h>
#include <Magnum/Ui/Style.h>
#include <Magnum/Ui/UserInterface.h>
#include <Magnum/Ui/ValidatedInput.h>

#include <tfError.h>
#include <tfSimulator.h>
#include <tf_system.h>
#include <types/tf_cast.h>

#include <regex>

using namespace TissueForge;


constexpr Magnum::Vector2 WidgetSize{80, 32};

static struct UserInterface* _ui = NULL;


template <typename T> const std::regex* regex_validator();

template<> const std::regex* regex_validator<int>() { return new std::regex{R"(^[-+]?[1-9]\d*\.?[0]*$)"}; }
template<> const std::regex* regex_validator<float>() { return new std::regex{R"(^([0-9]*|\d*\.\d{1}?\d*)$)"}; }
template<> const std::regex* regex_validator<double>() { return regex_validator<float>(); }
template<> const std::regex* regex_validator<std::string>() { return new std::regex(); }


namespace TissueForge {
    template <> std::string cast<std::string>(const std::string& s) { return s; }
}


static HRESULT WidgetRenderer_getApp(rendering::GlfwApplication** app) {
    Simulator* sim = Simulator::get();
    if(!sim || !sim->app) 
        return E_FAIL;

    *app = (rendering::GlfwApplication*)sim->app;
    return S_OK;
}

#define WIDGETRENDERER_GETAPP(retval)           \
    rendering::GlfwApplication* app;            \
    if(WidgetRenderer_getApp(&app) != S_OK)     \
        return retval;

static void startTextInput() {
    WIDGETRENDERER_GETAPP();
    app->startTextInput();
}

static void stopTextInput() {
    WIDGETRENDERER_GETAPP();
    app->stopTextInput();
}


//////////////////
// ModularPlane //
//////////////////


enum ModularPlaneWidgetType {
    Button = 0,
    Input,
    Label,
    ValidatedInput
};


void deleteWidget(Magnum::Ui::Widget* w, const ModularPlaneWidgetType& wtype) {
    if(wtype == ModularPlaneWidgetType::Button) 
        delete (Magnum::Ui::Button*)w;
    else if(wtype == ModularPlaneWidgetType::Input) 
        delete (Magnum::Ui::Input*)w;
    else if(wtype == ModularPlaneWidgetType::Label) 
        delete (Magnum::Ui::Label*)w;
    else if(wtype == ModularPlaneWidgetType::ValidatedInput) 
        delete (Magnum::Ui::ValidatedInput*)w;
}


struct ModularPlane : Magnum::Ui::Plane {

    explicit ModularPlane(Magnum::Ui::UserInterface& ui) : 
        Magnum::Ui::Plane{
            ui, 
            Magnum::Ui::Snap::Top|Magnum::Ui::Snap::Bottom|Magnum::Ui::Snap::Left|Magnum::Ui::Snap::Right,
            0, 16, 256
        }
    {}

    ~ModularPlane() {
        for(auto& d : _data) {
            ModularPlaneWidgetType wtype;
            Magnum::Ui::Widget* widget;
            Magnum::Ui::Label* label;
            std::tie(wtype, widget, label) = d;
            deleteWidget(widget, wtype);
            if(label) delete label;
        }
        _data.clear();
    }

    template <typename F> 
    unsigned int addButton(const F& cb, const std::string& label) {
        Magnum::Ui::Button* widget = new Magnum::Ui::Button(
            *(Magnum::Ui::Plane*)this,
            generateAnchor(),
            label,
            15
        );
        _data.push_back({ModularPlaneWidgetType::Button, widget, NULL});
        Corrade::Interconnect::connect(
            *widget, 
            &Magnum::Ui::Button::tapped,
            cb
        );
        return _data.size() - 1;
    }

    template <typename T> 
    unsigned int addOutputField(const T& val) {
        return _addOutputField(cast<T, std::string>(val));
    }

    template <typename T, typename F> 
    unsigned int addInputField(const F& cb, const T& val) {
        Magnum::Ui::ValidatedInput* widget = new Magnum::Ui::ValidatedInput(
            *(Magnum::Ui::Plane*)this, 
            generateAnchor(), 
            *regex_validator<T>(), 
            cast<T, std::string>(val), 
            15
        );
        _data.push_back({ModularPlaneWidgetType::ValidatedInput, widget, NULL});
        Corrade::Interconnect::connect(
            *widget, 
            &Magnum::Ui::ValidatedInput::valueChanged, 
            std::bind(&ModularPlane::_apply_cb<T, F>, this, cb, std::placeholders::_1)
        );
        return _data.size() - 1;
    }

    template <typename T, typename F> 
    void _apply_cb(const F& cb, const std::string& val) { cb(cast<std::string, T>(val)); }

    HRESULT pressButton(const unsigned int& idx) {
        if(_checkDataIndex(idx) != S_OK) return E_FAIL;

        ModularPlaneWidgetType wtype;
        Magnum::Ui::Widget* widget;
        Magnum::Ui::Label* label;
        std::tie(wtype, widget, label) = _data[idx];

        if(_checkWidgetType(wtype, ModularPlaneWidgetType::Button) != S_OK) return E_FAIL;
        ((Magnum::Ui::Button*)widget)->tapped();
        return S_OK;
    }

    template <typename T> 
    HRESULT getInput(const unsigned int& idx, T& val) {
        if(_checkDataIndex(idx) != S_OK) return E_FAIL;

        ModularPlaneWidgetType wtype;
        Magnum::Ui::Widget* widget;
        Magnum::Ui::Label* label;
        std::tie(wtype, widget, label) = _data[idx];

        if(_checkWidgetType(wtype, ModularPlaneWidgetType::ValidatedInput) != S_OK) return E_FAIL;
        val = cast<std::string, T>(((Magnum::Ui::ValidatedInput*)widget)->value());
        return S_OK;
    }

    template <typename T> 
    HRESULT getOutput(const unsigned int& idx, T& val) {
        if(_checkDataIndex(idx) != S_OK) return E_FAIL;

        ModularPlaneWidgetType wtype;
        Magnum::Ui::Widget* widget;
        Magnum::Ui::Label* label;
        std::tie(wtype, widget, label) = _data[idx];

        if(_checkWidgetType(wtype, ModularPlaneWidgetType::Input) != S_OK) return E_FAIL;
        val = cast<std::string, T>(((Magnum::Ui::Input*)widget)->value());
        return S_OK;
    }

    template <typename T> 
    HRESULT setOutput(const unsigned int& idx, const T& val) {
        if(_checkDataIndex(idx) != S_OK) return E_FAIL;

        ModularPlaneWidgetType wtype;
        Magnum::Ui::Widget* widget;
        Magnum::Ui::Label* label;
        std::tie(wtype, widget, label) = _data[idx];

        if(_checkWidgetType(wtype, ModularPlaneWidgetType::Input) != S_OK) return E_FAIL;
        ((Magnum::Ui::Input*)widget)->setValue(cast<T, std::string>(val));
        return S_OK;
    }

    HRESULT addSetLabel(const unsigned int& idx, const std::string& lbl) {
        if(_checkDataIndex(idx) != S_OK) return E_FAIL;

        ModularPlaneWidgetType wtype;
        Magnum::Ui::Widget* widget;
        Magnum::Ui::Label* label;
        std::tie(wtype, widget, label) = _data[idx];

        if(wtype == ModularPlaneWidgetType::Button) {
            // Button has its own text
            ((Magnum::Ui::Button*)widget)->setText(lbl);
        }
        else {
            if(!label) {
                label = new Magnum::Ui::Label{
                    *(Magnum::Ui::Plane*)this, 
                    {Magnum::Ui::Snap::Left, *widget}, 
                    lbl, 
                    Magnum::Text::Alignment::MiddleRight
                };
                _data[idx] = {wtype, widget, label};
            } 
            else {
                label->setText(lbl);
            }
        }

        return S_OK;
    }

private:

    std::vector<std::tuple<ModularPlaneWidgetType, Magnum::Ui::Widget*, Magnum::Ui::Label*> > _data;

    Magnum::Ui::Anchor generateAnchor() {
        return _data.size() == 0 ? 
            Magnum::Ui::Anchor{Magnum::Ui::Snap::Top|Magnum::Ui::Snap::Right, WidgetSize} : 
            Magnum::Ui::Anchor{Magnum::Ui::Snap::Bottom, *std::get<1>(_data.back()), WidgetSize};
    }

    unsigned int _addOutputField(const std::string& val) {
        Magnum::Ui::Input* widget = new Magnum::Ui::Input(*(Magnum::Ui::Plane*)this, generateAnchor(), val, 15);
        _data.push_back({ModularPlaneWidgetType::Input, widget, NULL});
        return _data.size() - 1;
    }

    HRESULT _checkDataIndex(const unsigned int& idx) {
        if(idx >= _data.size()) { return tf_error(E_FAIL, "Request exceeds available widgets"); }
        return S_OK;
    }

    HRESULT _checkWidgetType(const ModularPlaneWidgetType& wtype1, const ModularPlaneWidgetType& wtype2) {
        if(wtype1 != wtype2) { return tf_error(E_FAIL, "Requested widget is of incorrect type"); }
        return S_OK;
    }

};


///////////////////
// UserInterface //
///////////////////


struct UserInterface : Magnum::Ui::UserInterface {

    UserInterface(rendering::GlfwApplication* app) : 
        Magnum::Ui::UserInterface{
            Vector2(app->windowSize()) / app->dpiScaling(), 
            app->windowSize(), 
            app->framebufferSize(), 
            Magnum::Ui::mcssDarkStyleConfiguration()
        },
        plane{*this}
    {}
    ~UserInterface() {}

    static UserInterface* get()  {
        if(_ui) return _ui;

        WIDGETRENDERER_GETAPP(NULL);

        _ui = new UserInterface(app);
        Corrade::Interconnect::connect(
            *_ui, 
            &Magnum::Ui::UserInterface::inputWidgetFocused, 
            &startTextInput
        );
        Corrade::Interconnect::connect(
            *_ui, 
            &Magnum::Ui::UserInterface::inputWidgetBlurred, 
            &stopTextInput
        );

        return _ui;
    }

    template <typename T> 
    unsigned int addOutputField(const T& val) {
        return plane.addOutputField(val);
    }

    template <typename T> 
    unsigned int addOutputField(const T& val, const std::string& label) {
        unsigned int idx = addOutputField(val);
        plane.addSetLabel(idx, label);
        return idx;
    }

    template <typename T, typename F> 
    unsigned int addInputField(const F& cb, const T& val) {
        return plane.addInputField(cb, val);
    }

    template <typename T, typename F> 
    unsigned int addInputField(const F& cb, const T& val, const std::string& label) {
        unsigned int idx = addInputField(cb, val);
        plane.addSetLabel(idx, label);
        return idx;
    }

    template <typename F> 
    unsigned int addButton(const F& cb, const std::string& label) {
        return plane.addButton(cb, label);
    }

    template <typename T> 
    HRESULT getOutput(const unsigned int& idx, T& val) {
        return plane.getOutput(idx, val);
    }

    template <typename T> 
    HRESULT getInput(const unsigned int& idx, T& val) {
        return plane.getInput(idx, val);
    }

    template <typename T> 
    HRESULT setOutput(const unsigned int& idx, const T& val) {
        return plane.setOutput(idx, val);
    }

    HRESULT pressButton(const unsigned int& idx) {
        return plane.pressButton(idx);
    }

private:

    ModularPlane plane;

};


#define WIDGETRENDERER_GETUI(name, retval)                                              \
    UserInterface* name = UserInterface::get();                                         \
    if(!name) {tf_error(E_FAIL, "Could not retrieve user interface"); return retval; }


////////////////////
// WidgetRenderer //
////////////////////


rendering::WidgetRenderer::~WidgetRenderer() {
    if(_ui) {
        delete _ui;
        _ui = NULL;
    }
}

HRESULT rendering::WidgetRenderer::start(const std::vector<fVector4> &clipPlanes) {
    return S_OK;
}

HRESULT rendering::WidgetRenderer::draw(
    ArcBallCamera *camera, 
    const iVector2 &viewportSize, 
    const fMatrix4 &modelViewMat
) {
    WIDGETRENDERER_GETUI(ui, E_FAIL);

    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);
    ui->draw();
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);

    return S_OK;
}

rendering::WidgetRenderer *rendering::WidgetRenderer::get() {
    auto *renderer = system::getRenderer();
    return (rendering::WidgetRenderer*)renderer->getSubRenderer(rendering::SubRendererFlag::SUBRENDERER_WIDGET);
}

void rendering::WidgetRenderer::keyPressEvent(Magnum::Platform::GlfwApplication::KeyEvent& event) {
    WIDGETRENDERER_GETAPP();

    if(app->isTextInputActive() && _ui->focusedInputWidget() && _ui->focusedInputWidget()->handleKeyPress(event)) 
        event.setAccepted();
}

void rendering::WidgetRenderer::mousePressEvent(Magnum::Platform::GlfwApplication::MouseEvent& event) {
    if(!_ui) return;

    if(_ui->handlePressEvent(event.position())) 
        event.setAccepted();
}

void rendering::WidgetRenderer::mouseReleaseEvent(Magnum::Platform::GlfwApplication::MouseEvent& event) {
    if(!_ui) return;

    if(_ui->handleReleaseEvent(event.position())) 
        event.setAccepted();
}

void rendering::WidgetRenderer::textInputEvent(Magnum::Platform::GlfwApplication::TextInputEvent& event) {
    WIDGETRENDERER_GETAPP();

    if(app->isTextInputActive() && _ui->focusedInputWidget() && _ui->focusedInputWidget()->handleTextInput(event)) {
        event.setAccepted();
    }
}

template <typename T> 
int WidgetRenderer_addOutputField(const T& val, const std::string& label) {
    WIDGETRENDERER_GETUI(ui, -1);
    return ui->addOutputField(val, label);
}

template <typename T> 
int WidgetRenderer_addOutputField(const T& val) {
    WIDGETRENDERER_GETUI(ui, -1);
    return ui->addOutputField(val);
}

template <typename T, typename F> 
int WidgetRenderer_addInputField(const F& cb, const T& val, const std::string& label) {
    WIDGETRENDERER_GETUI(ui, -1);
    return ui->addInputField(cb, val, label);
}

template <typename T, typename F> 
int WidgetRenderer_addInputField(const F& cb, const T& val) {
    WIDGETRENDERER_GETUI(ui, -1);
    return ui->addInputField(cb, val);
}

template <typename T> 
int WidgetRenderer_addInputField(rendering::WidgetRenderer::CallbackInput<T> cb, const T& val) {
    return WidgetRenderer_addInputField([&](const T& _v) -> void { cb(_v); }, val);
}

int rendering::WidgetRenderer::addButton(
    rendering::WidgetRenderer::FunctionOutput cb, 
    const std::string& label
) {
    WIDGETRENDERER_GETUI(ui, -1);
    return ui->addButton(cb, label);
}

int rendering::WidgetRenderer::addButton(
    rendering::WidgetRenderer::CallbackOutput cb, 
    const std::string& label
) {
    return addButton([&]() -> void { cb(); }, label);
}

template <typename T> 
HRESULT WidgetRenderer_getOutput(const unsigned int& idx, T& val) {
    WIDGETRENDERER_GETUI(ui, E_FAIL);
    return ui->getOutput(idx, val);
}

template <typename T> 
HRESULT WidgetRenderer_getInput(const unsigned int& idx, T& val) {
    WIDGETRENDERER_GETUI(ui, E_FAIL);
    return ui->getInput(idx, val);
}

template <typename T> 
HRESULT WidgetRenderer_setOutput(const unsigned int& idx, const T& val) {
    WIDGETRENDERER_GETUI(ui, E_FAIL);
    return ui->setOutput<T>(idx, val);
}

int rendering::WidgetRenderer::addOutputField(const int& val, const std::string& label) { return WidgetRenderer_addOutputField(val, label); }
int rendering::WidgetRenderer::addOutputField(const int& val) { return WidgetRenderer_addOutputField(val); }
int rendering::WidgetRenderer::addOutputField(const float& val, const std::string& label) { return WidgetRenderer_addOutputField(val, label); }
int rendering::WidgetRenderer::addOutputField(const float& val) { return WidgetRenderer_addOutputField(val); }
int rendering::WidgetRenderer::addOutputField(const double& val, const std::string& label) { return WidgetRenderer_addOutputField(val, label); }
int rendering::WidgetRenderer::addOutputField(const double& val) { return WidgetRenderer_addOutputField(val); }
int rendering::WidgetRenderer::addOutputField(const std::string& val, const std::string& label) { return WidgetRenderer_addOutputField(val, label); }
int rendering::WidgetRenderer::addOutputField(const std::string& val) { return WidgetRenderer_addOutputField(val); }

int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::FunctionInput<int> cb, const int& val, const std::string& label) { return WidgetRenderer_addInputField(cb, val, label); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::FunctionInput<int> cb, const int& val) { return WidgetRenderer_addInputField(cb, val); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::FunctionInput<float> cb, const float& val, const std::string& label) { return WidgetRenderer_addInputField(cb, val, label); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::FunctionInput<float> cb, const float& val) { return WidgetRenderer_addInputField(cb, val); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::FunctionInput<double> cb, const double& val, const std::string& label) { return WidgetRenderer_addInputField(cb, val, label); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::FunctionInput<double> cb, const double& val) { return WidgetRenderer_addInputField(cb, val); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::FunctionInput<std::string> cb, const std::string& val, const std::string& label) { return WidgetRenderer_addInputField(cb, val, label); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::FunctionInput<std::string> cb, const std::string& val) { return WidgetRenderer_addInputField(cb, val); }

int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::CallbackInput<int> cb, const int& val, const std::string& label) { return WidgetRenderer_addInputField(cb, val, label); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::CallbackInput<int> cb, const int& val) { return WidgetRenderer_addInputField(cb, val); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::CallbackInput<float> cb, const float& val, const std::string& label) { return WidgetRenderer_addInputField(cb, val, label); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::CallbackInput<float> cb, const float& val) { return WidgetRenderer_addInputField(cb, val); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::CallbackInput<double> cb, const double& val, const std::string& label) { return WidgetRenderer_addInputField(cb, val, label); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::CallbackInput<double> cb, const double& val) { return WidgetRenderer_addInputField(cb, val); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::CallbackInput<std::string> cb, const std::string& val, const std::string& label) { return WidgetRenderer_addInputField(cb, val, label); }
int rendering::WidgetRenderer::addInputField(rendering::WidgetRenderer::CallbackInput<std::string> cb, const std::string& val) { return WidgetRenderer_addInputField(cb, val); }

HRESULT rendering::WidgetRenderer::pressButton(const unsigned int& idx) {
    WIDGETRENDERER_GETUI(ui, E_FAIL);
    return ui->pressButton(idx);
}

HRESULT rendering::WidgetRenderer::getOutputInt(const unsigned int& idx, int& val) { return WidgetRenderer_getOutput(idx, val); }
HRESULT rendering::WidgetRenderer::getOutputFloat(const unsigned int& idx, float& val) { return WidgetRenderer_getOutput(idx, val); }
HRESULT rendering::WidgetRenderer::getOutputDouble(const unsigned int& idx, double& val) { return WidgetRenderer_getOutput(idx, val); }
HRESULT rendering::WidgetRenderer::getOutputString(const unsigned int& idx, std::string& val) { return WidgetRenderer_getOutput(idx, val); }

HRESULT rendering::WidgetRenderer::getInputInt(const unsigned int& idx, int& val) { return WidgetRenderer_getInput(idx, val); }
HRESULT rendering::WidgetRenderer::getInputFloat(const unsigned int& idx, float& val) { return WidgetRenderer_getInput(idx, val); }
HRESULT rendering::WidgetRenderer::getInputDouble(const unsigned int& idx, double& val) { return WidgetRenderer_getInput(idx, val); }
HRESULT rendering::WidgetRenderer::getInputString(const unsigned int& idx, std::string& val) { return WidgetRenderer_getInput(idx, val); }

HRESULT rendering::WidgetRenderer::setOutputInt(const unsigned int& idx, const int& val) { return WidgetRenderer_setOutput<int>(idx, val); }
HRESULT rendering::WidgetRenderer::setOutputFloat(const unsigned int& idx, const float& val) { return WidgetRenderer_setOutput<float>(idx, val); }
HRESULT rendering::WidgetRenderer::setOutputDouble(const unsigned int& idx, const double& val) { return WidgetRenderer_setOutput<double>(idx, val); }
HRESULT rendering::WidgetRenderer::setOutputString(const unsigned int& idx, const std::string& val) { return WidgetRenderer_setOutput<std::string>(idx, val); }

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

#include "tfSimulatorPy.h"

#include "tfBoundaryConditionsPy.h"
#include "tf_systemPy.h"

#include <tfLogger.h>
#include <tf_util.h>
#include <tfError.h>
#include <types/tf_cast.h>
#include <rendering/tfApplication.h>
#include <rendering/tfClipPlane.h>
#include <rendering/tfGlfwApplication.h>
#include <rendering/tfWindowlessApplication.h>


using namespace TissueForge;


static TissueForge::ErrorCallback *pySimErrCb = NULL;
static unsigned int pySimErrCbId = 0;


#define TF_SIMPY_CHECK(hr) \
    if(SUCCEEDED(hr)) { Py_RETURN_NONE; } \
    else {return NULL;}

#define TF_SIMULATORPY_CHECK()  if (!Simulator::get()) { return tf_error(E_INVALIDARG, "Simulator is not initialized"); }

#define TF_SIMPY_TRY() \
    try {\
        if(_Engine.flags == 0) { \
            std::string err = TF_FUNCTION; \
            err += "universe not initialized"; \
            tf_exp(std::domain_error(err.c_str())); \
        }

#define TF_SIMPY_FINALLY(retval) \
    } \
    catch(const std::exception &e) { \
        tf_exp(e); return retval; \
    }

/**
 * Make a Arguments struct from a string list,
 * Magnum has different args for different app types,
 * so this needs to be a template.
 */
template<typename T>
struct ArgumentsWrapper  {

    ArgumentsWrapper(PyObject *args) {

        for(int i = 0; i < PyList_Size(args); ++i) {
            PyObject *o = PyList_GetItem(args, i);
            strings.push_back(cast<PyObject, std::string>(o));
            cstrings.push_back(strings.back().c_str());
            if(Logger::getLevel() < LOG_INFORMATION) {
                cstrings.push_back("--magnum-log");
                cstrings.push_back("quiet");
            }

            TF_Log(LOG_INFORMATION) <<  "args: " << cstrings.back() ;;
        }

        // int reference, keep an ivar around for it to point to.
        argsIntReference = cstrings.size();
        char** constRef = const_cast<char**>(cstrings.data());

        pArgs = new T(argsIntReference, constRef);
    }

    ~ArgumentsWrapper() {
        delete pArgs;
    }

    std::vector<std::string> strings;
    std::vector<const char*> cstrings;
    T *pArgs = NULL;
    int argsIntReference;
};

static void SimulatorPy_ErrorCallback(const TissueForge::Error &err) {
    PyErr_SetString(PyExc_RuntimeError, errStr(err).c_str());
}

static HRESULT SimulatorPy_setErrorCallback() {
    if(!pySimErrCb) {
        pySimErrCb = new ErrorCallback(SimulatorPy_ErrorCallback);
        pySimErrCbId = addErrorCallback(*pySimErrCb);
    }
    return S_OK;
}

static HRESULT SimulatorPy_unsetErrorCallback() {
    if(pySimErrCb) {
        removeErrorCallback(pySimErrCbId);
        delete pySimErrCb;
        pySimErrCb = 0;
        pySimErrCbId = 0;
    }
    return S_OK;
}

HRESULT py::SimulatorPy::_throwExceptions(const bool &_throw) {
    if(_throw && !pySimErrCb) SimulatorPy_setErrorCallback();
    else if(pySimErrCb) SimulatorPy_unsetErrorCallback();
    return S_OK;
}

bool py::SimulatorPy::_throwingExceptions() {
    return pySimErrCb;
}

static void parse_kwargs(PyObject *kwargs, Simulator::Config &conf) {

    TF_Log(LOG_INFORMATION) << "parsing python dictionary input";

    PyObject *o;

    std::string loadPath;
    if((o = PyDict_GetItemString(kwargs, "load_file"))) {
        loadPath = cast<PyObject, std::string>(o);

        TF_Log(LOG_INFORMATION) << "got load file: " << loadPath;
        
        if(initSimConfigFromFile(loadPath, conf) != S_OK) 
            return;
    }

    FVector3 *dim;
    if((o = PyDict_GetItemString(kwargs, "dim"))) {
        dim = new FVector3(cast<PyObject, Magnum::Vector3>(o));

        TF_Log(LOG_INFORMATION) << "got dim: " 
                             << std::to_string(dim->x()) << "," 
                             << std::to_string(dim->y()) << "," 
                             << std::to_string(dim->z());
    }
    else dim = NULL;

    FloatP_t *cutoff;
    if((o = PyDict_GetItemString(kwargs, "cutoff"))) {
        cutoff = new FloatP_t(cast<PyObject, FloatP_t>(o));

        TF_Log(LOG_INFORMATION) << "got cutoff: " << std::to_string(*cutoff);
    }
    else cutoff = NULL;

    iVector3 *cells;
    if((o = PyDict_GetItemString(kwargs, "cells"))) {
        cells = new iVector3(cast<PyObject, iVector3>(o));

        TF_Log(LOG_INFORMATION) << "got cells: " 
                             << std::to_string(cells->x()) << "," 
                             << std::to_string(cells->y()) << "," 
                             << std::to_string(cells->z());
    }
    else cells = NULL;

    unsigned *threads;
    if((o = PyDict_GetItemString(kwargs, "threads"))) {
        threads = new unsigned(cast<PyObject, unsigned>(o));

        TF_Log(LOG_INFORMATION) << "got threads: " << std::to_string(*threads);
    }
    else threads = NULL;

    int *integrator;
    if((o = PyDict_GetItemString(kwargs, "integrator"))) {
        integrator = new int(cast<PyObject, int>(o));

        TF_Log(LOG_INFORMATION) << "got integrator: " << std::to_string(*integrator);
    }
    else integrator = NULL;

    FloatP_t *dt;
    if((o = PyDict_GetItemString(kwargs, "dt"))) {
        dt = new FloatP_t(cast<PyObject, FloatP_t>(o));

        TF_Log(LOG_INFORMATION) << "got dt: " << std::to_string(*dt);
    }
    else dt = NULL;

    py::BoundaryConditionsArgsContainerPy *bcArgs;
    if((o = PyDict_GetItemString(kwargs, "bc"))) {
        bcArgs = new py::BoundaryConditionsArgsContainerPy(o);
        
        TF_Log(LOG_INFORMATION) << "Got boundary conditions";
    }
    else bcArgs = NULL;

    FloatP_t *max_distance;
    if((o = PyDict_GetItemString(kwargs, "max_distance"))) {
        max_distance = new FloatP_t(cast<PyObject, FloatP_t>(o));

        TF_Log(LOG_INFORMATION) << "got max_distance: " << std::to_string(*max_distance);
    }
    else max_distance = NULL;

    bool *windowless;
    if((o = PyDict_GetItemString(kwargs, "windowless"))) {
        windowless = new bool(cast<PyObject, bool>(o));

        TF_Log(LOG_INFORMATION) << "got windowless " << (*windowless ? "True" : "False");
    }
    else windowless = NULL;

    iVector2 *window_size;
    if((o = PyDict_GetItemString(kwargs, "window_size"))) {
        window_size = new iVector2(cast<PyObject, Magnum::Vector2i>(o));

        TF_Log(LOG_INFORMATION) << "got window_size: " << std::to_string(window_size->x()) << "," << std::to_string(window_size->y());
    }
    else window_size = NULL;

    unsigned int *seed;
    if((o = PyDict_GetItemString(kwargs, "seed"))) {
        seed = new unsigned int(cast<PyObject, unsigned int>(o));

        TF_Log(LOG_INFORMATION) << "got seed: " << std::to_string(*seed);
    } 
    else seed = NULL;

    bool *throw_exc;
    if((o = PyDict_GetItemString(kwargs, "throw_exc"))) {
        throw_exc = new bool(cast<PyObject, bool>(o));

        TF_Log(LOG_INFORMATION) << "got throw_exc: " << std::to_string(*throw_exc);
    }
    else throw_exc = NULL;

    uint32_t *perfcounters;
    if((o = PyDict_GetItemString(kwargs, "perfcounters"))) {
        perfcounters = new uint32_t(cast<PyObject, uint32_t>(o));

        TF_Log(LOG_INFORMATION) << "got perfcounters: " << std::to_string(*perfcounters);
    }
    else perfcounters = NULL;

    int *perfcounter_period;
    if((o = PyDict_GetItemString(kwargs, "perfcounter_period"))) {
        perfcounter_period = new int(cast<PyObject, int>(o));

        TF_Log(LOG_INFORMATION) << "got perfcounter_period: " << std::to_string(*perfcounter_period);
    }
    else perfcounter_period = NULL;
    
    int *logger_level;
    if((o = PyDict_GetItemString(kwargs, "logger_level"))) {
        logger_level = new int(cast<PyObject, int>(o));

        TF_Log(LOG_INFORMATION) << "got logger_level: " << std::to_string(*logger_level);
    }
    else logger_level = NULL;
    
    std::vector<std::tuple<FVector3, FVector3> > *clip_planes;
    PyObject *pyTuple;
    PyObject *pyTupleItem;
    if((o = PyDict_GetItemString(kwargs, "clip_planes"))) {
        clip_planes = new std::vector<std::tuple<FVector3, FVector3> >();
        for(unsigned int i=0; i < PyList_Size(o); ++i) {
            pyTuple = PyList_GetItem(o, i);
            if(!pyTuple || !PyTuple_Check(pyTuple)) {
                TF_Log(LOG_ERROR) << "Clip plane input entry not a tuple";
                continue;
            }
            
            pyTupleItem = PyTuple_GetItem(pyTuple, 0);
            if(!PyList_Check(pyTupleItem)) {
                TF_Log(LOG_ERROR) << "Clip plane point entry not a list";
                continue;
            }
            FVector3 point = cast<PyObject, FVector3>(pyTupleItem);

            pyTupleItem = PyTuple_GetItem(pyTuple, 1);
            if(!PyList_Check(pyTupleItem)) {
                TF_Log(LOG_ERROR) << "Clip plane normal entry not a list";
                continue;
            }
            FVector3 normal = cast<PyObject, FVector3>(pyTupleItem);

            clip_planes->push_back(std::make_tuple(point, normal));
            TF_Log(LOG_INFORMATION) << "got clip plane: " << point << ", " << normal;
        }
    }
    else clip_planes = NULL;

    if(dim)

    if(dim) conf.universeConfig.dim = *dim;
    if(cutoff) conf.universeConfig.cutoff = *cutoff;
    if(cells) conf.universeConfig.spaceGridSize = *cells;
    if(threads) conf.universeConfig.threads = *threads;
    if(integrator) {
        int kind = *integrator;
        switch (kind) {
            case FORWARD_EULER:
            case RUNGE_KUTTA_4:
                conf.universeConfig.integrator = (EngineIntegrator)kind;
                break;
            default: {
                std::string msg = "invalid integrator kind: ";
                msg += std::to_string(kind);
                tf_exp(std::logic_error(msg));
            }
        }
    }
    if(dt) conf.universeConfig.dt = *dt;

    if(bcArgs) conf.universeConfig.setBoundaryConditions((BoundaryConditionsArgsContainer*)bcArgs);
    
    if(max_distance) conf.universeConfig.max_distance = *max_distance;
    if(windowless) conf.setWindowless(*windowless);
    if(window_size) conf.setWindowSize(*window_size);
    if(seed) conf.setSeed(*seed);
    if(throw_exc) conf.setThrowingExceptions(*throw_exc);
    if(perfcounters) conf.universeConfig.timers_mask = *perfcounters;
    if(perfcounter_period) conf.universeConfig.timer_output_period = *perfcounter_period;
    if(logger_level) Logger::setLevel(*logger_level);
    if(clip_planes) {
        std::vector<std::tuple<fVector3, fVector3> > _clip_planes;
        for(auto &v : *clip_planes) _clip_planes.push_back(v);
        for(auto &e : parsePlaneEquation(_clip_planes)) conf.clipPlanes.push_back(e);
    }
}

static PyObject *ih = NULL;

HRESULT py::_setIPythonInputHook(PyObject *_ih) {
    ih = _ih;
    return S_OK;
}

HRESULT py::_onIPythonNotReady() {
    Simulator::get()->app->mainLoopIteration(0.001);
    return S_OK;
}

static void simulator_interactive_run() {
    TF_Log(LOG_INFORMATION) <<  "entering ";

    if (Universe_Flag(Universe::Flags::POLLING_MSGLOOP)) {
        return;
    }

    // interactive run only works in terminal ipytythn.
    PyObject *ipy = py::iPython_Get();
    const char* ipyname = ipy ? ipy->ob_type->tp_name : "NULL";
    TF_Log(LOG_INFORMATION) <<  "ipy type: " << ipyname;

    if(ipy && strcmp("TerminalInteractiveShell", ipy->ob_type->tp_name) == 0) {

        TF_Log(LOG_DEBUG) << "calling python interactive loop";
        
        PyObject *tf_str = cast<std::string, PyObject*>(std::string("tissue_forge"));

        // Try to import ipython

        /**
         *        """
            Registers the Tissue Forge input hook with the ipython pt_inputhooks
            class.

            The ipython TerminalInteractiveShell.enable_gui('name') method
            looks in the registered input hooks in pt_inputhooks, and if it
            finds one, it activtes that hook.

            To acrtivate the gui mode, call:

            ip = IPython.get_ipython()
            ip.
            """
            import IPython.terminal.pt_inputhooks as pt_inputhooks
            pt_inputhooks.register("tissue_forge", inputhook)
         *
         */

        PyObject *pt_inputhooks = py::Import_ImportString("IPython.terminal.pt_inputhooks");
        
        TF_Log(LOG_INFORMATION) <<  "pt_inputhooks: " << py::str(pt_inputhooks);
        
        PyObject *reg = PyObject_GetAttrString(pt_inputhooks, "register");
        
        TF_Log(LOG_INFORMATION) <<  "reg: " << py::str(reg);
        
        TF_Log(LOG_INFORMATION) <<  "ih: " << py::str(ih);
        
        TF_Log(LOG_INFORMATION) <<  "calling reg....";
        
        PyObject *args = PyTuple_Pack(2, tf_str, ih);
        PyObject *reg_result = PyObject_Call(reg, args, NULL);
        Py_XDECREF(args);
        
        if(reg_result == NULL) {
            tf_exp(std::logic_error("error calling IPython.terminal.pt_inputhooks.register()"));
        }
        
        Py_XDECREF(reg_result);

        // import IPython
        // ip = IPython.get_ipython()
        PyObject *ipython = py::Import_ImportString("IPython");
        TF_Log(LOG_INFORMATION) <<  "ipython: " << py::str(ipython);
        
        PyObject *get_ipython = PyObject_GetAttrString(ipython, "get_ipython");
        TF_Log(LOG_INFORMATION) <<  "get_ipython: " << py::str(get_ipython);
        
        args = PyTuple_New(0);
        PyObject *ip = PyObject_Call(get_ipython, args, NULL);
        Py_XDECREF(args);
        
        if(ip == NULL) {
            tf_exp(std::logic_error("error calling IPython.get_ipython()"));
        }
        
        PyObject *enable_gui = PyObject_GetAttrString(ip, "enable_gui");
        
        if(enable_gui == NULL) {
            tf_exp(std::logic_error("error calling ipython has no enable_gui attribute"));
        }
        
        args = PyTuple_Pack(1, tf_str);
        PyObject *enable_gui_result = PyObject_Call(enable_gui, args, NULL);
        Py_XDECREF(args);
        Py_XDECREF(tf_str);
        
        if(enable_gui_result == NULL) {
            tf_exp(std::logic_error("error calling ipython.enable_gui(\"tissue_forge\")"));
        }
        
        Py_XDECREF(enable_gui_result);

        Universe_SetFlag(Universe::Flags::IPYTHON_MSGLOOP, true);

        // show the app
        Simulator::get()->app->show();
    }
    else {
        // not in ipython, so run regular run.
        Simulator::get()->run(-1);
        return;
    }

    Py_XDECREF(ipy);
    TF_Log(LOG_INFORMATION) << "leaving ";
}

py::SimulatorPy *py::SimulatorPy::get() {
    return (py::SimulatorPy*)Simulator::get();
}

PyObject *py::SimulatorPy_init(PyObject *args, PyObject *kwargs) {

    std::thread::id id = std::this_thread::get_id();
    TF_Log(LOG_INFORMATION) << "thread id: " << id;

    try {

        if(Simulator::get()) {
            tf_exp(std::domain_error( "Error, Simulator is already initialized" ));
        }
        
        Simulator *sim = new Simulator();

        #ifdef TF_WITHCUDA
        cuda::init();
        sim->makeCUDAConfigCurrent(new cuda::SimulatorConfig());
        #endif

        TF_Log(LOG_INFORMATION) << "successfully created new simulator";

        // get the argv,
        PyObject * argv = NULL;
        if(kwargs == NULL || (argv = PyDict_GetItemString(kwargs, "argv")) == NULL) {
            TF_Log(LOG_INFORMATION) << "Getting command-line args";

            PyObject *sys_name = cast<std::string, PyObject*>(std::string("sys"));
            PyObject *sys = PyImport_Import(sys_name);
            argv = PyObject_GetAttrString(sys, "argv");
            
            Py_DECREF(sys_name);
            Py_DECREF(sys);
            
            if(!argv) {
                tf_exp(std::logic_error("could not get argv from sys module"));
            }
        }

        Simulator::Config conf;
        
        if(PyList_Size(argv) > 0) {
            std::string name = cast<PyObject, std::string>(PyList_GetItem(argv, 0));
            _Universe.name = name;
            conf.setTitle(name);
        }

        TF_Log(LOG_INFORMATION) << "got universe name: " << _Universe.name;
        
        // find out if we are in jupyter, set default state of config,
        // not sure if this makes more sense in config constructor or here...
        if(py::ZMQInteractiveShell()) {
            TF_Log(LOG_INFORMATION) << "in zmq shell, setting windowless default to true";
            conf.setWindowless(true);
        }
        else {
            TF_Log(LOG_INFORMATION) << "not zmq shell, setting windowless default to false";
            conf.setWindowless(false);
        }

        if(kwargs && PyDict_Size(kwargs) > 0) {
            parse_kwargs(kwargs, conf);
        }

        TF_Log(LOG_INFORMATION) << "successfully parsed args";

        SimulatorPy::_throwExceptions(conf.throwingExceptions());
        
        if(!conf.windowless() && py::ZMQInteractiveShell()) {
            TF_Log(LOG_WARNING) << "requested window mode in Jupyter notebook, will fail badly if there is no X-server";
        }

        setSeed(const_cast<Simulator::Config&>(conf).seed());

        // init the engine first
        /* Initialize scene particles */
        universe_init(conf.universeConfig);

        TF_Log(LOG_INFORMATION) << "successfully initialized universe";

        if(conf.windowless()) {
            TF_Log(LOG_INFORMATION) <<  "creating Windowless app" ;
            
            ArgumentsWrapper<rendering::WindowlessApplication::Arguments> margs(argv);

            rendering::WindowlessApplication *windowlessApp = new rendering::WindowlessApplication(*margs.pArgs);

            if(FAILED(windowlessApp->createContext(conf))) {
                delete windowlessApp;

                tf_exp(std::domain_error("could not create windowless gl context"));
            }
            else {
                sim->app = windowlessApp;
            }

	    TF_Log(LOG_TRACE) << "sucessfully created windowless app";
        }
        else {
            TF_Log(LOG_INFORMATION) <<  "creating GLFW app" ;
            
            ArgumentsWrapper<rendering::GlfwApplication::Arguments> margs(argv);

            rendering::GlfwApplication *glfwApp = new rendering::GlfwApplication(*margs.pArgs);
            
            if(FAILED(glfwApp->createContext(conf))) {
                TF_Log(LOG_DEBUG) << "deleting failed glfwApp";
                delete glfwApp;
                tf_exp(std::domain_error("could not create  gl context"));
            }
            else {
                sim->app = glfwApp;
            }
        }

        TF_Log(LOG_INFORMATION) << "sucessfully created application";

        sim->makeCurrent();
        
        if(py::ZMQInteractiveShell()) {
            TF_Log(LOG_INFORMATION) << "in jupyter notebook, calling widget init";
            PyObject *widgetInit = py::jwidget_init(args, kwargs);
            if(!widgetInit) {
                TF_Log(LOG_ERROR) << "could not create jupyter widget";
                return NULL;
            }
            else {
                Py_DECREF(widgetInit);
            }
        }
        
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        TF_Log(LOG_CRITICAL) << "Initializing simulator failed!";

        tf_exp(e); return NULL;
    }
}

HRESULT py::SimulatorPy::irun()
{
    TF_Log(LOG_TRACE);
    
    TF_SIMULATORPY_CHECK();

    Universe_SetFlag(Universe::Flags::RUNNING, true);

    TF_Log(LOG_DEBUG) << "checking for ipython";

    bool interactive = py::terminalInteractiveShell();
    setIsTerminalInteractiveShell(interactive);

    if (interactive) {

        if (!Universe_Flag(Universe::Flags::IPYTHON_MSGLOOP)) {
            // ipython message loop, this exits right away
            simulator_interactive_run();
        }

        TF_Log(LOG_DEBUG) <<  "in ipython, calling interactive";

        Simulator::get()->app->show();
        
        TF_Log(LOG_DEBUG) << "finished";

        return S_OK;
    }
    else {
        TF_Log(LOG_DEBUG) << "not ipython, returning Simulator_Run";
        return Simulator::get()->run(-1);
    }
}

HRESULT py::SimulatorPy::_show()
{
    TF_SIMULATORPY_CHECK();
    
    TF_Log(LOG_DEBUG) << "checking for ipython";

    bool interactive = py::terminalInteractiveShell();
    setIsTerminalInteractiveShell(interactive);
    
    if (interactive) {

        if (!Universe_Flag(Universe::Flags::IPYTHON_MSGLOOP)) {
            // ipython message loop, this exits right away
            simulator_interactive_run();
        }

        TF_Log(LOG_TRACE) << "in ipython, calling interactive";

        Simulator::get()->app->show();
        
        TF_Log(LOG_INFORMATION) << ", Simulator::get()->app->show() all done" ;

        return S_OK;
    }
    else {
        TF_Log(LOG_TRACE) << "not ipython, returning Simulator::get()->app->show()";
        return Simulator::get()->show();
    }
}

void *py::SimulatorPy::wait_events(const FloatP_t &timeout) {
    TF_SIMPY_TRY();
    if(timeout < 0) {
        TF_SIMPY_CHECK(Simulator::waitEvents());
    }
    else {
        TF_SIMPY_CHECK(Simulator::waitEventsTimeout(timeout));
    }
    TF_SIMPY_FINALLY(NULL);
}

PyObject *py::SimulatorPy::_run(PyObject *args, PyObject *kwargs) {
    TF_SIMPY_TRY();

    if (py::ZMQInteractiveShell()) {
        PyObject* result = py::jwidget_run(args, kwargs);
        if (!result) {
            TF_Log(LOG_ERROR) << "failed to call tissue_forge.jwidget.run";
            return NULL;
        }

        if (result == Py_True) {
            Py_DECREF(result);
            Py_RETURN_NONE;
        }
        else if (result == Py_False) {
            TF_Log(LOG_INFORMATION) << "returned false from  tissue_forge.jwidget.run, performing normal simulation";
        }
        else {
            TF_Log(LOG_WARNING) << "unexpected result from tissue_forge.jwidget.run , performing normal simulation"; 
        }

        Py_DECREF(result);
    }


    FloatP_t et = py::arg("et", 0, args, kwargs, -1.0);
    TF_SIMPY_CHECK(Simulator::get()->run(et));
    TF_SIMPY_FINALLY(NULL);
}

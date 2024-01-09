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

#include "tfSimulator.h"
#include "rendering/tfUI.h"

#include <Magnum/Magnum.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Version.h>
#include <Magnum/Platform/Implementation/DpiScaling.h>

#include "rendering/tfApplication.h"
#include "rendering/tfUniverseRenderer.h"
#include "rendering/tfGlfwApplication.h"
#include "rendering/tfWindowlessApplication.h"
#include "rendering/tfClipPlane.h"
#include <map>
#include <sstream>
#include "tfUniverse.h"
#include "tf_system.h"
#include <tfCluster.h>
#include "tfLogger.h"
#include "tfError.h"
#include "tf_parse.h"
#include "types/tf_cast.h"
#include "types/tfMagnum.h"
#include "tf_util.h"
#include <tf_errs.h>
#include <thread>


using namespace TissueForge;


/* What to do if ENGINE_FLAGS was not defined? */
#ifndef ENGINE_FLAGS
#define ENGINE_FLAGS engine_flag_none
#endif
#ifndef CPU_TPS
#define CPU_TPS 2.67e+9
#endif


#define TF_SIM_TRY() \
    try {\
        if(_Engine.flags == 0) { \
            std::string err = TF_FUNCTION; \
            err += "universe not initialized"; \
            tf_exp(std::domain_error(err.c_str())); \
        }

#define TF_SIM_FINALLY(retval) \
    } \
    catch(const std::exception &e) { \
        tf_exp(e); return retval; \
    }

static Simulator* _Simulator = NULL;
#ifdef TF_WITHCUDA
static cuda::SimulatorConfig *_SimulatorCUDAConfig = NULL;
#endif
static bool _isTerminalInteractiveShell = false;

static TissueForge::ErrorCallback *simErrCb = NULL;
static unsigned int simErrCbId = 0;

static void simulator_interactive_run();


Simulator::Config::Config():
            _title{"Tissue Forge Application"},
            _size{800, 600},
            _dpiScalingPolicy{Simulator::DpiScalingPolicy::Default},
            queues{4},
           _windowless{ false }
{
    _windowFlags = Simulator::WindowFlags::Resizable |
                   Simulator::WindowFlags::Focused   |
                   Simulator::WindowFlags::Hidden;  // make the window initially hidden
}



Simulator::GLConfig::GLConfig():
    _colorBufferSize{8, 8, 8, 0}, 
    _depthBufferSize{24}, 
    _stencilBufferSize{0},
    _sampleCount{0}, 
    _version{TissueForge::cast<GL::Version, std::int32_t>(GL::Version::None)},
    #ifndef MAGNUM_TARGET_GLES
    _flags{Flag::ForwardCompatible},
    #else
    _flags{},
    #endif
    _srgbCapable{false} 
{}

Simulator::GLConfig::~GLConfig() = default;




#define TF_SIMULATOR_CHECK()  if (!_Simulator) { return tf_error(E_INVALIDARG, "Simulator is not initialized"); }

/**
 * Make a Arguments struct from a string list,
 * Magnum has different args for different app types,
 * so this needs to be a template.
 */
template<typename T>
struct ArgumentsWrapper  {

    ArgumentsWrapper(const std::vector<std::string> &args) {

        for(auto &a : args) {
            strings.push_back(a);
            cstrings.push_back(a.c_str());
            if(Logger::getLevel() < LOG_INFORMATION) {
                cstrings.push_back("--magnum-log");
                cstrings.push_back("quiet");
            }

            TF_Log(LOG_INFORMATION) <<  "args: " << a ;;
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

HRESULT TissueForge::initSimConfigFromFile(Simulator::Config &conf) {

    if(io::FIO::hasImport()) {
        return tf_error(E_FAIL, "Cannot load from multiple files");
    }

    if(!conf.importDataFilePath()) {
        return tf_error(E_FAIL, "No file specified");;
    }

    io::IOElement fe;
    if(io::FIO::fromFile(*conf.importDataFilePath(), fe) != S_OK) {
        return tf_error(E_FAIL, "Error loading file");
    }

    io::MetaData metaData, metaDataFile;

    auto feItr = fe.get()->children.find(io::FIO::KEY_METADATA);
    if(feItr == fe.get()->children.end() || io::fromFile(feItr->second, metaData, &metaDataFile) != S_OK) {
        return tf_error(E_FAIL, "Error loading metadata");
    }

    feItr = fe.get()->children.find(io::FIO::KEY_SIMULATOR);
    if(feItr == fe.get()->children.end() || io::fromFile(feItr->second, metaDataFile, &conf) != S_OK) {
        return tf_error(E_FAIL, "Error loading simulator");
    }

    return S_OK;
}

static void parse_kwargs(
    Simulator::Config &conf, 
    FVector3 *dim=NULL, 
    FloatP_t *cutoff=NULL, 
    TissueForge::iVector3 *cells=NULL, 
    unsigned *threads=NULL, 
    unsigned *nr_fluxsteps=NULL,
    int *integrator=NULL, 
    FloatP_t *dt=NULL, 
    int *bcValue=NULL, 
    std::unordered_map<std::string, unsigned int> *bcVals=NULL, 
    std::unordered_map<std::string, FVector3> *bcVels=NULL, 
    std::unordered_map<std::string, FloatP_t> *bcRestores=NULL, 
    BoundaryConditionsArgsContainer *bcArgs=NULL, 
    FloatP_t *max_distance=NULL, 
    bool *windowless=NULL, 
    TissueForge::iVector2 *window_size=NULL, 
    unsigned int *seed=NULL,
    bool *throw_exc=NULL, 
    uint32_t *perfcounters=NULL, 
    int *perfcounter_period=NULL, 
    int *logger_level=NULL, 
    std::vector<std::tuple<FVector3, FVector3> > *clip_planes=NULL) 
{
    if(dim) conf.universeConfig.dim = *dim;
    if(cutoff) conf.universeConfig.cutoff = *cutoff;
    if(cells) conf.universeConfig.spaceGridSize = *cells;
    if(threads) conf.universeConfig.threads = *threads;
    if(nr_fluxsteps) conf.universeConfig.nr_fluxsteps = *nr_fluxsteps;
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

    if(bcArgs) conf.universeConfig.setBoundaryConditions(bcArgs);
    else conf.universeConfig.setBoundaryConditions(new BoundaryConditionsArgsContainer(bcValue, bcVals, bcVels, bcRestores));
    
    if(max_distance) conf.universeConfig.max_distance = *max_distance;
    if(windowless) conf.setWindowless(*windowless);
    if(window_size) conf.setWindowSize(*window_size);
    if(seed) conf.setSeed(*seed);
    if(throw_exc) conf.setThrowingExceptions(*throw_exc);
    if(perfcounters) conf.universeConfig.timers_mask = *perfcounters;
    if(perfcounter_period) conf.universeConfig.timer_output_period = *perfcounter_period;
    if(logger_level) Logger::setLevel(*logger_level);
    if(clip_planes) {
        conf.clipPlanes = std::vector<FVector4>();
        std::vector<std::tuple<fVector3, fVector3> > _clip_planes;
        for(auto &v : *clip_planes) _clip_planes.push_back(v);
        for(auto &e : parsePlaneEquation(_clip_planes)) conf.clipPlanes.push_back(e);
    }
}

// intermediate kwarg parsing
static void parse_kwargs(const std::vector<std::string> &kwargs, Simulator::Config &conf) {

    TF_Log(LOG_INFORMATION) << "parsing vector string input";

    std::string s;

    if(parse::has_kwarg(kwargs, "load_file")) {
        s = parse::kwargVal(kwargs, "load_file");

        TF_Log(LOG_INFORMATION) << "got load file: " << s;

        conf.setImportDataFilePath(s);
        if(initSimConfigFromFile(conf) != S_OK) 
            return;
    }

    FVector3 *dim;
    if(parse::has_kwarg(kwargs, "dim")) {
        s = parse::kwargVal(kwargs, "dim");
        dim = new FVector3(parse::strToVec<FloatP_t>(s));

        TF_Log(LOG_INFORMATION) << "got dim: " 
                             << std::to_string(dim->x()) << "," 
                             << std::to_string(dim->y()) << "," 
                             << std::to_string(dim->z());
    }
    else dim = NULL;

    FloatP_t *cutoff;
    if(parse::has_kwarg(kwargs, "cutoff")) {
        s = parse::kwargVal(kwargs, "cutoff");
        cutoff = new FloatP_t(TissueForge::cast<std::string, FloatP_t>(s));

        TF_Log(LOG_INFORMATION) << "got cutoff: " << std::to_string(*cutoff);
    }
    else cutoff = NULL;

    TissueForge::iVector3 *cells;
    if(parse::has_kwarg(kwargs, "cells")) {
        s = parse::kwargVal(kwargs, "cells");
        cells = new TissueForge::iVector3(parse::strToVec<int>(s));

        TF_Log(LOG_INFORMATION) << "got cells: " 
                             << std::to_string(cells->x()) << "," 
                             << std::to_string(cells->y()) << "," 
                             << std::to_string(cells->z());
    }
    else cells = NULL;

    unsigned *threads;
    if(parse::has_kwarg(kwargs, "threads")) {
        s = parse::kwargVal(kwargs, "threads");
        threads = new unsigned(TissueForge::cast<std::string, unsigned>(s));

        TF_Log(LOG_INFORMATION) << "got threads: " << std::to_string(*threads);
    }
    else threads = NULL;

    unsigned *nr_fluxsteps;
    if(parse::has_kwarg(kwargs, "flux_steps")) {
        s = parse::kwargVal(kwargs, "flux_steps");
        nr_fluxsteps = new unsigned(TissueForge::cast<std::string, unsigned>(s));

        TF_Log(LOG_INFORMATION) << "got flux steps: " << std::to_string(*nr_fluxsteps);
    } 
    else nr_fluxsteps = NULL;

    int *integrator;
    if(parse::has_kwarg(kwargs, "integrator")) {
        s = parse::kwargVal(kwargs, "integrator");
        integrator = new int(TissueForge::cast<std::string, int>(s));

        TF_Log(LOG_INFORMATION) << "got integrator: " << std::to_string(*integrator);
    }
    else integrator = NULL;

    FloatP_t *dt;
    if(parse::has_kwarg(kwargs, "dt")) {
        s = parse::kwargVal(kwargs, "dt");
        dt = new FloatP_t(TissueForge::cast<std::string, FloatP_t>(s));

        TF_Log(LOG_INFORMATION) << "got dt: " << std::to_string(*dt);
    }
    else dt = NULL;

    BoundaryConditionsArgsContainer *bcArgs;
    if(parse::has_mapKwarg(kwargs, "bc")) {
        // example: 
        // bc={left={velocity={x=0;y=2};restore=1.0};bottom={type=no_slip}}
        s = parse::kwargVal(kwargs, "bc");
        std::vector<std::string> mapEntries = parse::mapStrToStrVec(parse::mapStrip(s));

        std::unordered_map<std::string, unsigned int> *bcVals = new std::unordered_map<std::string, unsigned int>();
        std::unordered_map<std::string, FVector3> *bcVels = new std::unordered_map<std::string, FVector3>();
        std::unordered_map<std::string, FloatP_t> *bcRestores = new std::unordered_map<std::string, FloatP_t>();

        std::string name;
        std::vector<std::string> val;
        for(auto &ss : mapEntries) {
            std::tie(name, val) = parse::kwarg_getNameMapVals(ss);

            if(parse::has_kwarg(val, "type")) {
                std::string ss = parse::kwargVal(val, "type");
                (*bcVals)[name] = BoundaryConditions::boundaryKindFromString(ss);

                TF_Log(LOG_INFORMATION) << "got bc type: " << name << "->" << ss;
            }
            else if(parse::has_kwarg(val, "velocity")) {
                std::string ss = parse::mapStrip(parse::kwarg_strMapVal(val, "velocity"));
                std::vector<std::string> sv = parse::mapStrToStrVec(ss);
                FloatP_t x, y, z;
                if(parse::has_kwarg(sv, "x")) x = TissueForge::cast<std::string, FloatP_t>(parse::kwargVal(sv, "x"));
                else x = 0.0;
                if(parse::has_kwarg(sv, "y")) y = TissueForge::cast<std::string, FloatP_t>(parse::kwargVal(sv, "y"));
                else y = 0.0;
                if(parse::has_kwarg(sv, "z")) z = TissueForge::cast<std::string, FloatP_t>(parse::kwargVal(sv, "z"));
                else z = 0.0;

                auto vel = FVector3(x, y, z);
                (*bcVels)[name] = vel;

                TF_Log(LOG_INFORMATION) << "got bc velocity: " << name << "->" << vel;

                if(parse::has_kwarg(val, "restore")) {
                    std::string ss = parse::kwargVal(val, "restore");
                    (*bcRestores)[name] = TissueForge::cast<std::string, FloatP_t>(ss);

                    TF_Log(LOG_INFORMATION) << "got bc restore: " << name << "->" << ss;
                }
            }
        }
        
        bcArgs = new BoundaryConditionsArgsContainer(NULL, bcVals, bcVels, bcRestores);
    }
    else if(parse::has_kwarg(kwargs, "bc")) {
        // example: 
        // bc=no_slip
        s = parse::kwargVal(kwargs, "bc");
        int *bcValue = new int(BoundaryConditions::boundaryKindFromString(s));
        bcArgs = new BoundaryConditionsArgsContainer(bcValue, NULL, NULL, NULL);

        TF_Log(LOG_INFORMATION) << "got bc val: " << std::to_string(*bcValue);
    }
    else bcArgs = NULL;

    FloatP_t *max_distance;
    if(parse::has_kwarg(kwargs, "max_distance")) {
        s = parse::kwargVal(kwargs, "max_distance");
        max_distance = new FloatP_t(TissueForge::cast<std::string, FloatP_t>(s));

        TF_Log(LOG_INFORMATION) << "got max_distance: " << std::to_string(*max_distance);
    }
    else max_distance = NULL;

    bool *windowless;
    if(parse::has_kwarg(kwargs, "windowless")) {
        s = parse::kwargVal(kwargs, "windowless");
        windowless = new bool(TissueForge::cast<std::string, bool>(s));

        TF_Log(LOG_INFORMATION) << "got windowless: " << (*windowless ? "True" : "False");
    }
    else windowless = NULL;

    TissueForge::iVector2 *window_size;
    if(parse::has_kwarg(kwargs, "window_size")) {
        s = parse::kwargVal(kwargs, "window_size");
        window_size = new TissueForge::iVector2(parse::strToVec<int>(s));

        TF_Log(LOG_INFORMATION) << "got window_size: " << std::to_string(window_size->x()) << "," << std::to_string(window_size->y());
    }
    else window_size = NULL;

    unsigned int *seed;
    if(parse::has_kwarg(kwargs, "seed")) {
        s = parse::kwargVal(kwargs, "seed");
        seed = new unsigned int(TissueForge::cast<std::string, unsigned int>(s));

        TF_Log(LOG_INFORMATION) << "got seed: " << std::to_string(*seed);
    }
    else seed = NULL;

    bool *throw_exc;
    if(parse::has_kwarg(kwargs, "throw_exc")) {
        throw_exc = new bool(TissueForge::cast<std::string, bool>(parse::kwargVal(kwargs, "throw_exc")));

        TF_Log(LOG_INFORMATION) << "got throw_exc: " << std::to_string(*throw_exc);
    }
    else throw_exc = NULL;

    uint32_t *perfcounters;
    if(parse::has_kwarg(kwargs, "perfcounters")) {
        s = parse::kwargVal(kwargs, "perfcounters");
        perfcounters = new uint32_t(TissueForge::cast<std::string, uint32_t>(s));

        TF_Log(LOG_INFORMATION) << "got perfcounters: " << std::to_string(*perfcounters);
    }
    else perfcounters = NULL;

    int *perfcounter_period;
    if(parse::has_kwarg(kwargs, "perfcounter_period")) {
        s = parse::kwargVal(kwargs, "perfcounter_period");
        perfcounter_period = new int(TissueForge::cast<std::string, int>(s));

        TF_Log(LOG_INFORMATION) << "got perfcounter_period: " << std::to_string(*perfcounter_period);
    }
    else perfcounter_period = NULL;
    
    int *logger_level;
    if(parse::has_kwarg(kwargs, "logger_level")) {
        s = parse::kwargVal(kwargs, "logger_level");
        logger_level = new int(TissueForge::cast<std::string, int>(s));

        TF_Log(LOG_INFORMATION) << "got logger_level: " << std::to_string(*logger_level);
    }
    else logger_level = NULL;
    
    std::vector<std::tuple<FVector3, FVector3> > *clip_planes;
    FVector3 point, normal;
    if(parse::has_kwarg(kwargs, "clip_planes")) {
        // ex: clip_planes=0,1,2,3,4,5;1,2,3,4,5,6
        clip_planes = new std::vector<std::tuple<FVector3, FVector3> >();

        s = parse::kwargVal(kwargs, "clip_planes");
        std::vector<std::string> sc = parse::mapStrToStrVec(s);
        for (auto &ss : sc) {
            std::vector<float> svec = parse::strToVec<float>(ss);
            point = FVector3(svec[0], svec[1], svec[2]);
            normal = FVector3(svec[3], svec[4], svec[5]);
            clip_planes->push_back(std::make_tuple(point, normal));

            TF_Log(LOG_INFORMATION) << "got clip plane: " << point << ", " << normal;
        }
    }
    else clip_planes = NULL;
    parse_kwargs(
        conf, 
        dim, 
        cutoff, 
        cells, 
        threads, 
        nr_fluxsteps, 
        integrator, 
        dt, 
        NULL, NULL, NULL, NULL, 
        bcArgs, 
        max_distance, 
        windowless, 
        window_size, 
        seed, 
        throw_exc, 
        perfcounters, 
        perfcounter_period, 
        logger_level, 
        clip_planes
    );
}

// (5) Initializer list constructor
const std::map<std::string, int> configItemMap {
    {"none", Simulator::Key::NONE},
    {"windowless", Simulator::Key::WINDOWLESS},
    {"glfw", Simulator::Key::GLFW}
};

#define TF_CLASS METH_CLASS | METH_VARARGS | METH_KEYWORDS


CAPI_FUNC(HRESULT) Simulator::pollEvents()
{
    TF_SIMULATOR_CHECK();
    return _Simulator->app->pollEvents();
}

HRESULT Simulator::waitEvents()
{
    TF_SIMULATOR_CHECK();
    return _Simulator->app->waitEvents();
}

HRESULT Simulator::waitEventsTimeout(FloatP_t timeout)
{
    TF_SIMULATOR_CHECK();
    return _Simulator->app->waitEventsTimeout((double)timeout);
}

HRESULT Simulator::postEmptyEvent()
{
    TF_SIMULATOR_CHECK();
    return _Simulator->app->postEmptyEvent();
}

HRESULT Simulator::swapInterval(int si)
{
    TF_SIMULATOR_CHECK();
    return _Simulator->app->setSwapInterval(si);
}

const int Simulator::getNumThreads() {
    TF_SIM_TRY();
    return _Engine.nr_runners;
    TF_SIM_FINALLY(0);
}

const rendering::GlfwWindow *Simulator::getWindow() {
    TF_SIM_TRY();
    return _Simulator->app->getWindow();
    TF_SIM_FINALLY(0);
}

#ifdef TF_WITHCUDA
cuda::SimulatorConfig *Simulator::getCUDAConfig() {
    return _SimulatorCUDAConfig;
}

HRESULT Simulator::makeCUDAConfigCurrent(cuda::SimulatorConfig *config) {
    if(_SimulatorCUDAConfig) {
        tf_exp(std::domain_error("Error, Simulator is already initialized" ));
        return E_FAIL;
    }
    _SimulatorCUDAConfig = config;
    return S_OK;
}
#endif

static void Simulator_ErrorCallback(const TissueForge::Error &err) {
    throw std::runtime_error(errStr(err));
}

static HRESULT Simulator_setErrorCallback() {
    if(!simErrCb) {
        simErrCb = new ErrorCallback(Simulator_ErrorCallback);
        simErrCbId = addErrorCallback(*simErrCb);
    }
    return S_OK;
}

static HRESULT Simulator_unsetErrorCallback() {
    if(simErrCb) {
        removeErrorCallback(simErrCbId);
        delete simErrCb;
        simErrCb = 0;
        simErrCbId = 0;
    }
    return S_OK;
}

HRESULT Simulator::throwExceptions(const bool &_throw) {
    if(_throw && !simErrCb) Simulator_setErrorCallback();
    else if(simErrCb) Simulator_unsetErrorCallback();

    return S_OK;
}

bool Simulator::throwingExceptions() {
    return simErrCb;
}

bool TissueForge::isTerminalInteractiveShell() {
    return _isTerminalInteractiveShell;
}

HRESULT TissueForge::setIsTerminalInteractiveShell(const bool &_interactive) {
    _isTerminalInteractiveShell = _interactive;
    return S_OK;
}

HRESULT TissueForge::modules_init() {
    TF_Log(LOG_DEBUG) << ", initializing modules... " ;

    _Particle_init();
    _Cluster_init();

    return S_OK;
}

HRESULT TissueForge::universe_init(const UniverseConfig &conf ) {

    _Universe.events = new event::EventBaseList();

    iVector3 cells = conf.spaceGridSize;

    FloatP_t cutoff = conf.cutoff;

    int nr_runners = conf.threads;

    FloatP_t _origin[] = {0.0, 0.0, 0.0};
    FloatP_t _dim[3];
    for(int i = 0; i < 3; ++i) {
        _dim[i] = conf.dim[i];
    }


    TF_Log(LOG_INFORMATION) << "main: initializing the engine... ";
    
    if ( engine_init( &_Engine , _origin , _dim , cells.data() , cutoff , conf.boundaryConditionsPtr ,
            conf.maxTypes , engine_flag_none, conf.nr_fluxsteps ) != S_OK ) 
        return tf_error(E_FAIL, errs_err_msg[MDCERR_engine]);

    _Engine.dt = conf.dt;
    _Engine.dt_flux = conf.dt / conf.nr_fluxsteps;
    _Engine.time = conf.start_step;
    _Engine.temperature = conf.temp;
    _Engine.integrator = conf.integrator;

    _Engine.timers_mask = conf.timers_mask;
    _Engine.timer_output_period = conf.timer_output_period;

    if(conf.max_distance >= 0) {
        // max_velocity is in absolute units, convert
        // to scale fraction.

        _Engine.particle_max_dist_fraction = conf.max_distance / _Engine.s.h[0];
    }

    const char* inte = NULL;

    switch(_Engine.integrator) {
    case EngineIntegrator::FORWARD_EULER:
        inte = "Forward Euler";
        break;
    case EngineIntegrator::RUNGE_KUTTA_4:
        inte = "Ruge-Kutta-4";
        break;
    }

    TF_Log(LOG_INFORMATION) << "engine integrator: " << inte;
    TF_Log(LOG_INFORMATION) << "engine: n_cells: " << _Engine.s.nr_cells << ", cell width set to " << cutoff;
    TF_Log(LOG_INFORMATION) << "engine: cell dimensions = [" << _Engine.s.cdim[0] << ", " << _Engine.s.cdim[1] << ", " << _Engine.s.cdim[2] << "]";
    TF_Log(LOG_INFORMATION) << "engine: cell size = [" << _Engine.s.h[0]  << ", " <<_Engine.s.h[1] << ", " << _Engine.s.h[2] << "]";
    TF_Log(LOG_INFORMATION) << "engine: cutoff set to " << cutoff;
    TF_Log(LOG_INFORMATION) << "engine: nr tasks: " << _Engine.s.nr_tasks;
    TF_Log(LOG_INFORMATION) << "engine: nr cell pairs: " <<_Engine.s.nr_pairs;
    TF_Log(LOG_INFORMATION) << "engine: dt: " << _Engine.dt;
    TF_Log(LOG_INFORMATION) << "engine: max distance fraction: " << _Engine.particle_max_dist_fraction;

    // start the engine

    if(engine_start( &_Engine , nr_runners , nr_runners ) != S_OK || modules_init() != S_OK) 
        return tf_error(E_FAIL, errs_err_msg[MDCERR_engine]);

    // if loading from file, populate universe if data is available
    
    if(io::FIO::hasImport()) {
        TF_Log(LOG_INFORMATION) << "Populating universe from file";

        io::MetaData metaData, metaDataFile;

        io::IOElement currentRootElement;
        io::FIO::getCurrentIORootElement(&currentRootElement);

        auto feItr = currentRootElement.get()->children.find(io::FIO::KEY_METADATA);
        if(feItr == currentRootElement.get()->children.end() || io::fromFile(feItr->second, metaData, &metaDataFile) != S_OK) 
            return tf_error(E_FAIL, errs_err_msg[MDCERR_io]);

        feItr = currentRootElement.get()->children.find(io::FIO::KEY_UNIVERSE);
        if(feItr != currentRootElement.get()->children.end()) {
            if(io::fromFile(feItr->second, metaDataFile, Universe::get()) != S_OK) 
                return tf_error(E_FAIL, errs_err_msg[MDCERR_io]);
        }
    }

    fflush(stdout);

    return 0;
}

HRESULT Simulator::run(FloatP_t et)
{
    TF_SIMULATOR_CHECK();

    TF_Log(LOG_INFORMATION) <<  "simulator run(" << et << ")" ;

    return _Simulator->app->run((double)et);
}

HRESULT TissueForge::Simulator_init(Simulator::Config &conf, const std::vector<std::string> &appArgv) {

    std::thread::id id = std::this_thread::get_id();
    TF_Log(LOG_INFORMATION) << "thread id: " << id;

    // Last check for import from file
    if(conf.importDataFilePath() && !io::FIO::hasImport()) 
        initSimConfigFromFile(conf);

    if(_Simulator) 
        return tf_error(E_FAIL, "Error, Simulator is already initialized");
    
    Simulator *sim = new Simulator();

    #ifdef TF_WITHCUDA
    cuda::init();
    _SimulatorCUDAConfig = new cuda::SimulatorConfig();
    #endif
    
    _Universe.name = conf.title();

    TF_Log(LOG_INFORMATION) << "got universe name: " << _Universe.name;

    Simulator::throwExceptions(conf.throwingExceptions());

    setSeed(const_cast<Simulator::Config&>(conf).seed());

    // init the engine first
    /* Initialize scene particles */
    if(universe_init(conf.universeConfig) != S_OK) 
        return tf_error(E_FAIL, "Error initializing the universe");

    if(conf.windowless()) {
        TF_Log(LOG_INFORMATION) <<  "creating Windowless app" ;
        
        ArgumentsWrapper<rendering::WindowlessApplication::Arguments> margs(appArgv);

        rendering::WindowlessApplication *windowlessApp = new rendering::WindowlessApplication(*margs.pArgs);

        if(FAILED(windowlessApp->createContext(conf))) {
            TF_Log(LOG_DEBUG) << "deleting failed windowless app";
            delete windowlessApp;
            return tf_error(E_FAIL, "Could not create windowless gl context");
        }
        else {
            sim->app = windowlessApp;
        }

    TF_Log(LOG_TRACE) << "sucessfully created windowless app";
    }
    else {
        TF_Log(LOG_INFORMATION) <<  "creating GLFW app" ;
        
        ArgumentsWrapper<rendering::GlfwApplication::Arguments> margs(appArgv);

        rendering::GlfwApplication *glfwApp = new rendering::GlfwApplication(*margs.pArgs);
        
        if(FAILED(glfwApp->createContext(conf))) {
            TF_Log(LOG_DEBUG) << "deleting failed glfw app";
            delete glfwApp;
            return tf_error(E_FAIL, "Could not create gl context");
        }
        else {
            sim->app = glfwApp;
        }
    }

    TF_Log(LOG_INFORMATION) << "sucessfully created application";

    sim->makeCurrent();
    
    return S_OK;
}

HRESULT TissueForge::Simulator_init(const std::vector<std::string> &argv) {

    try {

        Simulator::Config conf;
        
        if(argv.size() > 0) {
            std::string name = argv[0];
            conf.setTitle(name);
        }

        TF_Log(LOG_INFORMATION) << "got universe name: " << _Universe.name;
        
        // set default state of config
        conf.setWindowless(false);

        if(argv.size() > 1) {
            parse_kwargs(argv, conf);
        }

        TF_Log(LOG_INFORMATION) << "successfully parsed args";

        return Simulator_init(conf, argv);
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT Simulator::show()
{
    TF_SIMULATOR_CHECK();

    return _Simulator->app->show();
}

HRESULT Simulator::redraw()
{
    TF_SIMULATOR_CHECK();
    return _Simulator->app->redraw();
}

HRESULT Simulator::initConfig(const Simulator::Config &conf, const Simulator::GLConfig &glConf)
{
    if(_Simulator) {
        return tf_error(E_FAIL, "simulator already initialized");
    }

    Simulator *sim = new Simulator();

    #ifdef TF_WITHCUDA
    cuda::init();
    _SimulatorCUDAConfig = new cuda::SimulatorConfig();
    #endif

    // init the engine first
    /* Initialize scene particles */
    if(universe_init(conf.universeConfig) != S_OK) 
        return tf_error(E_FAIL, "Error initializing the universe");


    if(conf.windowless()) {

        /*



        rendering::WindowlessApplication::Configuration windowlessConf;

        rendering::WindowlessApplication *windowlessApp = new rendering::WindowlessApplication(*margs.pArgs);

        if(!windowlessApp->tryCreateContext(conf)) {
            delete windowlessApp;

            tf_exp(std::domain_error("could not create windowless gl context"));
        }
        else {
            sim->app = windowlessApp;
        }
        */
    }
    else {

        TF_Log(LOG_INFORMATION) <<  "creating GLFW app" ;;

        int argc = conf.argc;

        rendering::GlfwApplication::Arguments args{argc, conf.argv};

        rendering::GlfwApplication *glfwApp = new rendering::GlfwApplication(args);

        glfwApp->createContext(conf);

        sim->app = glfwApp;
    }

    TF_Log(LOG_INFORMATION);

    sim->makeCurrent();

    return S_OK;
}

HRESULT Simulator::close()
{
    TF_SIMULATOR_CHECK();
    return _Simulator->app->close();
}

HRESULT Simulator::destroy()
{
    TF_SIMULATOR_CHECK();
    return _Simulator->app->destroy();
}

/**
 * gets the global simulator object, returns NULL if fail.
 */
Simulator *Simulator::get() {
    if(_Simulator) {
        return _Simulator;
    }
    TF_Log(LOG_WARNING) << "Simulator is not initialized";
    return NULL;
}

HRESULT Simulator::makeCurrent() {
    if(_Simulator) {
        tf_exp(std::logic_error("Simulator is already initialized"));
        return E_FAIL;
    }
    _Simulator = this;
    return S_OK;
}


namespace TissueForge::io {

    template <>
    HRESULT toFile(const Simulator &dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "dim", FVector3::from(_Engine.s.dim));
        TF_IOTOEASY(fileElement, metaData, "cutoff", _Engine.s.cutoff);
        TF_IOTOEASY(fileElement, metaData, "cells", iVector3::from(_Engine.s.cdim));
        TF_IOTOEASY(fileElement, metaData, "integrator", (int)_Engine.integrator);
        TF_IOTOEASY(fileElement, metaData, "dt", _Engine.dt);
        TF_IOTOEASY(fileElement, metaData, "time", _Engine.time);
        TF_IOTOEASY(fileElement, metaData, "boundary_conditions", _Engine.boundary_conditions);
        TF_IOTOEASY(fileElement, metaData, "max_distance", _Engine.particle_max_dist_fraction * _Engine.s.h[0]);
        TF_IOTOEASY(fileElement, metaData, "seed", getSeed());
        TF_IOTOEASY(fileElement, metaData, "nr_fluxsteps", _Engine.nr_fluxsteps);
        
        if(dataElement.app != NULL) {
            auto renderer = dataElement.app->getRenderer();

            if(renderer != NULL && renderer->clipPlaneCount() > 0) {

                FVector4 clipPlaneEq;
                FVector3 normal, point;
                std::vector<FVector3> normals, points;

                for(unsigned int i = 0; i < renderer->clipPlaneCount(); i++) {
                    std::tie(normal, point) = planeEquation(renderer->getClipPlaneEquation(i));
                    normals.push_back(normal);
                    points.push_back(point);
                }

                TF_IOTOEASY(fileElement, metaData, "clipPlaneNormals", normals);
                TF_IOTOEASY(fileElement, metaData, "clipPlanePoints", points);

            }

        }

        fileElement.get()->type = "Simulator";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Simulator::Config *dataElement) { 

        // Do sim setup

        FVector3 dim(0.);
        TF_IOFROMEASY(fileElement, metaData, "dim", &dim);
        dataElement->universeConfig.dim = FVector3(dim);

        TF_IOFROMEASY(fileElement, metaData, "cutoff", &dataElement->universeConfig.cutoff);

        iVector3 cells(0);
        TF_IOFROMEASY(fileElement, metaData, "cells", &cells);
        dataElement->universeConfig.spaceGridSize = cells;
        
        int integrator;
        TF_IOFROMEASY(fileElement, metaData, "integrator", &integrator); 
        dataElement->universeConfig.integrator = (EngineIntegrator)integrator;
        
        TF_IOFROMEASY(fileElement, metaData, "dt", &dataElement->universeConfig.dt);
        TF_IOFROMEASY(fileElement, metaData, "time", &dataElement->universeConfig.start_step);
        
        BoundaryConditionsArgsContainer *bcArgs = new BoundaryConditionsArgsContainer();
        TF_IOFROMEASY(fileElement, metaData, "boundary_conditions", bcArgs);
        dataElement->universeConfig.setBoundaryConditions(bcArgs);
        
        TF_IOFROMEASY(fileElement, metaData, "max_distance", &dataElement->universeConfig.max_distance);

        unsigned int seed;
        TF_IOFROMEASY(fileElement, metaData, "seed", &seed);
        dataElement->setSeed(seed);

        IOChildMap fec = IOElement::children(fileElement);

        if(fec.find("nr_fluxsteps") != fec.end()) {
            unsigned int nr_fluxsteps;
            TF_IOFROMEASY(fileElement, metaData, "nr_fluxsteps", &nr_fluxsteps);
            dataElement->universeConfig.nr_fluxsteps = nr_fluxsteps;
        }
        
        if(fec.find("clipPlaneNormals") != fec.end()) {
            std::vector<fVector3> normals, points;
            std::vector<std::tuple<fVector3, fVector3> > clipPlanes;
            TF_IOFROMEASY(fileElement, metaData, "clipPlaneNormals", &normals);
            TF_IOFROMEASY(fileElement, metaData, "clipPlanePoints", &points);
            if(normals.size() > 0) {
                for(unsigned int i = 0; i < normals.size(); i++) 
                    clipPlanes.push_back(std::make_tuple(normals[i], points[i]));
                dataElement->clipPlanes = std::vector<FVector4>();
                for(auto &e : parsePlaneEquation(clipPlanes)) dataElement->clipPlanes.push_back(e);
            }
        }

        return S_OK;
    }

};

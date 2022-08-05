/*******************************************************************************
 * This file is part of mdcore.
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

/**
 * @file tfForce.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFFORCE_H_
#define _MDCORE_INCLUDE_TFFORCE_H_

#include "tf_platform.h"
#include "tf_fptype.h"
#include <types/tf_types.h>
#include <io/tf_io.h>

#include <limits>


namespace TissueForge { 


    enum FORCE_KIND {
        FORCE_ONEBODY,
        FORCE_PAIRWISE
    };

    enum FORCE_TYPE {
        FORCE_FORCE         = 0, 
        FORCE_BERENDSEN     = 1 << 0, 
        FORCE_GAUSSIAN      = 1 << 1, 
        FORCE_FRICTION      = 1 << 2, 
        FORCE_SUM           = 1 << 3, 
        FORCE_CUSTOM        = 1 << 4
    };

    /**
     * single body force function.
     */
    typedef void (*Force_EvalFcn)(struct Force*, struct Particle*, FPTYPE*);

    struct Berendsen;
    struct Gaussian;
    struct Friction;

    /**
     * @brief Force is a metatype, in that Tissue Forge has lots of 
     * different instances of force functions, that have different attributes, but
     * only have one base type. 
     * 
     * Forces are one of the fundamental processes in Tissue Forge that cause objects to move. 
     */
    struct CAPI_EXPORT Force {
        FORCE_TYPE type = FORCE_FORCE;

        Force_EvalFcn func;

        int stateVectorIndex = -1;

        /**
         * @brief Tests whether this object is a custom force type.
         * 
         * @return true if custom
         */
        virtual bool isCustom() { return false; }

        /**
         * @brief Bind a force to a species. 
         * 
         * When a force is bound to a species, the magnitude of the force is scaled by the concentration of the species. 
         * 
         * @param a_type particle type containing the species
         * @param coupling_symbol symbol of the species
         * @return HRESULT 
         */
        HRESULT bind_species(struct ParticleType *a_type, const std::string &coupling_symbol);

        /**
         * @brief Creates a Berendsen thermostat. 
         * 
         * The thermostat uses the target temperature @f$ T_0 @f$ from the object 
         * to which it is bound. 
         * The Berendsen thermostat effectively re-scales the velocities of an object in 
         * order to make the temperature of that family of objects match a specified 
         * temperature.
         * 
         * The Berendsen thermostat force has the function form: 
         * 
         * @f[
         * 
         *      \frac{\mathbf{p}}{\tau_T} \left(\frac{T_0}{T} - 1 \right),
         * 
         * @f]
         * 
         * where @f$ \mathbf{p} @f$ is the momentum, 
         * @f$ T @f$ is the measured temperature of a family of 
         * particles, @f$ T_0 @f$ is the control temperature, and 
         * @f$ \tau_T @f$ is the coupling constant. The coupling constant is a measure 
         * of the time scale on which the thermostat operates, and has units of 
         * time. Smaller values of @f$ \tau_T @f$ result in a faster acting thermostat, 
         * and larger values result in a slower acting thermostat.
         * 
         * @param tau time constant that determines how rapidly the thermostat effects the system.
         * @return Berendsen* 
         */
        static Berendsen* berendsen_tstat(const FPTYPE &tau);

        /**
         * @brief Creates a random force. 
         * 
         * A random force has a randomly selected orientation and magnitude. 
         * 
         * Orientation is selected according to a uniform distribution on the unit sphere. 
         * 
         * Magnitude is selected according to a prescribed mean and standard deviation. 
         * 
         * @param std standard deviation of magnitude
         * @param mean mean of magnitude
         * @param duration duration of force. Defaults to 0.01. 
         * @return Gaussian* 
         */
        static Gaussian* random(const FPTYPE &std, const FPTYPE &mean, const FPTYPE &duration=0.01);

        /**
         * @brief Creates a friction force. 
         * 
         * A friction force has the form: 
         * 
         * @f[
         * 
         *      - \frac{|| \mathbf{v} ||}{\tau} \mathbf{v} ,
         * 
         * @f]
         * 
         * where @f$ \mathbf{v} @f$ is the velocity of a particle and @f$ \tau @f$ is a time constant. 
         * 
         * @param coef time constant
         * @return Friction* 
         */
        static Friction* friction(const FPTYPE &coef);

        Force& operator+(const Force& rhs);

        /**
         * @brief Get a JSON string representation
         * 
         * @return std::string 
         */
        virtual std::string toString();

        /**
         * @brief Create from a JSON string representation
         * 
         * @param str 
         * @return Force* 
         */
        static Force *fromString(const std::string &str);
    };

    struct CAPI_EXPORT ForceSum : Force {
        Force *f1, *f2;

        /**
         * @brief Convert basic force to force sum. 
         * 
         * If the basic force is not a force sum, then NULL is returned. 
         * 
         * @param f 
         * @return ForceSum* 
         */
        static ForceSum *fromForce(Force *f);
    };

    CAPI_FUNC(Force*) Force_add(Force *f1, Force *f2);

    struct CustomForce;
    using UserForceFuncType = FVector3(*)(CustomForce*);

    /**
     * @brief A custom force function. 
     * 
     * The force is updated according to an update frequency.
     */
    struct CAPI_EXPORT CustomForce : Force {
        UserForceFuncType *userFunc;
        FPTYPE updateInterval;
        FPTYPE lastUpdate;
        
        FVector3 force;
        
        /**
         * notify this user force object of a simulation time step,
         *
         * this will check if interval has elapsed, and update the function.
         *
         * throws std::exception if userfunc is not a valid kind.
         */
        virtual void onTime(FPTYPE time);

        virtual FVector3 getValue();
        
        /**
         * sets the value of the force to a vector
         *
         * throws std::exception if invalid value.
         */
        void setValue(const FVector3 &f);
        
        /**
         * sets the value of the force from a user function. 
         * if a user function is passed, then it is stored as the user function of the force
         *
         * throws std::exception if invalid value.
         */
        void setValue(UserForceFuncType *_userFunc=NULL);

        FPTYPE getPeriod();
        void setPeriod(const FPTYPE &period);

        bool isCustom() { return true; }

        CustomForce();
        CustomForce(const FVector3 &f, const FPTYPE &period=std::numeric_limits<FPTYPE>::max());
        CustomForce(UserForceFuncType *f, const FPTYPE &period=std::numeric_limits<FPTYPE>::max());
        virtual ~CustomForce(){}

        /**
         * @brief Convert basic force to CustomForce. 
         * 
         * If the basic force is not a CustomForce, then NULL is returned. 
         * 
         * @param f 
         * @return CustomForce* 
         */
        static CustomForce *fromForce(Force *f);
    };

    /**
     * @brief Berendsen force. 
     * 
     * Create one with :meth:`Force.berendsen_tstat`. 
     */
    struct CAPI_EXPORT Berendsen : Force {
        /**
         * @brief time constant
         */
        FPTYPE itau;

        /**
         * @brief Convert basic force to Berendsen. 
         * 
         * If the basic force is not a Berendsen, then NULL is returned. 
         * 
         * @param f 
         * @return Berendsen* 
         */
        static Berendsen *fromForce(Force *f);
    };

    /**
     * @brief Random force. 
     * 
     * Create one with :meth:`Force.random`. 
     */
    struct CAPI_EXPORT Gaussian : Force {
        /**
         * @brief standard deviation of magnitude
         */
        FPTYPE std;

        /**
         * @brief mean of magnitude
         */
        FPTYPE mean;

        /**
         * @brief duration of force.
         */
        unsigned durration_steps;

        /**
         * @brief Convert basic force to Gaussian. 
         * 
         * If the basic force is not a Gaussian, then NULL is returned. 
         * 
         * @param f 
         * @return Gaussian* 
         */
        static Gaussian *fromForce(Force *f);
    };

    /**
     * @brief Friction force. 
     * 
     * Create one with :meth:`Force.friction`. 
     */
    struct CAPI_EXPORT Friction : Force {
        /**
         * @brief time constant
         */
        FPTYPE coef;

        /**
         * @brief Convert basic force to Friction. 
         * 
         * If the basic force is not a Friction, then NULL is returned. 
         * 
         * @param f 
         * @return Friction* 
         */
        static Friction *fromForce(Force *f);
    };


    CPPAPI_FUNC(ForceSum*) ForceSum_fromStr(const std::string &str);
    CPPAPI_FUNC(Berendsen*) Berendsen_fromStr(const std::string &str);
    CPPAPI_FUNC(Gaussian*) Gaussian_fromStr(const std::string &str);
    CPPAPI_FUNC(Friction*) Friction_fromStr(const std::string &str);

};

#endif // _MDCORE_INCLUDE_TFFORCE_H_
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

/**
 * @file tf_util.h
 * 
 */

#ifndef _SOURCE_TF_UTIL_H_
#define _SOURCE_TF_UTIL_H_

#include "TissueForge_private.h"

#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>
#include <bitset>
#include <cycle.h>
#include <string>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <random>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <limits>
#include <type_traits>


namespace TissueForge { 


    typedef std::mt19937 RandomType;
    RandomType &randomEngine();

    /**
     * @brief Get the current seed for the pseudo-random number generator
     * 
     */
    CPPAPI_FUNC(unsigned int) getSeed();

    /**
     * @brief Set the current seed for the pseudo-random number generator
     * 
     * @param _seed 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setSeed(const unsigned int *_seed=0);

    enum class PointsType : unsigned int {
        Sphere,
        SolidSphere,
        Disk,
        SolidCube,
        Cube,
        Ring
    };

    /**
     * @brief Get the names of all available colors
     * 
     * @return std::vector<std::string> 
     */
    CPPAPI_FUNC(std::vector<std::string>) color3Names();

    /**
     * @brief Get the coefficients of a plane equation for a normal vector and point
     * 
     * @param normal plane normal
     * @param point plane point
     * @return coefficients of a plane equation 
     */
    CPPAPI_FUNC(FVector4) planeEquation(const FVector3 &normal, const FVector3 &point);

    /**
     * @brief Get the plane normal and a plane point from the coefficients of a plane equation
     * 
     * @param planeEq coefficients of a plane equation
     * @return normal and point
     */
    CPPAPI_FUNC(std::tuple<FVector3, FVector3>) planeEquation(const FVector4 &planeEq);

    /**
     * @brief Get the coordinates of a random point in a kind of shape. 
     * 
     * Currently supports sphere, disk, solid cube and solid sphere. 
     * 
     * @param kind kind of shape
     * @param dr thickness parameter; only applicable to solid sphere kind
     * @param phi0 angle lower bound; only applicable to solid sphere kind
     * @param phi1 angle upper bound; only applicable to solid sphere kind
     * @return FVector3 
     */
    CPPAPI_FUNC(FVector3) randomPoint(
        const PointsType &kind, 
        const FloatP_t &dr=0, 
        const FloatP_t &phi0=0, 
        const FloatP_t &phi1=M_PI
    );

    /**
     * @brief Get the coordinates of random points in a kind of shape. 
     * 
     * Currently supports sphere, disk, solid cube and solid sphere.
     * 
     * @param kind kind of shape
     * @param n number of points
     * @param dr thickness parameter; only applicable to solid sphere kind
     * @param phi0 angle lower bound; only applicable to solid sphere kind
     * @param phi1 angle upper bound; only applicable to solid sphere kind
     * @return std::vector<FVector3> 
     */
    CPPAPI_FUNC(std::vector<FVector3>) randomPoints(
        const PointsType &kind, 
        const int &n=1, 
        const FloatP_t &dr=0, 
        const FloatP_t &phi0=0, 
        const FloatP_t &phi1=M_PI
    );

    /**
     * @brief Get the coordinates of uniform points in a kind of shape. 
     * 
     * Currently supports ring and sphere. 
     * 
     * @param kind kind of shape
     * @param n number of points
     * @return std::vector<FVector3> 
     */
    CPPAPI_FUNC(std::vector<FVector3>) points(const PointsType &kind, const unsigned int &n=1);

    /**
     * @brief Get the coordinates of a uniformly filled cube. 
     * 
     * @param corner1 first corner of cube
     * @param corner2 second corner of cube
     * @param nParticlesX number of particles along x-direction of filling axes (>=2)
     * @param nParticlesY number of particles along y-direction of filling axes (>=2)
     * @param nParticlesZ number of particles along z-direction of filling axes (>=2)
     * @return std::vector<FVector3> 
     */
    CPPAPI_FUNC(std::vector<FVector3>) filledCubeUniform(
        const FVector3 &corner1, 
        const FVector3 &corner2, 
        const unsigned int &nParticlesX=2, 
        const unsigned int &nParticlesY=2, 
        const unsigned int &nParticlesZ=2
    );

    /**
     * @brief Get the coordinates of a randomly filled cube. 
     * 
     * @param corner1 first corner of cube
     * @param corner2 second corner of cube
     * @param nParticles number of points in the cube
     * @return std::vector<FVector3> 
     */
    CPPAPI_FUNC(std::vector<FVector3>) filledCubeRandom(const FVector3 &corner1, const FVector3 &corner2, const int &nParticles);

    /**
     * @brief Get the coordinates of an icosphere. 
     * 
     * @param subdivisions number of subdivisions
     * @param phi0 angle lower bound
     * @param phi1 angle upper bound
     * @param verts returned vertices
     * @param inds returned indices
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) icosphere(
        const int subdivisions, 
        FloatP_t phi0, 
        FloatP_t phi1,
        std::vector<FVector3> &verts, 
        std::vector<int32_t> &inds
    );

    /**
     * @brief Generates a randomly oriented vector with random magnitude 
     * with given mean and standard deviation according to a normal 
     * distribution.
     * 
     * @param mean magnitude mean
     * @param std magnitude standard deviation
     * @return FVector3 
     */
    CPPAPI_FUNC(FVector3) randomVector(FloatP_t mean, FloatP_t std);

    /**
     * @brief Generates a randomly oriented unit vector.
     * 
     * @return FVector3 
     */
    CPPAPI_FUNC(FVector3) randomUnitVector();

    template<class T>
    typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp = 2)
    {
        // the machine epsilon has to be scaled to the magnitude of the values used
        // and multiplied by the desired precision in ULPs (units in the last place)
        return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
        // unless the result is subnormal
        || std::fabs(x-y) < std::numeric_limits<T>::min();
    }

    #ifdef _WIN32
    // windows
    inline void* aligned_Malloc(size_t size, size_t alignment) {
        return _aligned_malloc(size,  alignment);
    }

    inline void aligned_Free(void *mem) {
        return _aligned_free(mem);
    }

    #elif __APPLE__
    // mac
    inline void* aligned_Malloc(size_t size, size_t alignment)
    {
        enum {
            void_size = sizeof(void*)
        };
        if (!size) {
            return 0;
        }
        if (alignment < void_size) {
            alignment = void_size;
        }
        void* p;
        if (::posix_memalign(&p, alignment, size) != 0) {
            p = 0;
        }
        return p;
    }

    inline void aligned_Free(void *mem) {
        return free(mem);
    }

    #else
    // linux
    inline void* aligned_Malloc(size_t size, size_t alignment) {
        return aligned_alloc(alignment,  size);
    }

    inline void aligned_Free(void *mem) {
        return free(mem);
    }

    #endif


    namespace util {


        extern const char* color3_Names[];

        Magnum::Color3 Color3_Parse(const std::string &str);

        /**
         * modulus for negative numbers
         *
         * General mod for integer or floating point
         *
         * int mod(int x, int divisor)
         * {
         *    int m = x % divisor;
         *    return m + (m < 0 ? divisor : 0);
         * }
         */
        template<typename XType, typename DivType> XType mod(XType x, DivType divisor)
        {
            return (divisor + (x%divisor)) % divisor;
        }

        //Returns floor(a/n) (with the division done exactly).
        //Let ÷ be mathematical division, and / be C++ division.
        //We know
        //    a÷b = a/b + f (f is the remainder, not all
        //                   divisions have exact Integral results)
        //and
        //    (a/b)*b + a%b == a (from the standard).
        //Together, these imply (through algebraic manipulation):
        //    sign(f) == sign(a%b)*sign(b)
        //We want the remainder (f) to always be >=0 (by definition of flooredDivision),
        //so when sign(f) < 0, we subtract 1 from a/n to make f > 0.
        template<typename TA, typename TN>
        TA flooredDivision(TA a, TN n) {
            TA q(a/n);
            if ((a%n < 0 && n > 0) || (a%n > 0 && n < 0)) --q;
            return q;
        }

        //flooredModulo: Modulo function for use in the construction
        //looping topologies. The result will always be between 0 and the
        //denominator, and will loop in a natural fashion (rather than swapping
        //the looping direction over the zero point (as in C++11),
        //or being unspecified (as in earlier C++)).
        //Returns x such that:
        //
        //Real a = Real(numerator)
        //Real n = Real(denominator)
        //Real r = a - n*floor(n/d)
        //x = Integral(r)
        template<typename TA, typename TN>
        TA flooredModulo(TA a, TN n) {
            return a - n * flooredDivision(a, n);
        }

        template<typename TA, typename TN>
        TA loopIndex(TA index, TN range) {
            return mod(index + range, range);
        }

        /**
         * searches for the item in the container. If the item is found,
         * returns the index, otherwise returns -1.
         */
        template<typename Vec, typename Val>
        int indexOf(const Vec& vec, const Val& val) {
            int result = std::find(vec.begin(), vec.end(), val) - vec.begin();
            return result < vec.size() ? result : -1;
        }

        template<typename ContainerType, typename SizeType>
        typename ContainerType::value_type wrappedAt(ContainerType &container, SizeType index) {
            SizeType wrappedIndex = loopIndex(index, container.size());
            return container.at(wrappedIndex);
        }

        template <typename Type, typename Klass>
        inline constexpr size_t offset_of(Type Klass::*member) {
            constexpr Klass object {};
            return size_t(&(object.*member)) - size_t(&object);
        }

        enum InstructionSetFlags : std::int64_t {
            IS_3DNOW              = 1ll << 0,
            IS_3DNOWEXT           = 1ll << 1,
            IS_ABM                = 1ll << 2,
            IS_ADX                = 1ll << 3,
            IS_AES                = 1ll << 4,
            IS_AVX                = 1ll << 5,
            IS_AVX2               = 1ll << 6,
            IS_AVX512CD           = 1ll << 7,
            IS_AVX512ER           = 1ll << 8,
            IS_AVX512F            = 1ll << 9,
            IS_AVX512PF           = 1ll << 10,
            IS_BMI1               = 1ll << 11,
            IS_BMI2               = 1ll << 12,
            IS_CLFSH              = 1ll << 13,
            IS_CMPXCHG16B         = 1ll << 14,
            IS_CX8                = 1ll << 15,
            IS_ERMS               = 1ll << 16,
            IS_F16C               = 1ll << 17,
            IS_FMA                = 1ll << 18,
            IS_FSGSBASE           = 1ll << 19,
            IS_FXSR               = 1ll << 20,
            IS_HLE                = 1ll << 21,
            IS_INVPCID            = 1ll << 23,
            IS_LAHF               = 1ll << 24,
            IS_LZCNT              = 1ll << 25,
            IS_MMX                = 1ll << 26,
            IS_MMXEXT             = 1ll << 27,
            IS_MONITOR            = 1ll << 28,
            IS_MOVBE              = 1ll << 28,
            IS_MSR                = 1ll << 29,
            IS_OSXSAVE            = 1ll << 30,
            IS_PCLMULQDQ          = 1ll << 31,
            IS_POPCNT             = 1ll << 32,
            IS_PREFETCHWT1        = 1ll << 33,
            IS_RDRAND             = 1ll << 34,
            IS_RDSEED             = 1ll << 35,
            IS_RDTSCP             = 1ll << 36,
            IS_RTM                = 1ll << 37,
            IS_SEP                = 1ll << 38,
            IS_SHA                = 1ll << 39,
            IS_SSE                = 1ll << 40,
            IS_SSE2               = 1ll << 41,
            IS_SSE3               = 1ll << 42,
            IS_SSE41              = 1ll << 43,
            IS_SSE42              = 1ll << 44,
            IS_SSE4a              = 1ll << 45,
            IS_SSSE3              = 1ll << 46,
            IS_SYSCALL            = 1ll << 47,
            IS_TBM                = 1ll << 48,
            IS_XOP                = 1ll << 49,
            IS_XSAVE              = 1ll << 50,
        };

        #if defined(__x86_64__) || defined(_M_X64)

        // Yes, Windows has the __cpuid and __cpuidx macros in the #include <intrin.h>
        // header file, but it seg-faults when we try to call them from clang.
        // this version of the cpuid seems to work with clang on both Windows and mac.

        // adapted from https://github.com/01org/linux-sgx/blob/master/common/inc/internal/linux/cpuid_gnu.h
        /* This is a PIC-compliant version of CPUID */
        static inline void __tf_cpuid(int *eax, int *ebx, int *ecx, int *edx)
        {
        #if defined(__x86_64__)
            asm("cpuid"
                    : "=a" (*eax),
                    "=b" (*ebx),
                    "=c" (*ecx),
                    "=d" (*edx)
                    : "0" (*eax), "2" (*ecx));

        #else
            /*on 32bit, ebx can NOT be used as PIC code*/
            asm volatile ("xchgl %%ebx, %1; cpuid; xchgl %%ebx, %1"
                    : "=a" (*eax), "=r" (*ebx), "=c" (*ecx), "=d" (*edx)
                    : "0" (*eax), "2" (*ecx));
        #endif
        }

        #ifdef _WIN32

        // TODO: PATHETIC HACK for windows. 
        // don't know why, but calling cpuid in release mode, and ONLY in release 
        // mode causes a segfault. Hack is to flush stdout, push some junk on the stack. 
        // and force a task switch. 
        // dont know why this works, but if any of these are not here, then it segfaults
        // in release mode. 
        // this also seems to work, but force it non-inline and return random
        // number. 
        // Maybe the optimizer is inlining it, and inlining causes issues
        // calling cpuid??? 

        static __declspec(noinline) int tf_cpuid(int a[4], int b)
        {
            a[0] = b;
            a[2] = 0;
            __tf_cpuid(&a[0], &a[1], &a[2], &a[3]);
            return std::rand();
        }

        static __declspec(noinline) int tf_cpuidex(int a[4], int b, int c)
        {
            a[0] = b;
            a[2] = c;
            __tf_cpuid(&a[0], &a[1], &a[2], &a[3]);
            return std::rand();
        }

        #else 

        static  void tf_cpuid(int a[4], int b)
        {
            a[0] = b;
            a[2] = 0;
            __tf_cpuid(&a[0], &a[1], &a[2], &a[3]);
        }

        static void tf_cpuidex(int a[4], int b, int c)
        {
            a[0] = b;
            a[2] = c;
            __tf_cpuid(&a[0], &a[1], &a[2], &a[3]);
        }

        #endif

        class CAPI_EXPORT InstructionSet
        {

        private:
            
            typedef iVector4 VectorType;

            class InstructionSet_Internal
            {
            public:
                InstructionSet_Internal();

                int nIds_;
                int nExIds_;
                std::string vendor_;
                std::string brand_;
                bool isIntel_;
                bool isAMD_;
                std::bitset<32> f_1_ECX_;
                std::bitset<32> f_1_EDX_;
                std::bitset<32> f_7_EBX_;
                std::bitset<32> f_7_ECX_;
                std::bitset<32> f_81_ECX_;
                std::bitset<32> f_81_EDX_;
                std::vector<VectorType> data_;
                std::vector<VectorType> extdata_;
            };
            
            InstructionSet_Internal CPU_Rep;


        public:
            // getters
            std::string Vendor(void);
            std::string Brand(void);

            inline bool SSE3(void);
            inline bool PCLMULQDQ(void);
            inline bool MONITOR(void);
            inline bool SSSE3(void);
            inline bool FMA(void);
            inline bool CMPXCHG16B(void);
            inline bool SSE41(void);
            inline bool SSE42(void);
            inline bool MOVBE(void);
            inline bool POPCNT(void);
            inline bool AES(void);
            inline bool XSAVE(void);
            inline bool OSXSAVE(void);
            inline bool AVX(void);
            inline bool F16C(void);
            inline bool RDRAND(void);
            inline bool MSR(void);
            inline bool CX8(void);
            inline bool SEP(void);
            inline bool CMOV(void);
            inline bool CLFSH(void);
            inline bool MMX(void);
            inline bool FXSR(void);
            inline bool SSE(void);
            inline bool SSE2(void);
            inline bool FSGSBASE(void);
            inline bool BMI1(void);
            inline bool HLE(void);
            inline bool AVX2(void);
            inline bool BMI2(void);
            inline bool ERMS(void);
            inline bool INVPCID(void);
            inline bool RTM(void);
            inline bool AVX512F(void);
            inline bool RDSEED(void);
            inline bool ADX(void);
            inline bool AVX512PF(void);
            inline bool AVX512ER(void);
            inline bool AVX512CD(void);
            inline bool SHA(void);
            inline bool PREFETCHWT1(void);
            inline bool LAHF(void);
            inline bool LZCNT(void);
            inline bool ABM(void);
            inline bool SSE4a(void);
            inline bool XOP(void);
            inline bool TBM(void);
            inline bool SYSCALL(void);
            inline bool MMXEXT(void);
            inline bool RDTSCP(void);
            inline bool _3DNOWEXT(void);
            inline bool _3DNOW(void);

            std::unordered_map<std::string, bool> featuresMap;

            InstructionSet();
        };

        CPPAPI_FUNC(std::unordered_map<std::string, bool>) getFeaturesMap();

        #else // #if defined(__x86_64__) || defined(_M_X64)

        CPPAPI_FUNC(std::unordered_map<std::string, bool>) getFeaturesMap();

        #endif // #if defined(__x86_64__) || defined(_M_X64)

        class CAPI_EXPORT CompileFlags {

            std::unordered_map<std::string, unsigned int> flags;
            std::list<std::string> flagNames;

        public:

            CompileFlags();
            ~CompileFlags() {};

            const std::list<std::string> getFlags();
            const int getFlag(const std::string &_flag);

        };


        CPPAPI_FUNC(double) wallTime();

        CPPAPI_FUNC(double) CPUTime();


        class WallTime {
        public:
            WallTime();
            ~WallTime();
            double start;
        };

        class PerformanceTimer {
        public:
            PerformanceTimer(unsigned id);
            ~PerformanceTimer();
            ticks _start;
            unsigned _id;
        };

        CPPAPI_FUNC(uint64_t) nextPrime(const uint64_t &start_prime);

        /**
         * @brief Get prime numbers, beginning with a starting prime number.
         * 
         * @param start_prime Starting prime number
         * @param n Number of prime numbers to get
         * @return std::vector<uint64_t> 
         */
        CPPAPI_FUNC(std::vector<uint64_t>) findPrimes(const uint64_t &start_prime, int n);


        struct Differentiator {
            FloatP_t (*func)(FloatP_t);
            FloatP_t xmin, xmax, inc_cf = 1e-3;

            Differentiator(FloatP_t (*f)(FloatP_t), const FloatP_t &xmin, const FloatP_t &xmax, const FloatP_t &inc_cf=1e-3);

            FloatP_t fnp(const FloatP_t &x, const unsigned int &order=0);
            FloatP_t operator() (const FloatP_t &x);
        };


        /**
         * @brief Get the unique elements of a vector
         * 
         * @tparam T element type
         * @param vec vector of elements
         * @return std::vector<T> unique elements
         */
        template <typename T>
        std::vector<T> unique(const std::vector<T> &vec) {
            std::vector<T> result_vec;
            std::unordered_set<T> result_us;

            result_vec.reserve(vec.size());
            
            for(auto f : vec) {
                if(result_us.find(f) == result_us.end()) {
                    result_vec.push_back(f);
                    result_us.insert(f);
                }
            }

            result_vec.shrink_to_fit();
            return result_vec;
        }

}};

#endif // _SOURCE_TF_UTIL_H_
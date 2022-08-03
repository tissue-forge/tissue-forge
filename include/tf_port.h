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

#ifndef _INCLUDE_TF_PORT_H_
#define _INCLUDE_TF_PORT_H_

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#if (!defined(__cplusplus) || defined(_TF_ASC))
#   if !defined(TF_ASC)
#       define TF_ASC
#   endif
#else
#   if defined(TF_ASC)
#       undef TF_ASC
#   endif
#endif

#if defined(__cplusplus)
#   include <cstdio>
#   include <cstring>
#else
#   include <stdio.h>
#   include <string.h>
#endif

#include <tf_config.h>

// select support for old C++ stuff
#if (!defined(TF_ASC) && !defined(__GNUC__))
    namespace std {
        template<typename A, typename R>
        struct unary_function {
            typedef A argument_type;
            typedef R result_type;
        };
    };
#endif // (!defined(TF_ASC) && !defined(__GNUC__))


/**
 * Tissue Forge floating-point precision. 
 */
#if defined(TF_FPTYPE_SINGLE)
    typedef float tfFloatP_t;
#else
    typedef double tfFloatP_t;
#endif

#if !defined(TF_ASC)
    namespace TissueForge {
        typedef tfFloatP_t FloatP_t;
    }
#endif


#if !defined(__cplusplus)
    typedef uint8_t bool;
#   define true 1
#   define false 0
#endif


// Ensure sync with mdcore precision
#if defined(TF_FPTYPE_SINGLE)
#   if !defined(MDCORE_SINGLE)
#       define MDCORE_SINGLE
#   endif
#   if defined(MDCORE_DOUBLE)
#       undef MDCORE_DOUBLE
#   endif
#   if !defined(FPTYPE_SINGLE)
#       define FPTYPE_SINGLE
#   endif
#   if defined(FPTYPE_DOUBLE)
#       undef FPTYPE_DOUBLE
#   endif
#else
#   if defined(MDCORE_SINGLE)
#       undef MDCORE_SINGLE
#   endif
#   if !defined(MDCORE_DOUBLE)
#       define MDCORE_DOUBLE
#   endif
#   if defined(FPTYPE_SINGLE)
#       undef FPTYPE_SINGLE
#   endif
#   if !defined(FPTYPE_DOUBLE)
#       define FPTYPE_DOUBLE
#   endif
#endif


#ifndef __has_attribute         // Optional of course.
  #define __has_attribute(x) 0  // Compatibility with non-clang compilers.
#endif


/* Get the inlining right. */
#ifndef INLINE
# if __GNUC__ && !__GNUC_STDC_INLINE__
#  define TF_INLINE extern inline
# else
#  define TF_INLINE inline
# endif
#endif

#if __has_attribute(always_inline)
#define TF_ALWAYS_INLINE __attribute__((always_inline)) TF_INLINE
#else
#define TF_ALWAYS_INLINE TF_INLINE
#endif

#if defined(__CUDACC__)
  #define TF_ALIGNED(RTYPE, VAL) RTYPE __align__(VAL)
#elif __has_attribute(aligned)
  #define TF_ALIGNED(RTYPE, VAL) RTYPE __attribute__((aligned(VAL)))
#elif defined(_MSC_VER)
  #define TF_ALIGNED(RTYPE, VAL) __declspec(align(VAL)) RTYPE
#else
  #define TF_ALIGNED(RTYPE, VAL) RTYPE
#endif

/* Declarations for symbol visibility.

  CAPI_FUNC(TYPE): Declares a public Tissue Forge C API function and return type
  CPPAPI_FUNC(TYPE): Declares a public Tissue Forge C++ API function and return type
  CAPI_DATA(TYPE): Declares public Tissue Forge data and its type
  CAPI_STRUCT(TYPE): Declares opaque public Tissue Forge data types

  As a number of platforms support/require "__declspec(dllimport/dllexport)",
  we support a HAVE_DECLSPEC_DLL macro to save duplication.
*/

/*
  All windows ports, except cygwin, are handled in PC/pyconfig.h.

  Cygwin is the only other autoconf platform requiring special
  linkage handling and it uses __declspec().
*/

#if defined(__CYGWIN__)
#   define HAVE_DECLSPEC_DLL
#endif

/* Only get special linkage if built as shared */

#if defined(HAVE_DECLSPEC_DLL) 
    /* Building an extension module, or an embedded situation */
    /* public Tissue Forge functions and data are imported */
    /* Under Cygwin, auto-import functions to prevent compilation */
    /* failures similar to those described at the bottom of 4.1: */
    /* http://docs.python.org/extending/windows.html#a-cookbook-approach */
#   define CAPI_DATA(RTYPE) extern __declspec(dllimport) RTYPE
#endif // HAVE_DECLSPEC

/* If no external linkage macros defined by now, create defaults */

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   if defined(C_BUILDING_DLL)
#       define CAPI_EXPORT __declspec(dllexport)
#   else
#       define CAPI_EXPORT __declspec(dllimport)
#   endif
#else
#   define CAPI_EXPORT __attribute__((visibility("default")))
#endif

#if !defined(CAPI_FUNC)
#   if defined(__cplusplus)
#       define CAPI_FUNC(RTYPE) extern "C" CAPI_EXPORT RTYPE
#   else
#       define CAPI_FUNC(RTYPE) extern CAPI_EXPORT RTYPE
#   endif
#endif

#ifndef CPPAPI_FUNC
#   if defined(__cplusplus)
#       define CPPAPI_FUNC(...) extern CAPI_EXPORT __VA_ARGS__
#   else
#       define CPPAPI_FUNC(RTYPE) RTYPE
#   endif
#endif

#ifndef CAPI_DATA
#   if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#       define CAPI_DATA(RTYPE) extern "C" CAPI_EXPORT RTYPE
#   else
#       define CAPI_DATA(RTYPE) extern CAPI_EXPORT RTYPE 
#   endif
#endif

/** Macro for pre-defining opaque public data types */
#ifndef CAPI_STRUCT
#    if defined(__cplusplus)
#        define CAPI_STRUCT(...) struct __VA_ARGS__
#    else
#        define CAPI_STRUCT(TYPE) typedef struct TYPE TYPE
#    endif
#endif


#if !(defined(_WIN32) || defined(__WIN32__) || defined(WIN32))


/**
 * Tissue Forge return code, same as Windows.
 *
 * To test an HRESULT value, use the FAILED and SUCCEEDED macros.

 * The high-order bit in the HRESULT or SCODE indicates whether the
 * return value represents success or failure.
 * If set to 0, SEVERITY_SUCCESS, the value indicates success.
 * If set to 1, SEVERITY_ERROR, it indicates failure.
 *
 * The facility field indicates from bits 26-16 is the system
 * service responsible for the error. The FACILITY_ITF = 4 is used for most status codes
 * returned from interface methods. The actual meaning of the
 * error is defined by the interface. That is, two HRESULTs with
 * exactly the same 32-bit value returned from two different
 * interfaces might have different meanings.
 *
 * The code field from bits 15-0 is the application defined error
 * code.
 */
typedef int32_t HRESULT;

#define S_OK             0x00000000 // Operation successful
#define E_ABORT          0x80004004 // Operation aborted
#define E_ACCESSDENIED   0x80070005 // General access denied error
#define E_FAIL           0x80004005 // Unspecified failure
#define E_HANDLE         0x80070006 // Handle that is not valid
#define E_INVALIDARG     0x80070057 // One or more arguments are not valid
#define E_NOINTERFACE    0x80004002 // No such interface supported
#define E_NOTIMPL        0x80004001 // Not implemented
#define E_OUTOFMEMORY    0x8007000E // Failed to allocate necessary memory
#define E_POINTER        0x80004003 // Pointer that is not valid
#define E_UNEXPECTED     0x8000FFFF // Unexpected failure





/**
 * Provides a generic test for success on any status value.
 * Parameters:
 * hr: The status code. A non-negative number indicates success.
 * Return value: TRUE if hr represents a success status value;
 * otherwise, FALSE.
 */
#define SUCCEEDED(hr) (((HRESULT)(hr)) >= 0)

#define FAILED(hr) (((HRESULT)(hr)) < 0)

/**
 * Creates an HRESULT value from its component pieces.
 * Parameters
 * sev: The severity.
 * fac: The facility.
 * code: The code.
 * Return value: The HRESULT value.
 *
 * Note   Calling MAKE_HRESULT for S_OK verification carries a
 * performance penalty. You should not routinely use MAKE_HRESULT
 * for successful results.
 */
#define MAKE_HRESULT(sev,fac,code) \
    static_cast<HRESULT>((static_cast<uint32_t>(sev)<<31) | (static_cast<uint32_t>(fac)<<16) | (static_cast<uint32_t>(code)))

/**
 * Extracts the code portion of the specified HRESULT.
 * Parameters:
 * hr: The HRESULT value.
 * Return value: The code.
 */
#define HRESULT_CODE(hr)    ((hr) & 0xFFFF)

/**
 * Extracts the facility of the specified HRESULT.
 * Parameters:
 * hr: The HRESULT value.
 * Return value: The facility.
 */
#define HRESULT_FACILITY(hr)  (((hr) >> 16) & 0x1fff)

/**
 * Extracts the severity field of the specified HRESULT.
 * Parameters:
 * hr: The HRESULT.
 * Return value: The severity field.
 */
#define HRESULT_SEVERITY(hr)  (((hr) >> 31) & 0x1)

/**
 * print the function name
 */
#define TF_FUNCTION __PRETTY_FUNCTION__

#else
// Windows
#undef min
#undef max
#undef inline
#include <Windows.h>
#undef min
#undef max
#undef inline

#define TF_FUNCTION __func__
#endif

#if !defined(C_UNUSED)
#   define C_UNUSED(x) (void)(x);
#endif


#if defined(_WIN32)
#   define _USE_MATH_DEFINES
#   define bzero(b,len) memset((b), '\0', (len))
#else
#   define algined_free(x) free(x)
#endif

/**
 * debug verify an operation succeedes
 *
 */

#if !defined(NDEBUG)
#   define VERIFY(hr) assert(SUCCEEDED(hr))
#else
#   define VERIFY(hr) hr
#endif

/**
 * Error code faculties for Tissue Forge errors.
 */
#define FACULTY_MESH 10
#define FACULTY_MESHOPERATION 11

#define CE_ABORT                            ((HRESULT)0x80004004)
#define CE_INVALIDARG                       ((HRESULT)0x80070057)
#define CE_NOTIMPL                          ((HRESULT)0x80004001)
#define CE_OUTOFMEMORY                      ((HRESULT)0x8007000E)
#define CE_POINTER                          ((HRESULT)0x80004003)
#define CE_UNEXPECTED                       ((HRESULT)0x8000FFFF)
#define CE_FAIL                             ((HRESULT)0x80004005)
#define CE_ARGUMENT                         ((HRESULT)0x80070057)
#define CE_ARGUMENTOUTOFRANGE               ((HRESULT)0x80131502)
#define CE_TYPEMISMATCH                     ((HRESULT)0x80028ca0)





#define CE_HANDLE                           ((HRESULT)0x80070006)

#define CE_NOINTERFACE                      ((HRESULT)0x80004002)


#define CE_CLOSED                           ((HRESULT)0x80000013)
#define CE_BOUNDS                           ((HRESULT)0x8000000B)
#define CE_CHANGED_STATE                    ((HRESULT)0x8000000C)

#define CE_CLASSNOTREG                      ((HRESULT)0x80040154)
#define CE_AMBIGUOUSMATCH                   ((HRESULT)0x8000211D)
#define CE_APPDOMAINUNLOADED                ((HRESULT)0x80131014)
#define CE_APPLICATION                      ((HRESULT)0x80131600)

#define CE_ARITHMETIC                       ((HRESULT)0x80070216)
#define CE_ARRAYTYPEMISMATCH                ((HRESULT)0x80131503)
#define CE_BADIMAGEFORMAT                   ((HRESULT)0x8007000B)
#define CE_TYPEUNLOADED                     ((HRESULT)0x80131013)
#define CE_CANNOTUNLOADAPPDOMAIN            ((HRESULT)0x80131015)
#define CE_COMEMULATE                       ((HRESULT)0x80131535)
#define CE_CONTEXTMARSHAL                   ((HRESULT)0x80131504)
#define CE_DATAMISALIGNED                   ((HRESULT)0x80131541)
#define CE_TIMEOUT                          ((HRESULT)0x80131505)
#define CE_CUSTOMATTRIBUTEFORMAT            ((HRESULT)0x80131605)
#define CE_DIVIDEBYZERO                     ((HRESULT)0x80020012)
#define CE_DUPLICATEWAITOBJECT              ((HRESULT)0x80131529)
#define CE_EXCEPTION                        ((HRESULT)0x80131500)
#define CE_EXECUTIONENGINE                  ((HRESULT)0x80131506)
#define CE_FIELDACCESS                      ((HRESULT)0x80131507)
#define CE_FORMAT                           ((HRESULT)0x80131537)
#define CE_INDEXOUTOFRANGE                  ((HRESULT)0x80131508)
#define CE_INSUFFICIENTMEMORY               ((HRESULT)0x8013153D)
#define CE_INSUFFICIENTEXECUTIONSTACK       ((HRESULT)0x80131578)
#define CE_INVALIDCAST                      ((HRESULT)0x80004002)
#define CE_INVALIDCOMOBJECT                 ((HRESULT)0x80131527)
#define CE_INVALIDFILTERCRITERIA            ((HRESULT)0x80131601)
#define CE_INVALIDOLEVARIANTTYPE            ((HRESULT)0x80131531)
#define CE_INVALIDOPERATION                 ((HRESULT)0x80131509)
#define CE_INVALIDPROGRAM                   ((HRESULT)0x8013153A)
#define CE_KEYNOTFOUND                      ((HRESULT)0x80131577)
#define CE_MARSHALDIRECTIVE                 ((HRESULT)0x80131535)
#define CE_MEMBERACCESS                     ((HRESULT)0x8013151A)
#define CE_METHODACCESS                     ((HRESULT)0x80131510)
#define CE_MISSINGFIELD                     ((HRESULT)0x80131511)
#define CE_MISSINGMANIFESTRESOURCE          ((HRESULT)0x80131532)
#define CE_MISSINGMEMBER                    ((HRESULT)0x80131512)
#define CE_MISSINGMETHOD                    ((HRESULT)0x80131513)
#define CE_MISSINGSATELLITEASSEMBLY         ((HRESULT)0x80131536)
#define CE_MULTICASTNOTSUPPORTED            ((HRESULT)0x80131514)
#define CE_NOTFINITENUMBER                  ((HRESULT)0x80131528)
#define CE_PLATFORMNOTSUPPORTED             ((HRESULT)0x80131539)
#define CE_NOTSUPPORTED                     ((HRESULT)0x80131515)
#define CE_NULLREFERENCE                    ((HRESULT)0x80004003)
#define CE_OBJECTDISPOSED                   ((HRESULT)0x80131622)
#define CE_OPERATIONCANCELED                ((HRESULT)0x8013153B)
#define CE_OVERFLOW                         ((HRESULT)0x80131516)
#define CE_RANK                             ((HRESULT)0x80131517)
#define CE_REFLECTIONTYPELOAD               ((HRESULT)0x80131602)
#define CE_RUNTIMEWRAPPED                   ((HRESULT)0x8013153E)
#define CE_SAFEARRAYRANKMISMATCH            ((HRESULT)0x80131538)
#define CE_SAFEARRAYTYPEMISMATCH            ((HRESULT)0x80131533)
#define CE_SAFEHANDLEMISSINGATTRIBUTE       ((HRESULT)0x80131623)
#define CE_SECURITY                         ((HRESULT)0x8013150A)
#define CE_SERIALIZATION                    ((HRESULT)0x8013150C)
#define CE_SEMAPHOREFULL                    ((HRESULT)0x8013152B)
#define CE_WAITHANDLECANNOTBEOPENED         ((HRESULT)0x8013152C)
#define CE_ABANDONEDMUTEX                   ((HRESULT)0x8013152D)
#define CE_STACKOVERFLOW                    ((HRESULT)0x800703E9)
#define CE_SYNCHRONIZATIONLOCK              ((HRESULT)0x80131518)
#define CE_SYSTEM                           ((HRESULT)0x80131501)
#define CE_TARGET                           ((HRESULT)0x80131603)
#define CE_TARGETINVOCATION                 ((HRESULT)0x80131604)
#define CE_TARGETPARAMCOUNT                 ((HRESULT)0x8002000e)
#define CE_THREADABORTED                    ((HRESULT)0x80131530)
#define CE_THREADINTERRUPTED                ((HRESULT)0x80131519)
#define CE_THREADSTATE                      ((HRESULT)0x80131520)
#define CE_THREADSTOP                       ((HRESULT)0x80131521)
#define CE_THREADSTART                      ((HRESULT)0x80131525)
#define CE_TYPEACCESS                       ((HRESULT)0x80131543)
#define CE_TYPEINITIALIZATION               ((HRESULT)0x80131534)
#define CE_TYPELOAD                         ((HRESULT)0x80131522)
#define CE_ENTRYPOINTNOTFOUND               ((HRESULT)0x80131523)
#define CE_DLLNOTFOUND                      ((HRESULT)0x80131524)
#define CE_UNAUTHORIZEDACCESS               ((HRESULT)0x80070005)
#define CE_UNSUPPORTEDFORMAT                ((HRESULT)0x80131523)
#define CE_VERIFICATION                     ((HRESULT)0x8013150D)
#define CE_HOSTPROTECTION                   ((HRESULT)0x80131640)


#define CERR_EXCEP                          CE_FAIL
#define CERR_ARITHMETIC                     CE_ARITHMETIC
#define CERR_TYPE                           CE_TYPEMISMATCH
#define CERR_INVALIDARG                     CE_INVALIDARG
#define CERR_FAIL                           CE_FAIL

#endif // _INCLUDE_TF_PORT_H_
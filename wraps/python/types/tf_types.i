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

// Making return by reference comprehensible for basic types
%typemap(in) int* (int temp) {
    if($input == Py_None) $1 = NULL;
    else {
        temp = (int) PyInt_AsLong($input);
        $1 = &temp;
    }
}
%typemap(out) int* {
    if($1 == NULL) $result = Py_None;
    else $result = PyInt_FromLong(*$1);
}

%typemap(in) int& {
    $1 = (int) PyInt_AsLong($input);
}
%typemap(out) int& {
    $result = PyInt_FromLong(*$1);
}

%typemap(in) unsigned int* (unsigned int temp) {
    if($input == Py_None) $1 = NULL;
    else {
        temp = (unsigned int) PyInt_AsLong($input);
        $1 = &temp;
    }
}
%typemap(out) unsigned int* {
    if($1 == NULL) $result = Py_None;
    else $result = PyInt_FromLong(*$1);
}

%typemap(in) unsigned int& {
    $1 = (unsigned int) PyInt_AsLong($input);
}
%typemap(out) unsigned int& {
    $result = PyInt_FromLong(*$1);
}

%typemap(in) float* (float temp) {
    if($input == Py_None) $1 = NULL;
    else {
        temp = (float) PyFloat_AsDouble($input);
        $1 = &temp;
    }
}
%typemap(out) float* {
    if($1 == NULL) $result = Py_None;
    else $result = PyFloat_FromDouble((double) *$1);
}

%typemap(in) float& {
    $1 = (float) PyFloat_AsDouble($input);
}
%typemap(out) float& {
    $result = PyFloat_FromDouble((double) *$1);
}

%typemap(in) double* (double temp) {
    if($input == Py_None) $1 = NULL;
    else{
        temp = PyFloat_AsDouble($input);
        $1 = &temp;
    }
}
%typemap(out) double* {
    if($1 == NULL) $result = Py_None;
    else $result = PyFloat_FromDouble(*$1);
}

%typemap(in) double& {
    $1 = PyFloat_AsDouble($input);
}
%typemap(out) double& {
    $result = PyFloat_FromDouble(*$1);
}

%typemap(in) std::string* (std::string temp) {
    if($input == Py_None) $1 = NULL;
    else {
        std::string temp = std::string(PyUnicode_AsUTF8($input));
        $1 = &temp;
    }
}
%typemap(out) std::string* {
    if(!$1) $result = Py_None;
    else $result = PyUnicode_FromString($1->c_str());
}

%typemap(in) bool* (bool temp) {
    if($input == Py_None) $1 = NULL;
    else {
        if($input == Py_True) temp = true;
        else temp = false;
        $1 = &temp;
    }
}
%typemap(out) bool* {
    if(!$1) $result = Py_None;
    else $result = PyBool_FromLong(*$1);
}

%inline %{

using namespace std;

%}

%{

#include <types/tfVector.h>
#include <types/tfVector2.h>
#include <types/tfVector3.h>
#include <types/tfVector4.h>

#include <types/tfMatrix.h>
#include <types/tfMatrix3.h>
#include <types/tfMatrix4.h>

#include <types/tfQuaternion.h>

%}

%include <types/tfVector.h>
%include <types/tfVector2.h>
%include <types/tfVector3.h>
%include <types/tfVector4.h>

%include <types/tfMatrix.h>
%include <types/tfMatrix3.h>
%include <types/tfMatrix4.h>

%include <types/tfQuaternion.h>
%include <types/tf_types.h>

typedef int32_t HRESULT;
#ifdef TF_FPTYPE_SINGLE
typedef float FPTYPE;
typedef float FloatP_t;
#else
typedef double FPTYPE;
typedef double FloatP_t;
#endif

typedef TissueForge::types::TVector2<double> dVector2;
typedef TissueForge::types::TVector3<double> dVector3;
typedef TissueForge::types::TVector4<double> dVector4;

typedef TissueForge::types::TVector2<float> fVector2;
typedef TissueForge::types::TVector3<float> fVector3;
typedef TissueForge::types::TVector4<float> fVector4;

typedef TissueForge::types::TVector2<int> iVector2;
typedef TissueForge::types::TVector3<int> iVector3;
typedef TissueForge::types::TVector4<int> iVector4;

typedef TissueForge::types::TMatrix3<double> dMatrix3;
typedef TissueForge::types::TMatrix4<double> dMatrix4;

typedef TissueForge::types::TMatrix3<float> fMatrix3;
typedef TissueForge::types::TMatrix4<float> fMatrix4;

typedef TissueForge::types::TQuaternion<double> dQuaternion;
typedef TissueForge::types::TQuaternion<float> fQuaternion;

%template(lists) std::list<std::string>;
%template(pairff) std::pair<float, float>;
%template(pairdd) std::pair<double, double>;
%template(umapsb) std::unordered_map<std::string, bool>;
%template(umapss) std::unordered_map<std::string, std::string>;
%template(vectord) std::vector<double>;
%template(vectorf) std::vector<float>;
%template(vectori) std::vector<int16_t>;
%template(vectorl) std::vector<int32_t>;
%template(vectorll) std::vector<int64_t>;
%template(vectors) std::vector<std::string>;
%template(vectoru) std::vector<uint16_t>;
%template(vectorul) std::vector<uint32_t>;
%template(vectorull) std::vector<uint64_t>;

%template(vector2f) std::vector<std::vector<float>>;
%template(vector2d) std::vector<std::vector<double>>;

// Generic prep instantiations for floating-point vectors. 
// This partners with vector_template_init to implement
//  functionality exclusive to vectors with floating-point data 
//  that swig doesn't automatically pick up. 
%define vector_template_prep_float(name, dataType, wrappedName)
%template(_ ## wrappedName ## _length) name::length<dataType>;
%template(_ ## wrappedName ## _normalized) name::normalized<dataType>;
%template(_ ## wrappedName ## _resized) name::resized<dataType>;
%template(_ ## wrappedName ## _projected) name::projected<dataType>;
%template(_ ## wrappedName ## _projectedOntoNormalized) name::projectedOntoNormalized<dataType>;

%extend name<dataType> {
    dataType _length() { return $self->length(); }
    name<dataType> _normalized() { return $self->normalized(); }
    name<dataType> _resized(dataType length) { return $self->resized(length); }
    name<dataType> _projected(const name<dataType> &other) { return $self->projected(other); }
    name<dataType> _projectedOntoNormalized(const name<dataType> &other) { return $self->projectedOntoNormalized(other); }

    %pythoncode %{
        def length(self):
            """length of vector"""
            return self._length()

        def normalized(self):
            """vector normalized"""
            return self._normalized()

        def resized(self, length):
            """resize be a length"""
            return self._resize(length)

        def projected(self, other):
            """project onto another vector"""
            return self._projected(other)

        def projectedOntoNormalized(self, other):
            """project onto a normalized vector"""
            return self._projectedOntoNormalized(other)
    %}
}
%enddef

// Like vector_template_prep_float, but for TVector2
%define vector2_template_prep_float(dataType, wrappedName)
vector_template_prep_float(TissueForge::types::TVector2, dataType, wrappedName)

%rename(_ ## wrappedName ## _distance) TissueForge::types::TVector2::distance<dataType>;

%extend TissueForge::types::TVector2<dataType> {
    dataType _distance(
        const TissueForge::types::TVector2<dataType> &lineStartPt, 
        const TissueForge::types::TVector2<dataType> &lineEndPt) 
    { 
        return $self->distance(lineStartPt, lineEndPt); 
    }

    %pythoncode %{
        def distance(self, line_start_pt, line_end_pt):
            """distance from a line defined by two points"""
            return self._distance(line_start_pt, line_end_pt)

        def __reduce__(self):
            return self.__class__, (self.x(), self.y())
    %}
}

%enddef

// Like vector_template_prep_float, but for TVector3
%define vector3_template_prep_float(dataType, wrappedName)
vector_template_prep_float(TissueForge::types::TVector3, dataType, wrappedName)

%rename(_ ## wrappedName ## _distance) TissueForge::types::TVector3::distance<dataType>;
%rename(_ ## wrappedName ## _relativeTo) TissueForge::types::TVector3::relativeTo<dataType>;
%rename(_ ## wrappedName ## _xy) TissueForge::types::TVector3::xy<dataType>;

%extend TissueForge::types::TVector3<dataType> {

    dataType _distance(
        const TissueForge::types::TVector3<dataType> &lineStartPt, 
        const TissueForge::types::TVector3<dataType> &lineEndPt) 
    { 
        return $self->distance(lineStartPt, lineEndPt); 
    }
    TissueForge::types::TVector3<dataType> _relativeTo(
        const TVector3<dataType> &origin, 
        const TVector3<dataType> &dim, 
        const bool &periodic_x, 
        const bool &periodic_y, 
        const bool &periodic_z) 
    {
        return $self->relativeTo(origin, dim, periodic_x, periodic_y, periodic_z);
    }
    // Fixes allocation error under certain conditions (e.g., Universe.center.xy() returns nonsense)
    TissueForge::types::TVector2<dataType> _xy() {
        return TissueForge::types::TVector2<dataType>::from($self->data());
    }

    %pythoncode %{
        def distance(self, line_start_pt, line_end_pt):
            """distance from a line defined by two points"""
            return self._distance(line_start_pt, line_end_pt)

        def relative_to(self, origin, dim, periodic_x, periodic_y, periodic_z):
            """position relative to an origin in a space with some periodic boundary conditions"""
            return self._relativeTo(origin, dim, periodic_x, periodic_y, periodic_z)

        def xy(self):
            return self._xy()

        def __reduce__(self):
            return self.__class__, (self.x(), self.y(), self.z())
    %}
}
%enddef

// Like vector_template_prep_float, but for TVector4
%define vector4_template_prep_float(dataType, wrappedName)
vector_template_prep_float(TissueForge::types::TVector4, dataType, wrappedName)

%rename(_ ## wrappedName ## _distance) TissueForge::types::TVector4::distance<dataType>;
%rename(_ ## wrappedName ## _distanceScaled) TissueForge::types::TVector4::distanceScaled<dataType>;
%rename(_ ## wrappedName ## _planeEquation) TissueForge::types::TVector4::planeEquation<dataType>;
%rename(_ ## wrappedName ## _xyz) TissueForge::types::TVector4::xyz<dataType>;

%extend TissueForge::types::TVector4<dataType> {
    dataType _distance(const TissueForge::types::TVector3<dataType> &point) { return $self->distance(point); }
    dataType _distanceScaled(const TissueForge::types::TVector3<dataType> &point) { return $self->distanceScaled(point); }
    static TissueForge::types::TVector4<dataType> _planeEquation(
        const TissueForge::types::TVector3<dataType> &normal, 
        const TissueForge::types::TVector3<dataType> &point) 
    {
        return TissueForge::types::TVector4<dataType>::planeEquation(normal, point);
    }
    static TissueForge::types::TVector4<dataType> _planeEquation(
        const TissueForge::types::TVector3<dataType>& p0, 
        const TissueForge::types::TVector3<dataType>& p1, 
        const TissueForge::types::TVector3<dataType>& p2) 
    {
        return TissueForge::types::TVector4<dataType>::planeEquation(p0, p1, p2);
    }
    // Fixes allocation error under certain conditions
    TissueForge::types::TVector3<dataType> _xyz() {
        return TissueForge::types::TVector3<dataType>::from($self->data());
    }

    %pythoncode %{
        def distance(self, point):
            """distance from a point"""
            return self._distance(point)

        def distanceScaled(self, point):
            """scaled distance from a point"""
            return self._distanceScaled(point)

        @classmethod
        def planeEquation(cls, *args):
            """get a plane equation"""
            return cls._planeEquation(*args)

        def xyz(self):
            return self._xyz()

        def __reduce__(self):
            return self.__class__, (self.x(), self.y(), self.z(), self.w())
    %}
}
%enddef

// Do the vector template implementation
%define vector_template_init(name, dataType, wrappedName)
%ignore name<dataType>::length;
%ignore name<dataType>::normalized;
%ignore name<dataType>::resized;
%ignore name<dataType>::projected;
%ignore name<dataType>::projectedOntoNormalized;

%template(wrappedName) name<dataType>;
%enddef

// Like vector_template_init, but for TVector2
%define vector2_template_init(dataType, wrappedName)
%ignore TissueForge::types::TVector2<dataType>::distance;

vector_template_init(TissueForge::types::TVector2, dataType, wrappedName)
%enddef

// Like vector_template_init, but for TVector3
%define vector3_template_init(dataType, wrappedName)
%ignore TissueForge::types::TVector3<dataType>::distance;

vector_template_init(TissueForge::types::TVector3, dataType, wrappedName)
%enddef

// Like vector_template_init, but for TVector4
%define vector4_template_init(dataType, wrappedName)
%ignore TissueForge::types::TVector4<dataType>::distance;
%ignore TissueForge::types::TVector4<dataType>::distanceScaled;
%ignore TissueForge::types::TVector4<dataType>::planeEquation;

vector_template_init(TissueForge::types::TVector4, dataType, wrappedName)
%enddef

vector2_template_prep_float(double, dVector2)
vector2_template_prep_float(float, fVector2)
vector2_template_init(double, dVector2)
vector2_template_init(float, fVector2)
vector2_template_init(int, iVector2)

vector3_template_prep_float(double, dVector3)
vector3_template_prep_float(float, fVector3)
vector3_template_init(double, dVector3)
vector3_template_init(float, fVector3)
vector3_template_init(int, iVector3)

vector4_template_prep_float(double, dVector4)
vector4_template_prep_float(float, fVector4)
vector4_template_init(double, dVector4)
vector4_template_init(float, fVector4)
vector4_template_init(int, iVector4)

%template(dMatrix3) TissueForge::types::TMatrix3<double>;
%template(fMatrix3) TissueForge::types::TMatrix3<float>;

%template(dMatrix4) TissueForge::types::TMatrix4<double>;
%template(fMatrix4) TissueForge::types::TMatrix4<float>;

%template(dQuaternion) TissueForge::types::TQuaternion<double>;
%template(fQuaternion) TissueForge::types::TQuaternion<float>;

%define vector_list_cast_add(name, dataType, vectorName)

%template(vectorName) std::vector<name<dataType>>;
%template(vectorName ## _p) std::vector<name<dataType>*>;

%extend name<dataType>{
    %pythoncode %{
        def __getitem__(self, index: int):
            if index >= len(self):
                raise IndexError('Valid indices < ' + str(len(self)))
            return self._getitem(index)

        def __setitem__(self, index: int, val):
            if index >= len(self):
                raise IndexError('Valid indices < ' + str(len(self)))
            self._setitem(index, val)

        def as_list(self) -> list:
            """convert to a python list"""
            return list(self.asVector())

        def __str__(self) -> str:
            s = type(self).__name__
            s += ': ' + str(self.as_list())
            return s
    %}
}

%enddef

vector_list_cast_add(TissueForge::types::TVector2, double, vectordVector2)
vector_list_cast_add(TissueForge::types::TVector3, double, vectordVector3)
vector_list_cast_add(TissueForge::types::TVector4, double, vectordVector4)
vector_list_cast_add(TissueForge::types::TVector2, float, vectorfVector2)
vector_list_cast_add(TissueForge::types::TVector3, float, vectorfVector3)
vector_list_cast_add(TissueForge::types::TVector4, float, vectorfVector4)
vector_list_cast_add(TissueForge::types::TVector2, int, vectoriVector2)
vector_list_cast_add(TissueForge::types::TVector3, int, vectoriVector3)
vector_list_cast_add(TissueForge::types::TVector4, int, vectoriVector4)
vector_list_cast_add(TissueForge::types::TQuaternion, double, vectordQuaternion)
vector_list_cast_add(TissueForge::types::TQuaternion, float, vectorfQuaternion)

%define matrix_list_cast_add(name, dataType, vectorName)

%template(vectorName) std::vector<name<dataType>>;
%template(vectorName ## _p) std::vector<name<dataType>*>;

%extend name<dataType>{
    %pythoncode %{
        def __getitem__(self, index: int):
            if index >= len(self):
                raise IndexError('Valid indices < ' + str(len(self)))
            return self._getitem(index)

        def __setitem__(self, index: int, val):
            if index >= len(self):
                raise IndexError('Valid indices < ' + str(len(self)))
            self._setitem(index, val)

        def as_lists(self) -> list:
            """convert to a list of python lists"""
            return [list(v) for v in self.asVectors()]

        def __str__(self) -> str:
            s = type(self).__name__
            s += ': ' + str(self.as_lists())
            return s

        def __reduce__(self):
            return self.__class__, tuple([v for v in self.asVectors()])
    %}
}

%enddef

matrix_list_cast_add(TissueForge::types::TMatrix3, double,   vectordMatrix3)
matrix_list_cast_add(TissueForge::types::TMatrix4, double,   vectordMatrix4)
matrix_list_cast_add(TissueForge::types::TMatrix3, float,    vectorfMatrix3)
matrix_list_cast_add(TissueForge::types::TMatrix4, float,    vectorfMatrix4)

%template(pairfVecMat3) std::pair<TissueForge::types::TVector3<float>,  TissueForge::types::TMatrix3<float> >;
%template(pairfVecMat4) std::pair<TissueForge::types::TVector4<float>,  TissueForge::types::TMatrix4<float> >;
%template(pairdVecMat3) std::pair<TissueForge::types::TVector3<double>, TissueForge::types::TMatrix3<double> >;
%template(pairdVecMat4) std::pair<TissueForge::types::TVector4<double>, TissueForge::types::TMatrix4<double> >;


#ifdef TF_FPTYPE_SINGLE
%pythoncode %{

FVector2 = fVector2
FVector3 = fVector3
FVector4 = fVector4
FMatrix3 = fMatrix3
FMatrix4 = fMatrix4
FQuaternion = fQuaternion

vectorFVector2 = vectorfVector2
vectorFVector3 = vectorfVector3
vectorFVector4 = vectorfVector4
vectorFMatrix3 = vectorfMatrix3
vectorFMatrix4 = vectorfMatrix4

pairFF = pairff
vectorF = vectorf
vector2F = vector2f

pairFVecMat3 = pairfVecMat3
pairFVecMat4 = pairfVecMat4

%}
#else
%pythoncode %{

FVector2 = dVector2
FVector3 = dVector3
FVector4 = dVector4
FMatrix3 = dMatrix3
FMatrix4 = dMatrix4
FQuaternion = dQuaternion

vectorFVector2 = vectordVector2
vectorFVector3 = vectordVector3
vectorFVector4 = vectordVector4
vectorFMatrix3 = vectordMatrix3
vectorFMatrix4 = vectordMatrix4

pairFF = pairdd
vectorF = vectord
vector2F = vector2d

pairFVecMat3 = pairdVecMat3
pairFVecMat4 = pairdVecMat4

%}
#endif

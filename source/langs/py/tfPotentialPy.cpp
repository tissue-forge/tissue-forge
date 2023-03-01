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

#include "tfPotentialPy.h"


using namespace TissueForge;


static FloatP_t pyEval(PyObject *f, FloatP_t r) {
	PyObject *py_r = cast<FloatP_t, PyObject*>(r);
	PyObject *args = PyTuple_Pack(1, py_r);
	PyObject *py_result = PyObject_CallObject(f, args);
	Py_XDECREF(args);

	if (py_result == NULL) {
		PyObject *err = PyErr_Occurred();
		PyErr_Clear();
		return 0.0;
	}

	FloatP_t result = cast<PyObject, FloatP_t>(py_result);
	Py_DECREF(py_result);
	return result;
}

static PyObject *pyCustom_f, *pyCustom_fp, *pyCustom_f6p;

static FloatP_t pyEval_f(FloatP_t r) {
	return pyEval(pyCustom_f, r);
}

static FloatP_t pyEval_fp(FloatP_t r) {
	return pyEval(pyCustom_fp, r);
}

static FloatP_t pyEval_f6p(FloatP_t r) {
	return pyEval(pyCustom_f6p, r);
}

Potential *py::PotentialPy::customPy(FloatP_t min, FloatP_t max, PyObject *f, PyObject *fp, PyObject *f6p, FloatP_t *tol, uint32_t *flags) {
	pyCustom_f = f;
	FloatP_t (*eval_fp)(FloatP_t) = NULL;
	FloatP_t (*eval_f6p)(FloatP_t) = NULL;

	if (fp != Py_None) {
		pyCustom_fp = fp;
		eval_fp = &pyEval_fp;
	}
	
	if (f6p != Py_None) {
		pyCustom_f6p = f6p;
		eval_f6p = &pyEval_f6p;
	}

	auto p = Potential::custom(min, max, &pyEval_f, eval_fp, eval_f6p, tol, flags);

	pyCustom_f = NULL;
	pyCustom_fp = NULL;
	pyCustom_f6p = NULL;

	return p;
}

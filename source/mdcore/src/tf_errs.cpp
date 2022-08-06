/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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

/* System-wide includes. */
#include <stdlib.h>
#include <cstdio>


/* Local includes. */
#include <tf_errs.h>


using namespace TissueForge;


/* Global values. */
int TissueForge::errs_err = errs_err_ok;
const char *TissueForge::errs_err_msg[] = {
    "All is well.",
    "An IO-error has occurred." };

/* The error stack. */
struct {
    int id, line;
    const char *msg, *func, *file;
} errs_stack[ errs_maxstack ];
int errs_count = 0;


/**
 * @brief Re-set the error stack.
 */
void TissueForge::errs_clear() {

    errs_count = 0;
}


/**
 * @brief Print the error stack out to the given FILE pointer.
 *
 * @param out A pointer to a FILE structure.
 * 
 * @return #errs_err_ok or < 0 on failure.
 */
 
int TissueForge::errs_dump(FILE *out) {

    int k;
    
    /* Loop over the error stack, bottom-up. */
    for(k = 0 ; k < errs_count ; k++)
        fprintf(out, "%s:%s:%i: %s (%i)\n",
            errs_stack[k].file, errs_stack[k].func, errs_stack[k].line, errs_stack[k].msg, errs_stack[k].id);
            
    /* Clean up the stack. */
    errs_clear();
    
    /* End on a good note. */
    return errs_err_ok;

}

std::string TissueForge::errs_getstring(int i) {
    char buffer[256];
    std::string result;
    
    
    int k;
    
    /* Loop over the error stack, bottom-up. */
    for(k = 0 ; k < errs_count ; k++) {
        snprintf(buffer, 255*sizeof(char), "%s, id:%i, file:%s, func:%s, line:%i",
                 errs_stack[k].msg, errs_stack[k].id, errs_stack[k].file, errs_stack[k].func, errs_stack[k].line);
        result = result + buffer;
        if(k+2 < errs_count) {
            result += "\n";
        }
    }
    
    /* Clean up the stack. */
    errs_clear();

    return result;
}


/**
 * @brief Dump an error onto the stack.
 *
 * @param id An error identifier that will be returned.
 * @param msg A pointer to a string containing a descriptive error message.
 * @param line The line on which the error occured.
 * @param func The name of the function in which the error occured.
 * @param file The name of the file in which the error occured.
 * 
 * @return The value of @c id.
 */
 
int TissueForge::errs_register(int id, const char *msg, int line, const char *func, const char *file) {

    /* Is there any room left on the stack? */
    if(errs_count < errs_maxstack) {
    
        /* Register the error. */
        errs_stack[errs_count].id = id;
        errs_stack[errs_count].msg = msg;
        errs_stack[errs_count].line = line;
        errs_stack[errs_count].func = func;
        errs_stack[errs_count].file = file;
        
        /* Increase the counter. */
        errs_count += 1;
    
        }

    /* Return with the given error code. */
    return id;

}

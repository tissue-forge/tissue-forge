/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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

/* Include configuration header */
#include <mdcore_config.h>

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <limits.h>

/* Include some conditional headers. */
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include <tf_cycle.h>
#include <tf_errs.h>
#include <tf_fptype.h>
#include <tf_lock.h>
#include <tfParticle.h>
#include <tfSpace_cell.h>
#include <tfTask.h>
#include <tfSpace.h>
#include <tfPotential.h>
#include <tfEngine.h>
#include <tfQueue.h>
#include <tfError.h>

#if !defined(_MSC_VER)
#pragma clang diagnostic ignored "-Wwritable-strings"
#endif


using namespace TissueForge;


/* Global variables. */

/* the error macro. */
#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))


struct task *TissueForge::queue_get(struct queue *q, int rid, int keep) {

    int j, k, tid = -1, ind_best = -1, score, score_best = -1, hit = 0;
    struct task *t;
    struct space *s = q->space;
    char *cells_taboo = s->cells_taboo, *cells_owner = s->cells_owner;

    /* Check if the queue is empty first. */
    if(q->next >= q->count)
        return NULL;

    /* Lock the queue. */
    if(lock_lock(&q->lock) != 0) {
        error(MDCERR_lock);
        return NULL;
    }
        
    /* Loop over the entries. */
    for(k = q->next ; k < q->count && hit < queue_maxhit ; k++) {
    
        /* Increase the hit counter if we've got a potential solution. */
        if(ind_best >= 0)
            hit += 1;
    
        /* Put a finger on the kth pair. */
        t = &q->tasks[ q->ind[k] ];

        /* Is this task ready yet? */
        if(t->wait)
            continue;

        /* Get this pair's score. */
        if(t->type == task_type_sort || t->type == task_type_self)
            score = 2 *(cells_owner[ t->i ] == rid);
        else if(t->type == task_type_pair)
            score =(cells_owner[ t->i ] == rid) +(cells_owner[ t->j ] == rid);
        else
            score = 0;

        /* Is this better than what we've seen so far? */
        if(score <= score_best)
            continue;

        /* Is this pair ok? */
        if(t->type == task_type_pair) {
            if(rid & 1) {
                if(sync_val_compare_and_swap(&cells_taboo[ t->i ], 0, 1) == 0) {
                    if(sync_val_compare_and_swap(&cells_taboo[ t->j ], 0, 1) == 0) {
                        if(ind_best >= 0) {
                            t = &q->tasks[ q->ind[ ind_best ] ];
                            if(t->type == task_type_self || t->type == task_type_sort)
                                cells_taboo[ t->i ] = 0;
                            else if(t->type == task_type_pair) {
                                cells_taboo[ t->i ] = 0;
                                cells_taboo[ t->j ] = 0;
                            }
                        }
                        score_best = score;
                        ind_best = k;
                    }
                    else
                        cells_taboo[ t->i ] = 0;
                }
            }
            else {
                if(sync_val_compare_and_swap(&cells_taboo[ t->j ], 0, 1) == 0) {
                    if(sync_val_compare_and_swap(&cells_taboo[ t->i ], 0, 1) == 0) {
                        if(ind_best >= 0) {
                            t = &q->tasks[ q->ind[ ind_best ] ];
                            if(t->type == task_type_self || t->type == task_type_sort)
                                cells_taboo[ t->i ] = 0;
                            else if(t->type == task_type_pair) {
                                cells_taboo[ t->i ] = 0;
                                cells_taboo[ t->j ] = 0;
                            }
                        }
                        score_best = score;
                        ind_best = k;
                    }
                    else
                        cells_taboo[ t->j ] = 0;
                }
            }
        }
        else if(t->type == task_type_sort || t->type == task_type_self) {
            if(sync_val_compare_and_swap(&cells_taboo[ t->i ], 0, 1) == 0) {
                if(ind_best >= 0) {
                    t = &q->tasks[ q->ind[ ind_best ] ];
                    if(t->type == task_type_self || t->type == task_type_sort)
                        cells_taboo[ t->i ] = 0;
                    else if(t->type == task_type_pair) {
                        cells_taboo[ t->i ] = 0;
                        cells_taboo[ t->j ] = 0;
                        }
                    }
                score_best = score;
                ind_best = k;
            }
        }
            
        /* If we have the maximum score, break. */
        if(score_best == 2)
            break;
    
    } /* loop over the entries. */
        
    /* Did we get an entry? */
    if(ind_best >= 0) {
    
        /* Keep an eye on this index. */
        tid = q->ind[ ind_best ];
        
        /* Own this task's cells. */
        t = &q->tasks[ tid ];
        if(t->type == task_type_sort ||
             t->type == task_type_self)
            cells_owner[ t->i ] = rid;
        else if(t->type == task_type_pair) {
            cells_owner[ t->i ] = rid;
            cells_owner[ t->j ] = rid;
        }
    
        /* Remove this entry from the queue? */
        if(keep) {
        
            /* Shuffle all the indices down. */
            q->count -= 1;
            for(j = ind_best ; j < q->count ; j++)
                q->ind[j] = q->ind[j+1];
        
        }
            
        /* Otherwise, just shuffle it to the front. */
        else {
        
            /* Bubble down... */
            for(k = ind_best ; k > q->next ; k--)
                q->ind[k] = q->ind[k-1];
                
            /* Write the original index back to the list. */
            q->ind[ q->next ] = tid;
            
            /* Move the next pointer up a notch. */
            q->next += 1;
        
        }
    
    } /* did we get an entry? */

    /* Unlock the queue. */
    if(lock_unlock(&q->lock) != 0) {
        error(MDCERR_lock);
        return NULL;
    }
        
    /* Return whatever we've got. */
    if(tid == -1)
        return NULL;
    else {
        return &q->tasks[ tid ];
    }
        
}

void TissueForge::queue_reset(struct queue *q) {

    /* Set the next index to the start of the queue. */
    q->next = 0;

}

int TissueForge::queue_insert(struct queue *q, struct task *t) {

    int k;

    /* Should we even try? */
    if(q->count == q->size)
        return 0;
        
    /* Lock the queue. */
    if(lock_lock(&q->lock) != 0)
        return error(MDCERR_lock);
        
    /* Is there space left? */
    if(q->count == q->size) {
        if(lock_unlock(&q->lock) != 0)
            return error(MDCERR_lock);
        return 0;
    }
        
    /* Add the new index to the end of the queue. */
    for(k = q->count ; k > q->next ; k--)
        q->ind[ k ] = q->ind[ k-1 ];
    q->ind[ q->next ] = t - q->tasks;
    q->count += 1;
    q->next += 1;
        
    /* Unlock the queue. */
    if(lock_unlock(&q->lock) != 0)
        return error(MDCERR_lock);
        
    /* No news is good news. */
    return 1;

}

HRESULT TissueForge::queue_init(struct queue *q, int size, struct space *s, struct task *tasks) {

    /* Sanity check. */
    if(q == NULL || s == NULL || tasks == NULL)
        return error(MDCERR_null);
        
    /* Allocate the indices. */
    if((q->ind = (int*)malloc(sizeof(int) * size)) == NULL)
        return error(MDCERR_malloc);
        
    /* Init the queue data. */
    q->space = s;
    q->size = size;
    q->next = 0;
    q->count = 0;
    q->tasks = tasks;

    /* Init the lock. */
    if(lock_init(&q->lock) != 0)
        return error(MDCERR_lock);

    /* Nothing to see here. */
    return S_OK;

}

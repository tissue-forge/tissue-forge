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


/* include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <string.h>

/* Include conditional headers. */
#include <mdcore_config.h>
#ifdef WITH_MPI
    #include <mpi.h>
#endif
#ifdef HAVE_OPENMP
    #include <omp.h>
#endif

/* include local headers */
#include <cycle.h>
#include <tf_errs.h>
#include <tf_fptype.h>
#include <tf_lock.h>
#include <tfParticle.h>
#include <tfSpace_cell.h>
#include <tfSpace.h>
#include <tfPotential.h>
#include <tfRunner.h>
#include <tfBond.h>
#include <tfRigid.h>
#include <tfAngle.h>
#include <tfDihedral.h>
#include <tfExclusion.h>
#include <tfEngine.h>
#include <tfError.h>


using namespace TissueForge;


/* the error macro. */
#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))


#ifdef WITH_MPI
HRESULT engine_exchange_rigid_wait(struct engine *e) {

    /* Try to grab the xchg_mutex, which will only be free while
       the async routine is waiting on a condition. */
    if(pthread_mutex_lock(&e->xchg2_mutex) != 0)
        return error(MDCERR_pthread);
        
    /* If the async exchange was started but is not running,
       wait for a signal. */
    while(e->xchg2_started && ~e->xchg2_running)
        if(pthread_cond_wait(&e->xchg2_cond, &e->xchg2_mutex) != 0)
            return error(MDCERR_pthread);
        
    /* We don't actually need this, so release it again. */
    if(pthread_mutex_unlock(&e->xchg2_mutex) != 0)
        return error(MDCERR_pthread);
        
    /* The end of the tunnel. */
    return S_OK;

    }
#endif

#ifdef WITH_MPI 
HRESULT engine_exchange_rigid_async(struct engine *e) {

    /* Check the input. */
    if(e == NULL)
        return error(MDCERR_null);

    /* Bail if not in parallel. */
    if(!(e->flags & engine_flag_mpi) || e->nr_nodes <= 1)
        return S_OK;
        
    /* Get a hold of the exchange mutex. */
    if(pthread_mutex_lock(&e->xchg2_mutex) != 0)
        return error(MDCERR_pthread);
        
    /* Tell the async thread to get to work. */
    e->xchg2_started = 1;
    if(pthread_cond_signal(&e->xchg2_cond) != 0)
        return error(MDCERR_pthread);
        
    /* Release the exchange mutex and let the async run. */
    if(pthread_mutex_unlock(&e->xchg2_mutex) != 0)
        return error(MDCERR_pthread);
        
    /* Done (for now). */
    return S_OK;
        
    }
#endif


#ifdef WITH_MPI
HRESULT engine_exchange_rigid(struct engine *e) {

    int i, j, k, ind, res;
    int counts[ e->nr_nodes ], next[ e->nr_nodes ];
    int totals_send[ e->nr_nodes ], totals_recv[ e->nr_nodes ];
    MPI_Request reqs_send[ e->nr_nodes ], reqs_recv[ e->nr_nodes ];
    struct part *buff_send[ e->nr_nodes ], *buff_recv[ e->nr_nodes ];
    struct cell *c;

    /* Check the input. */
    if(e == NULL)
        return error(MDCERR_null);

    /* Initialize the request queues. */
    for(k = 0 ; k < e->nr_nodes ; k++) {
        reqs_recv[k] = MPI_REQUEST_NULL;
        reqs_send[k] = MPI_REQUEST_NULL;
        }
        
    /* Run through the cells and fill the total counts. */
    bzero(totals_send, sizeof(int) * e->nr_nodes);
    bzero(totals_recv, sizeof(int) * e->nr_nodes);
    for(k = e->rigids_local ; k < e->rigids_semilocal ; k++) {
    
        /* count the nr of parts on each node. */
        bzero(counts, sizeof(int) * e->nr_nodes);
        for(j = 0 ; j < e->rigids[k].nr_parts ; j++) {
            c = e->s.celllist[ e->rigids[k].parts[j] ];
            counts[ c->nodeID ] += 1;
            }
            
        /* Add the number of particles going out. */
        for(i = 0 ; i < e->nr_nodes ; i++)
            if(i != e->nodeID && counts[i] > 0) {
                totals_send[i] += counts[ e->nodeID ];
                totals_recv[i] += counts[i];
                }
            
        }
        
    /* Run through the cells again and fill the send buffers. */
    for(i = 0 ; i < e->nr_nodes ; i++)
        if(e->send[i].count > 0) {
            if((buff_send[i] = (struct part *)malloc(sizeof(struct part) * totals_send[i])) == NULL)
                return error(MDCERR_malloc);
            next[i] = 0;
            }
    for(k = e->rigids_local ; k < e->rigids_semilocal ; k++) {
    
        /* count the nr of parts on each node. */
        bzero(counts, sizeof(int) * e->nr_nodes);
        for(j = 0 ; j < e->rigids[k].nr_parts ; j++) {
            c = e->s.celllist[ e->rigids[k].parts[j] ];
            counts[ c->nodeID ] += 1;
            }
            
        /* Copy the local particles to the appropriate arrays */
        for(j = 0 ; j < e->rigids[k].nr_parts ; j++) {
            c = e->s.celllist[ e->rigids[k].parts[j] ];
            if(c->nodeID == e->nodeID)
                for(i = 0 ; i < e->nr_nodes ; i++)
                    if(i != e->nodeID && counts[i] > 0)
                        buff_send[i][ next[i]++ ] = *(e->s.partlist[ e->rigids[k].parts[j] ]);
            }
            
        }
        
        
    /* Send and receive data for each neighbour. */
    for(i = 0 ; i < e->nr_nodes ; i++) {
    
        /* Do we have anything to send? */
        if(e->send[i].count > 0) {
            
            /* File a send. */
            /* printf("engine_exchange[%i]: sending %i parts to node %i.\n", e->nodeID, totals_send[i], i); */
            res = MPI_Isend(buff_send[i], totals_send[i]*sizeof(struct part), MPI_BYTE, i, e->nodeID, e->comm, &reqs_send[i]);
            
            }
            
        /* Are we expecting any parts? */
        if(e->recv[i].count > 0) {
    
            /* Allocate a buffer for the send and recv queues. */
            buff_recv[i] = (struct part *)malloc(sizeof(struct part) * totals_recv[i]);

            /* File a recv. */
            /* printf("engine_exchange[%i]: recving %i parts from node %i.\n", e->nodeID, totals_recv[i], i); */
            res = MPI_Irecv(buff_recv[i], totals_recv[i]*sizeof(struct part), MPI_BYTE, i, i, e->comm, &reqs_recv[i]);
            
            }
            
        }

    /* Wait for all the recvs to come in. */
    /* if((res = MPI_Waitall(e->nr_nodes, reqs_recv, MPI_STATUSES_IGNORE)) != MPI_SUCCESS)
        return error(MDCERR_mpi); */
        
    /* Unpack the received data. */
    #pragma omp parallel for schedule(static), private(i,ind,res,k)
    for(i = 0 ; i < e->nr_nodes ; i++) {
    
        /* Wait for this recv to come in. */
        #pragma omp critical
        { res = MPI_Waitany(e->nr_nodes, reqs_recv, &ind, MPI_STATUS_IGNORE); }
        
        /* Did we get a propper index? */
        if(ind != MPI_UNDEFINED) {

            /* Loop over the data and pass it to the cells. */
            for(k = 0 ; k < totals_recv[ind] ; k++)
                *(e->s.partlist[ buff_recv[ind][k].id ]) = buff_recv[ind][k];
                
            }
                
        }
        
    /* Wait for all the sends to come in. */
    if((res = MPI_Waitall(e->nr_nodes, reqs_send, MPI_STATUSES_IGNORE)) != MPI_SUCCESS)
        return error(MDCERR_mpi);
    /* printf("engine_exchange[%i]: all send/recv completed.\n", e->nodeID); */
        
    /* Free the send and recv buffers. */
    for(i = 0 ; i < e->nr_nodes ; i++) {
        if(e->send[i].count > 0)
            free(buff_send[i]);
        if(e->recv[i].count > 0)
            free(buff_recv[i]);
        }
        

    /* The end of the tunnel. */
    return S_OK;

    }
#endif 


#ifdef WITH_MPI
HRESULT engine_exchange_rigid_async_run(struct engine *e) {

    int i, j, k, ind, res, nr_neigh;
    int counts[ e->nr_nodes ], next[ e->nr_nodes ];
    int totals_send[ e->nr_nodes ], totals_recv[ e->nr_nodes ];
    MPI_Request reqs_send[ e->nr_nodes ], reqs_recv[ e->nr_nodes ];
    struct part *buff_send[ e->nr_nodes ], *buff_recv[ e->nr_nodes ];
    struct cell *c;

    /* Check the input. */
    if(e == NULL)
        return error(MDCERR_null);

    /* Initialize the request queues. */
    for(k = 0 ; k < e->nr_nodes ; k++) {
        reqs_recv[k] = MPI_REQUEST_NULL;
        reqs_send[k] = MPI_REQUEST_NULL;
        }
        
    /* Set the number of concurrent threads in this context. */
    for(nr_neigh = 0, k = 0 ; k < e->nr_nodes ; k++)
        nr_neigh += (e->recv[k].count > 0);
    omp_set_num_threads(nr_neigh);
        
    /* Start by acquiring the xchg_mutex. */
    if(pthread_mutex_lock(&e->xchg2_mutex) != 0)
        return error(MDCERR_pthread);

    /* Main loop... */
    while(1) {

        /* Wait for a signal to start. */
        e->xchg_running = 0;
        if(pthread_cond_wait(&e->xchg2_cond, &e->xchg2_mutex) != 0)
            return error(MDCERR_pthread);
            
        /* Tell the world I'm alive! */
        e->xchg2_started = 0; e->xchg2_running = 1;
        if(pthread_cond_signal(&e->xchg2_cond) != 0)
            return error(MDCERR_pthread);
        
        /* Run through the cells and fill the total counts. */
        bzero(totals_send, sizeof(int) * e->nr_nodes);
        bzero(totals_recv, sizeof(int) * e->nr_nodes);
        for(k = e->rigids_local ; k < e->rigids_semilocal ; k++) {

            /* count the nr of parts on each node. */
            bzero(counts, sizeof(int) * e->nr_nodes);
            for(j = 0 ; j < e->rigids[k].nr_parts ; j++) {
                c = e->s.celllist[ e->rigids[k].parts[j] ];
                counts[ c->nodeID ] += 1;
                }

            /* Add the number of particles going out. */
            for(i = 0 ; i < e->nr_nodes ; i++)
                if(i != e->nodeID && counts[i] > 0) {
                    totals_send[i] += counts[ e->nodeID ];
                    totals_recv[i] += counts[i];
                    }

            }

        /* Run through the cells again and fill the send buffers. */
        for(i = 0 ; i < e->nr_nodes ; i++)
            if(e->send[i].count > 0) {
                if((buff_send[i] = (struct part *)malloc(sizeof(struct part) * totals_send[i])) == NULL)
                    return error(MDCERR_malloc);
                next[i] = 0;
                }
        for(k = e->rigids_local ; k < e->rigids_semilocal ; k++) {

            /* count the nr of parts on each node. */
            bzero(counts, sizeof(int) * e->nr_nodes);
            for(j = 0 ; j < e->rigids[k].nr_parts ; j++) {
                c = e->s.celllist[ e->rigids[k].parts[j] ];
                counts[ c->nodeID ] += 1;
                }

            /* Copy the local particles to the appropriate arrays */
            for(j = 0 ; j < e->rigids[k].nr_parts ; j++) {
                c = e->s.celllist[ e->rigids[k].parts[j] ];
                if(c->nodeID == e->nodeID)
                    for(i = 0 ; i < e->nr_nodes ; i++)
                        if(i != e->nodeID && counts[i] > 0)
                            buff_send[i][ next[i]++ ] = *(e->s.partlist[ e->rigids[k].parts[j] ]);
                }

            }


        /* Send and receive data for each neighbour. */
        for(i = 0 ; i < e->nr_nodes ; i++) {

            /* Do we have anything to send? */
            if(e->send[i].count > 0) {

                /* File a send. */
                /* printf("engine_exchange[%i]: sending %i parts to node %i.\n", e->nodeID, totals_send[i], i); */
                res = MPI_Isend(buff_send[i], totals_send[i]*sizeof(struct part), MPI_BYTE, i, e->nodeID, e->comm, &reqs_send[i]);

                }

            /* Are we expecting any parts? */
            if(e->recv[i].count > 0) {

                /* Allocate a buffer for the send and recv queues. */
                buff_recv[i] = (struct part *)malloc(sizeof(struct part) * totals_recv[i]);

                /* File a recv. */
                /* printf("engine_exchange[%i]: recving %i parts from node %i.\n", e->nodeID, totals_recv[i], i); */
                res = MPI_Irecv(buff_recv[i], totals_recv[i]*sizeof(struct part), MPI_BYTE, i, i, e->comm, &reqs_recv[i]);

                }

            }

        /* Wait for all the recvs to come in. */
        /* if((res = MPI_Waitall(e->nr_nodes, reqs_recv, MPI_STATUSES_IGNORE)) != MPI_SUCCESS)
            return error(MDCERR_mpi); */

        /* Unpack the received data. */
        #pragma omp parallel for schedule(static), private(i,ind,res,k)
        for(i = 0 ; i < nr_neigh ; i++) {

            /* Wait for this recv to come in. */
            #pragma omp critical
            MPI_Waitany(e->nr_nodes, reqs_recv, &ind, MPI_STATUS_IGNORE);

            /* Did we get a propper index? */
            if(ind != MPI_UNDEFINED) {

                /* Loop over the data and pass it to the cells. */
                for(k = 0 ; k < totals_recv[ind] ; k++)
                    *(e->s.partlist[ buff_recv[ind][k].id ]) = buff_recv[ind][k];

                }

            }

        /* Wait for all the sends to come in. */
        if((res = MPI_Waitall(e->nr_nodes, reqs_send, MPI_STATUSES_IGNORE)) != MPI_SUCCESS)
            return error(MDCERR_mpi);
        /* printf("engine_exchange[%i]: all send/recv completed.\n", e->nodeID); */

        /* Free the send and recv buffers. */
        for(i = 0 ; i < e->nr_nodes ; i++) {
            if(e->send[i].count > 0)
                free(buff_send[i]);
            if(e->recv[i].count > 0)
                free(buff_recv[i]);
            }
        
        } /* main loop. */
        

    /* The end of the tunnel. */
    return S_OK;

    }
#endif 

 
#ifdef WITH_MPI
HRESULT engine_exchange_wait(struct engine *e) {

    /* Try to grab the xchg_mutex, which will only be free while
       the async routine is waiting on a condition. */
    if(pthread_mutex_lock(&e->xchg_mutex) != 0)
        return error(MDCERR_pthread);
        
    /* If the async exchange was started but is not running,
       wait for a signal. */
    while(e->xchg_started && !(e->xchg_running))
        if(pthread_cond_wait(&e->xchg_cond, &e->xchg_mutex) != 0)
            return error(MDCERR_pthread);
        
    /* We don't actually need this, so release it again. */
    if(pthread_mutex_unlock(&e->xchg_mutex) != 0)
        return error(MDCERR_pthread);
        
    /* The end of the tunnel. */
    return S_OK;

    }
#endif


#ifdef WITH_MPI 
HRESULT engine_exchange_async_run(struct engine *e) {

    int i, k, ind, cid, res, nr_neigh;
    int *counts_in[ e->nr_nodes ], *counts_out[ e->nr_nodes ];
    int totals_send[ e->nr_nodes ], totals_recv[ e->nr_nodes ];
    MPI_Request reqs_send[ e->nr_nodes ], reqs_recv[ e->nr_nodes ];
    MPI_Request reqs_send2[ e->nr_nodes ], reqs_recv2[ e->nr_nodes ];
    struct part *buff_send[ e->nr_nodes ], *buff_recv[ e->nr_nodes ], *finger;
    struct cell *c;
    struct space *s;

    /* Check the input. */
    if(e == NULL)
        return error(MDCERR_null);

    /* Get local copies of some data. */
    s = &e->s;
        
    /* Initialize the request queues. */
    for(k = 0 ; k < e->nr_nodes ; k++) {
        reqs_recv[k] = MPI_REQUEST_NULL;
        reqs_recv2[k] = MPI_REQUEST_NULL;
        reqs_send[k] = MPI_REQUEST_NULL;
        reqs_send2[k] = MPI_REQUEST_NULL;
        }
        
    /* Set the number of concurrent threads in this context. */
    for(nr_neigh = 0, k = 0 ; k < e->nr_nodes ; k++)
        nr_neigh += (e->recv[k].count > 0);
    omp_set_num_threads(nr_neigh);
        
    /* Start by acquiring the xchg_mutex. */
    if(pthread_mutex_lock(&e->xchg_mutex) != 0)
        return error(MDCERR_pthread);

    /* Main loop... */
    while(1) {

        /* Wait for a signal to start. */
        e->xchg_running = 0;
        if(pthread_cond_wait(&e->xchg_cond, &e->xchg_mutex) != 0)
            return error(MDCERR_pthread);
            
        /* Tell the world I'm alive! */
        e->xchg_started = 0; e->xchg_running = 1;
        if(pthread_cond_signal(&e->xchg_cond) != 0)
            return error(MDCERR_pthread);
        
        /* Start by packing and sending/receiving a counts array for each send queue. */
        #pragma omp parallel for schedule(static), private(i,k)
        for(i = 0 ; i < e->nr_nodes ; i++) {

            /* Do we have anything to send? */
            if(e->send[i].count > 0) {

                /* Allocate a new lengths array. */
                counts_out[i] = (int *)malloc(sizeof(int) * e->send[i].count);

                /* Pack the array with the counts. */
                totals_send[i] = 0;
                for(k = 0 ; k < e->send[i].count ; k++)
                    totals_send[i] += (counts_out[i][k] = s->cells[ e->send[i].cellid[k] ].count);
                /* printf("engine_exchange[%i]: totals_send[%i]=%i.\n", e->nodeID, i, totals_send[i]); */

                /* Ship it off to the correct node. */
                MPI_Isend(counts_out[i], e->send[i].count, MPI_INT, i, e->nodeID, e->comm, &reqs_send[i]);
                /* printf("engine_exchange[%i]: sending %i counts to node %i.\n", e->nodeID, e->send[i].count, i); */

                }

            /* Are we expecting any parts? */
            if(e->recv[i].count > 0) {

                /* Allocate a new lengths array for the incomming data. */
                counts_in[i] = (int *)malloc(sizeof(int) * e->recv[i].count);

                /* Dispatch a recv request. */
                MPI_Irecv(counts_in[i], e->recv[i].count, MPI_INT, i, i, e->comm, &reqs_recv[i]);
                /* printf("engine_exchange[%i]: recving %i counts from node %i.\n", e->nodeID, e->recv[i].count, i); */

                }

            }

        /* Send and receive data. */
        #pragma omp parallel for schedule(static), private(i,ind,finger,k,c)
        for(ind = 0 ; ind < nr_neigh ; ind++) {

            /* Wait for this recv to come in. */
            #pragma omp critical
            MPI_Waitany(e->nr_nodes, reqs_recv, &i, MPI_STATUS_IGNORE);
            if(i == MPI_UNDEFINED)
                continue;
            
            /* Do we have anything to send? */
            if(e->send[i].count > 0) {

                /* Allocate a buffer for the send queue. */
                buff_send[i] = (struct part *)malloc(sizeof(struct part) * totals_send[i]);

                /* Fill the send buffer. */
                finger = buff_send[i];
                for(k = 0 ; k < e->send[i].count ; k++) {
                    c = &(s->cells[e->send[i].cellid[k]]);
                    memcpy(finger, c->parts, sizeof(struct part) * c->count);
                    finger = &(finger[ c->count ]);
                    }

                /* File a send. */
                MPI_Isend(buff_send[i], totals_send[i]*sizeof(struct part), MPI_BYTE, i, e->nodeID, e->comm, &reqs_send2[i]);
                /* printf("engine_exchange[%i]: sending %i parts to node %i.\n", e->nodeID, totals_send[i], i); */

                }

            /* Are we expecting any parts? */
            if(e->recv[i].count > 0) {

                /* Count the nr of parts to recv. */
                totals_recv[i] = 0;
                for(k = 0 ; k < e->recv[i].count ; k++)
                    totals_recv[i] += counts_in[i][k];

                /* Allocate a buffer for the send and recv queues. */
                buff_recv[i] = (struct part *)malloc(sizeof(struct part) * totals_recv[i]);

                /* File a recv. */
                MPI_Irecv(buff_recv[i], totals_recv[i]*sizeof(struct part), MPI_BYTE, i, i, e->comm, &reqs_recv2[i]);
                /* printf("engine_exchange[%i]: recving %i parts from node %i.\n", e->nodeID, totals_recv[i], i); */

                }

            }

        /* Wait for all the recvs to come in. */
        /* if((res = MPI_Waitall(e->nr_nodes, reqs_recv, MPI_STATUSES_IGNORE)) != MPI_SUCCESS)
            return error(MDCERR_mpi); */

        /* Unpack the received data. */
        #pragma omp parallel for schedule(static), private(i,ind,finger,k,c,cid)
        for(i = 0 ; i < nr_neigh ; i++) {

            /* Wait for this recv to come in. */
            #pragma omp critical
            MPI_Waitany(e->nr_nodes, reqs_recv2, &ind, MPI_STATUS_IGNORE);

            /* Did we get a propper index? */
            if(ind != MPI_UNDEFINED) {

                /* Loop over the data and pass it to the cells. */
                finger = buff_recv[ind];
                for(k = 0 ; k < e->recv[ind].count ; k++) {
                    cid = e->recv[ind].cellid[k];
                    c = &(s->cells[cid]);
                    cell_load(c, finger, counts_in[ind][k], s->partlist, s->celllist);
                    
                    /* Somewhat convoluted lock/signal/unlock dance to avoid
                       blocking if all runners are waiting. */
                    pthread_mutex_lock(&s->tasks_mutex);
                    s->cells_taboo[ cid ] = 0;
                    pthread_cond_broadcast(&s->tasks_avail);
                    pthread_mutex_unlock(&s->tasks_mutex);
                    
                    finger = &(finger[ counts_in[ind][k] ]);
                    }

                }

            }

        /* Wait for all the sends to come in. */
        if((res = MPI_Waitall(e->nr_nodes, reqs_send, MPI_STATUSES_IGNORE)) != MPI_SUCCESS)
            return error(MDCERR_mpi);
        if((res = MPI_Waitall(e->nr_nodes, reqs_send2, MPI_STATUSES_IGNORE)) != MPI_SUCCESS)
            return error(MDCERR_mpi);
        /* printf("engine_exchange[%i]: all send/recv completed.\n", e->nodeID); */

        /* Free the send and recv buffers. */
        for(i = 0 ; i < e->nr_nodes ; i++) {
            if(e->send[i].count > 0) {
                free(buff_send[i]);
                free(counts_out[i]);
                }
            if(e->recv[i].count > 0) {
                free(buff_recv[i]);
                free(counts_in[i]);
                }
            }

        } /* main loop. */
        
    }
#endif


#ifdef WITH_MPI 
HRESULT engine_exchange_async(struct engine *e) {

    int k, cid;

    /* Check the input. */
    if(e == NULL)
        return error(MDCERR_null);

    /* Bail if not in parallel. */
    if(!(e->flags & engine_flag_mpi) || e->nr_nodes <= 1)
        return S_OK;
        
    /* Mark all the ghost cells as taboo and flush them. */
    for(k = 0 ; k < e->s.nr_ghost ; k++) {
        cid = e->s.cid_ghost[k];
        e->s.cells_taboo[ cid ] += 2;
        if(cell_flush(&e->s.cells[cid], e->s.partlist, e->s.celllist) < 0)
            return error(MDCERR_cell);
        }
            
    /* Get a hold of the exchange mutex. */
    if(pthread_mutex_lock(&e->xchg_mutex) != 0)
        return error(MDCERR_pthread);
        
    /* Tell the async thread to get to work. */
    e->xchg_started = 1;
    if(pthread_cond_signal(&e->xchg_cond) != 0)
        return error(MDCERR_pthread);
        
    /* Release the exchange mutex and let the async run. */
    if(pthread_mutex_unlock(&e->xchg_mutex) != 0)
        return error(MDCERR_pthread);
        
    /* Done (for now). */
    return S_OK;
        
    }
#endif


#ifdef WITH_MPI 
HRESULT engine_exchange(struct engine *e) {

    int i, k, ind, res;
    int *counts_in[ e->nr_nodes ], *counts_out[ e->nr_nodes ];
    int totals_send[ e->nr_nodes ], totals_recv[ e->nr_nodes ];
    MPI_Request reqs_send[ e->nr_nodes ], reqs_recv[ e->nr_nodes ];
    MPI_Request reqs_send2[ e->nr_nodes ], reqs_recv2[ e->nr_nodes ];
    struct part *buff_send[ e->nr_nodes ], *buff_recv[ e->nr_nodes ], *finger;
    struct cell *c;
    struct space *s;
    
    /* Check the input. */
    if(e == NULL)
        return error(MDCERR_null);
        
    /* Bail if not in parallel. */
    if(!(e->flags & engine_flag_mpi) || e->nr_nodes <= 1)
        return S_OK;
        
    /* Get local copies of some data. */
    s = &e->s;
        
    /* Wait for any asynchronous calls to finish. */
    if(e->flags & engine_flag_async)
        if(engine_exchange_wait(e) < 0)
            return error(engine_err);
            
    /* Initialize the request queues. */
    for(k = 0 ; k < e->nr_nodes ; k++) {
        reqs_recv[k] = MPI_REQUEST_NULL;
        reqs_recv2[k] = MPI_REQUEST_NULL;
        reqs_send[k] = MPI_REQUEST_NULL;
        reqs_send2[k] = MPI_REQUEST_NULL;
        }
        
    /* Start by packing and sending/receiving a counts array for each send queue. */
    #pragma omp parallel for schedule(static), private(i,k,res)
    for(i = 0 ; i < e->nr_nodes ; i++) {
    
        /* Do we have anything to send? */
        if(e->send[i].count > 0) {
        
            /* Allocate a new lengths array. */
            counts_out[i] = (int *)malloc(sizeof(int) * e->send[i].count);

            /* Pack the array with the counts. */
            totals_send[i] = 0;
            for(k = 0 ; k < e->send[i].count ; k++)
                totals_send[i] += (counts_out[i][k] = s->cells[ e->send[i].cellid[k] ].count);
            /* printf("engine_exchange[%i]: totals_send[%i]=%i.\n", e->nodeID, i, totals_send[i]); */

            /* Ship it off to the correct node. */
            /* printf("engine_exchange[%i]: sending %i counts to node %i.\n", e->nodeID, e->send[i].count, i); */
            { res = MPI_Isend(counts_out[i], e->send[i].count, MPI_INT, i, e->nodeID, e->comm, &reqs_send[i]); }
            
            }
            
        /* Are we expecting any parts? */
        if(e->recv[i].count > 0) {
    
            /* Allocate a new lengths array for the incomming data. */
            counts_in[i] = (int *)malloc(sizeof(int) * e->recv[i].count);

            /* Dispatch a recv request. */
            /* printf("engine_exchange[%i]: recving %i counts from node %i.\n", e->nodeID, e->recv[i].count, i); */
            { res = MPI_Irecv(counts_in[i], e->recv[i].count, MPI_INT, i, i, e->comm, &reqs_recv[i]); }
            
            }
    
        }
        
    /* Send and receive data for each neighbour as the counts trickle in. */
    #pragma omp parallel for schedule(static), private(i,ind,finger,k,c,res)
    for(ind = 0 ; ind < e->nr_nodes-1 ; ind++) {
    
        /* Wait for this recv to come in. */
        #pragma omp critical
        MPI_Waitany(e->nr_nodes, reqs_recv, &i, MPI_STATUS_IGNORE);
        if(i == MPI_UNDEFINED)
            continue;
        
        /* Do we have anything to send? */
        if(e->send[i].count > 0) {
            
            /* Allocate a buffer for the send queue. */
            buff_send[i] = (struct part *)malloc(sizeof(struct part) * totals_send[i]);

            /* Fill the send buffer. */
            finger = buff_send[i];
            for(k = 0 ; k < e->send[i].count ; k++) {
                c = &(s->cells[e->send[i].cellid[k]]);
                memcpy(finger, c->parts, sizeof(struct part) * c->count);
                finger = &(finger[ c->count ]);
                }

            /* File a send. */
            /* printf("engine_exchange[%i]: sending %i parts to node %i.\n", e->nodeID, totals_send[i], i); */
            { res = MPI_Isend(buff_send[i], totals_send[i]*sizeof(struct part), MPI_BYTE, i, e->nodeID, e->comm, &reqs_send2[i]); }
            
            }
            
        /* Are we expecting any parts? */
        if(e->recv[i].count > 0) {
    
            /* Count the nr of parts to recv. */
            totals_recv[i] = 0;
            for(k = 0 ; k < e->recv[i].count ; k++)
                totals_recv[i] += counts_in[i][k];

            /* Allocate a buffer for the send and recv queues. */
            buff_recv[i] = (struct part *)malloc(sizeof(struct part) * totals_recv[i]);

            /* File a recv. */
            /* printf("engine_exchange[%i]: recving %i parts from node %i.\n", e->nodeID, totals_recv[i], i); */
            { res = MPI_Irecv(buff_recv[i], totals_recv[i]*sizeof(struct part), MPI_BYTE, i, i, e->comm, &reqs_recv2[i]); }
            
            }
            
        }

    /* Wait for all the recvs to come in. */
    /* if((res = MPI_Waitall(e->nr_nodes, reqs_recv, MPI_STATUSES_IGNORE)) != MPI_SUCCESS)
        return error(MDCERR_mpi); */
        
    /* Unpack the received data. */
    #pragma omp parallel for schedule(static), private(i,ind,res,finger,k,c)
    for(i = 0 ; i < e->nr_nodes-1 ; i++) {
    
        /* Wait for this recv to come in. */
        #pragma omp critical
        MPI_Waitany(e->nr_nodes, reqs_recv2, &ind, MPI_STATUS_IGNORE);
        
        /* Did we get a propper index? */
        if(ind != MPI_UNDEFINED) {

            /* Loop over the data and pass it to the cells. */
            finger = buff_recv[ind];
            for(k = 0 ; k < e->recv[ind].count ; k++) {
                c = &(s->cells[e->recv[ind].cellid[k]]);
                cell_flush(c, s->partlist, s->celllist);
                cell_load(c, finger, counts_in[ind][k], s->partlist, s->celllist);
                finger = &(finger[ counts_in[ind][k] ]);
                }
                
            }
                
        }
        
    /* Wait for all the sends to come in. */
    if((res = MPI_Waitall(e->nr_nodes, reqs_send, MPI_STATUSES_IGNORE)) != MPI_SUCCESS)
        return error(MDCERR_mpi);
    if((res = MPI_Waitall(e->nr_nodes, reqs_send2, MPI_STATUSES_IGNORE)) != MPI_SUCCESS)
        return error(MDCERR_mpi);
    /* printf("engine_exchange[%i]: all send/recv completed.\n", e->nodeID); */
        
    /* Free the send and recv buffers. */
    for(i = 0 ; i < e->nr_nodes ; i++) {
        if(e->send[i].count > 0) {
            free(buff_send[i]);
            free(counts_out[i]);
            }
        if(e->recv[i].count > 0) {
            free(buff_recv[i]);
            free(counts_in[i]);
            }
        }
        
    /* Make sure each part is in the right cell, but only if we're
       not using Verlet lists. */
    if(!(e->flags & engine_flag_verlet)) {
    
        /* Shuffle the space. */
        if(space_shuffle(s) < 0)
            return error(MDCERR_space);
            
        /* Welcome the parts into the respective cells. */
        #pragma omp parallel for schedule(static), private(i)
        for(i = 0 ; i < s->nr_marked ; i++)
            cell_welcome(&(s->cells[ s->cid_marked[i] ]), s->partlist);
    
        }
            
    /* Call it a day. */
    return S_OK;
        
    }
#endif


#ifdef WITH_MPI 
HRESULT engine_exchange_incomming(struct engine *e) {

    int i, j, k, ind, res;
    int *counts_in[ e->nr_nodes ], *counts_out[ e->nr_nodes ];
    int totals_send[ e->nr_nodes ], totals_recv[ e->nr_nodes ];
    MPI_Request reqs_send[ e->nr_nodes ], reqs_recv[ e->nr_nodes ];
    MPI_Request reqs_send2[ e->nr_nodes ], reqs_recv2[ e->nr_nodes ];
    struct part *buff_send[ e->nr_nodes ], *buff_recv[ e->nr_nodes ], *finger;
    struct cell *c;
    struct space *s;
    
    /* Check the input. */
    if(e == NULL)
        return error(MDCERR_null);
        
    /* Bail if not in parallel. */
    if(!(e->flags & engine_flag_mpi) || e->nr_nodes <= 1)
        return S_OK;
        
    /* Get local copies of some data. */
    s = &e->s;
        
    /* Initialize the request queues. */
    for(k = 0 ; k < e->nr_nodes ; k++) {
        reqs_recv[k] = MPI_REQUEST_NULL;
        reqs_recv2[k] = MPI_REQUEST_NULL;
        reqs_send[k] = MPI_REQUEST_NULL;
        reqs_send2[k] = MPI_REQUEST_NULL;
        }
        
    /* As opposed to #engine_exchange, we are going to send the incomming
       particles on ghost cells that do not belong to us. We therefore invert
       the send/recv queues, i.e. we send the incommings for the cells
       from which we usually receive data. */
        
    /* Start by packing and sending/receiving a counts array for each send queue. */
    #pragma omp parallel for schedule(static), private(i,k,res)
    for(i = 0 ; i < e->nr_nodes ; i++) {
    
        /* Do we have anything to send? */
        if(e->recv[i].count > 0) {
        
            /* Allocate a new lengths array. */
            counts_out[i] = (int *)malloc(sizeof(int) * e->recv[i].count);

            /* Pack the array with the counts. */
            totals_send[i] = 0;
            for(k = 0 ; k < e->recv[i].count ; k++)
                totals_send[i] += (counts_out[i][k] = s->cells[ e->recv[i].cellid[k] ].incomming_count);
            /* printf("engine_exchange[%i]: totals_send[%i]=%i.\n", e->nodeID, i, totals_send[i]); */

            /* Ship it off to the correct node. */
            /* printf("engine_exchange[%i]: sending %i counts to node %i.\n", e->nodeID, e->send[i].count, i); */
            { res = MPI_Isend(counts_out[i], e->recv[i].count, MPI_INT, i, e->nodeID, e->comm, &reqs_send[i]); }
            
            }
            
        /* Are we expecting any parts? */
        if(e->send[i].count > 0) {
    
            /* Allocate a new lengths array for the incomming data. */
            counts_in[i] = (int *)malloc(sizeof(int) * e->send[i].count);

            /* Dispatch a recv request. */
            /* printf("engine_exchange[%i]: recving %i counts from node %i.\n", e->nodeID, e->recv[i].count, i); */
            { res = MPI_Irecv(counts_in[i], e->send[i].count, MPI_INT, i, i, e->comm, &reqs_recv[i]); }
            
            }
    
        }
        
    /* Send and receive data. */
    #pragma omp parallel for schedule(static), private(ind,i,finger,k,c,res)
    for(ind = 0 ; ind < e->nr_nodes ; ind++) {
    
        /* Wait for this recv to come in. */
        #pragma omp critical
        MPI_Waitany(e->nr_nodes, reqs_recv, &i, MPI_STATUS_IGNORE);
        if(i == MPI_UNDEFINED)
            continue;
        
        /* Do we have anything to send? */
        if(e->recv[i].count > 0) {
            
            /* Allocate a buffer for the send queue. */
            buff_send[i] = (struct part *)malloc(sizeof(struct part) * totals_send[i]);

            /* Fill the send buffer. */
            finger = buff_send[i];
            for(k = 0 ; k < e->recv[i].count ; k++) {
                c = &(s->cells[e->recv[i].cellid[k]]);
                memcpy(finger, c->incomming, sizeof(struct part) * c->incomming_count);
                finger = &(finger[ c->incomming_count ]);
                }

            /* File a send. */
            /* printf("engine_exchange[%i]: sending %i parts to node %i.\n", e->nodeID, totals_send[i], i); */
            { res = MPI_Isend(buff_send[i], totals_send[i]*sizeof(struct part), MPI_BYTE, i, e->nodeID, e->comm, &reqs_send2[i]); }
            
            }
            
        /* Are we expecting any parts? */
        if(e->send[i].count > 0) {
    
            /* Count the nr of parts to recv. */
            totals_recv[i] = 0;
            for(k = 0 ; k < e->send[i].count ; k++)
                totals_recv[i] += counts_in[i][k];

            /* Allocate a buffer for the send and recv queues. */
            buff_recv[i] = (struct part *)malloc(sizeof(struct part) * totals_recv[i]);

            /* File a recv. */
            /* printf("engine_exchange[%i]: recving %i parts from node %i.\n", e->nodeID, totals_recv[i], i); */
            { res = MPI_Irecv(buff_recv[i], totals_recv[i]*sizeof(struct part), MPI_BYTE, i, i, e->comm, &reqs_recv2[i]); }
            
            }
            
        }

    /* Wait for all the recvs to come in. */
    /* if((res = MPI_Waitall(e->nr_nodes, reqs_recv, MPI_STATUSES_IGNORE)) != MPI_SUCCESS)
        return error(MDCERR_mpi); */
        
    /* Unpack the received data. */
    #pragma omp parallel for schedule(static), private(i,j,ind,res,finger,k,c)
    for(i = 0 ; i < e->nr_nodes ; i++) {
    
        /* Wait for this recv to come in. */
        #pragma omp critical
        MPI_Waitany(e->nr_nodes, reqs_recv2, &ind, MPI_STATUS_IGNORE);
        
        /* Did we get a propper index? */
        if(ind != MPI_UNDEFINED) {

            /* Loop over the data and pass it to the cells. */
            finger = buff_recv[ind];
            for(k = 0 ; k < e->send[ind].count ; k++) {
                c = &(s->cells[e->send[ind].cellid[k]]);
                pthread_mutex_lock(&c->cell_mutex);
                cell_add_incomming_multiple(c, finger, counts_in[ind][k]);
                pthread_mutex_unlock(&c->cell_mutex);
                for(j = 0 ; j < counts_in[ind][k] ; j++)
                    e->s.celllist[ finger[j].id ] = c;
                finger = &(finger[ counts_in[ind][k] ]);
                }
                
            }
                
        }
        
    /* Wait for all the sends to come in. */
    if((res = MPI_Waitall(e->nr_nodes, reqs_send, MPI_STATUSES_IGNORE)) != MPI_SUCCESS)
        return error(MDCERR_mpi);
    if((res = MPI_Waitall(e->nr_nodes, reqs_send2, MPI_STATUSES_IGNORE)) != MPI_SUCCESS)
        return error(MDCERR_mpi);
    /* printf("engine_exchange[%i]: all send/recv completed.\n", e->nodeID); */
        
    /* Free the send and recv buffers. */
    for(i = 0 ; i < e->nr_nodes ; i++) {
        if(e->send[i].count > 0) {
            free(buff_send[i]);
            free(counts_out[i]);
            }
        if(e->recv[i].count > 0) {
            free(buff_recv[i]);
            free(counts_in[i]);
            }
        }
        
    /* Call it a day. */
    return S_OK;
        
    }
#endif

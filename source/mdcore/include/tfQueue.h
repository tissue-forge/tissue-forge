/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * Coypright (c) 2017 Andy Somogyi (somogyie at indiana dot edu)
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

#ifndef _MDCORE_INCLUDE_TFQUEUE_H_
#define _MDCORE_INCLUDE_TFQUEUE_H_

#include "tf_platform.h"
#include "tf_lock.h"

MDCORE_BEGIN_DECLS

/* queue error codes */
#define queue_err_ok                    0
#define queue_err_null                  -1
#define queue_err_malloc                -2
#define queue_err_full                  -3
#define queue_err_lock                  -4

/* Some constants. */
#define queue_maxhit                    10


namespace TissueForge { 


	/** ID of the last error */
	CAPI_DATA(int) queue_err;

	/** The queue structure */
	typedef struct queue {

		/* Allocated size. */
		int size;

		/* The queue data. */
		struct task *tasks;

		/* The space in which this queue lives. */
		struct space *space;

		/* The queue indices. */
		int *ind;

		/* Index of next entry. */
		int next;

		/* Index of last entry. */
		int count;

		/* Lock for this queue. */
		lock_type lock;

	} queue;


	/* Associated functions */
	int queue_init(struct queue *q, int size, struct space *s, struct task *tasks);
	void queue_reset(struct queue *q);
	int queue_insert(struct queue *q, struct task *t);
	struct task *queue_get(struct queue *q, int rid, int keep);

};

MDCORE_END_DECLS
#endif // _MDCORE_INCLUDE_TFQUEUE_H_
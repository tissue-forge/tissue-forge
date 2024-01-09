/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * Coypright (c) 2017 Andy Somogyi (somogyie at indiana dot edu)
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

/**
 * @file tfQueue.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFQUEUE_H_
#define _MDCORE_INCLUDE_TFQUEUE_H_

#include "tf_platform.h"
#include "tf_lock.h"

MDCORE_BEGIN_DECLS

/* Some constants. */
#define queue_maxhit                    10


namespace TissueForge { 


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

	/**
	 * @brief Initialize a task queue.
	 *
	 * @param q The #queue to initialize.
	 * @param size The maximum number of cellpairs in this queue.
	 * @param s The space with which this queue is associated.
	 * @param tasks An array containing the #task to which the queue
	 *        indices will refer to.
	 *
	 * Initializes a queue of the maximum given size. The initial queue
	 * is empty and can be filled with pair ids.
	 *
	 * @sa #queue_tuples_init
	 */
	HRESULT queue_init(struct queue *q, int size, struct space *s, struct task *tasks);

	/**
	 * @brief Reset the queue.
	 * 
	 * @param q The #queue.
	 */
	void queue_reset(struct queue *q);

	/**
	 * @brief Add an index to the given queue.
	 * 
	 * @param q The #queue.
	 * @param thing The thing to be inserted.
	 *
	 * Inserts a task into the queue at the location of the next pointer
	 * and moves all remaining tasks up by one. Thus, if the queue is executing,
	 * the inserted task is considered to already have been taken.
	 *
	 * @return 1 on success, 0 if the queue is full and <0 on error (see #queue_err).
	 */
	int queue_insert(struct queue *q, struct task *t);

	/**
	 * @brief Get a task from the queue.
	 * 
	 * @param q The #queue.
	 * @param rid #runner ID for ownership issues.
	 * @param keep If true, remove the returned index from the queue.
	 *
	 * @return A #task with no unresolved dependencies or conflicts
	 *      or @c NULL if none could be found.
	 */
	struct task *queue_get(struct queue *q, int rid, int keep);

};

MDCORE_END_DECLS
#endif // _MDCORE_INCLUDE_TFQUEUE_H_
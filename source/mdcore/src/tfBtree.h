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

#ifndef _MDCORE_SOURCE_TFBTREE_H_
#define _MDCORE_SOURCE_TFBTREE_H_

/** BTree constants. */
#define btree_maxnodes				8
#define btree_minnodes				4
#define btree_cache					256


/** Error constants. */
#define btree_err_ok				0
#define btree_err_null              -1
#define btree_err_malloc            -2
#define btree_err_map               -3


/** Flags. */
#define btree_flag_freeable         1
#define btree_flag_leaf             2


/** Define type for mapping function */
#define btree_maptype				int (*)(void *, void *)


namespace TissueForge { 


	/** ID of the last error */
	extern int btree_err;

	/** Data structure for a btree node. */
	struct btree_node {

		short int fill;					// nr of nodes in node  (2 bytes)
		unsigned short int flags; 		// node flags           (2 bytes)
		void *data[btree_maxnodes + 1]; // node content         ((N+1)*4 bytes)
		int keys[btree_maxnodes + 1];   // node keys            ((N+1)*4 bytes)
		struct btree_node *nodes[btree_maxnodes + 2];
										// node branches;       ((N+2)*4 bytes)

	};

	/** The btree itself. */
	struct btree {

		struct btree_node *first;		// first node in tree
		
		struct btree_node *cache;       // cached nodes

	};

	int btree_init(struct btree *b);
	struct btree_node *btree_getnode(struct btree *b);
	int btree_insert(struct btree *b, int key, void *data);
	int btree_map(struct btree *b, int (*func)(void *, void *), void *data);
	int btree_find(struct btree *b, int key, void **res);
	int btree_releasenode(struct btree *b, struct btree_node *n);
	int btree_delete(struct btree *b, int key, void **res);

};

#endif // _MDCORE_SOURCE_TFBTREE_H_
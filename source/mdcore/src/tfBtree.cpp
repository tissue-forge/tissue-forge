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

/* include some standard headers. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* local includes. */
#include <tf_errs.h>
#include "tfBtree.h"


using namespace TissueForge;


/* the error macro. */
#define error(id)				(btree_err = errs_register(id, btree_err_msg[-(id)], __LINE__, __FUNCTION__, __FILE__))

/* list of error messages. */
const char *btree_err_msg[4] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "The user-supplied mapping function returned an error."
	};


/** ID of the last error. */
int btree_err = btree_err_ok;


/**
 * @brief Delete a key from the #btree.
 *
 * @param b the #btree.
 * @param key the key to search for and remove
 * @param res a pointer to a pointer in which the address of the
 *      data will be written, if found (may be @c NULL)
 *
 * @return 1 if the @c key was found, 0 if not and < 0 on error
 *      (see #btree_err).
 */
 
int TissueForge::btree_delete(struct btree *b, int key, void **res) {

    int key_up = 0, wasfound = 0;
    void *data_up = NULL;
    struct btree_node *n;

    /*
     * @brief Recursive call of #btree_delete.
     * @param n the #btree_node over which to recurse
     * @param mode mode of operation
     * @return 1 if the node underflowed, 0 otherwise.
     */
    int btree_delete_rec(struct btree_node *n, int mode) {
    
        int found = 0, underflow = 0, k, l, m = 0, r;
        struct btree_node *nl, *nr;
    
        /* Three modes of operation: if mode is zero, then
            we're just looking for the key. If mode is +/-1,
            then we are looking for the rightmost or leftmost
            node respectively. */
        if(mode == 0) {
        
            /* look for the key (bisection). */
            l = -1; r = n->fill;
            while(r-l > 1) {
                m = (l + r) / 2;
                if(n->keys[m] > key)
                    r = m;
                else if(n->keys[m] < key)
                    l = m;
                else {
                    found = 1;
                    wasfound = 1;
                    if(res != NULL)
                        *res = n->data[m];
                    break;
                    }
                }
                
            /* are we a leaf ? */
            if (n->flags & btree_flag_leaf) {
            
                /* if the node was not found, just bail. */
                if(!found)
                    return btree_err_ok;
                
                /* store the data if requested. */
                if(res != NULL)
                    *res = n->data[m];

                /* shrink this node. */
                n->fill -= 1;

                /* fill the gap. */
                for(k = m ; k < n->fill ; k++) {
                    n->keys[k] = n->keys[k+1];
                    n->data[k] = n->data[k+1];
                    }

                }
                
            /* not a leaf, more complicated... */
            else {
            
                /* if found, replace the node with the rightmost node
                    on the left. */
                if(found) {
                
                    /* recurse. */
                    r = m;
                    underflow = btree_delete_rec(n->nodes[r], 1);
                    
                    /* replace the key and data with whatever came up. */
                    n->keys[m] = key_up;
                    n->data[m] = data_up;
                
                    }
                    
                /* otherwise, just recurse. */
                else 
                    underflow = btree_delete_rec(n->nodes[r], 0);
                    
                }
        
            }
        
        /* just looking for the rightmost node in a leaf. */
        else if(mode == 1) {
        
            /* is this a leaf? */
            if(n->flags & btree_flag_leaf) {
            
                /* remove the key and data. */
                n->fill -= 1;
                key_up = n->keys[n->fill];
                data_up = n->data[n->fill];
            
                }
                
            /* otherwise, just recurse. */
            else {
                r = n->fill;
                underflow = btree_delete_rec(n->nodes[r], 1);
                }
        
            }
                    
        /* did the previous node underflow? */
        if(underflow) {

            /* try to borrow from the left... */
            if((r > 0) && (n->nodes[r-1]->fill > btree_minnodes)) {

                /* get a finger on the nodes downstream. */
                nl = n->nodes[r-1]; nr = n->nodes[r];

                /* make some space on the right. */
                for(k = nr->fill ; k > 0 ; k--) {
                    nr->keys[k] = nr->keys[k-1];
                    nr->data[k] = nr->data[k-1];
                    nr->nodes[k+1] = nr->nodes[k];
                    }
                nr->nodes[1] = nr->nodes[0];
                nr->fill += 1;

                /* copy the key and data from n. */
                nr->keys[0] = n->keys[r-1];
                nr->data[0] = n->data[r-1];

                /* copy the node pointer from nl to nr. */
                nr->nodes[0] = nl->nodes[nl->fill];

                /* shrink nl. */
                nl->fill -= 1;

                /* copy the data from nl to n. */
                n->keys[r-1] = nl->keys[nl->fill];
                n->data[r-1] = nl->data[nl->fill];

                }

            /* try to borrow from the right... */
            else if((r < n->fill) && (n->nodes[r+1]->fill > btree_minnodes)) {

                /* get a finger on the nodes downstream. */
                nl = n->nodes[r]; nr = n->nodes[r+1];

                /* copy the data from n to nl. */
                nl->keys[nl->fill] = n->keys[r];
                nl->data[nl->fill] = n->data[r];
                nl->fill += 1;

                /* copy nodes from nr to nl. */
                nl->nodes[nl->fill] = nr->nodes[0];

                /* copy data from nr to n. */
                n->keys[r] = nr->keys[0];
                n->data[r] = nr->data[0];

                /* shrink nr. */
                nr->fill -= 1;
                for(k = 0 ; k < nr->fill ; k++) {
                    nr->keys[k] = nr->keys[k+1];
                    nr->data[k] = nr->data[k+1];
                    nr->nodes[k] = nr->nodes[k+1];
                    }
                nr->nodes[nr->fill] = nr->nodes[nr->fill+1];

                }

            /* merge with either left or right. */
            else {
            
                /* get handles on the downstream nodes. */
                if(r == n->fill)
                    r -= 1;
                nl = n->nodes[r]; nr = n->nodes[r+1];

                /* copy the data from n to nl. */
                nl->keys[nl->fill] = n->keys[r];
                nl->data[nl->fill] = n->data[r];
                nl->fill += 1;

                /* copy data and nodes from nr to nl. */
                for(k = 0 ; k < nr->fill ; k++) {
                    nl->keys[nl->fill+k] = nr->keys[k];
                    nl->data[nl->fill+k] = nr->data[k];
                    nl->nodes[nl->fill+k] = nr->nodes[k];
                    }
                nl->nodes[nl->fill+nr->fill] = nr->nodes[nr->fill];
                nl->fill += nr->fill;

                /* release nr. */
                btree_releasenode(b, nr);

                /* shrink n. */
                n->fill -= 1;
                for(k = r ; k < n->fill ; k++) {
                    n->keys[k] = n->keys[k+1];
                    n->data[k] = n->data[k+1];
                    n->nodes[k+1] = n->nodes[k+2];
                    }

                }

            } /* if underflow. */
            
        /* did this node underflow? */
        return (n->fill < btree_minnodes);
    
        }
        

    /* check for null pointers. */
    if(b == NULL)
        return error(btree_err_null);
        
    /* if the tree is not empty, call the recursion. */
    if(b->first != NULL) {
    
        /* did something get deleted? */
        if(btree_delete_rec(b->first, 0)) {

            /* check if the root node is empty. */
            if(b->first->fill == 0) {

                /* unhook the first node. */
                n = b->first;
                b->first = b->first->nodes[0];

                /* release the free node. */
                btree_releasenode(b, n);

                }

            }
            
        }
        
    /* return ok. */
    return wasfound;

    }


/**
 * @brief Retrieve the data for a given key.
 * 
 * @param b the #btree in which to search.
 * @param key the key to search for.
 * @param res a pointer to a pointer in which the address of the
 *      data will be written.
 *
 * @return 1 if found, 0 if not and < 0 on err (see #btree_err).
 *
 * Looks for the given @c key and copies its data pointer to the
 * address given by @c res.
 */
 
int TissueForge::btree_find(struct btree *b, int key, void **res) {

    int l, m, r;
    struct btree_node *n;

    /* check for the usual nonsense. */
    if(b == NULL || res == NULL)
        return error(btree_err_null);
        
    /* if the btree is empty, just bail. */
    if(b->first == NULL)
        return btree_err_ok;
        
    /* get a hold of the first node. */
    n = b->first;
    
    /* loop down the tree. */
    while(1) {
    
        /* look for the key in this node. */
        l = -1; r = n->fill;
        while(r-l > 1) {
            m = (l + r) / 2;
            if(n->keys[m] > key)
                r = m;
            else if(n->keys[m] < key)
                l = m;
            else {
                *res = n->data[m];
                return 1;
                }
            }
            
        /* is this node already a leaf? */
        if(n->flags & btree_flag_leaf)
            break;
            
        /* otherwise, get the node in between l and r. */
        else
            n = n->nodes[r];
    
        }
        
    /* bail if nothing found. */
    return btree_err_ok;

    }


/**
 * @brief Apply a given function to all data in a btree.
 *
 * @param b the #btree.
 * @param func the funciton, which should be of the type #btree_maptype.
 * @param data a pointer that will be passed to @c func with each call.
 *
 * @return #btree_err_ok or < 0 on error (see #btree_err).
 *
 * If @c func returns < 0 for any node, the traversal is interrupted and
 * an error is returned.
 */
 
int TissueForge::btree_map(struct btree *b, int (*func)(void *, void *), void *data) {

    
    /*
     * @brief Recursive call of #btree_map.
     * @param n the #btree_node over which to recurse
     */
    int btree_map_rec(struct btree_node *n) {
    
        int k;
    
        /* is this node a leaf? */
        if(n->flags & btree_flag_leaf) {
        
            /* loop over the data. */
            for(k = 0 ; k < n->fill ; k++)
                if(func(n->data[k], data) < 0)
                    return error(btree_err_map);
                    
            }
            
        /* it's an internal node, call recursively too... */
        else {
        
            /* recurse over the first node. */
            if(btree_map_rec(n->nodes[0]) < 0)
                return btree_err;
        
            /* loop over the data. */
            for(k = 0 ; k < n->fill ; k++) {
                if(func(n->data[k], data) < 0)
                    return error(btree_err_map);
                if(btree_map_rec(n->nodes[k+1]) < 0)
                    return btree_err;
                }
                    
            }
            
        /* over and out. */
        return btree_err_ok;
            
        }
    
    /* check for the usual nonsense. */
    if(b == NULL || func == NULL)
        return error(btree_err_null);
        
    /* if the btree is not empty, call the recursion. */
    if(b->first != NULL && btree_map_rec(b->first) < 0)
        return btree_err;
    else
        return btree_err_ok;

    }


/**
 * @brief Insert a key/data pair into the given #btree
 *
 * @param b the #btree.
 * @param key the integer key.
 * @param data a pointer to the data associated with @c key.
 *
 * @return #btree_err_ok or < 0 on error (see #btree_err).
 *
 * If a node with the given key already exists, the data pointer
 * is replaced.
 */
 
int TissueForge::btree_insert(struct btree *b, int key, void *data) {

    struct btree_node *n, *split_left = NULL, *split_right = NULL;
    int key_up = 0;
    void *data_up = NULL;
    
    /*
     * @brief recursive function for node insertion.
     * @param n the node over which to recurse.
     * @param key the key to insert.
     * @param data a pointer to the data associated with that node.
     * @return #btree_err_ok or 1 if the node has been split.
     */
    int btree_insert_rec(struct btree_node *n, int key, void *data) {
    
        int k, m, l = -1, r = n->fill;
        
        /* start by finding out where the key should go (binary search). */
        while(r-l > 1) {
        
            /* do the normal bisection thing. */
            m = (l + r) / 2;
            if(n->keys[m] > key)
                r = m;
            else if(n->keys[m] < key)
                l = m;
                
            /* did we hit the key? if so, just replace and quit. */
            else {
                n->data[m] = data;
                return btree_err_ok;
                }
                
            }
            
        /* is this node a leaf? */
        if(n->flags & btree_flag_leaf) {
            
            /* insert the new key and data at position r. */
            for(k = n->fill ; k > r ; k--) {
                n->keys[k] = n->keys[k-1];
                n->data[k] = n->data[k-1];
                }
            n->keys[r] = key;
            n->data[r] = data;
            
            /* increase the fill */
            n->fill += 1;
            
            }
            
        /* call recursively and pick-up any splits. */
        else {
        
            /* did the recursive call split? */
            if(btree_insert_rec(n->nodes[r], key, data)) {
            
                /* insert the new key and data at position l. */
                for(k = n->fill ; k > r ; k--) {
                    n->keys[k] = n->keys[k-1];
                    n->data[k] = n->data[k-1];
                    n->nodes[k+1] = n->nodes[k];
                    }
                n->keys[r] = key_up;
                n->data[r] = data_up;
                n->nodes[r] = split_left; n->nodes[r+1] = split_right;

                /* increase the fill */
                n->fill += 1;
            
                }
        
            }
            
        /* now check for overflow... */
        if(n->fill > btree_maxnodes) {
        
            /* get a new node */
            split_right = btree_getnode(b);
            
            /* copy data to the right of the middle to the new node. */
            m = btree_maxnodes/2;
            for(k = 0 ; k < m ; k++) {
                split_right->keys[k] = n->keys[m+k+1];
                split_right->data[k] = n->data[m+k+1];
                split_right->nodes[k] = n->nodes[m+k+1];
                }
            split_right->nodes[m] = n->nodes[n->fill];
            split_right->fill = m;
            
            /* this node is a leaf if the previous one was. */
            split_right->flags |= (n->flags & btree_flag_leaf);
            
            /* set key_up and data_up. */
            key_up = n->keys[m];
            data_up = n->data[m];
            
            /* reduce the current node. */
            n->fill = m;
            split_left = n;
            
            /* signal that there was a split. */
            return 1;
        
            }
    
        /* all done! */
        return btree_err_ok;
    
        }


    /* check for bad pointers. */
    if(b == NULL)
        return error(btree_err_null);
        
    /* simplest case: b is empty. */
    if(b->first == NULL) {
    
        /* get a new node. */
        if((n = btree_getnode(b)) == NULL)
            return error(btree_err);
            
        /* fill it with the sole key and data. */
        n->fill = 1;
        n->flags |= btree_flag_leaf;
        n->data[0] = data;
        n->keys[0] = key;
        n->nodes[0] = NULL; n->nodes[1] = NULL;
        
        /* set the fist node. */
        b->first = n;
    
        }
        
    /* otherwise, call recursively. */
    else {
    
        /* insert in first node and check for split. */
        if(btree_insert_rec(b->first, key, data)) {
        
            /* get a new node. */
            if((n = btree_getnode(b)) == NULL)
                return error(btree_err);

            /* fill it with the sole key and data. */
            n->fill = 1; n->flags &= ~btree_flag_leaf;
            n->data[0] = data_up;
            n->keys[0] = key_up;
            n->nodes[0] = split_left; n->nodes[1] = split_right;

            /* set the fist node. */
            b->first = n;
    
            }
    
        }
        
    /* end on a good note. */
    return btree_err_ok;

    }
    
    
/** 
 * @brief Return a #btree_node to the btree's cache.
 *
 * @param b the #btree.
 * @param n the #btree_node
 *
 * @return A pointer to a #btree_node or @c NULL if an error
 *      occured (see #btree_err).
 */
 
int TissueForge::btree_releasenode(struct btree *b, struct btree_node *n) {

    /* check for null pointers. */
    if(b == NULL || n == NULL)
        return error(btree_err_null);
        
    /* hook-up the node. */
    n->nodes[0] = b->cache;
    b->cache = n;
    
    /* all is well... */
    return btree_err_ok;
    
    }
    

/** 
 * @brief get a #btree_node from the btree's cache.
 *
 * @param the #btree.
 *
 * @return A pointer to a #btree_node or @c NULL if an error
 *      occured (see #btree_err).
 */
 
struct btree_node *TissueForge::btree_getnode(struct btree *b) {

    struct btree_node *n;
    int k;

    /* check for nonsense. */
    if(b == NULL) {
        error(btree_err_null);
        return NULL;
        }
        
    /* if the cache is empty, fill it. */
    if(b->cache == NULL) {
    
        /* allocate some nodes for the cache. */
        if((n = (struct btree_node *)malloc(sizeof(struct btree_node) * btree_cache)) == NULL) {
            error(btree_err_malloc);
            return NULL;
            }
        bzero(n, sizeof(struct btree_node) * btree_cache);
        n[0].flags = btree_flag_freeable;

        /* link each node to its neighbour. */
        for(k = 0 ; k < btree_cache-1 ; k++)
            n[k].nodes[0] = &(n[k+1]);

        /* link the cache. */
        b->cache = n;
        
        }
        
    /* return the first node in the cache. */
    n = b->cache;
    b->cache = n->nodes[0];
    return n;
    
    }
    

/**
 * @brief Initialize the given #btree.
 *
 * @param b the #btree.
 *
 * @return #btree_err_ok or < 0 on error.
 */
 
int TissueForge::btree_init(struct btree *b) {

    struct btree_node *n;
    int k;

    /* check for rotten input. */
    if(b == NULL)
        return error(btree_err_null);
        
    /* clear the first node. */
    b->first = NULL;
    
    /* allocate some nodes for the cache. */
    if((n = (struct btree_node *)malloc(sizeof(struct btree_node) * btree_cache)) == NULL)
        return error(btree_err_malloc);
    bzero(n, sizeof(struct btree_node) * btree_cache);
    n[0].flags = btree_flag_freeable;
        
    /* link each node to its neighbour. */
    for(k = 0 ; k < btree_cache-1 ; k++)
        n[k].nodes[0] = &(n[k+1]);
    
    /* link the cache. */
    b->cache = n;
    
    /* all done? */
    return btree_err_ok;
    
    }

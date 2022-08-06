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
#include <tfLogger.h>

#pragma clang diagnostic ignored "-Wwritable-strings"


using namespace TissueForge;


/* the error macro. */
#define error(id) (engine_err = errs_register(id, engine_err_msg[-(id)], __LINE__, __FUNCTION__, __FILE__))


/**
 * @brief Compute all bonded interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Does the same as #engine_bond_eval, #engine_angle_eval and
 * #engine_dihedral eval, yet all in one go to avoid excessive
 * updates of the particle forces.
 */

int TissueForge::engine_bonded_eval_sets(struct engine *e) {

	FPTYPE epot_bond = 0.0, epot_angle = 0.0, epot_dihedral = 0.0, epot_exclusion = 0.0;
#ifdef HAVE_OPENMP
	int sets_taboo[ e->nr_sets];
	int k, j, set_curr, sets_next = 0, sets_ind[ e->nr_sets ];
	FPTYPE epot_local_bond = 0.0, epot_local_angle = 0.0, epot_local_dihedral = 0.0, epot_local_exclusion = 0.0;
	ticks toc_bonds, toc_angles, toc_dihedrals, toc_exclusions;
#endif
	ticks tic;

#ifdef HAVE_OPENMP

	/* Fill the indices. */
	for(k = 0 ; k < e->nr_sets ; k++) {
		sets_ind[k] = k;
		sets_taboo[k] = 0;
	}

#pragma omp parallel private(k,j,set_curr,epot_local_bond,epot_local_angle,epot_local_dihedral,epot_local_exclusion,toc_bonds,toc_angles,toc_dihedrals,toc_exclusions)
	if(e->nr_sets > 0 && omp_get_num_threads() > 1) {

		/* Init local counters. */
		toc_bonds = 0; toc_angles = 0; toc_dihedrals = 0; toc_exclusions = 0;
		epot_local_bond = 0.0;
		epot_local_angle = 0.0;
		epot_local_dihedral = 0.0;
		epot_local_exclusion = 0.0;
		set_curr = -1;

		/* Main loop. */
		while(sets_next < e->nr_sets) {

			/* Try to grab a set. */
			set_curr = -1;
#pragma omp critical (setlist)
			while(sets_next < e->nr_sets) {

				/* Find the next free set. */
				for(k = sets_next ; k < e->nr_sets && sets_taboo[ sets_ind[k] ] ; k++);

				/* If a set was found... */
				if(k < e->nr_sets) {

					/* Swap it to the top and put a finger on it. */
					set_curr = sets_ind[k];
					sets_ind[k] = sets_ind[sets_next];
					sets_ind[sets_next] = set_curr;
					sets_next += 1;

					/* Mark conflicting sets as taboo. */
#pragma omp critical (taboo)
					for(j = 0 ; j < e->sets[set_curr].nr_confl ; j++)
						sets_taboo[ e->sets[set_curr].confl[j] ] += 1;

					/* And exit the loop. */
					break;

				}

			}

			/* Did we even get a set? */
			if(set_curr < 0)
				break;

			/* Evaluate the bonded interaction in the set. */
			/* Do exclusions. */
			tic = getticks();
			exclusion_eval(e->sets[set_curr].exclusions, e->sets[set_curr].nr_exclusions, e, &epot_local_exclusion);
			toc_exclusions += getticks() - tic;

			/* Do bonds. */
			tic = getticks();
			bond_eval(e->sets[set_curr].bonds, e->sets[set_curr].nr_bonds, e, &epot_local_bond);
			toc_bonds += getticks() - tic;

			/* Do angles. */
			tic = getticks();
			angle_eval(e->sets[set_curr].angles, e->sets[set_curr].nr_angles, e, &epot_local_angle);
			toc_angles += getticks() - tic;

			/* Do dihedrals. */
			tic = getticks();
			dihedral_eval(e->sets[set_curr].dihedrals, e->sets[set_curr].nr_dihedrals, e, &epot_local_dihedral);
			toc_dihedrals += getticks() - tic;

			/* Un-mark conflicting sets. */
#pragma omp critical (taboo)
			for(k = 0 ; k < e->sets[set_curr].nr_confl ; k++)
				sets_taboo[ e->sets[set_curr].confl[k] ] -= 1;

		} /* main loop. */

		/* Write-back global data. */
#pragma omp critical (writeback)
		{
			e->timers[engine_timer_bonds] += toc_bonds;
			e->timers[engine_timer_angles] += toc_angles;
			e->timers[engine_timer_dihedrals] += toc_dihedrals;
			e->timers[engine_timer_exclusions] += toc_exclusions;
			epot_bond += epot_local_bond;
			epot_angle += epot_local_angle;
			epot_dihedral += epot_local_dihedral;
			epot_exclusion += epot_local_exclusion;
		}

	}

	/* Otherwise, just do the sequential thing. */
	else {

		/* Do exclusions. */
		tic = getticks();
		exclusion_eval(e->exclusions, e->nr_exclusions, e, &epot_exclusion);
		e->timers[engine_timer_exclusions] += getticks() - tic;

		/* Do bonds. */
		tic = getticks();
		bond_eval(e->bonds, e->nr_bonds, e, &epot_bond);
		e->timers[engine_timer_bonds] += getticks() - tic;

		/* Do angles. */
		tic = getticks();
		angle_eval(e->angles, e->nr_angles, e, &epot_angle);
		e->timers[engine_timer_angles] += getticks() - tic;

		/* Do dihedrals. */
		tic = getticks();
		dihedral_eval(e->dihedrals, e->nr_dihedrals, e, &epot_dihedral);
		e->timers[engine_timer_dihedrals] += getticks() - tic;

	}
#else

	/* Do exclusions. */
	tic = getticks();
	if(exclusion_eval(e->exclusions, e->nr_exclusions, e, &epot_exclusion) < 0)
		return error(engine_err_exclusion);
	e->timers[engine_timer_exclusions] += getticks() - tic;

	/* Do bonds. */
	tic = getticks();
	if(bond_eval(e->bonds, e->nr_bonds, e, &epot_bond) < 0)
		return error(engine_err_bond);
	e->timers[engine_timer_bonds] += getticks() - tic;

	/* Do angles. */
	tic = getticks();
	if(angle_eval(e->angles, e->nr_angles, e, &epot_angle) < 0)
		return error(engine_err_angle);
	e->timers[engine_timer_angles] += getticks() - tic;

	/* Do dihedrals. */
	tic = getticks();
	if(dihedral_eval(e->dihedrals, e->nr_dihedrals, e, &epot_dihedral) < 0)
		return error(engine_err_dihedral);
	e->timers[engine_timer_dihedrals] += getticks() - tic;

#endif

/* Store the potential energy. */
	e->s.epot += epot_bond + epot_angle + epot_dihedral + epot_exclusion;
	e->s.epot_bond += epot_bond;
	e->s.epot_angle += epot_angle;
	e->s.epot_dihedral += epot_dihedral;
	e->s.epot_exclusion += epot_exclusion;

	/* I'll be back... */
	return engine_err_ok;

}

typedef struct _bonded_set{
	int i, j;
} bonded_set;

typedef struct _bonded_sets {
	bonded_set *confl, *confl_sorted;
	int confl_count;
	int confl_size;
	int *nconfl;

} bonded_sets;

/* Function to add a conflict. */
static int confl_add (bonded_sets *bs, int i, int j) {
	if(bs->confl_count == bs->confl_size &&
       (bs->confl = (bonded_set*)realloc(bs->confl, sizeof(bonded_set) * (bs->confl_size *= 2))) == NULL)
		return error(engine_err_malloc);
	bs->confl[bs->confl_count].i = i; bs->confl[bs->confl_count].j = j;
	bs->nconfl[i] += 1; bs->nconfl[j] += 1;
	bs->confl_count += 1;
	return engine_err_ok;
}


/* Recursive quicksort for the conflicts. */
static void confl_qsort (bonded_sets *bs, int l, int r) {

	int i = l, j = r;
	int pivot_i = bs->confl_sorted[ (l + r)/2 ].i;
	bonded_set temp;

	/* Too small? */
	if(r - l < 10) {

		/* Use Insertion Sort. */
		for(i = l+1 ; i <= r ; i++) {
			pivot_i = bs->confl_sorted[i].i;
			for(j = i-1 ; j >= l ; j--)
				if(bs->confl_sorted[j].i > pivot_i) {
					temp = bs->confl_sorted[j];
					bs->confl_sorted[j] = bs->confl_sorted[j+1];
					bs->confl_sorted[j+1] = temp;
				}
				else
					break;
		}

	}

	else {

		/* Partition. */
		while(i <= j) {
			while(bs->confl_sorted[i].i < pivot_i)
				i += 1;
			while(bs->confl_sorted[j].i > pivot_i)
				j -= 1;
			if(i <= j) {
				temp = bs->confl_sorted[i];
				bs->confl_sorted[i] = bs->confl_sorted[j];
				bs->confl_sorted[j] = temp;
				i += 1;
				j -= 1;
			}
		}

		/* Recurse. */
		if(l < j)
			confl_qsort(bs, l, j);
		if(i < r)
			confl_qsort(bs, i, r);
	}
}



/**
 * @brief Assemble non-conflicting sets of bonded interactions.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int TissueForge::engine_bonded_sets(struct engine *e, int max_sets) {


	int *confl_index;
	int *weight;
	int *setid_bonds, *setid_angles, *setid_dihedrals, *setid_exclusions, *setid_rigids;
	int nr_sets;
	int i, jj, j, k, min_i, min_j, min_weight, max_weight, max_confl, nr_confl;
	FPTYPE avg_weight;
	char *confl_counts;

	bonded_sets bs;
	bs.confl_count = 0;

	/* Start with one set per bonded interaction. */
	nr_sets = e->nr_bonds + e->nr_angles + e->nr_dihedrals + e->nr_exclusions + e->nr_rigids;
	if((weight = (int *)malloc(sizeof(int) * nr_sets)) == NULL ||
			(bs.nconfl = (int *)calloc(nr_sets, sizeof(int))) == NULL ||
			(confl_counts = (char *)malloc(sizeof(char) * nr_sets)) == NULL)
		return error(engine_err_malloc);

	/* Allocate the initial setids. */
	if((setid_bonds = (int *)malloc(sizeof(int) * e->nr_bonds)) == NULL ||
			(setid_angles = (int *)malloc(sizeof(int) * e->nr_angles)) == NULL ||
			(setid_dihedrals = (int *)malloc(sizeof(int) * e->nr_dihedrals)) == NULL ||
			(setid_exclusions = (int *)malloc(sizeof(int) * e->nr_exclusions)) == NULL ||
			(setid_rigids = (int *)malloc(sizeof(int) * e->nr_rigids)) == NULL)
		return error(engine_err_malloc);

	/* Generate the set of conflicts. */
	bs.confl_size = nr_sets;
	if((bs.confl = (_bonded_set*)malloc(sizeof(int) * 2 * bs.confl_size)) == NULL)
		return error(engine_err_malloc);
	nr_sets = 0;

	/* Loop over all dihedrals. */
	for(k = 0 ; k < e->nr_dihedrals ; k++) {

		/* This dihedral gets its own id. */
		weight[ nr_sets ] = 3;
		setid_dihedrals[k] = nr_sets++;

		/* Loop over other dihedrals... */
		for(j = 0 ; j < k ; j++)
			if(e->dihedrals[k].i == e->dihedrals[j].i || e->dihedrals[k].i == e->dihedrals[j].j || e->dihedrals[k].i == e->dihedrals[j].k || e->dihedrals[k].i == e->dihedrals[j].l ||
					e->dihedrals[k].j == e->dihedrals[j].i || e->dihedrals[k].j == e->dihedrals[j].j || e->dihedrals[k].j == e->dihedrals[j].k || e->dihedrals[k].j == e->dihedrals[j].l ||
					e->dihedrals[k].k == e->dihedrals[j].i || e->dihedrals[k].k == e->dihedrals[j].j || e->dihedrals[k].k == e->dihedrals[j].k || e->dihedrals[k].k == e->dihedrals[j].l ||
					e->dihedrals[k].l == e->dihedrals[j].i || e->dihedrals[k].l == e->dihedrals[j].j || e->dihedrals[k].l == e->dihedrals[j].k || e->dihedrals[k].l == e->dihedrals[j].l)
				if(confl_add(&bs, setid_dihedrals[k], setid_dihedrals[j]) < 0)
					return error(engine_err);

	} /* Loop over dihedrals. */

	/* Loop over all angles. */
	for(k = 0 ; k < e->nr_angles ; k++) {

		/* Loop over dihedrals, looking for matches... */
		for(j = 0 ; j < e->nr_dihedrals ; j++)
			if((e->angles[k].i == e->dihedrals[j].i && e->angles[k].j == e->dihedrals[j].j && e->angles[k].k == e->dihedrals[j].k) ||
					(e->angles[k].i == e->dihedrals[j].j && e->angles[k].j == e->dihedrals[j].k && e->angles[k].k == e->dihedrals[j].l) ||
					(e->angles[k].k == e->dihedrals[j].j && e->angles[k].j == e->dihedrals[j].k && e->angles[k].i == e->dihedrals[j].l) ||
					(e->angles[k].k == e->dihedrals[j].i && e->angles[k].j == e->dihedrals[j].j && e->angles[k].i == e->dihedrals[j].k)) {
				setid_angles[k] = -setid_dihedrals[j];
				weight[ setid_dihedrals[j] ] += 2;
				break;
			}

		/* Does this angle get its own id? */
		if(j < e->nr_dihedrals)
			continue;
		else {
			weight[ nr_sets ] = 2;
			setid_angles[k] = nr_sets++;
		}

		/* Loop over dihedrals, looking for conflicts... */
		for(j = 0 ; j < e->nr_dihedrals ; j++)
			if(e->angles[k].i == e->dihedrals[j].i || e->angles[k].i == e->dihedrals[j].j || e->angles[k].i == e->dihedrals[j].k || e->angles[k].i == e->dihedrals[j].l ||
					e->angles[k].j == e->dihedrals[j].i || e->angles[k].j == e->dihedrals[j].j || e->angles[k].j == e->dihedrals[j].k || e->angles[k].j == e->dihedrals[j].l ||
					e->angles[k].k == e->dihedrals[j].i || e->angles[k].k == e->dihedrals[j].j || e->angles[k].k == e->dihedrals[j].k || e->angles[k].k == e->dihedrals[j].l)
				if(confl_add(&bs, setid_angles[k], setid_dihedrals[j]) < 0)
					return error(engine_err);

		/* Loop over previous angles... */
		for(j = 0 ; j < k ; j++)
			if(setid_angles[j] >= 0 &&
					(e->angles[k].i == e->angles[j].i || e->angles[k].i == e->angles[j].j || e->angles[k].i == e->angles[j].k ||
							e->angles[k].j == e->angles[j].i || e->angles[k].j == e->angles[j].j || e->angles[k].j == e->angles[j].k ||
							e->angles[k].k == e->angles[j].i || e->angles[k].k == e->angles[j].j || e->angles[k].k == e->angles[j].k))
				if(confl_add(&bs, setid_angles[k], setid_angles[j]) < 0)
					return error(engine_err);

	} /* Loop over angles. */

	/* Loop over all bonds. */
	for(k = 0 ; k < e->nr_bonds ; k++) {

		/* Loop over dihedrals, looking for overlap... */
		for(j = 0 ; j < e->nr_dihedrals ; j++)
			if((e->bonds[k].i == e->dihedrals[j].i && e->bonds[k].j == e->dihedrals[j].j) ||
					(e->bonds[k].j == e->dihedrals[j].i && e->bonds[k].i == e->dihedrals[j].j) ||
					(e->bonds[k].i == e->dihedrals[j].j && e->bonds[k].j == e->dihedrals[j].k) ||
					(e->bonds[k].j == e->dihedrals[j].j && e->bonds[k].i == e->dihedrals[j].k) ||
					(e->bonds[k].i == e->dihedrals[j].k && e->bonds[k].j == e->dihedrals[j].l) ||
					(e->bonds[k].j == e->dihedrals[j].k && e->bonds[k].i == e->dihedrals[j].l)) {
				setid_bonds[k] = -setid_dihedrals[j];
				weight[ setid_dihedrals[j] ] += 1;
				break;
			}

		/* Does this bond get its own id? */
		if(j < e->nr_dihedrals)
			continue;

		/* Loop over angles, looking for overlap... */
		for(j = 0 ; j < e->nr_angles ; j++)
			if(setid_angles[j] >= 0 &&
					((e->bonds[k].i == e->angles[j].i && e->bonds[k].j == e->angles[j].j) ||
							(e->bonds[k].j == e->angles[j].i && e->bonds[k].i == e->angles[j].j) ||
							(e->bonds[k].i == e->angles[j].j && e->bonds[k].j == e->angles[j].k) ||
							(e->bonds[k].j == e->angles[j].j && e->bonds[k].i == e->angles[j].k))) {
				setid_bonds[k] = -setid_angles[j];
				weight[ setid_angles[j] ] += 1;
				break;
			}

		/* Does this bond get its own id? */
		if(j < e->nr_angles)
			continue;
		else {
			weight[ nr_sets ] = 1;
			setid_bonds[k] = nr_sets++;
		}

		/* Loop over dihedrals... */
		for(j = 0 ; j < e->nr_dihedrals ; j++)
			if(e->bonds[k].i == e->dihedrals[j].i || e->bonds[k].i == e->dihedrals[j].j || e->bonds[k].i == e->dihedrals[j].k || e->bonds[k].i == e->dihedrals[j].l ||
					e->bonds[k].j == e->dihedrals[j].i || e->bonds[k].j == e->dihedrals[j].j || e->bonds[k].j == e->dihedrals[j].k || e->bonds[k].j == e->dihedrals[j].l)
				if(confl_add(&bs, setid_bonds[k], setid_dihedrals[j]) < 0)
					return error(engine_err);

		/* Loop over angles... */
		for(j = 0 ; j < e->nr_angles ; j++)
			if(setid_angles[j] >= 0 &&
					(e->bonds[k].i == e->angles[j].i || e->bonds[k].i == e->angles[j].j || e->bonds[k].i == e->angles[j].k ||
							e->bonds[k].j == e->angles[j].i || e->bonds[k].j == e->angles[j].j || e->bonds[k].j == e->angles[j].k))
				if(confl_add(&bs, setid_bonds[k], setid_angles[j]) < 0)
					return error(engine_err);

		/* Loop over previous bonds... */
		for(j = 0 ; j < k ; j++)
			if(setid_bonds[j] >= 0 &&
					(e->bonds[k].i == e->bonds[j].i || e->bonds[k].i == e->bonds[j].j ||
							e->bonds[k].j == e->bonds[j].i || e->bonds[k].j == e->bonds[j].j))
				if(confl_add(&bs, setid_bonds[k], setid_bonds[j]) < 0)
					return error(engine_err);

	} /* Loop over bonds. */

	/* Blindly add all the rigids as sets. */
	for(k = 0 ; k < e->nr_rigids ; k++) {

		/* Add this rigid as a set. */
		weight[ nr_sets ] = 0;
		setid_rigids[k] = nr_sets++;

		/* Loop over dihedrals, looking for overlap. */
		for(j = 0 ; j < e->nr_dihedrals ; j++) {
			for(i = 0 ; i < e->rigids[k].nr_parts; i ++)
				if(e->rigids[k].parts[i] == e->dihedrals[j].i || e->rigids[k].parts[i] == e->dihedrals[j].j || e->rigids[k].parts[i] == e->dihedrals[j].k || e->rigids[k].parts[i] == e->dihedrals[j].l)
					break;
			if(i < e->rigids[k].nr_parts && confl_add(&bs, setid_rigids[k], setid_dihedrals[j]))
				return error(engine_err);
		}

		/* Loop over angles, looking for overlap. */
		for(j = 0 ; j < e->nr_angles ; j++) {
			if(setid_angles[j] < 0)
				continue;
			for(i = 0 ; i < e->rigids[k].nr_parts; i ++)
				if(e->rigids[k].parts[i] == e->angles[j].i || e->rigids[k].parts[i] == e->angles[j].j || e->rigids[k].parts[i] == e->angles[j].k)
					break;
			if(i < e->rigids[k].nr_parts && confl_add(&bs, setid_rigids[k], setid_angles[j]))
				return error(engine_err);
		}

		/* Loop over bonds, looking for overlap. */
		for(j = 0 ; j < e->nr_bonds ; j++) {
			if(setid_bonds[j] < 0)
				continue;
			for(i = 0 ; i < e->rigids[k].nr_parts; i ++)
				if(e->rigids[k].parts[i] == e->bonds[j].i || e->rigids[k].parts[i] == e->bonds[j].j)
					break;
			if(i < e->rigids[k].nr_parts && confl_add(&bs, setid_rigids[k], setid_bonds[j]))
				return error(engine_err);
		}

	}

	/* Loop over all exclusions. */
	for(k = 0 ; k < e->nr_exclusions ; k++) {

		/* Loop over rigids, looking for overlap. */
		for(j = 0 ; j < e->nr_rigids ; j++) {
			for(i = 0 ; i < e->rigids[j].nr_constr ; i++)
				if((e->exclusions[k].i == e->rigids[j].parts[ e->rigids[j].constr[i].i ] && e->exclusions[k].j == e->rigids[j].parts[ e->rigids[j].constr[i].j ]) ||
						(e->exclusions[k].j == e->rigids[j].parts[ e->rigids[j].constr[i].i ] && e->exclusions[k].i == e->rigids[j].parts[ e->rigids[j].constr[i].j ]))
					break;
			if(i < e->rigids[j].nr_constr) {
				setid_exclusions[k] = -setid_rigids[j];
				weight[ setid_rigids[j] ] += 1;
				break;
			}
		}

		/* Does this bond get its own id? */
		if(j < e->nr_rigids)
			continue;

		/* Loop over dihedrals, looking for overlap... */
		for(j = 0 ; j < e->nr_dihedrals ; j++)
			if((e->exclusions[k].i == e->dihedrals[j].i && e->exclusions[k].j == e->dihedrals[j].j) ||
					(e->exclusions[k].j == e->dihedrals[j].i && e->exclusions[k].i == e->dihedrals[j].j) ||
					(e->exclusions[k].i == e->dihedrals[j].j && e->exclusions[k].j == e->dihedrals[j].k) ||
					(e->exclusions[k].j == e->dihedrals[j].j && e->exclusions[k].i == e->dihedrals[j].k) ||
					(e->exclusions[k].i == e->dihedrals[j].k && e->exclusions[k].j == e->dihedrals[j].l) ||
					(e->exclusions[k].j == e->dihedrals[j].k && e->exclusions[k].i == e->dihedrals[j].l) ||
					(e->exclusions[k].i == e->dihedrals[j].i && e->exclusions[k].j == e->dihedrals[j].k) ||
					(e->exclusions[k].j == e->dihedrals[j].i && e->exclusions[k].i == e->dihedrals[j].k) ||
					(e->exclusions[k].i == e->dihedrals[j].j && e->exclusions[k].j == e->dihedrals[j].l) ||
					(e->exclusions[k].j == e->dihedrals[j].j && e->exclusions[k].i == e->dihedrals[j].l) ||
					(e->exclusions[k].i == e->dihedrals[j].i && e->exclusions[k].j == e->dihedrals[j].l) ||
					(e->exclusions[k].j == e->dihedrals[j].i && e->exclusions[k].i == e->dihedrals[j].l)) {
				setid_exclusions[k] = -setid_dihedrals[j];
				weight[ setid_dihedrals[j] ] += 1;
				break;
			}

		/* Does this bond get its own id? */
		if(j < e->nr_dihedrals)
			continue;

		/* Loop over angles, looking for overlap... */
		for(j = 0 ; j < e->nr_angles ; j++)
			if(setid_angles[j] >= 0 &&
					((e->exclusions[k].i == e->angles[j].i && e->exclusions[k].j == e->angles[j].j) ||
							(e->exclusions[k].j == e->angles[j].i && e->exclusions[k].i == e->angles[j].j) ||
							(e->exclusions[k].i == e->angles[j].j && e->exclusions[k].j == e->angles[j].k) ||
							(e->exclusions[k].j == e->angles[j].j && e->exclusions[k].i == e->angles[j].k) ||
							(e->exclusions[k].i == e->angles[j].i && e->exclusions[k].j == e->angles[j].k) ||
							(e->exclusions[k].j == e->angles[j].i && e->exclusions[k].i == e->angles[j].k))) {
				setid_exclusions[k] = -setid_angles[j];
				weight[ setid_angles[j] ] += 1;
				break;
			}

		/* Does this bond get its own id? */
		if(j < e->nr_angles)
			continue;

		/* Loop over bonds, looking for overlap... */
		for(j = 0 ; j < e->nr_bonds ; j++)
			if(setid_bonds[j] >= 0 &&
					((e->exclusions[k].i == e->bonds[j].i && e->exclusions[k].j == e->bonds[j].j) ||
							(e->exclusions[k].j == e->bonds[j].i && e->exclusions[k].i == e->bonds[j].j))) {
				setid_exclusions[k] = -setid_bonds[j];
				weight[ setid_bonds[j] ] += 1;
				break;
			}

		/* Does this bond get its own id? */
		if(j < e->nr_bonds)
			continue;
		else {
			weight[ nr_sets ] = 1;
			setid_exclusions[k] = nr_sets++;
		}

		/* Loop over dihedrals... */
		for(j = 0 ; j < e->nr_dihedrals ; j++)
			if(e->exclusions[k].i == e->dihedrals[j].i || e->exclusions[k].i == e->dihedrals[j].j || e->exclusions[k].i == e->dihedrals[j].k || e->exclusions[k].i == e->dihedrals[j].l ||
					e->exclusions[k].j == e->dihedrals[j].i || e->exclusions[k].j == e->dihedrals[j].j || e->exclusions[k].j == e->dihedrals[j].k || e->exclusions[k].j == e->dihedrals[j].l)
				if(confl_add(&bs, setid_exclusions[k], setid_dihedrals[j]) < 0)
					return error(engine_err);

		/* Loop over angles... */
		for(j = 0 ; j < e->nr_angles ; j++)
			if(setid_angles[j] >= 0 &&
					(e->exclusions[k].i == e->angles[j].i || e->exclusions[k].i == e->angles[j].j || e->exclusions[k].i == e->angles[j].k ||
							e->exclusions[k].j == e->angles[j].i || e->exclusions[k].j == e->angles[j].j || e->exclusions[k].j == e->angles[j].k))
				if(confl_add(&bs, setid_exclusions[k], setid_angles[j]) < 0)
					return error(engine_err);

		/* Loop over  bonds... */
		for(j = 0 ; j < e->nr_bonds ; j++)
			if(setid_bonds[j] >= 0 &&
					(e->exclusions[k].i == e->bonds[j].i || e->exclusions[k].i == e->bonds[j].j ||
							e->exclusions[k].j == e->bonds[j].i || e->exclusions[k].j == e->bonds[j].j))
				if(confl_add(&bs, setid_exclusions[k], setid_bonds[j]) < 0)
					return error(engine_err);

		/* Loop over previous exclusions... */
		for(j = 0 ; j < k ; j++)
			if(setid_exclusions[j] >= 0 &&
					(e->exclusions[k].i == e->exclusions[j].i || e->exclusions[k].i == e->exclusions[j].j ||
							e->exclusions[k].j == e->exclusions[j].i || e->exclusions[k].j == e->exclusions[j].j))
				if(confl_add(&bs, setid_exclusions[k], setid_exclusions[j]) < 0)
					return error(engine_err);

	} /* Loop over exclusions. */

	/* Make the setids positive again. */
	for(k = 0 ; k < e->nr_angles ; k++)
		setid_angles[k] = abs(setid_angles[k]);
	for(k = 0 ; k < e->nr_bonds ; k++)
		setid_bonds[k] = abs(setid_bonds[k]);
	for(k = 0 ; k < e->nr_exclusions ; k++)
		setid_exclusions[k] = abs(setid_exclusions[k]);

	/* Allocate the sorted conflict data. */
	if((bs.confl_sorted = (_bonded_set*)malloc(sizeof(int) * 4 * bs.confl_size)) == NULL ||
			(confl_index = (int *)malloc(sizeof(int) * (nr_sets + 1))) == NULL)
		return error(engine_err_malloc);


	/* As of here, the data structure has been set-up! */


	/* Main loop... */
	while(nr_sets > max_sets) {

		/* Get the average number of conflicts. */
		min_weight = weight[0]; max_weight = weight[0];
		for(k = 1 ; k < nr_sets ; k++)
			if(weight[k] < min_weight)
				min_weight = weight[k];
			else if(weight[k] > max_weight)
				max_weight = weight[k];
		avg_weight =(2.0*min_weight + max_weight) / 3;

		/* First try to do the cheap thing: find a pair with
           zero conflicts each. */
		for(min_i = 0 ; min_i < nr_sets && (weight[min_i] >= avg_weight || bs.nconfl[min_i] > 0) ; min_i++);
		for(min_j = min_i+1 ; min_j < nr_sets && (weight[min_j] >= avg_weight || bs.nconfl[min_j] > 0) ; min_j++);

		/* Did we find a mergeable pair? */
		if(min_i < nr_sets && min_j < nr_sets) {}

		/* Otherwise, look for a pair sharing a conflict. */
		else {

			/* Assemble and sort the conflicts array. */
			for(k = 0 ; k < bs.confl_count ; k++) {
				bs.confl_sorted[k] = bs.confl[k];
				bs.confl_sorted[bs.confl_count+k].i = bs.confl[k].j;
				bs.confl_sorted[bs.confl_count+k].j = bs.confl[k].i;
			}
			confl_qsort(&bs, 0, 2*bs.confl_count - 1);
			confl_index[0] = 0;
			for(j = 0, k = 0 ; k < 2*bs.confl_count ; k++)
				while(bs.confl_sorted[k].i > j)
					confl_index[++j] = k;
			while(j < nr_sets)
				confl_index[ ++j ] = 2*bs.confl_count;
			bzero(confl_counts, sizeof(char) * nr_sets);

			/* Init min_i, min_j and min_confl. */
			min_i = -1;
			min_j = -1;
			max_confl = -1;

			/* For every pair of sets i and j... */
			for(i = 0; i < nr_sets ; i++) {

				/* Skip i? */
				if(weight[i] > avg_weight || bs.nconfl[i] <= max_confl)
					continue;

				/* Mark the conflicts in the ith set. */
				for(k = confl_index[i] ; k < confl_index[i+1] ; k++)
					confl_counts[ bs.confl_sorted[k].j ] = 1;
				confl_counts[i] = 1;

				/* Loop over all following sets. */
				for(jj = confl_index[i] ; jj < confl_index[i+1] ; jj++) {

					/* Skip j? */
					j = bs.confl_sorted[jj].j;
					if(weight[j] > avg_weight || bs.nconfl[j] <= max_confl)
						continue;

					/* Get the number of conflicts in the combined set of i and j. */
					for(nr_confl = 0, k = confl_index[j] ; k < confl_index[j+1] ; k++)
						if(confl_counts[ bs.confl_sorted[k].j ])
							nr_confl += 1;

					/* Is this value larger than the current maximum? */
					if(nr_confl > max_confl) {
						max_confl = nr_confl; min_i = i; min_j = j;
					}

				} /* loop over following sets. */

				/* Un-mark the conflicts in the ith set. */
				for(k = confl_index[i] ; k < confl_index[i+1] ; k++)
					confl_counts[ bs.confl_sorted[k].j ] = 0;
				confl_counts[i] = 0;

			} /* for every pair of sets i and j. */


			/* If we didn't find anything, look for non-related set pairs (more expensive). */
			if(min_i < 0 || min_j < 0) {

				/* For every pair of sets i and j... */
				for(i = 0; i < nr_sets ; i++) {

					/* Skip i? */
					if(weight[i] > avg_weight || bs.nconfl[i] <= max_confl)
						continue;

					/* Mark the conflicts in the ith set. */
					for(k = confl_index[i] ; k < confl_index[i+1] ; k++)
						confl_counts[ bs.confl_sorted[k].j ] = 1;
					confl_counts[i] = 1;

					/* Loop over all following sets. */
					for(j = i+1 ; j < nr_sets ; j++) {

						/* Skip j? */
						if(weight[j] > avg_weight || bs.nconfl[j] <= max_confl)
							continue;

						/* Get the number of conflicts in the combined set of i and j. */
						for(nr_confl = 0, k = confl_index[j] ; k < confl_index[j+1] ; k++)
							if(confl_counts[ bs.confl_sorted[k].j ])
								nr_confl += 1;

						/* Is this value larger than the current maximum? */
						if(nr_confl > max_confl) {
							max_confl = nr_confl; min_i = i; min_j = j;
						}

					} /* loop over following sets. */

					/* Un-mark the conflicts in the ith set. */
					for(k = confl_index[i] ; k < confl_index[i+1] ; k++)
						confl_counts[ bs.confl_sorted[k].j ] = 0;
					confl_counts[i] = 0;

				} /* for every pair of sets i and j. */

			}

			/* If we didn't find anything, merge the pairs with the lowest weight. */
			if(min_i < 0 || min_j < 0) {

				/* Find the set with the minimum weight. */
				for(min_i = 0, i = 1 ; i < nr_sets ; i++)
					if(weight[i] < weight[min_i])
						min_i = i;

				/* Find the set with the second-minimum weight. */
				min_j = (min_i == 0 ? 1 : 0);
				for(j = 0 ; j < nr_sets ; j++)
					if(j != min_i && weight[j] < weight[min_j])
						min_j = j;

			}

			/* Did we catch any pair? */
			if(min_i < 0 || min_j < 0) {
				printf("engine_bonded_sets: could not find a pair to merge!\n");
				return error(engine_err_sets);
			}

			/* Mark the sets with which min_i conflicts. */
			for(k = confl_index[min_i] ; k < confl_index[min_i+1] ; k++)
				confl_counts[ bs.confl_sorted[k].j ] = 1;
			confl_counts[ min_i ] = 1;

			/* Re-label or remove conflicts with min_j. */
			for(k = 0 ; k < bs.confl_count ; k++)
				if(bs.confl[k].i == min_j) {
					if(confl_counts[ bs.confl[k].j ]) {
						bs.nconfl[ bs.confl[k].j ] -= 1;
						bs.confl[ k-- ] = bs.confl[ --bs.confl_count ];
					}
					else {
						bs.confl[k].i = min_i;
						bs.nconfl[min_i] += 1;
					}
				}
				else if(bs.confl[k].j == min_j) {
					if(confl_counts[ bs.confl[k].i ]) {
						bs.nconfl[ bs.confl[k].i ] -= 1;
						bs.confl[ k-- ] = bs.confl[ --bs.confl_count ];
					}
					else {
						bs.confl[k].j = min_i;
						bs.nconfl[min_i] += 1;
					}
				}

		}

		/* Merge the sets min_i and min_j. */
		for(k = 0 ; k < e->nr_bonds ; k++)
			if(setid_bonds[k] == min_j)
				setid_bonds[k] = min_i;
		for(k = 0 ; k < e->nr_angles ; k++)
			if(setid_angles[k] == min_j)
				setid_angles[k] = min_i;
		for(k = 0 ; k < e->nr_dihedrals ; k++)
			if(setid_dihedrals[k] == min_j)
				setid_dihedrals[k] = min_i;
		for(k = 0 ; k < e->nr_exclusions ; k++)
			if(setid_exclusions[k] == min_j)
				setid_exclusions[k] = min_i;

		/* Remove the set min_j (replace by last). */
				weight[min_i] += weight[min_j];
		nr_sets -= 1;
		weight[min_j] = weight[nr_sets];
		bs.nconfl[min_j] = bs.nconfl[nr_sets];
		for(k = 0 ; k < e->nr_bonds ; k++)
			if(setid_bonds[k] == nr_sets)
				setid_bonds[k] = min_j;
		for(k = 0 ; k < e->nr_angles ; k++)
			if(setid_angles[k] == nr_sets)
				setid_angles[k] = min_j;
		for(k = 0 ; k < e->nr_dihedrals ; k++)
			if(setid_dihedrals[k] == nr_sets)
				setid_dihedrals[k] = min_j;
		for(k = 0 ; k < e->nr_exclusions ; k++)
			if(setid_exclusions[k] == nr_sets)
				setid_exclusions[k] = min_j;
		for(k = 0 ; k < bs.confl_count ; k++)
			if(bs.confl[k].i == nr_sets)
				bs.confl[k].i = min_j;
			else if(bs.confl[k].j == nr_sets)
				bs.confl[k].j = min_j;

	} /* main loop. */


	/* Allocate the sets. */
	e->nr_sets = nr_sets;
	if((e->sets = (struct engine_set *)malloc(sizeof(struct engine_set) * nr_sets)) == NULL)
		return error(engine_err_malloc);
	bzero(e->sets, sizeof(struct engine_set) * nr_sets);

	/* Fill in the counts. */
	for(k = 0 ; k < e->nr_bonds ; k++)
		e->sets[ setid_bonds[k] ].nr_bonds += 1;
	for(k = 0 ; k < e->nr_angles ; k++)
		e->sets[ setid_angles[k] ].nr_angles += 1;
	for(k = 0 ; k < e->nr_dihedrals ; k++)
		e->sets[ setid_dihedrals[k] ].nr_dihedrals += 1;
	for(k = 0 ; k < e->nr_exclusions ; k++)
		e->sets[ setid_exclusions[k] ].nr_exclusions += 1;

	/* Allocate the index lists. */
	for(k = 0 ; k < nr_sets ; k++) {
		if((e->sets[k].bonds = (struct Bond *)malloc(sizeof(struct Bond) * e->sets[k].nr_bonds)) == NULL ||
				(e->sets[k].angles = (struct Angle *)malloc(sizeof(struct Angle) * e->sets[k].nr_angles)) == NULL ||
				(e->sets[k].dihedrals = (struct Dihedral *)malloc(sizeof(struct Dihedral) * e->sets[k].nr_dihedrals)) == NULL ||
				(e->sets[k].exclusions = (struct exclusion *)malloc(sizeof(struct exclusion) * e->sets[k].nr_exclusions)) == NULL ||
				(e->sets[k].confl = (int *)malloc(sizeof(int) * bs.nconfl[k])) == NULL)
			return error(engine_err_malloc);
		e->sets[k].weight = e->sets[k].nr_bonds + e->sets[k].nr_exclusions + 2*e->sets[k].nr_angles + 3*e->sets[k].nr_dihedrals;
		e->sets[k].nr_bonds = 0;
		e->sets[k].nr_angles = 0;
		e->sets[k].nr_dihedrals = 0;
		e->sets[k].nr_exclusions = 0;
	}

	/* Fill in the indices. */
	for(k = 0 ; k < e->nr_bonds ; k++) {
		j = setid_bonds[k];
		e->sets[j].bonds[ e->sets[j].nr_bonds++ ] = e->bonds[ k ];
	}
	for(k = 0 ; k < e->nr_angles ; k++) {
		j = setid_angles[k];
		e->sets[j].angles[ e->sets[j].nr_angles++ ] = e->angles[ k ];
	}
	for(k = 0 ; k < e->nr_dihedrals ; k++) {
		j = setid_dihedrals[k];
		e->sets[j].dihedrals[ e->sets[j].nr_dihedrals++ ] = e->dihedrals[ k ];
	}
	for(k = 0 ; k < e->nr_exclusions ; k++) {
		j = setid_exclusions[k];
		e->sets[j].exclusions[ e->sets[j].nr_exclusions++ ] = e->exclusions[ k ];
	}

	/* Fill in the conflicts. */
	for(k = 0 ; k < bs.confl_count ; k++) {
		i = bs.confl[k].i; j = bs.confl[k].j;
		e->sets[i].confl[ e->sets[i].nr_confl++ ] = j;
		e->sets[j].confl[ e->sets[j].nr_confl++ ] = i;
	}

	/* Clean up the allocated memory. */
	free(bs.nconfl);
	free(weight);
	free(bs.confl);
	free(bs.confl_sorted);
	free(setid_bonds); free(setid_angles); free(setid_dihedrals);
	free(setid_rigids); free(setid_exclusions);


	/* It's the end of the world as we know it... */
	return engine_err_ok;

}

/**
 * allocates a new dihedral, returns its id.
 */
int TissueForge::engine_dihedral_alloc(struct engine *e, Dihedral **out) {
    
    struct Dihedral *dummy;
	int dihedral_id = -1;
    
    /* Check inputs. */
    if (e == NULL) 
		return error(engine_err_null);
    
    // first check if we have any deleted dihedrals we can re-use
	if(e->nr_active_dihedrals < e->nr_dihedrals) {
        for(int i = 0; i < e->nr_dihedrals; ++i) {
            if(!(e->dihedrals[i].flags & DIHEDRAL_ACTIVE)) {
                dihedral_id = i;
                break;
            }
        }
        assert(dihedral_id >= 0 && dihedral_id < e->dihedrals_size);
	}
	else {
		/* Do we need to grow the dihedral array? */
		if (e->nr_dihedrals == e->dihedrals_size) {
			e->dihedrals_size *= 1.414;
			if ((dummy = (struct Dihedral *)malloc(sizeof(struct Dihedral) * e->dihedrals_size)) == NULL)
				return error(engine_err_malloc);
			memcpy(dummy, e->dihedrals, sizeof(struct Dihedral) * e->nr_dihedrals);
			free(e->dihedrals);
			e->dihedrals = dummy;
		}
		dihedral_id = e->nr_dihedrals;
		e->nr_dihedrals += 1;
	}

	bzero(&e->dihedrals[dihedral_id], sizeof(Dihedral));
    
    int result = e->dihedrals[dihedral_id].id = dihedral_id;

	*out = &e->dihedrals[dihedral_id];

	TF_Log(LOG_TRACE) << "Allocated dihedral: " << dihedral_id;
    
    return result;
}

/**
 * allocates a new angle, returns its id.
 */
int TissueForge::engine_angle_alloc (struct engine *e, Angle **out) {
    
    struct Angle *dummy;
	int angle_id = -1;
    
    /* Check inputs. */
    if(e == NULL)
    	return error(engine_err_null);
    
	// first check if we have any deleted angles we can re-use
    if(e->nr_active_angles < e->nr_angles) {
        for(int i = 0; i < e->nr_angles; ++i) {
            if(!(e->angles[i].flags & ANGLE_ACTIVE)) {
                angle_id = i;
                break;
            }
        }
        assert(angle_id >= 0 && angle_id < e->angles_size);
    }
    else {
		/* Do we need to grow the angles array? */
		if(e->nr_angles == e->angles_size) {
			e->angles_size *= 1.414;
			if((dummy = (struct Angle *)malloc(sizeof(struct Angle) * e->angles_size)) == NULL)
			return error(engine_err_malloc);
			memcpy(dummy, e->angles, sizeof(struct Angle) * e->nr_angles);
			free(e->angles);
			e->angles = dummy;
		}
		angle_id = e->nr_angles;
		e->nr_angles += 1;
	}

	bzero(&e->angles[angle_id], sizeof(Angle));
    
    int result = e->angles[angle_id].id = angle_id;

	*out = &e->angles[angle_id];
	
	TF_Log(LOG_TRACE) << "Allocated angle: " << angle_id;
    
    /* It's the end of the world as we know it. */
    return result;
}

/* Recursive quicksort for the exclusions. */
static void exclusion_qsort (struct engine *e,  int l, int r) {

	int i = l, j = r;
	int pivot_i = e->exclusions[ (l + r)/2 ].i;
	int pivot_j = e->exclusions[ (l + r)/2 ].j;
	struct exclusion temp;

	/* Too small? */
	if(r - l < 10) {

		/* Use Insertion Sort. */
		for(i = l+1 ; i <= r ; i++) {
			pivot_i = e->exclusions[i].i;
			pivot_j = e->exclusions[i].j;
			for(j = i-1 ; j >= l ; j--)
				if(e->exclusions[j].i > pivot_i ||
						(e->exclusions[j].i == pivot_i && e->exclusions[j].j > pivot_j)) {
					temp = e->exclusions[j];
					e->exclusions[j] = e->exclusions[j+1];
					e->exclusions[j+1] = temp;
				}
				else
					break;
		}

	}

	else {

		/* Partition. */
		while(i <= j) {
			while(e->exclusions[i].i < pivot_i ||
					(e->exclusions[i].i == pivot_i && e->exclusions[i].j < pivot_j))
				i += 1;
			while(e->exclusions[j].i > pivot_i ||
					(e->exclusions[j].i == pivot_i && e->exclusions[j].j > pivot_j))
				j -= 1;
			if(i <= j) {
				temp = e->exclusions[i];
				e->exclusions[i] = e->exclusions[j];
				e->exclusions[j] = temp;
				i += 1;
				j -= 1;
			}
		}

		/* Recurse. */
		if(l < j)
			exclusion_qsort(e, l, j);
		if(i < r)
			exclusion_qsort(e, i, r);

	}

}


/**
 * @brief Remove duplicate exclusions.
 *
 * @param e The #engine.
 *
 * @return The number of unique exclusions or < 0 on error (see #engine_err).
 */

int TissueForge::engine_exclusion_shrink(struct engine *e) {

	int j, k;

	/* Sort the exclusions. */
	exclusion_qsort(e, 0, e->nr_exclusions-1);

	/* Run through the exclusions and skip duplicates. */
	for(j = 0, k = 1 ; k < e->nr_exclusions ; k++)
		if(e->exclusions[k].j != e->exclusions[j].j ||
				e->exclusions[k].i != e->exclusions[j].i) {
			j += 1;
			e->exclusions[j] = e->exclusions[k];
		}

	/* Set the number of exclusions to j. */
	e->nr_exclusions = j+1;
	if((e->exclusions = (struct exclusion *)realloc(e->exclusions, sizeof(struct exclusion) * e->nr_exclusions)) == NULL)
		return error(engine_err_malloc);

	/* Go home. */
	return engine_err_ok;

}


/**
 * @brief Add a exclusioned interaction to the engine.
 *
 * @param e The #engine.
 * @param i The ID of the first #part.
 * @param j The ID of the second #part.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int TissueForge::engine_exclusion_add(struct engine *e, int i, int j) {

	struct exclusion *dummy;

	/* Check inputs. */
	if(e == NULL)
		return error(engine_err_null);

	/* Do we need to grow the exclusions array? */
	if(e->nr_exclusions == e->exclusions_size) {
		e->exclusions_size *= 1.414;
		if((dummy = (struct exclusion *)malloc(sizeof(struct exclusion) * e->exclusions_size)) == NULL)
			return error(engine_err_malloc);
		memcpy(dummy, e->exclusions, sizeof(struct exclusion) * e->nr_exclusions);
		free(e->exclusions);
		e->exclusions = dummy;
	}

	/* Store this exclusion. */
	if(i <= j) {
		e->exclusions[ e->nr_exclusions ].i = i;
		e->exclusions[ e->nr_exclusions ].j = j;
	}
	else {
		e->exclusions[ e->nr_exclusions ].i = j;
		e->exclusions[ e->nr_exclusions ].j = i;
	}
	e->nr_exclusions += 1;

	/* It's the end of the world as we know it. */
	return engine_err_ok;

}


/**
 * allocates a new bond, returns a pointer to it.
 */
int TissueForge::engine_bond_alloc (struct engine *e, Bond **out) {

    struct Bond *dummy;
    int bond_id = -1;
    
    /* Check inputs. */
    if(e == NULL)
        return error(engine_err_null);
    
    // first check if we have any deleted bonds we can re-use
    if(e->nr_active_bonds < e->nr_bonds) {
        for(int i = 0; i < e->nr_bonds; ++i) {
            if(!(e->bonds[i].flags & BOND_ACTIVE)) {
                bond_id = i;
                break;
            }
        }
        assert(bond_id >= 0 && bond_id < e->bonds_size);
    }
    else {
        /* Do we need to grow the bonds array? */
        if(e->nr_bonds == e->bonds_size) {
            e->bonds_size  *= 1.414;
            if((dummy = (struct Bond *)malloc(sizeof(struct Bond) * e->bonds_size)) == NULL)
                return error(engine_err_malloc);
            memcpy(dummy, e->bonds, sizeof(struct Bond) * e->nr_bonds);
            free(e->bonds);
            e->bonds = dummy;
        }
        bond_id = e->nr_bonds;
        e->nr_bonds += 1;
    }

    bzero(&e->bonds[bond_id], sizeof(Bond));
    
    int result = e->bonds[bond_id].id = bond_id;

    *out = &e->bonds[bond_id];

	TF_Log(LOG_TRACE) << "Allocated bond: " << bond_id;

    /* It's the end of the world as we know it. */
    return result;
}


/**
 * @brief Compute all bonded interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Does the same as #engine_bond_eval, #engine_angle_eval and
 * #engine_dihedral eval, yet all in one go to avoid excessive
 * updates of the particle forces.
 */

int TissueForge::engine_bonded_eval(struct engine *e) {

	FPTYPE epot_bond = 0.0, epot_angle = 0.0, epot_dihedral = 0.0, epot_exclusion = 0.0;
	struct space *s;
	struct Dihedral dtemp;
	struct Angle atemp;
	struct Bond btemp;
	struct exclusion etemp;
	int nr_dihedrals = e->nr_dihedrals, nr_bonds = e->nr_bonds;
	int nr_angles = e->nr_angles, nr_exclusions = e->nr_exclusions;
	int i, j, k;
	ticks tic;

	/* Bail if there are no bonded interaction. */
	if(nr_bonds == 0 && nr_angles == 0 && nr_dihedrals == 0 && nr_exclusions == 0)
		return engine_err_ok;

	/* Get a handle on the space. */
	s = &e->s;

	/* If in parallel... */
	if(e->nr_nodes > 1) {

		tic = getticks();

#pragma omp parallel for schedule(static), private(i,j,dtemp,atemp,btemp,etemp)
		for(k = 0 ; k < 4 ; k++) {

			if(k == 0) {
				/* Sort the dihedrals. */
				i = 0; j = nr_dihedrals-1;
				while(i < j) {
					while(i < nr_dihedrals &&
							s->partlist[e->dihedrals[i].i] != NULL &&
							s->partlist[e->dihedrals[i].j] != NULL &&
							s->partlist[e->dihedrals[i].k] != NULL &&
							s->partlist[e->dihedrals[i].l] != NULL)
						i += 1;
					while(j >= 0 &&
							(s->partlist[e->dihedrals[j].i] == NULL ||
									s->partlist[e->dihedrals[j].j] == NULL ||
									s->partlist[e->dihedrals[j].k] == NULL ||
									s->partlist[e->dihedrals[j].l] == NULL))
						j -= 1;
					if(i < j) {
						dtemp = e->dihedrals[i];
						e->dihedrals[i] = e->dihedrals[j];
						e->dihedrals[j] = dtemp;
					}
				}
				nr_dihedrals = i;
			}

			else if(k == 1) {
				/* Sort the angles. */
				i = 0; j = nr_angles-1;
				while(i < j) {
					while(i < nr_angles &&
							s->partlist[e->angles[i].i] != NULL &&
							s->partlist[e->angles[i].j] != NULL &&
							s->partlist[e->angles[i].k] != NULL)
						i += 1;
					while(j >= 0 &&
							(s->partlist[e->angles[j].i] == NULL ||
									s->partlist[e->angles[j].j] == NULL ||
									s->partlist[e->angles[j].k] == NULL))
						j -= 1;
					if(i < j) {
						atemp = e->angles[i];
						e->angles[i] = e->angles[j];
						e->angles[j] = atemp;
					}
				}
				nr_angles = i;
			}

			else if(k == 2) {
				/* Sort the bonds. */
				i = 0; j = nr_bonds-1;
				while(i < j) {
					while(i < nr_bonds &&
							s->partlist[e->bonds[i].i] != NULL &&
							s->partlist[e->bonds[i].j] != NULL)
						i += 1;
					while(j >= 0 &&
							(s->partlist[e->bonds[j].i] == NULL ||
						      s->partlist[e->bonds[j].j] == NULL))
						j -= 1;
					if(i < j) {
						btemp = e->bonds[i];
						e->bonds[i] = e->bonds[j];
						e->bonds[j] = btemp;
					}
				}
				nr_bonds = i;
			}

			else if(k == 3) {
				/* Sort the exclusions. */
				i = 0; j = nr_exclusions-1;
				while(i < j) {
					while(i < nr_exclusions &&
							s->partlist[e->exclusions[i].i] != NULL &&
							s->partlist[e->exclusions[i].j] != NULL)
						i += 1;
					while(j >= 0 &&
							(s->partlist[e->exclusions[j].i] == NULL ||
									s->partlist[e->exclusions[j].j] == NULL))
						j -= 1;
					if(i < j) {
						etemp = e->exclusions[i];
						e->exclusions[i] = e->exclusions[j];
						e->exclusions[j] = etemp;
					}
				}
				nr_exclusions = i;
			}

		}

		/* Stop the clock. */
		e->timers[engine_timer_bonded_sort] += getticks() - tic;

	}


	/* Do exclusions. */
	tic = getticks();
	if(exclusion_eval(e->exclusions, nr_exclusions, e, &epot_exclusion) < 0)
		return error(engine_err_exclusion);
	e->timers[engine_timer_exclusions] += getticks() - tic;

	/* Do bonds. */
	tic = getticks();
	if(bond_eval(e->bonds, nr_bonds, e, &epot_bond) < 0)
		return error(engine_err_bond);
	e->timers[engine_timer_bonds] += getticks() - tic;

	/* Do angles. */
	tic = getticks();
	if(angle_eval(e->angles, nr_angles, e, &epot_angle) < 0)
		return error(engine_err_angle);
	e->timers[engine_timer_angles] += getticks() - tic;

	/* Do dihedrals. */
	tic = getticks();
	if(dihedral_eval(e->dihedrals, nr_dihedrals, e, &epot_dihedral) < 0)
		return error(engine_err_dihedral);
	e->timers[engine_timer_dihedrals] += getticks() - tic;


	/* Store the potential energy. */
	s->epot += epot_bond + epot_angle + epot_dihedral + epot_exclusion;
	s->epot_bond += epot_bond;
	s->epot_angle += epot_angle;
	s->epot_dihedral += epot_dihedral;
	s->epot_exclusion += epot_exclusion;

	/* I'll be back... */
	return engine_err_ok;

}


/**
 * @brief Compute the dihedral interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int TissueForge::engine_dihedral_eval(struct engine *e) {

	FPTYPE epot = 0.0;
	struct space *s;
	struct Dihedral temp;
	int nr_dihedrals = e->nr_dihedrals, i, j;
#ifdef HAVE_OPENMP
	FPTYPE *eff;
	int nr_threads, cid, pid, gpid, k;
	struct part *p;
	struct cell *c;
	FPTYPE epot_local;
#endif

/* Get a handle on the space. */
	s = &e->s;

	/* Sort the dihedrals (if in parallel). */
	if(e->nr_nodes > 1) {
		i = 0; j = nr_dihedrals-1;
		while(i < j) {
			while(i < nr_dihedrals &&
					s->partlist[e->dihedrals[i].i] != NULL &&
					s->partlist[e->dihedrals[i].j] != NULL &&
					s->partlist[e->dihedrals[i].k] != NULL &&
					s->partlist[e->dihedrals[i].l] != NULL)
				i += 1;
			while(j >= 0 &&
					(s->partlist[e->dihedrals[j].i] == NULL ||
							s->partlist[e->dihedrals[j].j] == NULL ||
							s->partlist[e->dihedrals[j].k] == NULL ||
							s->partlist[e->dihedrals[j].l] == NULL))
				j -= 1;
			if(i < j) {
				temp = e->dihedrals[i];
				e->dihedrals[i] = e->dihedrals[j];
				e->dihedrals[j] = temp;
			}
		}
		nr_dihedrals = i;
	}

#ifdef HAVE_OPENMP

	/* Is it worth parallelizing? */
#pragma omp parallel private(k,nr_threads,c,p,cid,pid,gpid,eff,epot_local)
	if((e->flags & engine_flag_parbonded) &&
			((nr_threads = omp_get_num_threads()) > 1) &&
			(nr_dihedrals > engine_dihedrals_chunk)) {

		/* Init the local potential energy. */
		epot_local = 0.0;

		/* Allocate a buffer for the forces. */
		eff = (FPTYPE *)malloc(sizeof(FPTYPE) * 4 * s->nr_parts);
		bzero(eff, sizeof(FPTYPE) * 4 * s->nr_parts);

		/* Compute the dihedral interactions. */
		k = omp_get_thread_num();
		dihedral_evalf(&e->dihedrals[k*nr_dihedrals/nr_threads], (k+1)*nr_dihedrals/nr_threads - k*nr_dihedrals/nr_threads, e, eff, &epot_local);

		/* Write-back the forces (if anything was done). */
		for(cid = 0 ; cid < s->nr_real ; cid++) {
			c = &s->cells[ s->cid_real[cid] ];
			pthread_mutex_lock(&c->cell_mutex);
			for(pid = 0 ; pid < c->count ; pid++) {
				p = &c->parts[ pid ];
				gpid = p->id;
				for(k = 0 ; k < 3 ; k++)
					p->f[k] += eff[ gpid*4 + k ];
			}
			pthread_mutex_unlock(&c->cell_mutex);
		}
		free(eff);

		/* Aggregate the global potential energy. */
#pragma omp atomic
		epot += epot_local;

	}

	/* Otherwise, evaluate directly. */
	else if(omp_get_thread_num() == 0)
		dihedral_eval(e->dihedrals, nr_dihedrals, e, &epot);
#else
	if(dihedral_eval(e->dihedrals, nr_dihedrals, e, &epot) < 0)
		return error(engine_err_dihedral);
#endif

	/* Store the potential energy. */
	s->epot += epot;
	s->epot_dihedral += epot;

	/* I'll be back... */
	return engine_err_ok;

}


/**
 * @brief Compute the angled interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int TissueForge::engine_angle_eval(struct engine *e) {

	FPTYPE epot = 0.0;
	struct space *s;
	struct Angle temp;
	int nr_angles = e->nr_angles, i, j;
#ifdef HAVE_OPENMP
	FPTYPE *eff;
	int nr_threads, cid, pid, gpid, k;
	struct part *p;
	struct cell *c;
	FPTYPE epot_local;
#endif

/* Get a handle on the space. */
	s = &e->s;

	/* Sort the angles (if in parallel). */
	if(e->nr_nodes > 1) {
		i = 0; j = nr_angles-1;
		while(i < j) {
			while(i < nr_angles &&
					s->partlist[e->angles[i].i] != NULL &&
					s->partlist[e->angles[i].j] != NULL &&
					s->partlist[e->angles[i].k] != NULL)
				i += 1;
			while(j >= 0 &&
					(s->partlist[e->angles[j].i] == NULL ||
							s->partlist[e->angles[j].j] == NULL ||
							s->partlist[e->angles[j].k] == NULL))
				j -= 1;
			if(i < j) {
				temp = e->angles[i];
				e->angles[i] = e->angles[j];
				e->angles[j] = temp;
			}
		}
		nr_angles = i;
	}

#ifdef HAVE_OPENMP

	/* Is it worth parallelizing? */
#pragma omp parallel private(k,nr_threads,c,p,cid,pid,gpid,eff,epot_local)
	if((e->flags & engine_flag_parbonded) &&
			((nr_threads = omp_get_num_threads()) > 1) &&
			(nr_angles > engine_angles_chunk)) {

		/* Init the local potential energy. */
		epot_local = 0.0;

		/* Allocate a buffer for the forces. */
		eff = (FPTYPE *)malloc(sizeof(FPTYPE) * 4 * s->nr_parts);
		bzero(eff, sizeof(FPTYPE) * 4 * s->nr_parts);

		/* Compute the angle interactions. */
		k = omp_get_thread_num();
		angle_evalf(&e->angles[k*nr_angles/nr_threads], (k+1)*nr_angles/nr_threads - k*nr_angles/nr_threads, e, eff, &epot_local);

		/* Write-back the forces (if anything was done). */
		for(cid = 0 ; cid < s->nr_real ; cid++) {
			c = &s->cells[ s->cid_real[cid] ];
			pthread_mutex_lock(&c->cell_mutex);
			for(pid = 0 ; pid < c->count ; pid++) {
				p = &c->parts[ pid ];
				gpid = p->id;
				for(k = 0 ; k < 3 ; k++)
					p->f[k] += eff[ gpid*4 + k ];
			}
			pthread_mutex_unlock(&c->cell_mutex);
		}
		free(eff);

		/* Aggregate the global potential energy. */
#pragma omp atomic
		epot += epot_local;

	}

	/* Otherwise, evaluate directly. */
	else if(omp_get_thread_num() == 0)
		angle_eval(e->angles, nr_angles, e, &epot);
#else
	if(angle_eval(e->angles, nr_angles, e, &epot) < 0)
		return error(engine_err_angle);
#endif

	/* Store the potential energy. */
	s->epot += epot;
	s->epot_angle += epot;

	/* I'll be back... */
	return engine_err_ok;

}


/**
 * @brief Compute the exclusioned interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int TissueForge::engine_exclusion_eval(struct engine *e) {

	FPTYPE epot = 0.0;
	struct space *s;
	int nr_exclusions = e->nr_exclusions, i, j;
	struct exclusion temp;
#ifdef HAVE_OPENMP
	FPTYPE *eff;
	int nr_threads, cid, pid, gpid, k;
	struct part *p;
	struct cell *c;
	FPTYPE epot_local;
#endif

/* Get a handle on the space. */
	s = &e->s;

	/* Sort the exclusions (if in parallel). */
	if(e->nr_nodes > 1) {
		i = 0; j = nr_exclusions-1;
		while(i < j) {
			while(i < nr_exclusions &&
					s->partlist[e->exclusions[i].i] != NULL &&
					s->partlist[e->exclusions[i].j] != NULL)
				i += 1;
			while(j >= 0 &&
					(s->partlist[e->exclusions[j].i] == NULL ||
							s->partlist[e->exclusions[j].j] == NULL))
				j -= 1;
			if(i < j) {
				temp = e->exclusions[i];
				e->exclusions[i] = e->exclusions[j];
				e->exclusions[j] = temp;
			}
		}
		nr_exclusions = i;
	}

#ifdef HAVE_OPENMP

	/* Is it worth parallelizing? */
#pragma omp parallel private(k,nr_threads,c,p,cid,pid,gpid,eff,epot_local)
	if((e->flags & engine_flag_parbonded) &&
			((nr_threads = omp_get_num_threads()) > 1) &&
			(nr_exclusions > engine_exclusions_chunk)) {

		/* Init the local potential energy. */
		epot_local = 0.0;

		/* Allocate a buffer for the forces. */
		eff = (FPTYPE *)malloc(sizeof(FPTYPE) * 4 * s->nr_parts);
		bzero(eff, sizeof(FPTYPE) * 4 * s->nr_parts);

		/* Compute the exclusioned interactions. */
		k = omp_get_thread_num();
		exclusion_evalf(&e->exclusions[k*nr_exclusions/nr_threads], (k+1)*nr_exclusions/nr_threads - k*nr_exclusions/nr_threads, e, eff, &epot_local);

		/* Write-back the forces (if anything was done). */
		for(cid = 0 ; cid < s->nr_real ; cid++) {
			c = &s->cells[ s->cid_real[cid] ];
			pthread_mutex_lock(&c->cell_mutex);
			for(pid = 0 ; pid < c->count ; pid++) {
				p = &c->parts[ pid ];
				gpid = p->id;
				for(k = 0 ; k < 3 ; k++)
					p->f[k] += eff[ gpid*4 + k ];
			}
			pthread_mutex_unlock(&c->cell_mutex);
		}
		free(eff);

		/* Aggregate the global potential energy. */
#pragma omp atomic
		epot += epot_local;

	}

	/* Otherwise, evaluate directly. */
	else if(omp_get_thread_num() == 0)
		exclusion_eval(e->exclusions, nr_exclusions, e, &epot);
#else
	if(exclusion_eval(e->exclusions, nr_exclusions, e, &epot) < 0)
		return error(engine_err_exclusion);
#endif

	/* Store the potential energy. */
	s->epot += epot;
	s->epot_exclusion += epot;

	/* I'll be back... */
	return engine_err_ok;

}


/**
 * @brief Compute the bonded interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int TissueForge::engine_bond_eval(struct engine *e) {

	FPTYPE epot = 0.0;
	struct space *s;
	int nr_bonds = e->nr_bonds, i, j;
	struct Bond temp;
#ifdef HAVE_OPENMP
	FPTYPE *eff;
	int nr_threads, cid, pid, gpid, k;
	struct part *p;
	struct cell *c;
	FPTYPE epot_local;
#endif

/* Get a handle on the space. */
	s = &e->s;

	/* Sort the bonds (if in parallel). */
	if(e->nr_nodes > 1) {
		i = 0; j = nr_bonds-1;
		while(i < j) {
			while(i < nr_bonds &&
					s->partlist[e->bonds[i].i] != NULL &&
					s->partlist[e->bonds[i].j] != NULL)
				i += 1;
			while(j >= 0 &&
					(s->partlist[e->bonds[j].i] == NULL ||
							s->partlist[e->bonds[j].j] == NULL))
				j -= 1;
			if(i < j) {
				temp = e->bonds[i];
				e->bonds[i] = e->bonds[j];
				e->bonds[j] = temp;
			}
		}
		nr_bonds = i;
	}

#ifdef HAVE_OPENMP

	/* Is it worth parallelizing? */
#pragma omp parallel private(k,nr_threads,c,p,cid,pid,gpid,eff,epot_local)
	if((e->flags & engine_flag_parbonded) &&
			((nr_threads = omp_get_num_threads()) > 1) &&
			(nr_bonds > engine_bonds_chunk)) {

		/* Init the local potential energy. */
		epot_local = 0.0;

		/* Allocate a buffer for the forces. */
		eff = (FPTYPE *)malloc(sizeof(FPTYPE) * 4 * s->nr_parts);
		bzero(eff, sizeof(FPTYPE) * 4 * s->nr_parts);

		/* Compute the bonded interactions. */
		k = omp_get_thread_num();
		bond_evalf(&e->bonds[k*nr_bonds/nr_threads], (k+1)*nr_bonds/nr_threads - k*nr_bonds/nr_threads, e, eff, &epot_local);

		/* Write-back the forces (if anything was done). */
		for(cid = 0 ; cid < s->nr_real ; cid++) {
			c = &s->cells[ s->cid_real[cid] ];
			pthread_mutex_lock(&c->cell_mutex);
			for(pid = 0 ; pid < c->count ; pid++) {
				p = &c->parts[ pid ];
				gpid = p->id;
				for(k = 0 ; k < 3 ; k++)
					p->f[k] += eff[ gpid*4 + k ];
			}
			pthread_mutex_unlock(&c->cell_mutex);
		}
		free(eff);

		/* Aggregate the global potential energy. */
#pragma omp atomic
		epot += epot_local;

	}

	/* Otherwise, evaluate directly. */
	else if(omp_get_thread_num() == 0)
		bond_eval(e->bonds, nr_bonds, e, &epot);
#else
	if(bond_eval(e->bonds, nr_bonds, e, &epot) < 0)
		return error(engine_err_bond);
#endif

	/* Store the potential energy. */
	s->epot += epot;
	s->epot_bond += epot;

	/* I'll be back... */
	return engine_err_ok;

}

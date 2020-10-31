import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance, self_capped_distance
import numpy as np
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
import pandas as pd
import multiprocessing


class CustomAnalysis():
    def __init__(self, input_dict, analysis_dict):
        """
        This class is initialized with the dictionary containing the system file (top, traj) and the
        analysis dict containing what part of the system you are interested in and the start,
        stop, stride.

        :param input_dict:
        :param analysis_dict:
        """
        self.top = input_dict['top']
        self.traj = input_dict['traj']
        self.u = mda.Universe(self.top, self.traj, in_memory=False)
        _sel = self.u.select_atoms(analysis_dict['system'])
        _residxs = np.unique(_sel.resindices)
        self.nres = len(_residxs)

        self.start = analysis_dict['start']
        self.stop = analysis_dict['stop']
        self.stride = analysis_dict['stride']

        if self.stop == None:
            self.nframes = (self.u.trajectory.n_frames - self.start) // self.stride
        else:
            self.nframes = (self.stop - self.start) // self.stride
        #print(self.nframes)

class SaltBridges(CustomAnalysis):
    """ Compute salt bridge.
    Usage:
    sb = SaltBridges(dictionaries)
    matrix  = sb.run()
    """
    def __init__(self, input_dict, analysis_dict, sb_dict):
        super(SaltBridges, self).__init__(input_dict, analysis_dict)
        # Removing spaces for erroneous input
        # Acidic selection
        if len(sb_dict['cter_sel'].strip()) != 0:
            self.sel_acidic = f"({sb_dict['sel_acidic']}) or ({sb_dict['cter_sel']})"
        else:
            self.sel_acidic = sb_dict['sel_acidic']

        # Basic selection
        if (len(sb_dict['nter_sel'].strip()) != 0) and (len(sb_dict['histidine_sel'].strip()) != 0):
            self.sel_basic = f"({sb_dict['sel_basic']}) or ({sb_dict['nter_sel']}) or \
                                ({sb_dict['histidine_sel']})"
        elif len(sb_dict['nter_sel'].strip()) != 0:
            self.sel_basic = f"({sb_dict['sel_basic']}) or ({sb_dict['nter_sel']})"
        elif len(sb_dict['histidine_sel'].strip()) != 0:
            self.sel_basic = f"({sb_dict['sel_basic']}) or ({sb_dict['histidine_sel']})"
        else:
            self.sel_basic = sb_dict['sel_basic']

        self.sb_file = sb_dict['sb_file']
        self.sb_cutoff = sb_dict['sb_dist']


    def run(self):
        # creating the atom group
        _acidic = self.u.select_atoms(self.sel_acidic)
        _basic = self.u.select_atoms(self.sel_basic)
        #defining the matrix
        _salt_bridge = np.zeros((self.nres, self.nres), int)

        for ts in self.u.trajectory[self.start:self.stop:self.stride]:
            _dists = capped_distance(_acidic.positions, _basic.positions,
                                     max_cutoff=self.sb_cutoff,
                                     return_distances=False,
                                     box=_acidic.dimensions)
            # At least for salt bridges, we need only that one bridge is formed.
            # Therefore we will map the pairs of atoms indices into the pairs of
            # residues and take the unique pair to compose the matrix.
            # It is symmetric, but we can use the upper diagonal part.
            _residxs = []
            for resAcid, resBasic in zip(_acidic[_dists[:, 0]].resindices, _basic[_dists[:, 1]].resindices):
                _residxs.append([resAcid, resBasic])
            _residxs = np.unique(np.array(_residxs), axis=0)  # shifting back
            for r in _residxs:
                _salt_bridge[r[0], r[1]] += 1

        SB_matrix = (_salt_bridge + _salt_bridge.T) * 100 / self.nframes
        np.savetxt(self.sb_file, SB_matrix)

        return SB_matrix

class HydrophobicInteractions(CustomAnalysis):
    """ hydrophobic interactions.
    Usage:
    hc = HydrophobicInteraction(dictionaries)
    matrix = hc.run()"""
    def __init__(self, input_dict, analysis_dict, hc_dict):
        super(HydrophobicInteractions, self).__init__(input_dict, analysis_dict)
        # Removing spaces for erroneous input

        self.sidechain_sel = f"({hc_dict['hydrophobic_sel']}) and not ({hc_dict['backbone_sel']})"
        self.hc_cutoff = hc_dict['hc_cutoff']
        self.hc_file = hc_dict['hc_file']


    def run(self):
        _sel = self.u.select_atoms(self.sidechain_sel)
        _hc_matrix = np.zeros((self.nres, self.nres), int)

        for ts in self.u.trajectory[self.start:self.stop:self.stride]:
            hydroDists, _ = self_capped_distance(_sel.center_of_mass(compound='residues', pbc=True),
                                                 max_cutoff=self.hc_cutoff,
                                                 box=_sel.dimensions)

            for pair in hydroDists:
                res1 = _sel.residues[pair[0]].resindex
                res2 = _sel.residues[pair[1]].resindex
                _hc_matrix[res1, res2] += 1
                if res1 != res2:
                    _hc_matrix[res2, res1] += 1

        normed_hc_matrix = (_hc_matrix) * 100 / self.nframes
        np.savetxt(self.hc_file, normed_hc_matrix)
        return normed_hc_matrix


# Definitions of the functions for HBonds analysis, parallelized-ish

def parallel_hb(PDB, XTC, donor_sel, acceptor_sel, start=0, stop=None, stride=1,
                up_sels=False, dist=3.5, angle=120.0, output_csv="hb"):
    """
    Perform hbond analysis for a chunk of the trajectory

    :param PDB: pdb/psf file
    :param XTC: traj file
    :param start:   first frame of computation
    :param stop:    last frame of computation (uses python convention, so it is not computed)
    :param stride:  step
    :param donor_sel:       selection of atoms involved in hbond (def: "protein")
    :param acceptor_sel:    selection of atoms involved in hbond (def: "protein")
    :param dist:    distance criterion for hbond
    :param angle:   angle criterion for hbond
    :param accepts: additional acceptor atoms
    :param dons:    additional donor atoms
    :param output_csv:    beginning of
    :return:
    save the temporary df as csv.
    """
    print('Initialising: ' + multiprocessing.current_process().name)
    input_uni = mda.Universe(PDB, XTC, inmemory=False)
    input_hb = HBA(input_uni,
                   update_selections=up_sels,
                   donors_sel=donor_sel,
                   acceptors_sel=acceptor_sel,
                   d_a_cutoff=dist,
                   d_h_a_angle_cutoff=angle)
    #    input_hb.donors_sel=donor_sel
    #    input_hb.acceptors_sel=acceptor_sel
    #    input_hb.hydrogens_sel = input_hb.guess_hydrogens("name ?H*")

    #     print(f"distance: {input_hb.distance}")
    print(f"{multiprocessing.current_process().name} doing from {start} to {stop}, every {stride} frame")
    input_hb.run(start=start, stop=stop, step=stride)
    hb_csv = f'{output_csv}_{start}_{stop}.csv'
    print('Creating table: ' + multiprocessing.current_process().name)
    _h_results = input_hb.hbonds
    _df = pd.DataFrame(_h_results, index=_h_results[:, 0],
                       columns=['time', 'donor_idx', 'hydrogen_idx', 'acceptor_idx', 'distance', 'angle'])

    _df.to_csv(hb_csv)
    return True


def defineIntervals(nframe, ncores=0):
    """
    define the chunks of the sim to perform the analysis

    :param nframe: total frame of the trajectory
    :param ncores: available number of cores.
    Using ncores=0, means you are using all the cores available
    :return: list of start-stop frames.
    """

    if not ncores:
        print("using all the available cores")
        ncores = multiprocessing.cpu_count()
    traj_per_core = nframe // ncores
    intervals = []
    print(f"Using {ncores}")
    for i in range(ncores):
        if i == ncores - 1:
            # the None value allows the computation up to the last frame
            intervals.append((i * traj_per_core, None))
        else:
            intervals.append((i * traj_per_core, (i + 1) * traj_per_core))
    return intervals


def HBondInteraction(PDB, XTC, donor_sel, acceptor_sel, nframe, output_csv, start=0, stop=None, stride=1,
                     up_sels=False, dist=3.5, angle=120.0, ncores=0):
    """
    launch the analysis on the whole trajectory

    :param PDB: pdb/psf file
    :param XTC: traj file
    :param sel1:    selection of atoms involved in hbond (def: "protein")
    :param sel2:    selection of atoms involved in hbond (def: "protein")
    :param stride:  step
    :param output_csv: absolut path and beginning of the file name
    :param ncores:    available cores
    """
    intervals = defineIntervals(nframe, ncores)
    print("THe intervals are:", intervals)
    print(f"Pooling with {ncores} core(s)...")
    pool = multiprocessing.Pool()
    for start, stop in intervals:
        # print("Starting frames: ", start, "to", stop)
        pool.apply_async(parallel_hb,
                         args=(PDB, XTC, donor_sel, acceptor_sel),
                         kwds={'start': start,
                               'stop': stop,
                               'stride': stride,
                               'dist': 3.5,
                               'angle': 120.0,
                               'output_csv': f"{output_csv}"},
                         )
    pool.close()
    pool.join()


def MergingHBDataframe(output_csv, nframe, ncores=0):
    """
    merging the csv file

    :param output_csv:
    :param nframe:
    :param ncores:
    :return:
    """
    # merging data frames
    intervals = defineIntervals(nframe, ncores)
    df_list = []
    # We merge the dataframes using only columns of interest, like: time, donor_idx and acceptor_idx
    for start, stop in intervals:
        df_list.append(pd.read_csv(f'{output_csv}_{start}_{stop}.csv',
                                   usecols=['time', 'donor_idx', 'acceptor_idx'],
                                   dtype=int)
                       )

    hb_df = pd.concat(df_list, ignore_index=True)
    return hb_df


def HBMatrix(*, u, nres, nframes, hb_df, stride=1, hb_file="hb.dat"):
    """
    from dataframe to matrix

    :param u:
    :param nres:
    :param nframe: total number of frames, regardless of the stride
    :param stride:
    :param hb_df:
    :param hb_file:

    :return: save the matrix
    """
    # Associating the atom index to the resid-resname
    hb_df['donor_resname'] = u.atoms[hb_df['donor_idx']].resnames
    hb_df['acceptor_resname'] = u.atoms[hb_df['acceptor_idx']].resnames
    hb_df['donor_resid'] = u.atoms[hb_df['donor_idx']].resids
    hb_df['acceptor_resid'] = u.atoms[hb_df['acceptor_idx']].resids

    # removing non necessary columns
    hb_df.drop(['donor_idx', 'donor_resname', 'acceptor_idx', 'acceptor_resname'], axis=1, inplace=True)
    hb_df['check_pair'] = hb_df.apply(
        lambda row: '-'.join(sorted([f"{row['donor_resid']}", f"{row['acceptor_resid']}"])), axis=1)
    hb_df.drop_duplicates(subset=["time", "check_pair"], keep='first', inplace=True)

    # cleaning the dataframe
    hb_df.drop('check_pair', axis=1, inplace=True)

    # creating the matrix
    hbMatrix = np.zeros((nres, nres))

    for index, row in hb_df.iterrows():
        time, d_residx, a_residx = row
        i = d_residx
        j = a_residx
        hbMatrix[i, j] += 1
        if (i != j):
            hbMatrix[j, i] += 1

    fullHB = hbMatrix * 100 / (nframes // stride)
    # Saving matrices:
    np.savetxt(hb_file, fullHB)
#     return fullHB




# def saltBridge(u, nres, sel_acidic, sel_basic, start=0, stop=None, stride=1, sb_file='sb.dat', cutoff=4.5):
#     """
#     compute the saltbridges between charged group defined in `sel_acidic` and `sel_basic`.
#
#     :param u:    MDAnalysis universe
#     :param nres: number of residues for initialising the matrix
#     :param sel_acidic: (str) selection of atoms of charged acidic group
#     :param sel_basic:  (str) selection of atoms of charged basic group
#     :param start:      (int) initial frame [Default: 0]
#     :param stop:       (int) last frame [Default: None]
#     :param stride:     (int) compute analysis every `stride` frames
#     :param sb_file:    (str) filename to save the matrix
#     :param cutoff:     (float) cutoff for distances
#     :return: save the matrix
#     """
#     # reference groups (first frame of the trajectory, but you could also use a
#     # separate PDB, eg crystal structure)
#     acidic = u.select_atoms(sel_acidic)
#     basic = u.select_atoms(sel_basic)
#
#
#     salt_bridge = np.zeros((nres, nres), int)
#
#     # normalising the trajectory
#     if stop == None:
#         NFRAME = (u.trajectory.n_frames - start) // stride
#     else:
#         NFRAME = (stop - start)//stride
#
#     for ts in u.trajectory[start:stop:stride]:
#         dists = capped_distance(acidic.positions, basic.positions, max_cutoff=cutoff, return_distances=False, box=acidic.dimensions)
#         # At least for salt bridges, we need only that one bridge is formed.
#         # Therefore we will map the pairs of atoms indices into the pairs of
#         # residues and take the unique pair to compose the matrix.
#         # It is symmetric, but we can use the upper diagonal part.
#         residxs = []
#         for resAcid, resBasic in zip(acidic[dists[:,0]].resindices, basic[dists[:,1]].resindices):
#              residxs.append([resAcid, resBasic])
#         residxs = np.unique(np.array(residxs), axis=0) # shifting back
#         for r in residxs:
#             salt_bridge[r[0], r[1]] += 1
#
#
#     SSBB = (salt_bridge+salt_bridge.T)*100 / NFRAME
#     np.savetxt(sb_file, SSBB)
# #     return SSBB

# def HydrophobicInteraction(u, nres, sel, start=0, stop=None, stride=1, hc_file="hc.dat", cutoff=5.5):
#     """
#     compute the hydrophobic interaction.
#
#     :param u:    MDAnalysis universe
#     :param nres: number of residues for initialising the matrix
#     :param sel:        (str) hydrophobic residues from which to compute sidechain distances
#     :param start:      (int) initial frame [Default: 0]
#     :param stop:       (int) last frame [Default: None]
#     :param stride:     (int) compute analysis every `stride` frames
#     :param hc_file:    (str) filename to save the matrix
#     :param cutoff:     (float) cutoff for distances
#     :return: save the matrix
#     """
#     hcMatrix = np.zeros((nres, nres), int)
#
#     for ts in u.trajectory[::stride]:
#         hydroDists, _ = self_capped_distance(sel.center_of_mass(compound='residues', pbc=True),
#                                              max_cutoff=cutoff,
#                                              box=sel.dimensions)
#
#         for pair in hydroDists:
#             res1 = sel.residues[pair[0]].resindex
#             res2 = sel.residues[pair[1]].resindex
#             hcMatrix[res1, res2] += 1
#             if res1 != res2:
#                 hcMatrix[res2, res1] += 1
#
#
#     if stop == None:
#         NFRAME = (u.trajectory.n_frames - start) // stride
#     else:
#         NFRAME = (stop - start)//stride
#
#     normed_hcMatrix = (hcMatrix)*100/NFRAME
#     np.savetxt(hc_file, normed_hcMatrix)
# #     return normed_hcMatrix

# class ComputeHydrogenBonds(CustomAnalysis):
#     """ hydrophobic interactions.
#     Usage:
#     hc = HydrophobicInteraction(dictionaries)
#     matrix = hc.run()"""
#     def __init__(self, input_dict, analysis_dict, hb_dict):
#         super(ComputeHydrogenBonds, self).__init__(input_dict, analysis_dict)
#         # Removing spaces for erroneous input
#
#         self.acceptors_sel = hb_dict['acceptors']
#         self.donors_sel = hb_dict['donors']
#
#         self.hb_distance = hb_dict['d_a_dist']
#         self.angle = hb_dict['angle']
#         self.hb_file = hb_dict['hb_file']
#         self.update = hb_dict['update_sel']
#
#         self.tmp_output = "tmp_hb",
#
#     def parallel_hb(self, top, traj, donor_sel, acceptor_sel, start=0, stop=None, stride=1,
#                 up_sels=False, dist=3.5, angle=120.0, output_csv="hb_tmp"):
#         """
#         Perform hbond analysis for a chunk of the trajectory
#
#         :param PDB: pdb/psf file
#         :param XTC: traj file
#         :param start:   first frame of computation
#         :param stop:    last frame of computation (uses python convention, so it is not computed)
#         :param stride:  step
#         :param donor_sel:       selection of atoms involved in hbond (def: "protein")
#         :param acceptor_sel:    selection of atoms involved in hbond (def: "protein")
#         :param dist:    distance criterion for hbond
#         :param angle:   angle criterion for hbond
#         :param accepts: additional acceptor atoms
#         :param dons:    additional donor atoms
#         :param output_csv:    beginning of
#         :return:
#         save the temporary df as csv.
#         """
#         print('Initialising: ' + multiprocessing.current_process().name)
#         input_uni = mda.Universe(top, traj, inmemory=False)
#         input_hb = HBA(input_uni,
#                        update_selections=up_sels,
#                        donors_sel=donor_sel,
#                        acceptors_sel=acceptor_sel,
#                        d_a_cutoff=dist,
#                        d_h_a_angle_cutoff=angle
#                        )
#
#         print(f"{multiprocessing.current_process().name} doing from {start} to {stop}, every {stride} frame")
#         input_hb.run(start=start, stop=stop, step=stride)
#         hb_csv = f'{output_csv}_{start}_{stop}.csv'
#         print('Creating table: ' + multiprocessing.current_process().name)
#         _h_results = input_hb.hbonds
#         _df = pd.DataFrame(_h_results, index=_h_results[:, 0],
#                            columns=['time', 'donor_idx', 'hydrogen_idx', 'acceptor_idx', 'distance', 'angle'])
#
#         _df.to_csv(hb_csv)
#         return True
#
#     def defineIntervals(self, ncores=0):
#         """
#         define the chunks of the sim to perform the analysis
#
#         :param nframe: total frame of the trajectory
#         :param ncores: available number of cores.
#         Using ncores=0, means you are using all the cores available
#         :return: list of start-stop frames.
#         """
#
#         if not ncores:
#             print("using all the available cores")
#             ncores = multiprocessing.cpu_count()
#         traj_per_core = self.nframes // ncores
#         intervals = []
#         print(f"Using {ncores}")
#         for i in range(ncores):
#             if i == ncores - 1:
#                 # the None value allows the computation up to the last frame
#                 intervals.append((i * traj_per_core, None))
#             else:
#                 intervals.append((i * traj_per_core, (i + 1) * traj_per_core))
#         return intervals
#
#
#     def runHBonds(self, ncores=0):
#         """
#         launch the analysis on the whole trajectory
#
#         :param PDB: pdb/psf file
#         :param XTC: traj file
#         :param sel1:    selection of atoms involved in hbond (def: "protein")
#         :param sel2:    selection of atoms involved in hbond (def: "protein")
#         :param stride:  step
#         :param output_csv: absolut path and beginning of the file name
#         :param ncores:    available cores
#         """
#         intervals = self.defineIntervals(ncores)
#         print("THe intervals are:", intervals)
#         print(f"Pooling with {ncores} core(s)...")
#         pool = multiprocessing.Pool()
#         for _start, _stop in intervals:
#             # print("Starting frames: ", start, "to", stop)
#             pool.apply_async(self.parallel_hb,
#                              args=(self.top, self.traj,
#                                    self.donors_sel, self.acceptors_sel),
#                              kwds={'start': _start,
#                                    'stop': _stop,
#                                    'stride': self.stride,
#                                    'dist': self.hb_distance,
#                                    'angle': self.angle,
#                                    'output_csv': self.tmp_output},
#                              )
#         pool.close()
#         pool.join()
#
#     def MergingHBDataframe(self, ncores=0):
#         """
#         merging the csv file
#
#         :param output_csv:
#         :param nframe:
#         :param ncores:
#         :return:
#         """
#         # merging data frames
#         intervals = defineIntervals(self.nframes, ncores)
#         df_list = []
#         # We merge the dataframes using only columns of interest, like: time, donor_idx and acceptor_idx
#         for start, stop in intervals:
#             df_list.append(pd.read_csv(f'{self.tmp_output}_{start}_{stop}.csv',
#                                        usecols=['time', 'donor_idx', 'acceptor_idx'],
#                                        dtype=int)
#                            )
#
#         hb_df = pd.concat(df_list, ignore_index=True)
#         return hb_df
#
#     def HBMatrix(u, nres, nframe, hb_df, stride=1, hb_file="hb.dat"):
#         """
#         from dataframe to matrix
#
#         :param u:
#         :param nres:
#         :param nframe: total number of frames, regardless of the stride
#         :param stride:
#         :param hb_df:
#         :param hb_file:
#         :return: save the matrix
#         """
#         # Associating the atom index to the resid-resname
#         hb_df['donor_resname'] = u.atoms[hb_df['donor_idx']].resnames
#         hb_df['acceptor_resname'] = u.atoms[hb_df['acceptor_idx']].resnames
#         hb_df['donor_resid'] = u.atoms[hb_df['donor_idx']].resids
#         hb_df['acceptor_resid'] = u.atoms[hb_df['acceptor_idx']].resids
#
#         # removing non necessary columns
#         hb_df.drop(['donor_idx', 'donor_resname', 'acceptor_idx', 'acceptor_resname'], axis=1, inplace=True)
#         hb_df['check_pair'] = hb_df.apply(
#             lambda row: '-'.join(sorted([f"{row['donor_resid']}", f"{row['acceptor_resid']}"])), axis=1)
#         hb_df.drop_duplicates(subset=["time", "check_pair"], keep='first', inplace=True)
#
#         # cleaning the dataframe
#         hb_df.drop('check_pair', axis=1, inplace=True)
#
#         # creating the matrix
#         hbMatrix = np.zeros((nres, nres))
#
#         for index, row in hb_df.iterrows():
#             time, d_residx, a_residx = row
#             i = d_residx
#             j = a_residx
#             hbMatrix[i, j] += 1
#             if (i != j):
#                 hbMatrix[j, i] += 1
#
#         fullHB = hbMatrix * 100 / (nframe // stride)
#         # Saving matrices:
#         np.savetxt(hb_file, fullHB)
#
#     #     return fullHB
#     def run(self):
#         pass

# class ComputeHydrogenBonds(CustomAnalysis):
#     """ hydrophobic interactions.
#     Usage:
#     hc = HydrophobicInteraction(dictionaries)
#     matrix = hc.run()"""
#
#     def __init__(self, input_dict, analysis_dict, hb_dict):
#         super(ComputeHydrogenBonds, self).__init__(input_dict, analysis_dict)
#         # Removing spaces for erroneous input
#
#         self.acceptors_sel = hb_dict['acceptors']
#         self.donors_sel = hb_dict['donors']
#
#         self.hb_distance = hb_dict['d_a_dist']
#         self.angle = hb_dict['angle']
#         self.hb_file = hb_dict['hb_file']
#         self.update = hb_dict['update_sel']
#
#         self.tmp_output = "tmp_hb"
#
#     def dc(self, a, b):
#         print(a, "---", b)
#
#     def parallel_hb(self, start=0, stop=None, stride=1, output_csv="hb_tmp"):
#         #         top, traj, donor_sel, acceptor_sel, start=0, stop=None, stride=1,
#         #                up_sels=False, dist=3.5, angle=120.0,
#         """
#         Perform hbond analysis for a chunk of the trajectory
#
#         :param PDB: pdb/psf file
#         :param XTC: traj file
#         :param start:   first frame of computation
#         :param stop:    last frame of computation (uses python convention, so it is not computed)
#         :param stride:  step
#         :param donor_sel:       selection of atoms involved in hbond (def: "protein")
#         :param acceptor_sel:    selection of atoms involved in hbond (def: "protein")
#         :param dist:    distance criterion for hbond
#         :param angle:   angle criterion for hbond
#         :param accepts: additional acceptor atoms
#         :param dons:    additional donor atoms
#         :param output_csv:    beginning of
#         :return:
#         save the temporary df as csv.
#         """
#
#         print('Initialising: ' + multiprocessing.current_process().name)
#         input_uni = mda.Universe(self.top, self.traj, inmemory=False)
#         print(input_uni.atoms.n_atoms)
#         input_hb = HBA(input_uni,
#                        update_selections=self.update,
#                        donors_sel=self.donors_sel,
#                        acceptors_sel=self.acceptors_sel,
#                        d_a_cutoff=self.hb_distance,
#                        d_h_a_angle_cutoff=self.angle
#                        )
#         print(input_uni.atoms.n_atoms)
#         print(f"{multiprocessing.current_process().name} doing from {start} to {stop}, every {stride} frame")
#         input_hb.run(start=start, stop=stop, step=stride)
#         hb_csv = f'{output_csv}_{start}_{stop}.csv'
#         print('Creating table: ' + multiprocessing.current_process().name)
#         _h_results = input_hb.hbonds
#         _df = pd.DataFrame(_h_results, index=_h_results[:, 0],
#                            columns=['time', 'donor_idx', 'hydrogen_idx', 'acceptor_idx', 'distance', 'angle'])
#
#         _df.to_csv(hb_csv)
#         return True
#
#     def defineIntervals(self, ncores=0):
#         """
#         define the chunks of the sim to perform the analysis
#
#         :param nframe: total frame of the trajectory
#         :param ncores: available number of cores.
#         Using ncores=0, means you are using all the cores available
#         :return: list of start-stop frames.
#         """
#
#         if not ncores:
#             print("using all the available cores")
#             ncores = multiprocessing.cpu_count()
#         traj_per_core = self.nframes // ncores
#         intervals = []
#         print(f"Using {ncores}")
#         for i in range(ncores):
#             if i == ncores - 1:
#                 # the None value allows the computation up to the last frame
#                 intervals.append((i * traj_per_core, None))
#             else:
#                 intervals.append((i * traj_per_core, (i + 1) * traj_per_core))
#         return intervals
#
#     def runHBonds(self, ncores=0):
#         """
#         launch the analysis on the whole trajectory
#
#         :param PDB: pdb/psf file
#         :param XTC: traj file
#         :param sel1:    selection of atoms involved in hbond (def: "protein")
#         :param sel2:    selection of atoms involved in hbond (def: "protein")
#         :param stride:  step
#         :param output_csv: absolut path and beginning of the file name
#         :param ncores:    available cores
#         """
#         intervals = self.defineIntervals(ncores)
#         print("THe intervals are:", intervals)
#         print(f"Pooling with {ncores} core(s)...")
#         pool = multiprocessing.Pool()
#         for _start, _stop in intervals:
#             pool.apply_async(self.dc, args=(_start, _stop))
#             # pool.apply_async(self.parallel_hb,
#             #                  kwds={'start': _start,
#             #                        'stop': _stop,
#             #                        'stride': 1},
#             #                  )
#             #pool.apply_async(print, args="ciao")
#         pool.close()
#         pool.join()
#
#     def MergingHBDataframe(self, ncores=0):
#         """
#         merging the csv file
#
#         :param output_csv:
#         :param nframe:
#         :param ncores:
#         :return:
#         """
#         # merging data frames
#         intervals = defineIntervals(self.nframes, ncores)
#         df_list = []
#         # We merge the dataframes using only columns of interest, like: time, donor_idx and acceptor_idx
#         for start, stop in intervals:
#             df_list.append(pd.read_csv(f'{self.tmp_output}_{start}_{stop}.csv',
#                                        usecols=['time', 'donor_idx', 'acceptor_idx'],
#                                        dtype=int)
#                            )
#
#         hb_df = pd.concat(df_list, ignore_index=True)
#         return hb_df
#
#     def HBMatrix(u, nres, nframe, hb_df, stride=1, hb_file="hb.dat"):
#         """
#         from dataframe to matrix
#
#         :param u:
#         :param nres:
#         :param nframe: total number of frames, regardless of the stride
#         :param stride:
#         :param hb_df:
#         :param hb_file:
#         :return: save the matrix
#         """
#         # Associating the atom index to the resid-resname
#         hb_df['donor_resname'] = u.atoms[hb_df['donor_idx']].resnames
#         hb_df['acceptor_resname'] = u.atoms[hb_df['acceptor_idx']].resnames
#         hb_df['donor_resid'] = u.atoms[hb_df['donor_idx']].resids
#         hb_df['acceptor_resid'] = u.atoms[hb_df['acceptor_idx']].resids
#
#         # removing non necessary columns
#         hb_df.drop(['donor_idx', 'donor_resname', 'acceptor_idx', 'acceptor_resname'], axis=1, inplace=True)
#         hb_df['check_pair'] = hb_df.apply(
#             lambda row: '-'.join(sorted([f"{row['donor_resid']}", f"{row['acceptor_resid']}"])), axis=1)
#         hb_df.drop_duplicates(subset=["time", "check_pair"], keep='first', inplace=True)
#
#         # cleaning the dataframe
#         hb_df.drop('check_pair', axis=1, inplace=True)
#
#         # creating the matrix
#         hbMatrix = np.zeros((nres, nres))
#
#         for index, row in hb_df.iterrows():
#             time, d_residx, a_residx = row
#             i = d_residx
#             j = a_residx
#             hbMatrix[i, j] += 1
#             if (i != j):
#                 hbMatrix[j, i] += 1
#
#         fullHB = hbMatrix * 100 / (nframe // stride)
#         # Saving matrices:
#         np.savetxt(hb_file, fullHB)
#
#     #     return fullHB
#     def run(self):
#         pass





# def parallel_hb(PDB, XTC, start, stop, sel1, sel2, stride=1, up_sel1=False, up_sel2=False, dist=3.5, angle=120.0,
#                 accepts=[], dons=[], output_csv="hb"):
#     """
#     Perform hbond analysis for a chunk of the trajectory
#
#     :param PDB: pdb/psf file
#     :param XTC: traj file
#     :param start:   first frame of computation
#     :param stop:    last frame of computation (uses python convention, so it is not computed)
#     :param stride:  step
#     :param sel1:    selection of atoms involved in hbond (def: "protein")
#     :param sel2:    selection of atoms involved in hbond (def: "protein")
#     :param up_sel1: update the selection 1
#     :param up_sel2: update the selection 2
#     :param dist:    distance criterion for hbond
#     :param angle:   angle criterion for hbond
#     :param accepts: additional acceptor atoms
#     :param dons:    additional donor atoms
#     :param output_csv:    beginning of
#     :return:
#     save the temporary df as csv.
#     """
#     print('Initialising: ' + multiprocessing.current_process().name)
#     input_uni = mda.Universe(PDB, XTC, inmemory=False)
#     input_hb = hb.HydrogenBondAnalysis(input_uni, sel1, sel2,
#                                        update_selection1=up_sel1,
#                                        update_selection2=up_sel2,
#                                        distance=dist, angle=angle,
#                                        acceptors=accepts,
#                                        donors=dons,
#                                        pbc=True)
#     #     print(f"distance: {input_hb.distance}")
#     print(start, stop, stride)
#     input_hb.run(start=start, stop=stop, step=stride)
#     hb_csv = f'{output_csv}_{start}_{stop}.csv'
#     print('Creating table: ' + multiprocessing.current_process().name)
#     input_hb.generate_table()
#     df = pd.DataFrame.from_records(input_hb.table)
#     df.to_csv(hb_csv)
#
# def defineIntervals(nframe, ncores=0):
#     """
#     define the chunks of the sim to perform the analysis
#
#     :param nframe: total frame of the trajectory
#     :param ncores: available number of cores.
#     Using ncores=0, means you are using all the cores available
#     :return: list of start-stop frames.
#     """
#
#     if not ncores:
#         print("using all the available cores")
#         ncores = multiprocessing.cpu_count()
#     traj_per_core = nframe // ncores
#     intervals = []
#     print(f"Using {ncores}")
#     for i in range(ncores):
#         if i == ncores-1:
#             # the None value allows the computation up to the last frame
#             intervals.append((i*traj_per_core, None))
#         else:
#             intervals.append((i*traj_per_core, (i+1)*traj_per_core))
#     return intervals
#
# def HBondInteraction(PDB, XTC, sel1, sel2, nframe, stride, output_csv, ncores=0, lipidic_acceptors=[], lipidic_donors=[]):
#     """
#     launch the analysis on the whole trajectory
#
#     :param PDB: pdb/psf file
#     :param XTC: traj file
#     :param sel1:    selection of atoms involved in hbond (def: "protein")
#     :param sel2:    selection of atoms involved in hbond (def: "protein")
#     :param stride:  step
#     :param output_csv: absolut path and beginning of the file name
#     :param ncores:    available cores
#     """
#     intervals = defineIntervals(nframe, ncores)
#     print(intervals)
#     print(f"Pooling with {ncores}...")
#     pool = multiprocessing.Pool()
#     for start, stop in intervals:
#         print(start, stop)
#         pool.apply_async(parallel_hb, args=(PDB, XTC, start, stop, sel1, sel2),
#                                       kwds={'stride': stride,
#                                             'dist': 3.5,
#                                             'angle': 120.0,
#                                             'accepts': lipidic_acceptors,
#                                             'dons': lipidic_donors,
#                                             'output_csv': f"{output_csv}"})
#     pool.close()
#     pool.join()
#
# def MergingHBDataframe(output_csv, nframe, ncores=0):
#     """
#     merging the csv file
#
#     :param output_csv:
#     :param nframe:
#     :param ncores:
#     :return:
#     """
#     # merging data frames
#     intervals = defineIntervals(nframe, ncores)
#     df_list = []
#     for start, stop in intervals:
#         df_list.append(pd.read_csv(f'{output_csv}_{start}_{stop}.csv', usecols=['time', 'donor_index', 'acceptor_index']))
#
#     hb_df = pd.concat(df_list, ignore_index=True)
#     return hb_df
#
#
# def HBMatrix(u, nres, nframe, hb_df, stride=1, hb_file="hb.dat"):
#     """
#     from dataframe to matrix
#
#     :param u:
#     :param nres:
#     :param nframe: total number of frames, regardless of the stride
#     :param stride:
#     :param hb_df:
#     :param hb_file:
#     :return: save the matrix
#     """
#     # creating the dictionary to convert atom index into resindex
#     index2resindex = {}
#     for atom in u.atoms:
#         index2resindex[atom.index] = atom.resindex
#     # creating column with donor and acceptor resindices
#     hb_df['d_resindex'] = hb_df['donor_index'].map(index2resindex)
#     hb_df['a_resindex'] = hb_df['acceptor_index'].map(index2resindex)
#     # removing unnecessary columns
#     hb_df.drop(['donor_index', 'acceptor_index'], axis=1, inplace=True)
#     # creating a string to assess the equivalence of the pairs like 11-12 and 12-11
#     hb_df['check_pair'] = hb_df.apply(
#         lambda row: '-'.join(sorted([f"{row['d_resindex']:.0f}", f"{row['a_resindex']:.0f}"])), axis=1)
#     # removing the same hb at the same time step
#     hb_df.drop_duplicates(subset=['time', 'check_pair'], inplace=True)
#     # cleaning the dataframe
#     hb_df.drop('check_pair', axis=1, inplace=True)
#
#     # creating the matrix
#     hbMatrix = np.zeros((nres, nres))
#
#     for index, row in hb_df.iterrows():
#         time, d_residx, a_residx = row
#         i = int(d_residx)
#         j = int(a_residx)
#         hbMatrix[i, j] += 1
#         if (i != j):
#             hbMatrix[j, i] += 1
#
#     fullHB = hbMatrix * 100 / (nframe // stride)
#     # Saving matrices:
#     np.savetxt(hb_file, fullHB)
# #     return fullHB

import pickle

SPECTRA_2 = {
    'HSQC': {
        'peaks': ((('N', 0), ('H', 0)), ),
        'polarities': (1, ),
        'dimension': 2,
        'name': 'HSQC'
    },
    'HNCO': {
        'peaks': ((('N', 0), ('H', 0), ('C', -1)), ),
        'polarities': (1, ),
        'dimension': 3,
        'name': 'HNCO'
    },
    'HNCACB': {
        'peaks': ((('N', 0), ('H', 0), ('CA', 0)),
                  (('N', 0), ('H', 0), ('CB', 0)),
                  (('N', 0), ('H', 0), ('CA', -1)),
                  (('N', 0), ('H', 0), ('CB', -1))),
        'polarities': (1, -1, 1, -1),
        'dimension': 3,
        'name': 'HNCACB'
    },
    'HNCA': {
        'peaks': ((('N', 0), ('H', 0), ('CA', 0)),
                  (('N', 0), ('H', 0), ('CA', -1))),
        'polarities': (1, 1),
        'dimension': 3,
        'name': 'HNCA'
    },
    'HN(CO)CACB': {
        'peaks': ((('N', 0), ('H', 0), ('CA', -1)),
                  (('N', 0), ('H', 0), ('CB', -1))),
        'polarities': (1, 1),
        'dimension': 3,
        'name': 'HN(CO)CACB'
    },
    'HN(CO)CA': {
        'peaks': ((('N', 0), ('H', 0), ('CA', -1)), ),
        'polarities': (1, ),
        'dimension': 3,
        'name': 'HN(CO)CA'
    },
    'HN(CA)CO': {
        'peaks': ((('N', 0), ('H', 0), ('C', 0)),
                  (('N', 0), ('H', 0), ('C', -1))),
        'polarities': (1, 1),
        'dimension': 3,
        'name': 'HN(CA)CO'
    }
}

SCHEDULE = (
    ('HSQC', 0),
    ('HNCO', 0),
    ('HN(CO)CA', 0),
    ('HNCA', 1),
    ('HNCA', 0),
    ('HN(CO)CACB', 0),
    ('HN(CO)CACB', 1),
    ('HN(CA)CO', 1),
    ('HN(CA)CO', 0),
    ('HNCACB', 2),
    ('HNCACB', 0),
    ('HNCACB', 3),
    ('HNCACB', 1))

SPECTRA_ORDER = ('HSQC', 'HNCO', 'HN(CO)CA', 'HNCA',
                 'HN(CO)CACB', 'HN(CA)CO', 'HNCACB')

PARAMS = {
    'noise': {'C': 0.16, 'CA': 0.16, 'CB': 0.16, 'N': 0.16, 'H': 0.016},
    'tolerance': {'C': 0.4, 'CA': 0.4, 'CB': 0.4, 'N': 0.4, 'H': 0.03},
    'std': {'C': 0.1, 'CA': 0.1, 'CB': 0.1, 'N': 0.1, 'H': 0.03/4},
    'threshold': 2.,      # number of acceptable standard deviations for prior
    'fa_threshold': 3., 
    'half_tol_threshold': 0.3,
    'n_samples': 100
}

with open('chemical_shifts.dt', 'r') as fin:
    DATA = pickle.load(fin)

with open('spectra.pickle', 'r') as fin:
    INTERPRETER = pickle.load(fin)

THREE_TO_ONE = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'ASX': 'B',
    'CYS': 'C',
    'GLU': 'E',
    'GLN': 'Q',
    'GLX': 'Z',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V'
}

ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.iteritems()}


def three_to_one(sequence):
    return ''.join(THREE_TO_ONE[s] for s in sequence)


def one_to_three(sequence):
    return [ONE_TO_THREE[s] for s in sequence]


WINDOW_ORDER = ('N', 'H', 'C', 'CA', 'CB')
WINDOW_DICT = {'N': 0,
               'H': 1,
               'C': 2,
               'CA': 2,
               'CB': 2}

################
# SPIN SYSTEMS #
################

SPIN_SYSTEM_ORDER = (('N', 0), ('H', 0), ('CA', 0),
                     ('CA', -1), ('CB', 0), ('CB', -1))

# TESTS = {}
# TESTS['1gb1'] = 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
# TESTS['bmr4391'] = 'MKCKICNFDTCRAGELKVCASGEKYCFKESWREARGTRIERGCAATCPKGSVYG' + \
#                    'LYVLCCTTDDCN'
# TESTS['bmr4752'] = 'MEVNKKQLADIFGASIRTIQNWQEQGMPVLRGGGKGNEVLYDSAAVIKWYAERD' + \
#                    'AEIENEKLRREVEE'
# TESTS['bmr4144'] = 'MKVTDHILVPKHEIVPKEEVEEILKRYNIKIQQLPKIYEDDPVIQEIGAKEGDV' + \
#                    'VRVIRKSPTAGVSIAYRLVIKRII'
# TESTS['bmr4579'] = 'AVQELGRENQSLQIKHTQALNRKWAEDNEVQNCMACGKGFSVTVRRHHCRQCGN' + \
#                    'IFCAECSAKNALTPSSKKPVRVCDACFNDLQG'
# TESTS['bmr4316'] = 'MDMVLAKTVVLAASAVGAGAAMIAGIGPGVGQGYAAGKAVESVARQPEAKGDII' + \
#                    'STMVLGQAIAESTGIYSLVIALILLYANPFVGLLG'
# TESTS['bmr4288'] = 'GSELPWAVKSEDKAKYDAIFDSLSPVDGFLSGDKVKPVLLNSKLPVEILGRVWE' + \
#                    'LSDIDHDGKLDRDEFAVAMFLVYCALEKEPVPMSLPPALVPPSKRKTWVVS'
# TESTS['bmr4670'] = 'HMASRDQVKASHILIKHQGSRRKASWKDPEGKIILTTTREAAVEQLKSIREDIV' + \
#                    'SGKANFEEVATRVSDCSSAKRGGDLGSFGRGQMQKPFEEATYALKVGDISDIVD' + \
#                    'TDSGVHIIKRTA'
# TESTS['bmr4929'] = 'MEGVDPAVEEAAFVADDVSNIIKESIDAVLQNQQYSEAKVSQWTSSCLEHCIKR' + \
#                    'LTALNKPFKYVVTCIIMQKNGAGLHTAASCWWDSTTDGSRTVRWENKSMYCICT' + \
#                    'VFGLAI'
# TESTS['bmr4302'] = 'DKQPVKVLVGKNFEDVAFDEKKNVFVEFYAPWCGHCKQLAPIWDKLGETYKDHE' + \
#                    'NIVIAKMDSTANEVEAVKVHSFPTLKFFPASADRTVIDYNGERTLDGFKKFLES' + \
#                    'GGQDGAG'
# TESTS['bmr4353'] = 'KQTLPERTGYFLLLQHEDEVLLAQRPPSGLWGGLYCFPQFADEESLRQWLAQRQ' + \
#                    'IAADMLTQLTAFRHTFSHFHLDIVPMWLPVSSFTGCMDEGNALWYNLAQPPSVG' + \
#                    'LAAPVERLLQQLRTGAPV'
# TESTS['bmr4027'] = 'TLSILVAHDLQRVIGFENQLPWHLPNDLKHVKKLSTGHTLVMGRKTFESIGKPL' + \
#                    'PNRRNVVLTSDTSFNVEGVDVIHSIEDIYQLPGHVFIFGGQTLYEEMIDKVDDM' + \
#                    'YITVIEGKFRGDTFFPPYTFEDWEVASSVEGKLDEKNTIPHTFLHLIRKK'
# TESTS['bmr4318'] = 'MKLYIYDHCPYCLKARMIFGLKNIPVELHVLLNDDAETPTRMVGQKQVPILQKD' + \
#                    'DSRYMPESMDIVHYVDKLDGKPLLTGKRSPAIEEWLRKVNGYANKLLLPRFAKS' + \
#                    'AFDEFSTPAARKYFVDKKEASAGNFADLLAHSDGLIKNISDDLRALDKLIVKPN' + \
#                    'AVNGELSEDDIQLFPLLRNLTLVAGINWPSRVADYRDNMAKQTQINLLSSMAI'

TESTS = {}
TESTS['1gb1'] = './data/NMRStar/1gb1.txt'
TESTS['bmr4391'] = './data/NMRStar/4391.txt'
TESTS['bmr4752'] = './data/NMRStar/4752.txt'
TESTS['bmr4144'] = './data/NMRStar/4144.txt'
TESTS['bmr4579'] = './data/NMRStar/4579.txt'
TESTS['bmr4316'] = './data/NMRStar/4316.txt'
TESTS['bmr4288'] = './data/NMRStar/4288.txt'
TESTS['bmr4670'] = './data/NMRStar/4670.txt'
TESTS['bmr4929'] = './data/NMRStar/4929.txt'
TESTS['bmr4302'] = './data/NMRStar/4302.txt'
TESTS['bmr4353'] = './data/NMRStar/4353.txt'
TESTS['bmr4207'] = './data/NMRStar/4027.txt'
TESTS['bmr4318'] = './data/NMRStar/4318.txt'
TESTS['sh2'] = './data/NMRStar/SH2.txt'

IPASS_SETTINGS = {
    'tolerance': {
        'N': 0.4,
        'HN': 0.04,
        'H': 0.04,
        'CA': 0.4,
        'CB': 0.4,
        'C': 0.4
    },
    'std': {
        'N': 0.4/2.5,
        'HN': 0.04/2.5,
        'H': 0.04/2.5,
        'CA': 0.2/2.5,
        'CB': 0.4/2.5,
        'C': 0.4/2.5
    },
    'q': {
        'N': 0.4/2.5,
        'HN': 0.04/2.5,
        'H': 0.04/2.5,
        'CA': 0.2/2.5,
        'CB': 0.4/2.5,
        'C': 0.4/2.5
    }
}

SPIN_SYSTEM_TESTS = {}
SPIN_SYSTEM_TESTS['CASKIN'] = {
  'ppm': './data/ChemicalShifts_NMRView/CASKIN_ppm.out',
  'ss': './data/SpinSystems/CASKIN_SS.txt',
  'protein': './data/NMRStar/CASKIN.txt',
  'sequence': 'GSSHHHHHHSSGLVPRGSSLKVRALKDFWNLHDPTALNVRAGDVITVLEQHPDGRWKGH' + \
              'IHESQRGTDRIGYFPPGIVEVVSKR',
  'shift': 0
}
SPIN_SYSTEM_TESTS['HACS1'] = {
  'ppm': './data/ChemicalShifts_NMRView/HACS1_ppm.out',
  'ss': './data/SpinSystems/HACS1_SS.txt',
  'protein': './data/NMRStar/HACS1.txt',
  'sequence': 'GPFCGRARVHTDFTPSPYDTDSLKIKKGDIIDIICKTPMGMWTGMLNNKVGNFKFIYVDVISE',
  'shift': 0
}
SPIN_SYSTEM_TESTS['TM1112'] = {
  'ppm': './data/ChemicalShifts_NMRView/TM1112_ppm.out',
  'ss': './data/SpinSystems/TM1112_SS.txt',
  'protein': './data/NMRStar/TM1112.txt',
  'sequence': 'MEVKIEKPTPEKLKELSVEKWPIWEKEVSEFDWYYDTNETCYILEGKVEVTTEDGKKYV' + \
              'IEKGDLVTFPKGLRCRWKVLEPVRKHYNLF',
  'shift': 0
}
SPIN_SYSTEM_TESTS['VRAR'] = {
  'ppm': './data/ChemicalShifts_NMRView/VRAR_ppm.out',
  'ss': './data/SpinSystems/VRAR_SS.txt',
  'protein': './data/NMRStar/VRAR.txt',
  'sequence': 'GSSHHHHHHSSGLVPRGSHMKKRAELYEMLTEREMEILLLIAKGYSNQEIASASHITIK' + \
              'TVKTHVSNILSKLEVQDRTQAVIYAFQHNLIQ',
  'shift': 19
}

# Demo Names
DEMO_NAME_OOB = 'SDK Out of Box Demo'
DEMO_NAME_3DPC = '3D People Counting'
DEMO_NAME_VITALS = 'Vital Signs with People Tracking'
DEMO_NAME_LRPD = 'Long Range People Detection'
DEMO_NAME_MT = 'Mobile Tracker'
DEMO_NAME_SOD = 'Small Obstacle Detection'

# Different methods to color the points 
COLOR_MODE_SNR = 'SNR'
COLOR_MODE_HEIGHT = 'Height'
COLOR_MODE_DOPPLER = 'Doppler'
COLOR_MODE_TRACK = 'Associated Track'


# Com Port names
CLI_XDS_SERIAL_PORT_NAME = 'XDS110 Class Application/User UART'
DATA_XDS_SERIAL_PORT_NAME = 'XDS110 Class Auxiliary Data Port'
CLI_SIL_SERIAL_PORT_NAME = 'Enhanced COM Port'
DATA_SIL_SERIAL_PORT_NAME = 'Standard COM Port'


# Configurables
MAX_POINTS = 1000
MAX_PERSISTENT_FRAMES = 10

MAX_VITALS_PATIENTS = 2
NUM_FRAMES_PER_VITALS_PACKET = 15
NUM_VITALS_FRAMES_IN_PLOT = 150
NUM_HEART_RATES_FOR_MEDIAN = 10


# Magic Numbers for Target Index TLV
TRACK_INDEX_WEAK_SNR = 253 # Point not associated, SNR too weak
TRACK_INDEX_BOUNDS = 254 # Point not associated, located outside boundary of interest
TRACK_INDEX_NOISE = 255 # Point not associated, considered as noise


# Defined TLV's
MMWDEMO_OUTPUT_MSG_DETECTED_POINTS                      = 1
MMWDEMO_OUTPUT_MSG_RANGE_PROFILE                        = 2
MMWDEMO_OUTPUT_MSG_NOISE_PROFILE                        = 3
MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP               = 4
MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP               = 5
MMWDEMO_OUTPUT_MSG_STATS                                = 6
MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO            = 7
MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP     = 8
MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS                    = 9

MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS                     = 1000
MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST           = 1010
MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX             = 1011
MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT            = 1012
MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS                    = 1020
MMWDEMO_OUTPUT_MSG_PRESCENCE_INDICATION                 = 1021
MMWDEMO_OUTPUT_MSG_OCCUPANCY_STATE_MACHINE              = 1030

MMWDEMO_OUTPUT_MSG_VITALSIGNS                           = 1040

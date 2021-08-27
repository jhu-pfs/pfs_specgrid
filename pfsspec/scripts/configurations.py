from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.data.rbfgrid import RbfGrid
from pfsspec.surveys.sdssdatasetbuilder import SdssDatasetBuilder
from pfsspec.stellarmod.modelgriddatasetbuilder import ModelGridDatasetBuilder
from pfsspec.data.survey import Survey
from pfsspec.surveys.sdssseguesurveyreader import SdssSegueSurveyReader
from pfsspec.stellarmod.bosz import Bosz
from pfsspec.stellarmod.boszgridreader import BoszGridReader
#### I added in phoenix
from pfsspec.stellarmod.phoenix import Phoenix
from pfsspec.stellarmod.phoenixgridreader import PhoenixGridReader

from pfsspec.stellarmod.kuruczgrid import KuruczGrid
from pfsspec.pipelines.sdssbasicpipeline import SdssBasicPipeline
from pfsspec.pipelines.stellarmodelpipeline import StellarModelPipeline
from pfsspec.obsmod.gausspsf import GaussPsf
from pfsspec.obsmod.pcapsf import PcaPsf
from pfsspec.obsmod.pfsobservation import PfsObservation
from pfsspec.obsmod.simpleobservation import SimpleObservation
from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.stellarmod.modelgridfit import ModelGridFit
from pfsspec.stellarmod.modelgridconv import ModelGridConv
from pfsspec.stellarmod.modelpcagridbuilder import ModelPcaGridBuilder
from pfsspec.stellarmod.modelrbfgridbuilder import ModelRbfGridBuilder
from pfsspec.etc.etcpcapsfimporter import EtcPcaPsfImporter

from pfsspec.stellarmod.modelspectrum import ModelSpectrum
IMPORT_CONFIGURATIONS = {
    'grid': {
        'bosz': BoszGridReader
    },
        'grid': {
        'phoenix': PhoenixGridReader
        },
    'survey': {
        'segue': SdssSegueSurveyReader
    },
    'psf': {
        'etc': EtcPcaPsfImporter
    }
}

CONV_CONFIGURATIONS = {
    'grid': {
        'bosz': {
            'class': ModelGridConv,
            'config': Bosz()
        }
    }
}

PSF_CONFIGURATIONS = {
    'gauss': GaussPsf
}

FIT_CONFIGURATIONS = {
    'grid': {
        'bosz': {
            'class': ModelGridFit,
            'config': Bosz()
        }
    },
        'grid': {
        'phoenix': {
            'class': ModelGridFit,
            'config': Phoenix()
        }
    }
}

PCA_CONFIGURATIONS = {
    'grid': {
        'bosz': {
            'class': ModelPcaGridBuilder,
            'config': Bosz()
        }
    },
        'grid': {
        'bosz': {
            'class': ModelPcaGridBuilder,
            'config': Phoenix()
        }
    }
}

RBF_CONFIGURATIONS = {
    'grid': {
        'bosz': {
            'class': ModelRbfGridBuilder,
            'config': Bosz()
        }
    },
    'grid': {
        'bosz': {
            'class': ModelRbfGridBuilder,
            'config': Phoenix()
        }
    }
}

MODEL_PIPELINE_TYPES = {
    'pfs': {
        'pipeline': StellarModelPipeline,
        'obsmod': PfsObservation
    },
    'sdss': {
        'pipeline': StellarModelPipeline,
        'obsmod': SimpleObservation
    },
}

PREPARE_CONFIGURATIONS = {
    'survey': {
        'sdss': {
            'builder': SdssDatasetBuilder,
            'survey': Survey,
            'pipelines': {
                'basic': {
                    'pipeline': SdssBasicPipeline
                }
            }
        }
    },
    'model': {
        # TODO
        # 'kurucz': {
        #     'builder': ModelGridDatasetBuilder,
        #     'grid': KuruczGrid,
        #     'pipelines': MODEL_PIPELINE_TYPES
        # },
        'bosz': {
            'builder': ModelGridDatasetBuilder,
            'config': Bosz(),
            'grid': ArrayGrid,
            'pipelines': MODEL_PIPELINE_TYPES
        },
        'bosz-rbf': {
            'builder': ModelGridDatasetBuilder,
            'config': Bosz(pca=True),
            'grid': RbfGrid,
            'pipelines': MODEL_PIPELINE_TYPES
        },
        'phoenix': {
            'builder': ModelGridDatasetBuilder,
            'config': Phoenix(),
            'grid': ArrayGrid,
            'pipelines': MODEL_PIPELINE_TYPES
        },
        'phoenix-rbf': {
            'builder': ModelGridDatasetBuilder,
            'config': Phoenix(pca=True),
            'grid': RbfGrid,
            'pipelines': MODEL_PIPELINE_TYPES
        }
    }
}

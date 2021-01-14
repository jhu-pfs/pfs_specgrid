from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.surveys.sdssdatasetbuilder import SdssDatasetBuilder
from pfsspec.stellarmod.modelgriddatasetbuilder import ModelGridDatasetBuilder
from pfsspec.surveys.survey import Survey
from pfsspec.stellarmod.bosz import Bosz
from pfsspec.stellarmod.boszgridreader import BoszGridReader
from pfsspec.stellarmod.kuruczgrid import KuruczGrid
from pfsspec.pipelines.sdssbasicpipeline import SdssBasicPipeline
from pfsspec.pipelines.stellarmodelpipeline import StellarModelPipeline
from pfsspec.obsmod.pfsobservation import PfsObservation
from pfsspec.obsmod.simpleobservation import SimpleObservation
from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.stellarmod.modelgridfit import ModelGridFit
from pfsspec.stellarmod.modelpcagridbuilder import ModelPcaGridBuilder
from pfsspec.stellarmod.modelrbfgridbuilder import ModelRbfGridBuilder

IMPORT_CONFIGURATIONS = {
    'grid': {
        'bosz': BoszGridReader
    }
}
FIT_CONFIGURATIONS = {
    'grid': {
        'bosz': {
            'class': ModelGridFit,
            'config': Bosz()
        }
    }
}

PCA_CONFIGURATIONS = {
    'grid': {
        'bosz': {
            'class': ModelPcaGridBuilder,
            'config': Bosz()
        }
    }
}

RBF_CONFIGURATIONS = {
    'grid': {
        'bosz': {
            'class': ModelRbfGridBuilder,
            'config': Bosz()
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
            # TODO: imput grid type? array? rbf? pca?
            'builder': ModelGridDatasetBuilder,
            'config': Bosz(),
            'grid': ArrayGrid,
            'pipelines': MODEL_PIPELINE_TYPES
        }
    }
}

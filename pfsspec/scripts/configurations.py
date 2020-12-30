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
from pfsspec.stellarmod.boszgridcontinuumfit import BoszGridContinuumFit
from pfsspec.stellarmod.boszpcagridbuilder import BoszPcaGridBuilder

IMPORT_CONFIGURATIONS = {
    'grid': {
        'bosz': BoszGridReader
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
            'config': Bosz,
            'pipelines': MODEL_PIPELINE_TYPES
        }
    }
}

FIT_CONFIGURATIONS = {
    'grid': {
        'bosz': BoszGridContinuumFit
    }
}

PCA_CONFIGURATIONS = {
    'grid': {
        'bosz': BoszPcaGridBuilder,
    }
}

# RBF_CONFIGURATIONS = {
#     'grid': {
#         'bosz': BoszRbfGridBuilder,
#     }
# }
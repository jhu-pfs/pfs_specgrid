from pfsspec.surveys.sdssdatasetbuilder import SdssDatasetBuilder
from pfsspec.stellarmod.modelgriddatasetbuilder import ModelGridDatasetBuilder
from pfsspec.surveys.survey import Survey
from pfsspec.stellarmod.kuruczgrid import KuruczGrid
from pfsspec.stellarmod.boszmodelgrid import BoszModelGrid
from pfsspec.pipelines.sdssbasicpipeline import SdssBasicPipeline
from pfsspec.pipelines.stellarmodelpipeline import StellarModelPipeline
from pfsspec.obsmod.pfsobservation import PfsObservation
from pfsspec.obsmod.simpleobservation import SimpleObservation
from pfsspec.stellarmod.boszgridcontinuumfit import BoszGridContinuumFit
from pfsspec.stellarmod.boszpcagridbuilder import BoszPCAGridBuilder
from pfsspec.stellarmod.logchebyshevcontinuummodel import LogChebyshevContinuumModel

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
        'kurucz': {
            'builder': ModelGridDatasetBuilder,
            'grid': KuruczGrid,
            'pipelines': MODEL_PIPELINE_TYPES
        },
        'bosz': {
            'builder': ModelGridDatasetBuilder,
            'grid': BoszModelGrid,
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
        'bosz': BoszPCAGridBuilder,
    }
}
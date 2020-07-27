from pfsspec.surveys.sdssdatasetbuilder import SdssDatasetBuilder
from pfsspec.stellarmod.modelgriddatasetbuilder import ModelGridDatasetBuilder
from pfsspec.surveys.survey import Survey
from pfsspec.stellarmod.kuruczgrid import KuruczGrid
from pfsspec.stellarmod.boszgrid import BoszGrid
from pfsspec.pipelines.sdssbasicpipeline import SdssBasicPipeline
from pfsspec.pipelines.stellarmodelpipeline import StellarModelPipeline
from pfsspec.obsmod.pfsobservation import PfsObservation
from pfsspec.obsmod.sdssobservation import SdssObservation

MODEL_PIPELINE_TYPES = {
    'pfs': {
        'pipeline': StellarModelPipeline,
        'obsmod': PfsObservation
    },
    'sdss': {
        'pipeline': StellarModelPipeline,
        'obsmod': SdssObservation
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
            'grid': BoszGrid,
            'pipelines': MODEL_PIPELINE_TYPES
        }
    }
}

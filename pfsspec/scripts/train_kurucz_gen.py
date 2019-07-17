#!/usr/bin/env python

from keras import optimizers

from pfsspec.data.dataset import Dataset
from pfsspec.ml.dnn.keras.densegenerative import DenseGenerative
from pfsspec.ml.dnn.keras.cnngenerative import CnnGenerative
from pfsspec.stellarmod.kuruczgenerativeaugmenter import KuruczGenerativeAugmenter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=str, help="Training set or data file\n")
    parser.add_argument("--out", type=str, help="Model directory\n")
    parser.add_argument("--name", type=str, help="Model name prefix\n")
    parser.add_argument('--labels', type=str, nargs='+', help='Labels to train for\n')
    parser.add_argument('--wave', action='store_true', help='Include wavelength vector in training.\n')
    parser.add_argument('--gpus', type=str, help='GPUs to use\n')
    parser.add_argument('--type', type=str, help='Type of network\n')
    parser.add_argument('--levels', type=int, help='Number of levels\n')
    parser.add_argument('--units', type=int, help='Number of units\n')
    parser.add_argument('--split', type=float, default=0.5, help='Training/validation split\n')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs\n')
    parser.add_argument('--batch', type=int, default=None, help='Batch size\n')
    parser.add_argument('--patience', type=int, default=None, help='Number of epochs to wait before early stop.\n')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function')
    parser.add_argument('--aug', action='store_true', help='Augment data.\n')
    return parser.parse_args()

def train_dnn(args):
    if args.type == 'dense':
        model = DenseGenerative()
    elif args.type == 'cnn':
        model = CnnGenerative()
    else:
        raise NotImplementedError()

    if args.levels is not None:
        model.levels = args.levels
    if args.units is not None:
        model.units = args.units

    model.include_wave = args.wave
    model.gpus = args.gpus
    model.validation_split = args.split
    model.patience = args.patience
    model.epochs = args.epochs
    model.loss = args.loss
    model.generate_name()

    ######################
    # Override loss
    # model.loss = max_absolute_error
    model.dropout_rate = 0
    model.activation = 'relu'
    model.optimizer = optimizers.SGD(lr=0.1)
    ######################

    labels, coeffs = parse_labels_coeffs(args)

    outdir = model.name + '_' + '_'.join(labels)
    if args.name is not None:
        outdir = args.name + '_' + outdir
    outdir = os.path.join(args.out, outdir)
    create_output_dir(outdir)
    setup_logging(os.path.join(outdir, 'training.log'))
    dump_json(args, os.path.join(outdir, 'args.json'))
    model.checkpoint_path = os.path.join(outdir, 'best_model_weights.dat')

    dataset = Dataset()
    dataset.load(os.path.join(args.__dict__['in'], 'dataset.dat'))
    print('dataset.flux.shape', dataset.flux.shape)
    print('dataset.params.shape', dataset.params.shape)

    # Split doesn't apply here as models are in order
    # _, ts, vs = dataset.split(args.split)

    #training_generator = KuruczGenerativeAugmenter(ts, labels, coeffs, batch_size=args.batch)
    #validation_generator = KuruczGenerativeAugmenter(vs, labels, coeffs, batch_size=args.batch)

    generator = KuruczGenerativeAugmenter(dataset, labels, coeffs, shuffle=True)
    if args.batch == 0:
        generator.batch_size = generator.input_shape[0]
    else:
        generator.batch_size = args.batch
    training_generator = generator.copy()
    validation_generator = generator.copy()

    logging.info("Data input and labels shape: {}, {}"
                 .format(training_generator.input_shape, training_generator.output_shape))
    logging.info("Validation input and labels shape: {}, {}"
                 .format(validation_generator.input_shape, validation_generator.output_shape))
    #logging.info('data_generator.steps_per_epoch: {}'.format(data_generator.steps_per_epoch()))
    #logging.info('validation_generator.steps_per_epoch: {}'.format(validation_generator.steps_per_epoch()))

    model.ensure_model_created(training_generator.input_shape, training_generator.output_shape)
    model.print()

    model.train(training_generator, validation_generator)
    model.save(os.path.join(outdir, 'model.json'))
    model.save_history(os.path.join(outdir, 'history.csv'))

    # TODO: move this logic to model class
    predict_generator = KuruczGenerativeAugmenter(dataset, labels, coeffs, shuffle=False)
    output = model.predict(predict_generator)
    np.savez(os.path.join(outdir, 'prediction.npz'), output)

    logging.info('Results are written to {}'.format(outdir))

def main(args):
    setup_logging()
    train_dnn(args)

if __name__ == "__main__":
    main()
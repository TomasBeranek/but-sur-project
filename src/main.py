from ikrlib import wav16khz2mfcc, png2fea
from audio_gmm import ModelAudioGMM
from image_lin_classifier import ModelImageLinClassifier
from image_lin_regression import ModelImageLinRegression
from image_cnn import ModelImageCNN
import sys
from scipy import mean
import numpy as np

def train(train_target_dir, test_target_dir, train_non_target_dir, test_non_target_dir, verbose=True, cnn_file=None, gmm_file=None):
    if verbose:
        print("NOTE: Running training on\n\t-train_target_dir: '%s'\n\t-train_non_target_dir: '%s'\n\t-train_non_target_dir: '%s'\n\t-test_non_target_dir: '%s'" % (train_target_dir, test_target_dir, train_non_target_dir, test_non_target_dir))

    # # load audio data
    # audio_train_t = wav16khz2mfcc(train_target_dir, verbose)
    # audio_train_n = wav16khz2mfcc(train_non_target_dir, verbose)
    # audio_test_t  = wav16khz2mfcc(test_target_dir, verbose)
    # audio_test_n  = wav16khz2mfcc(test_non_target_dir, verbose)
    #
    # # audio data preprocessing
    #
    # # init audio models
    # model_audio_gmm = ModelAudioGMM(M_t=3, M_n=20, train_cycles=50, verbose=verbose)
    # model_audio_gmm.train(audio_train_t.values(), audio_train_n.values())
    #
    # # test audio models
    # results_audio_gmm_t = model_audio_gmm.test(audio_test_t, 0.5)
    # results_audio_gmm_n = model_audio_gmm.test(audio_test_n, 0.5)
    #
    # # print audio results
    # print_results(results_audio_gmm_t, "AudioGMM", "Target")
    # print_results(results_audio_gmm_n, "AudioGMM", "Non-target")

    # load image data
    image_train_t = png2fea(train_target_dir, verbose)
    image_train_n = png2fea(train_non_target_dir, verbose)
    image_test_t  = png2fea(test_target_dir, verbose)
    image_test_n  = png2fea(test_non_target_dir, verbose)

    # # init image models
    # model_image_lin_classifier = ModelImageLinClassifier(dimensions=45)
    # model_image_lin_classifier.train(list(image_train_t.values()), list(image_train_n.values()))
    #
    # # TODO: if the number of training epochs for linear regression is higher than +-40, the code fails
    # model_image_lin_regression = ModelImageLinRegression(dimensions=model_image_lin_classifier.dimensions, verbose=verbose)
    # model_image_lin_regression.train(list(image_train_t.values()), list(image_train_n.values()), epochs=5, init_w=model_image_lin_classifier.w, init_w0=model_image_lin_classifier.w0)

    model_image_cnn = ModelImageCNN(verbose=verbose)
    model_image_cnn.train(list(image_train_t.values()), list(image_train_n.values()), list(image_test_t.values()), list(image_test_n.values()), batch_size=16, epochs=25, path=cnn_file)

    # # test image models
    # results_model_image_lin_classifier_t = model_image_lin_classifier.test(image_test_t)
    # results_model_image_lin_classifier_n = model_image_lin_classifier.test(image_test_n)
    #
    # results_model_image_lin_regression_t = model_image_lin_regression.test(image_test_t)
    # results_model_image_lin_regression_n = model_image_lin_regression.test(image_test_n)

    results_mode_image_cnn_t = model_image_cnn.test(image_test_t)
    results_mode_image_cnn_n = model_image_cnn.test(image_test_n)

    # # print image results
    # print_results(results_model_image_lin_classifier_t, "ImageLinClassifier", "Target")
    # print_results(results_model_image_lin_classifier_n, "ImageLinClassifier", "Non-target")
    #
    # print_results(results_model_image_lin_regression_t, "ImageLinRegression", "Target")
    # print_results(results_model_image_lin_regression_n, "ImageLinRegression", "Non-target")

    print_results(results_mode_image_cnn_t, "ImageCNN", "Target")
    print_results(results_mode_image_cnn_n, "ImageCNN", "Non-target")

    # optionally save models
    # cnn model is saved in train method

    # if gmm_file:
    #     model_audio_gmm.save(gmm_file)


def print_results(results, model_name, data_type=None):
    precision = None

    if data_type == "Target":
        precision = len([pred for _,pred in results.values() if pred]) / len(results)
    elif data_type == "Non-target":
        precision = len([pred for _,pred in results.values() if not pred]) / len(results)
    else:
        detections = len([pred for _,pred in results.values() if pred]) / len(results)
        non_detections =  1 - detections

    if data_type:
        print("Model: %s     Samples: %s     Precision: %.2f %%" % (model_name, data_type, (100*precision)))
    else:
        print("Model: %s     Samples: %s     Detections: %.2f %%     Non-detections: %.2f %%" % (model_name, data_type, (100*detections), (100*non_detections)))

    for file, (log_prob, decision) in results.items():
        file = file.split('/')[-1].split('.')[0]
        print("%s %f %d" % (file, log_prob, decision))


def evaluate(test_dir, verbose=False, cnn_file=None, gmm_file=None):
    if not cnn_file and not gmm_file:
        print("ERROR: Missing models paths!")
        exit(1)

    if verbose:
        print("NOTE: Running evaluation on '%s'." % test_dir)

    if cnn_file:
        model_image_cnn = ModelImageCNN(verbose=verbose)
        model_image_cnn.load(cnn_file)
        image = png2fea(test_dir, verbose)
        results_cnn = model_image_cnn.test(image)
        print_results(results_cnn, "ImageCNN")

    if gmm_file:
        model_audio_gmm = ModelAudioGMM(verbose=verbose)
        model_audio_gmm.load(gmm_file)
        audio = wav16khz2mfcc(test_dir, verbose)
        results_gmm = model_audio_gmm.test(audio, 0.5)
        print_results(results_gmm, "AudioGMM")


if __name__ == '__main__':
    verbose = False
    cnn_file = None
    gmm_file = None

    if '--verbose' in sys.argv:
        verbose = True
        sys.argv.remove('--verbose')

    if '--cnn-file' in sys.argv:
        idx = sys.argv.index('--cnn-file')
        cnn_file = sys.argv[idx+1]
        del sys.argv[idx:idx+2]
        print(cnn_file)

    if '--gmm-file' in sys.argv:
        idx = sys.argv.index('--gmm-file')
        gmm_file = sys.argv[idx+1]
        del sys.argv[idx:idx+2]

    if '--test' in sys.argv:
        sys.argv.remove('--test')
        # only evaluate without ground truth
        evaluate(sys.argv[1], verbose=verbose, cnn_file=cnn_file, gmm_file=gmm_file)
    elif '--train' in sys.argv:
        sys.argv.remove('--train')
        # train and evaluate with GT
        train(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], verbose=verbose, cnn_file=cnn_file, gmm_file=gmm_file)
    else:
        print("ERROR: Incorrect number of arguments!", file=sys.stderr)
        exit(1)

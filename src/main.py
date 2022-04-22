from ikrlib import wav16khz2mfcc, png2fea
from audio_gmm import ModelAudioGMM
from image_lin_classifier import ModelImageLinClassifier
import sys
from scipy import mean
import numpy as np

def train(train_target_dir, test_target_dir, train_non_target_dir, test_non_target_dir, verbose=True):
    if verbose:
        print("NOTE: Running training on\n\t-train_target_dir: '%s'\n\t-train_non_target_dir: '%s'\n\t-train_non_target_dir: '%s'\n\t-test_non_target_dir: '%s'" % (train_target_dir, test_target_dir, train_non_target_dir, test_non_target_dir))

    # load audio data
    """audio_train_t = wav16khz2mfcc(train_target_dir, verbose)
    audio_train_n = wav16khz2mfcc(train_non_target_dir, verbose)
    audio_test_t  = wav16khz2mfcc(test_target_dir, verbose)
    audio_test_n  = wav16khz2mfcc(test_non_target_dir, verbose)

    # audio data preprocessing

    # init audio models
    model_audio_gmm = ModelAudioGMM(M_t=3, M_n=20, train_cycles=50, verbose=verbose)
    model_audio_gmm.train(audio_train_t.values(), audio_train_n.values())

    # test audio models
    results_audio_gmm_t = model_audio_gmm.test(audio_test_t, 0.5)
    results_audio_gmm_n = model_audio_gmm.test(audio_test_n, 0.5)

    # print audio results
    print_results(results_audio_gmm_t, "AudioGMM", "Target")
    print_results(results_audio_gmm_n, "AudioGMM", "Non-target")"""

    # load image data
    image_train_t = png2fea(train_target_dir, verbose)
    image_train_n = png2fea(train_non_target_dir, verbose)
    image_test_t  = png2fea(test_target_dir, verbose)
    image_test_n  = png2fea(test_non_target_dir, verbose)

    # init image models
    model_image_lin_classifier = ModelImageLinClassifier(dimensions=40)
    model_image_lin_classifier.train(list(image_train_t.values()), list(image_train_n.values()))

    # test image models
    results_model_image_lin_classifier_t = model_image_lin_classifier.test(image_test_t)
    results_model_image_lin_classifier_n = model_image_lin_classifier.test(image_test_n)

    # print image results
    print_results(results_model_image_lin_classifier_t, "ImageLinClassifier", "Target")
    print_results(results_model_image_lin_classifier_n, "ImageLinClassifier", "Non-target")


def print_results(results, model_name, data_type):
    precision = None

    if data_type == "Target":
        precision = len([pred for _,pred in results.values() if pred]) / len(results)
    elif data_type == "Non-target":
        precision = len([pred for _,pred in results.values() if not pred]) / len(results)

    print("\nModel: %s     Samples: %s     Precision: %.2f %%" % (model_name, data_type, (100*precision)))
    for file, (log_prob, decision) in results.items():
        file = file.split('/')[-1].split('.')[0]
        #print("%s %f %d" % (file, log_prob, decision))


def evaluate(test_dir, verbose=True):
    print("NOTE: Running evaluation on '%s'." % test_dir)

if __name__ == '__main__':
    verbose = False

    if '-v' in sys.argv:
        verbose = True
        sys.argv.remove('-v')

    if len(sys.argv) < 3:
        # only evaluate without ground truth
        evaluate(sys.argv[1], verbose=verbose)
    elif len(sys.argv) == 5:
        # train and evaluate with GT
        train(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], verbose=verbose)
    else:
        print("ERROR: Incorrect number of arguments! Try 'python3 main.py train_target_dir test_target_dir train_non_target_dir test_non_target_dir'", file=sys.stderr)

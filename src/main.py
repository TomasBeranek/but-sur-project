from ikrlib import wav16khz2mfcc
from audio_gmm import ModelAudioGMM
import sys
from scipy import mean

def train(train_target_dir, test_target_dir, train_non_target_dir, test_non_target_dir, verbose=True):
    if verbose:
        print("NOTE: Running training on\n\t-train_target_dir: '%s'\n\t-train_non_target_dir: '%s'\n\t-train_non_target_dir: '%s'\n\t-test_non_target_dir: '%s'" % (train_target_dir, test_target_dir, train_non_target_dir, test_non_target_dir))

    # load audio data
    train_t = wav16khz2mfcc(train_target_dir, verbose)
    train_n = wav16khz2mfcc(train_non_target_dir, verbose)
    test_t  = wav16khz2mfcc(test_target_dir, verbose)
    test_n  = wav16khz2mfcc(test_non_target_dir, verbose)

    # audio data preprocessing

    # init audio models
    model_audio_gmm = ModelAudioGMM(M_t=3, M_n=20, train_cycles=50, verbose=verbose)
    model_audio_gmm.train(train_t.values(), train_n.values())

    # load image data

    # test audio models
    results_audio_gmm_t = model_audio_gmm.test(test_t, 0.5)
    results_audio_gmm_n = model_audio_gmm.test(test_n, 0.5)

    # print results
    precision = len([pred for _,pred in results_audio_gmm_t.values() if pred]) / len(results_audio_gmm_t)
    print("\nModel: AudioGMM     Samples: Target     Precision: %.2f %%" % (100*precision))
    for file, result in results_audio_gmm_t.items():
        log_prob, decision = result
        file = file.split('/')[-1].split('.')[0]
        print("%s %f %d" % (file, log_prob, decision))

    precision = len([pred for _,pred in results_audio_gmm_n.values() if not pred]) / len(results_audio_gmm_n)
    print("\nModel: AudioGMM    Samples: Non-target     Precision: %.2f %%" % (100*precision))
    for file, result in results_audio_gmm_n.items():
        log_prob, decision = result
        file = file.split('/')[-1].split('.')[0]
        print("%s %f %d" % (file, log_prob, decision))

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

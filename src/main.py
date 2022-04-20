from ikrlib import wav16khz2mfcc
from audio_gmm import ModelAudioGMM
import sys

def train(train_target_dir, test_target_dir, train_non_target_dir, test_non_target_dir):
    print("NOTE: Running training on\n\t-train_target_dir: '%s'\n\t-train_non_target_dir: '%s'\n\t-train_non_target_dir: '%s'\n\t-test_non_target_dir: '%s'" % (train_target_dir, test_target_dir, train_non_target_dir, test_non_target_dir))

    # load audio data
    train_t = wav16khz2mfcc(train_target_dir).values()
    train_n = wav16khz2mfcc(train_non_target_dir).values()
    test_t  = wav16khz2mfcc(test_target_dir).values()
    test_n  = wav16khz2mfcc(test_non_target_dir).values()

    # audio data preprocessing

    # init audio models
    model_audio_gmm = ModelAudioGMM(M_t=3, M_n=20, train_cycles=30)
    model_audio_gmm.train(train_t, train_n)

    # load image data

    # test audio models
    log_prob_t, labels_t = model_audio_gmm.test(test_t, 0.5)
    log_prob_n, labels_n = model_audio_gmm.test(test_n, 0.5)

    print(labels_t)
    print(labels_n)


def evaluate(test_dir):
    print("NOTE: Running evaluation on '%s'." % test_dir)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        # only evaluate without ground truth
        evaluate(sys.argv[1])
    elif len(sys.argv) == 5:
        # train and evaluate with GT
        train(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("ERROR: Incorrect number of arguments! Try 'python3 main.py train_target_dir test_target_dir train_non_target_dir test_non_target_dir'", file=sys.stderr)

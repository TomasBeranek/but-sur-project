from ikrlib import wav16khz2mfcc, png2fea
from audio_gmm import ModelAudioGMM
from image_lin_classifier import ModelImageLinClassifier
from image_lin_regression import ModelImageLinRegression
from image_cnn import ModelImageCNN
import sys
from scipy import mean
import numpy as np

GMM_M_t=5
GMM_M_n=20
GMM_EPOCHS=10

def train(train_target_dir, test_target_dir, train_non_target_dir, test_non_target_dir, verbose=True, cnn_file=None, gmm_file=None):
    if verbose:
        print("NOTE: Running training on\n\t-train_target_dir: '%s'\n\t-train_non_target_dir: '%s'\n\t-train_non_target_dir: '%s'\n\t-test_non_target_dir: '%s'" % (train_target_dir, test_target_dir, train_non_target_dir, test_non_target_dir))

    # load audio data
    audio_train_t = wav16khz2mfcc(train_target_dir, verbose)
    audio_train_n = wav16khz2mfcc(train_non_target_dir, verbose)
    audio_test_t  = wav16khz2mfcc(test_target_dir, verbose)
    audio_test_n  = wav16khz2mfcc(test_non_target_dir, verbose)

    # init audio models
    model_audio_gmm = ModelAudioGMM(M_t=GMM_M_t, M_n=GMM_M_n, train_cycles=GMM_EPOCHS, verbose=verbose)
    model_audio_gmm.train(audio_train_t.values(), audio_train_n.values())

    # test audio models
    results_audio_gmm_t = model_audio_gmm.test(audio_test_t, 0.5)
    results_audio_gmm_n = model_audio_gmm.test(audio_test_n, 0.5)

    # print audio results
    print_results(results_audio_gmm_t, "AudioGMM", "Target")
    print_results(results_audio_gmm_n, "AudioGMM", "Non-target")

    # load image data
    image_train_t = png2fea(train_target_dir, verbose)
    image_train_n = png2fea(train_non_target_dir, verbose)
    image_test_t  = png2fea(test_target_dir, verbose)
    image_test_n  = png2fea(test_non_target_dir, verbose)

    # init image models
    # model_image_lin_classifier = ModelImageLinClassifier(dimensions=45)
    # model_image_lin_classifier.train(list(image_train_t.values()), list(image_train_n.values()))

    # TODO: if the number of training epochs for linear regression is higher than +-40, the code fails
    # model_image_lin_regression = ModelImageLinRegression(dimensions=model_image_lin_classifier.dimensions, verbose=verbose)
    # model_image_lin_regression.train(list(image_train_t.values()), list(image_train_n.values()), epochs=5, init_w=model_image_lin_classifier.w, init_w0=model_image_lin_classifier.w0)

    model_image_cnn = ModelImageCNN(verbose=verbose)
    model_image_cnn.train(list(image_train_t.values()), list(image_train_n.values()), list(image_test_t.values()), list(image_test_n.values()), batch_size=16, epochs=25, path=cnn_file)

    # test image models
    # results_model_image_lin_classifier_t = model_image_lin_classifier.test(image_test_t)
    # results_model_image_lin_classifier_n = model_image_lin_classifier.test(image_test_n)
    #
    # results_model_image_lin_regression_t = model_image_lin_regression.test(image_test_t)
    # results_model_image_lin_regression_n = model_image_lin_regression.test(image_test_n)

    results_mode_image_cnn_t = model_image_cnn.test(image_test_t)
    results_mode_image_cnn_n = model_image_cnn.test(image_test_n)

    # print image results
    # print_results(results_model_image_lin_classifier_t, "ImageLinClassifier", "Target")
    # print_results(results_model_image_lin_classifier_n, "ImageLinClassifier", "Non-target")
    #
    # print_results(results_model_image_lin_regression_t, "ImageLinRegression", "Target")
    # print_results(results_model_image_lin_regression_n, "ImageLinRegression", "Non-target")

    print_results(results_mode_image_cnn_t, "ImageCNN", "Target")
    print_results(results_mode_image_cnn_n, "ImageCNN", "Non-target")

    # optionally save models
    # cnn model is saved in train method

    if gmm_file:
        model_audio_gmm.save(gmm_file)


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

def mean_target_score(res):
    results = list(res.values())
    positive_scores = [score for score, label in results if score > 0]
    return np.mean(positive_scores)

def min_non_target_score(res):
    results = list(res.values())
    negative_scores = [score for score, label in results if score < 0]
    return min(negative_scores)

def max_target_score(res):
    results = list(res.values())
    positive_scores = [score for score, label in results if score > 0]
    return max(positive_scores)

def final_evaluate(cnns, gmms, test_dir):
    image_data = png2fea(test_dir, verbose=False)
    audio_data = wav16khz2mfcc(test_dir, verbose=False)

    CNN_res = [None]*3
    GMM_res = [None]*3

    for i in range(0, len(cnns)):
        CNN_model = ModelImageCNN(verbose=False)
        CNN_model.load(cnns[i])
        CNN_res[i] = CNN_model.test(image_data)

    for i in range(0, len(gmms)):
        GMM_model = ModelAudioGMM(verbose=False)
        GMM_model.load(gmms[i])
        GMM_res[i] = GMM_model.test(audio_data, 0.5)

    # remove extensions
    files = [''.join(file.split('/')[-1].split('.')[:-1]) for file in image_data.keys()]

    # apply thresholds and normalize scores
    for file in files:
        for i in range(0, len(CNN_res)):
            score, decision = CNN_res[i][file]

            if score > 0.8:
                decision = 1
            elif score < 0.2:
                decision = 0
            else:
                decision = -1

            CNN_res[i][file] = (score, decision)

        for i in range(0, len(GMM_res)):
            score, decision = GMM_res[i][file]
            mean_t_score = mean_target_score(GMM_res[i])

            if score > mean_t_score/4.0:
                decision = 1
            elif score < -100:
                decision = 0
            else:
                decision = -1

            norm_score = 0.5

            if score < 0:
                min_score = min_non_target_score(GMM_res[i])
                norm_score -= (score/min_score)/2
            else:
                max_score = max_target_score(GMM_res[i])
                norm_score += (score/max_score)/2.0

            GMM_res[i][file] = (norm_score, decision)

        final_res = {}

    # make final decision based of individual models's decisions
    for file in files:
        file_score_unknown = []
        file_scores_known = []
        file_t_labels = 0
        file_n_labels = 0
        cnn_vote = []

        for i in range(0,3):
            # CNNs
            score, label = CNN_res[i][file]
            if label == 1:
                file_t_labels += 1
                file_scores_known += [score]
                cnn_vote += [1]
            elif label == 0:
                file_n_labels += 1
                file_scores_known += [score]
                cnn_vote += [0]
            else:
                file_score_unknown += [score]

            #print('\tCNN%d     %f     %d' % (i, score, label))

            # GMMs
            score, label = GMM_res[i][file]
            if label == 1:
                file_t_labels += 1
                file_scores_known += [score]
            elif label == 0:
                file_n_labels += 1
                file_scores_known += [score]
            else:
                file_score_unknown += [score]

            #print('\tGMM%d     %f     %d' % (i, score, label))


        # more models vote for DETECTION
        if file_t_labels > file_n_labels:
            final_res[file] = (np.mean(file_scores_known), 1)
        # more models vote for NON-DETECTION
        elif file_t_labels < file_n_labels:
            final_res[file] = (np.mean(file_scores_known), 0)
        # its a draw, but atleast two models voted
        elif file_t_labels != 0:
            score = np.mean(file_scores_known)

            # if its very close trust CNNs
            if score > 0.45 and score < 0.55:
                cnns_label = np.mean(cnn_vote) > 0.5
                if cnns_label == 1:
                    final_res[file] = (0.5001, 1)
                else:
                    final_res[file] = (0.4999, 0)
            else:
                final_res[file] = (score, score > 0.5)
        # all models didnt vote -- they are not sure
        else:
            final_res[file] = (np.mean(file_score_unknown), np.mean(file_score_unknown) > 0.5)

    # print stats to file
    f = open('voting_model.txt', 'w')
    for file, (score, label) in final_res.items():
        f.write("%s %f %d\n" % (file, score, label))
    f.close()

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
    elif '--final' in sys.argv:
        sys.argv.remove('--final')
        cnns = sys.argv[1:4]
        gmms = sys.argv[4:7]
        test_dir = sys.argv[7]
        final_evaluate(cnns, gmms, test_dir)
    else:
        print("ERROR: Incorrect number of arguments!", file=sys.stderr)
        exit(1)

import sys

def train(train_target_dir, test_target_dir, train_non_target_dir, test_non_target_dir):
    print("NOTE: Running training on\n\t-train_target_dir: '%s'\n\t-train_non_target_dir: '%s'\n\t-train_non_target_dir: '%s'\n\t-test_non_target_dir: '%s'" % (train_target_dir, test_target_dir, train_non_target_dir, test_non_target_dir))

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
